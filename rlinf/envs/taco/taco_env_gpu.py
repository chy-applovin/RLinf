# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPU-batched TACO environment backed by MuJoCo Warp (mjwarp).

Selected with ``env.*.sim_backend: gpu`` (default ``cpu`` keeps the original
``TacoEnv`` untouched). One mjwarp model is replicated across ``num_envs``
worlds on a single CUDA device; physics, observation synthesis (point-cloud
re-posing) and the tracking reward are all computed batched in torch on GPU.

v1 limitations (asserted, documented):

* all worlds share ONE episode (mjwarp batches identical models) - select a
  single episode via ``episodes: [name]``;
* reward types: ``tracking`` / ``zero`` (batched re-implementation with the
  same semantics and config keys as the CPU ``TrackingReward``);
* no video capture (``video_cfg.save_video`` must stay False);
* mjwarp contact physics is numerically close to, but not bit-identical
  with, CPU MuJoCo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import gymnasium as gym
import mujoco
import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.envs.taco.scene import (
    HAND_DIM,
    TARGET_OBJ_QPOS,
    TOOL_OBJ_QPOS,
    load_episode_data,
    select_episodes,
)

__all__ = ["TacoEnvGPU"]


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Batched MuJoCo quaternion (w, x, y, z) -> rotation matrices (N, 3, 3)."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


class TacoEnvGPU(gym.Env):
    """mjwarp-batched TACO env exposing the same interface as ``TacoEnv``."""

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info=None,
        record_metrics: bool = True,
    ):
        import warp as wp

        try:
            wp.init()
        except RuntimeError:
            pass
        self._wp = wp

        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.seed = int(cfg.seed) + int(seed_offset)
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.record_metrics = record_metrics

        assert not bool(cfg.auto_reset), "TacoEnvGPU supports auto_reset=False only"
        assert not bool(cfg.video_cfg.save_video), (
            "TacoEnvGPU does not support video capture (use sim_backend: cpu)"
        )
        self.ignore_terminations = bool(cfg.ignore_terminations)
        self.group_size = int(cfg.group_size)

        self.obs_mode = str(cfg.obs_mode)
        assert self.obs_mode in ("pc_qpos", "pc2_qpos")
        self.need_tool_cloud = self.obs_mode == "pc2_qpos"
        self.obs_horizon = int(cfg.obs_horizon)
        self.num_points = int(cfg.num_points)
        self.max_episode_steps = int(cfg.max_episode_steps)
        self.success_threshold_m = float(cfg.get("success_threshold_m", 0.1))

        # ----------------------------------------------------------- episode
        episode_names = cfg.get("episodes", None)
        episode_dirs = select_episodes(
            Path(cfg.dataset_root),
            str(cfg.trajectory_file),
            names=list(episode_names) if episode_names is not None else None,
            categories=list(cfg.categories) if cfg.get("categories") else None,
            use_one_per_category=bool(cfg.get("one_per_category", False)),
            max_episodes=cfg.get("max_episodes", None),
        )
        assert len(episode_dirs) == 1, (
            "sim_backend=gpu batches identical worlds and currently supports "
            f"exactly ONE episode per worker; got {len(episode_dirs)}. "
            "Set env.*.episodes: [<episode_name>]."
        )
        self.episode = load_episode_data(
            episode_dirs[0],
            Path(cfg.scene_root) / f"gpu_proc_{seed_offset}",
            Path(cfg.allegro_assets_root),
            str(cfg.trajectory_file),
            self.num_points,
            self.need_tool_cloud,
        )
        self.ep_len = min(self.max_episode_steps, self.episode.num_frames - 1)

        # ------------------------------------------------------------ reward
        reward_cfg = cfg.get("reward", None)
        if reward_cfg is not None and OmegaConf.is_config(reward_cfg):
            reward_cfg = OmegaConf.to_container(reward_cfg, resolve=True)
        reward_cfg = dict(reward_cfg) if reward_cfg else {}
        self.reward_type = str(reward_cfg.get("type", "tracking"))
        assert self.reward_type in ("tracking", "zero"), (
            f"TacoEnvGPU supports tracking/zero rewards, got {self.reward_type}"
        )
        self.w_tool = float(reward_cfg.get("tool_pos_weight", 1.0))
        self.w_target = float(reward_cfg.get("target_pos_weight", 1.0))
        self.w_hand = float(reward_cfg.get("hand_qpos_weight", 0.1))
        self.s_tool = float(reward_cfg.get("tool_pos_scale", 0.05))
        self.s_target = float(reward_cfg.get("target_pos_scale", 0.05))
        self.s_hand = float(reward_cfg.get("hand_qpos_scale", 0.5))
        self.reward_norm = (
            (self.w_tool + self.w_target + self.w_hand)
            if bool(reward_cfg.get("normalize_by_weights", True))
            else 1.0
        )

        # --------------------------------------------------------- simulator
        self.device = torch.device("cuda")
        self._wp_device = str(wp.get_device())
        self._use_cuda_graph = bool(cfg.get("gpu_cuda_graph", True))
        self._nconmax = int(cfg.get("gpu_nconmax_per_env", 100))
        self._njmax = int(cfg.get("gpu_njmax_per_env", 350))

        self.model_cpu = mujoco.MjModel.from_xml_path(self.episode.scene_xml)
        assert self.model_cpu.nu == HAND_DIM
        self.substeps = max(
            1,
            round((1.0 / self.episode.frequency) / float(self.model_cpu.opt.timestep)),
        )
        self._build_warp_env()

        # ------------------------------------------------- cached GPU tensors
        f32 = dict(dtype=torch.float32, device=self.device)
        self.demo_qpos = torch.as_tensor(self.episode.qpos_demo, **f32)  # (T, 58)
        self.target_local = torch.as_tensor(self.episode.target_local, **f32)
        self.tool_local = (
            torch.as_tensor(self.episode.tool_local, **f32)
            if self.need_tool_cloud
            else None
        )

        # rolling obs history buffers (B, To, ...)
        b, to, k = self.num_envs, self.obs_horizon, self.num_points
        self._hist_qpos = torch.zeros(b, to, HAND_DIM, **f32)
        self._hist_pc = torch.zeros(b, to, k, 3, **f32)
        self._hist_tool_pc = (
            torch.zeros(b, to, k, 3, **f32) if self.need_tool_cloud else None
        )

        # metrics
        self.steps = 0
        self._returns = torch.zeros(b, **f32)
        self._final_tool_err = torch.full((b,), float("nan"), **f32)
        self._final_target_err = torch.full((b,), float("nan"), **f32)
        self._final_hand_err = torch.full((b,), float("nan"), **f32)
        self._is_start = True

    # --------------------------------------------------------------- warp env
    def _build_warp_env(self) -> None:
        import mujoco_warp as mjwarp

        wp = self._wp
        data_cpu = mujoco.MjData(self.model_cpu)
        data_cpu.qpos[:] = self.episode.qpos_demo[0]
        data_cpu.qvel[:] = self.episode.qvel_demo[0]
        mujoco.mj_forward(self.model_cpu, data_cpu)
        self._data_cpu = data_cpu

        with wp.ScopedDevice(self._wp_device):
            self.model_wp = mjwarp.put_model(self.model_cpu)
            self.data_wp = mjwarp.put_data(
                self.model_cpu,
                data_cpu,
                nworld=self.num_envs,
                nconmax=self._nconmax * self.num_envs,
                njmax=self._njmax,
            )
            # Warm-up step OUTSIDE capture: kernel modules cannot be loaded
            # while CUDA stream capture is active. State is restored by the
            # reset() that precedes every rollout.
            mjwarp.step(self.model_wp, self.data_wp)
            wp.synchronize()
            self._graph = None
            if self._use_cuda_graph:
                try:
                    with wp.ScopedCapture() as capture:
                        mjwarp.step(self.model_wp, self.data_wp)
                    wp.synchronize()
                    self._graph = capture.graph
                except RuntimeError as e:
                    # e.g. "Conditional graph nodes require CUDA driver 12.4+"
                    import logging

                    logging.getLogger(__name__).warning(
                        "CUDA graph capture unavailable (%s); "
                        "falling back to direct mjwarp.step launches.",
                        e,
                    )
                    self._graph = None
        self._mjwarp = mjwarp

    def _reset_warp_state(self) -> None:
        """Reset all worlds to demo frame 0 and recompute derived quantities."""
        wp = self._wp
        b = self.num_envs
        f32 = dict(dtype=torch.float32, device=self.device)
        qpos0 = torch.as_tensor(self.episode.qpos_demo[0], **f32).repeat(b, 1)
        qvel0 = torch.as_tensor(self.episode.qvel_demo[0], **f32).repeat(b, 1)
        with wp.ScopedDevice(self._wp_device):
            wp.copy(self.data_wp.qpos, wp.from_torch(qpos0.contiguous()))
            wp.copy(self.data_wp.qvel, wp.from_torch(qvel0.contiguous()))
            wp.copy(
                self.data_wp.ctrl,
                wp.from_torch(torch.zeros(b, HAND_DIM, **f32)),
            )
            wp.copy(
                self.data_wp.qacc_warmstart,
                wp.from_torch(torch.zeros(b, self.model_cpu.nv, **f32)),
            )
            wp.copy(self.data_wp.time, wp.from_torch(torch.zeros(b, **f32)))
            self._mjwarp.forward(self.model_wp, self.data_wp)
            wp.synchronize()

    # ------------------------------------------------------------------ props
    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = value

    def update_reset_state_ids(self) -> None:
        """Single-episode backend: nothing to resample."""

    # ------------------------------------------------------------------ state
    def _qpos(self) -> torch.Tensor:
        """(B, nq) live qpos as a torch CUDA view (zero-copy)."""
        return self._wp.to_torch(self.data_wp.qpos)

    def _push_obs_frame(self, qpos: torch.Tensor) -> None:
        """Append the current frame to the rolling obs history (batched)."""
        hand = qpos[:, :HAND_DIM]
        p_t = qpos[:, TARGET_OBJ_QPOS.start : TARGET_OBJ_QPOS.start + 3]
        r_t = _quat_to_rotmat(qpos[:, TARGET_OBJ_QPOS.start + 3 : TARGET_OBJ_QPOS.stop])
        # (B, K, 3) = local (K,3) @ R^T (B,3,3) + p
        pc = torch.einsum("kj,bij->bki", self.target_local, r_t) + p_t[:, None, :]

        self._hist_qpos = torch.roll(self._hist_qpos, -1, dims=1)
        self._hist_qpos[:, -1] = hand
        self._hist_pc = torch.roll(self._hist_pc, -1, dims=1)
        self._hist_pc[:, -1] = pc
        if self.need_tool_cloud:
            p_r = qpos[:, TOOL_OBJ_QPOS.start : TOOL_OBJ_QPOS.start + 3]
            r_r = _quat_to_rotmat(
                qpos[:, TOOL_OBJ_QPOS.start + 3 : TOOL_OBJ_QPOS.stop]
            )
            tool_pc = (
                torch.einsum("kj,bij->bki", self.tool_local, r_r) + p_r[:, None, :]
            )
            self._hist_tool_pc = torch.roll(self._hist_tool_pc, -1, dims=1)
            self._hist_tool_pc[:, -1] = tool_pc

    def _fill_obs_history(self, qpos: torch.Tensor) -> None:
        for _ in range(self.obs_horizon):
            self._push_obs_frame(qpos)

    def _wrap_obs(self) -> dict[str, torch.Tensor]:
        obs = {
            "states": self._hist_qpos.clone(),
            "pointcloud": self._hist_pc.clone(),
        }
        if self.need_tool_cloud:
            obs["tool_pointcloud"] = self._hist_tool_pc.clone()
        return obs

    # ------------------------------------------------------------------ errors
    def _errors(self, qpos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Frame-aligned (tool_err, target_err, hand_err), each (B,)."""
        demo = self.demo_qpos[min(self.steps, self.episode.num_frames - 1)]
        tool_err = torch.linalg.norm(
            qpos[:, TOOL_OBJ_QPOS.start : TOOL_OBJ_QPOS.start + 3]
            - demo[TOOL_OBJ_QPOS.start : TOOL_OBJ_QPOS.start + 3],
            dim=1,
        )
        target_err = torch.linalg.norm(
            qpos[:, TARGET_OBJ_QPOS.start : TARGET_OBJ_QPOS.start + 3]
            - demo[TARGET_OBJ_QPOS.start : TARGET_OBJ_QPOS.start + 3],
            dim=1,
        )
        hand_err = (qpos[:, :HAND_DIM] - demo[:HAND_DIM]).abs().mean(dim=1)
        return tool_err, target_err, hand_err

    def _compute_rewards(self, qpos: torch.Tensor) -> torch.Tensor:
        if self.reward_type == "zero":
            return torch.zeros(self.num_envs, device=self.device)
        tool_err, target_err, hand_err = self._errors(qpos)
        reward = (
            self.w_tool * torch.exp(-tool_err / self.s_tool)
            + self.w_target * torch.exp(-target_err / self.s_target)
            + self.w_hand * torch.exp(-hand_err / self.s_hand)
        ) / self.reward_norm
        return reward

    # ------------------------------------------------------------------- reset
    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = None,
    ):
        self._reset_warp_state()
        self.steps = 0
        self._returns.zero_()
        for buf in (self._final_tool_err, self._final_target_err, self._final_hand_err):
            buf.fill_(float("nan"))
        qpos = self._qpos()
        self._fill_obs_history(qpos)
        return self._wrap_obs(), {}

    # -------------------------------------------------------------------- step
    def step(self, actions, build_obs: bool = True):
        wp = self._wp
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        actions = actions.to(device=self.device, dtype=torch.float32)
        assert actions.shape == (self.num_envs, HAND_DIM)

        done_before = self.steps >= self.ep_len
        with wp.ScopedDevice(self._wp_device):
            wp.copy(self.data_wp.ctrl, wp.from_torch(actions.contiguous()))
            for _ in range(self.substeps):
                if self._graph is not None:
                    wp.capture_launch(self._graph)
                else:
                    self._mjwarp.step(self.model_wp, self.data_wp)
            wp.synchronize()

        if not done_before:
            self.steps += 1
        qpos = self._qpos()
        rewards = self._compute_rewards(qpos)
        if done_before:
            rewards = torch.zeros_like(rewards)
        self._returns += rewards
        self._push_obs_frame(qpos)

        done_now = self.steps >= self.ep_len
        if done_now and torch.isnan(self._final_tool_err).any():
            tool_err, target_err, hand_err = self._errors(qpos)
            self._final_tool_err.copy_(tool_err)
            self._final_target_err.copy_(target_err)
            self._final_hand_err.copy_(hand_err)

        truncations = torch.full((self.num_envs,), done_now, dtype=torch.bool)
        terminations = torch.zeros(self.num_envs, dtype=torch.bool)

        infos: dict[str, Any] = {}
        if self.record_metrics:
            infos["episode"] = self._episode_metrics()
        obs = self._wrap_obs() if build_obs else None
        return obs, rewards.float().cpu(), terminations, truncations, infos

    def chunk_step(self, chunk_actions):
        if isinstance(chunk_actions, np.ndarray):
            chunk_actions = torch.from_numpy(chunk_actions)
        chunk_actions = chunk_actions.to(device=self.device, dtype=torch.float32)
        assert chunk_actions.ndim == 3 and chunk_actions.shape[2] == HAND_DIM
        chunk_size = chunk_actions.shape[1]

        obs_list, infos_list, chunk_rewards = [], [], []
        raw_terminations, raw_truncations = [], []
        for i in range(chunk_size):
            is_last = i == chunk_size - 1
            obs, rewards, terminations, truncations, infos = self.step(
                chunk_actions[:, i], build_obs=is_last
            )
            obs_list.append(obs)
            infos_list.append(infos)
            chunk_rewards.append(rewards)
            raw_terminations.append(terminations)
            raw_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)
        raw_terminations = torch.stack(raw_terminations, dim=1)
        raw_truncations = torch.stack(raw_truncations, dim=1)

        chunk_terminations = torch.zeros_like(raw_terminations)
        chunk_terminations[:, -1] = raw_terminations.any(dim=1)
        chunk_truncations = torch.zeros_like(raw_truncations)
        chunk_truncations[:, -1] = raw_truncations.any(dim=1)
        return obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list

    # ----------------------------------------------------------------- metrics
    def _episode_metrics(self) -> dict[str, torch.Tensor]:
        rets = self._returns.float().cpu()
        lens = torch.full((self.num_envs,), float(max(self.steps, 1)))
        tool_err = self._final_tool_err.float().cpu()
        target_err = self._final_target_err.float().cpu()
        hand_err = self._final_hand_err.float().cpu()
        done = self.steps >= self.ep_len
        success = (tool_err < self.success_threshold_m) & torch.full(
            (self.num_envs,), done
        )
        return {
            "return": rets,
            "episode_len": lens,
            "reward": rets / lens,
            "success_once": success,
            "tool_pos_err_final_m": tool_err,
            "target_pos_err_final_m": target_err,
            "hand_qpos_err_final": hand_err,
        }

    def close(self):
        pass
