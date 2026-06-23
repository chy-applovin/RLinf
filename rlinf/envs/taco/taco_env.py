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

"""Vectorized TACO bimanual-Allegro MuJoCo environment.

One ``TacoEnv`` hosts ``num_envs`` CPU MuJoCo sub-simulations. Each sub-env is
instantiated from one TACO episode folder (its own ``scene.xml`` + objects) and
is initialized from the demonstration's first frame. Actions are 44-dim Allegro
hand qpos targets written to the position actuators at the demo control rate
(30 Hz, ~3 physics substeps of 10 ms each).

Observations match the flow-matching policy's IL training distribution:

* ``states``           - (B, To, 44) hand qpos history;
* ``pointcloud``       - (B, To, K, 3) target-object cloud, synthesized by
  re-posing the canonical (demo frame-0, object-frame) cloud with the live sim
  object pose;
* ``tool_pointcloud``  - (B, To, K, 3), only when ``obs_mode == pc2_qpos``.

Episodes are fixed-length: truncation at ``min(max_episode_steps, T_demo - 1)``
control steps (no early termination - the success signal is logged as a metric,
and reward design is pluggable; see ``rlinf/envs/taco/rewards.py``).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import gymnasium as gym
import mujoco
import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.envs.taco.deepmimic import EarlyTermination, RSISampler
from rlinf.envs.taco.rewards import BaseTacoReward, build_reward
from rlinf.envs.taco.robots import get_robot_spec
from rlinf.envs.taco.scene import (
    EpisodeData,
    load_episode_data,
    select_episodes,
    synth_obs_frame,
)
from rlinf.envs.taco.single_step import DemoTimestepSampler, SingleStepSampler

__all__ = ["TacoEnv"]


@dataclass
class _SubEnv:
    """One MuJoCo sub-simulation bound to a TACO episode."""

    episode: EpisodeData
    model: mujoco.MjModel
    data: mujoco.MjData
    substeps: int
    ep_len: int  # control steps until truncation
    steps: int = 0
    # DeepMimic RSI: episode starts at this demo frame (0 = original behavior)
    start_frame: int = 0
    obs_hist: list[dict[str, np.ndarray]] = field(default_factory=list)
    # metric accumulators
    ret: float = 0.0
    done: bool = False
    terminated: bool = False  # DeepMimic ET: failure termination (vs truncation)
    final_tool_err: float = float("nan")
    final_target_err: float = float("nan")
    final_hand_err: float = float("nan")
    renderer: Optional[mujoco.Renderer] = None

    def demo_qpos(self, executed_steps: int) -> np.ndarray:
        """Frame-aligned demo state: executed step k <-> frame min(s0+k, T-1).

        Dataset convention is ``action_offset=1``: the action executed at step
        ``k`` targets demo frame ``s0 + k`` where ``s0`` is the episode's RSI
        start frame (0 without RSI).
        """
        t = min(self.start_frame + executed_steps, self.episode.num_frames - 1)
        return self.episode.qpos_demo[t]

    @property
    def demo_frame(self) -> int:
        """Frame-aligned demo index of the current state."""
        return min(self.start_frame + self.steps, self.episode.num_frames - 1)

    def obs_frame(self, need_tool: bool) -> dict[str, np.ndarray]:
        """Single-frame observation synthesized from the live sim state."""
        return synth_obs_frame(self.data.qpos, self.episode, need_tool)


class TacoEnv(gym.Env):
    """Vectorized TACO env following the RLinf embodied env conventions.

    Required cfg fields (see ``examples/embodiment/config/env/taco_brush_allegro.yaml``):
    dataset_root, allegro_assets_root, scene_root, trajectory_file, obs_mode,
    obs_horizon, num_points, seed, group_size, max_episode_steps, auto_reset,
    ignore_terminations, reward (dict), video_cfg.
    """

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info=None,
        record_metrics: bool = True,
    ):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.seed = int(cfg.seed) + int(seed_offset)
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.record_metrics = record_metrics

        self.auto_reset = bool(cfg.auto_reset)
        if self.auto_reset:
            raise NotImplementedError(
                "TacoEnv currently supports auto_reset=False only "
                "(fixed-length episodes per rollout epoch)."
            )
        self.ignore_terminations = bool(cfg.ignore_terminations)
        self.group_size = int(cfg.group_size)
        assert self.num_envs % self.group_size == 0, (
            f"num_envs={self.num_envs} not divisible by group_size={self.group_size}"
        )
        self.num_group = self.num_envs // self.group_size
        self.use_fixed_reset_state_ids = bool(
            cfg.get("use_fixed_reset_state_ids", True)
        )

        # ------------------------------------------------ obs / policy contract
        self.obs_mode = str(cfg.obs_mode)
        assert self.obs_mode in ("pc_qpos", "pc2_qpos"), (
            f"Unsupported obs_mode {self.obs_mode}"
        )
        self.need_tool_cloud = self.obs_mode == "pc2_qpos"
        self.obs_horizon = int(cfg.obs_horizon)
        self.num_points = int(cfg.num_points)
        self.max_episode_steps = int(cfg.max_episode_steps)
        self.success_threshold_m = float(cfg.get("success_threshold_m", 0.1))

        # ------------------------------------------------------------- episodes
        dataset_root = Path(cfg.dataset_root)
        self.trajectory_file = str(cfg.trajectory_file)
        # robot qpos layout + asset materialization (rlinf/envs/taco/robots.py)
        self.spec = get_robot_spec(str(cfg.get("robot", "allegro")))
        episode_names = cfg.get("episodes", None)
        if episode_names is not None:
            episode_names = list(episode_names)
        categories = cfg.get("categories", None)
        if categories is not None:
            categories = list(categories)
        self.episode_dirs = select_episodes(
            dataset_root,
            self.trajectory_file,
            names=episode_names,
            categories=categories,
            use_one_per_category=bool(cfg.get("one_per_category", False)),
            max_episodes=cfg.get("max_episodes", None),
        )
        # scene_root holds the symlinked Spider asset layout; one root per
        # process is enough (links are content-stable across episodes).
        scene_root = Path(cfg.scene_root) / f"proc_{seed_offset}"
        self._scene_root = scene_root
        # Prefer the generic robot_assets_root; fall back to the legacy
        # allegro_assets_root so existing allegro configs / R01 overrides work.
        self._robot_assets = Path(
            cfg.get("robot_assets_root", None) or cfg.allegro_assets_root
        )
        self._episode_cache: dict[str, tuple[EpisodeData, mujoco.MjModel]] = {}

        # ------------------------------------------------------------- reward
        reward_cfg = cfg.get("reward", None)
        if reward_cfg is not None and OmegaConf.is_config(reward_cfg):
            reward_cfg = OmegaConf.to_container(reward_cfg, resolve=True)
        self.reward_fn: BaseTacoReward = build_reward(reward_cfg)

        # ------------------------------------- DeepMimic aids (default off)
        rsi_cfg = cfg.get("rsi", None)
        if rsi_cfg is not None and OmegaConf.is_config(rsi_cfg):
            rsi_cfg = OmegaConf.to_container(rsi_cfg, resolve=True)
        self.rsi = RSISampler(rsi_cfg, seed=self.seed + 1)

        et_cfg = cfg.get("early_termination", None)
        if et_cfg is not None and OmegaConf.is_config(et_cfg):
            et_cfg = OmegaConf.to_container(et_cfg, resolve=True)
        self._et_cfg = dict(et_cfg) if et_cfg else {}
        self.et_enabled = bool(self._et_cfg.get("enabled", False))
        if self.et_enabled:
            assert not self.ignore_terminations, (
                "early_termination requires ignore_terminations=False "
                "(terminations must reach the loss mask / GAE)"
            )
        self._et_cache: dict[str, EarlyTermination] = {}

        # ----------------------- demo-timestep starts / single-step RL (default off)
        ss_cfg = cfg.get("single_step", None)
        if ss_cfg is not None and OmegaConf.is_config(ss_cfg):
            ss_cfg = OmegaConf.to_container(ss_cfg, resolve=True)
        self.single_step = SingleStepSampler(ss_cfg, seed=self.seed + 2)

        demo_start_cfg = cfg.get("demo_start", None)
        if demo_start_cfg is not None and OmegaConf.is_config(demo_start_cfg):
            demo_start_cfg = OmegaConf.to_container(demo_start_cfg, resolve=True)
        self.demo_start = DemoTimestepSampler(demo_start_cfg, seed=self.seed + 3)
        assert not (self.single_step.enabled and self.demo_start.enabled), (
            "single_step and demo_start are mutually exclusive; use single_step "
            "for one-transition RL and demo_start for short-horizon curriculum RL"
        )
        self._demo_timestep_sampler = (
            self.single_step if self.single_step.enabled else self.demo_start
        )
        if self.single_step.enabled:
            assert self.max_episode_steps == 1, (
                "single_step RL runs exactly one control step per episode; "
                "set env.*.max_episode_steps=1 (and max_steps_per_rollout_epoch=1, "
                "actor.model.num_action_chunks=1)"
            )
        if self._demo_timestep_sampler.enabled:
            assert not self.rsi.enabled and not self.et_enabled, (
                "demo-timestep sampling is mutually exclusive with RSI / "
                "early-termination; it samples its own reference start state"
            )

        # ------------------------------------------------------------- runtime
        self.video_cfg = cfg.video_cfg
        self._video_num_envs = int(cfg.get("video_num_envs", 4))
        self._render_size = (
            int(cfg.get("render_height", 240)),
            int(cfg.get("render_width", 320)),
        )
        num_threads = int(cfg.get("num_threads", 0)) or min(self.num_envs, 16)
        self._pool = ThreadPoolExecutor(max_workers=num_threads)

        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.reset_state_ids: torch.Tensor | None = None
        self.update_reset_state_ids()

        self.subenvs: list[_SubEnv] = [None] * self.num_envs
        self._is_start = True

    # ------------------------------------------------------------------ props
    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = value

    @property
    def total_num_group_envs(self) -> int:
        return len(self.episode_dirs)

    # ------------------------------------------------------------- episode mgmt
    def update_reset_state_ids(self) -> None:
        """Resample one episode id per group (called by EnvWorker per rollout)."""
        ids = torch.randint(
            low=0,
            high=len(self.episode_dirs),
            size=(self.num_group,),
            generator=self._generator,
        )
        self.reset_state_ids = ids.repeat_interleave(self.group_size)

    def _get_episode(self, episode_id: int) -> tuple[EpisodeData, mujoco.MjModel]:
        ep_dir = self.episode_dirs[int(episode_id)]
        key = ep_dir.name
        if key not in self._episode_cache:
            ep = load_episode_data(
                ep_dir,
                self._scene_root,
                self._robot_assets,
                self.trajectory_file,
                self.num_points,
                self.need_tool_cloud,
                self.spec,
            )
            model = mujoco.MjModel.from_xml_path(ep.scene_xml)
            assert model.nu == self.spec.hand_dim, (
                f"{key}: expected nu={self.spec.hand_dim}, got {model.nu}"
            )
            self._episode_cache[key] = (ep, model)
        return self._episode_cache[key]

    def _get_early_termination(self, ep, model) -> Optional[EarlyTermination]:
        if not self.et_enabled:
            return None
        if ep.name not in self._et_cache:
            self._et_cache[ep.name] = EarlyTermination(self._et_cfg, model, ep)
        return self._et_cache[ep.name]

    def _make_subenv(self, episode_id: int) -> _SubEnv:
        if self._demo_timestep_sampler.enabled:
            return self._make_demo_timestep_subenv(episode_id)
        ep, model = self._get_episode(episode_id)
        # DeepMimic RSI: start from a randomly sampled reference frame
        start_frame = self.rsi.sample_start_frame(ep.num_frames)
        data = mujoco.MjData(model)
        data.qpos[:] = ep.qpos_demo[start_frame]
        data.qvel[:] = ep.qvel_demo[start_frame]
        mujoco.mj_forward(model, data)
        substeps = max(1, round((1.0 / ep.frequency) / float(model.opt.timestep)))
        ep_len = min(self.max_episode_steps, ep.num_frames - 1 - start_frame)
        sub = _SubEnv(
            episode=ep,
            model=model,
            data=data,
            substeps=substeps,
            ep_len=ep_len,
            start_frame=start_frame,
        )
        frame0 = sub.obs_frame(self.need_tool_cloud)
        sub.obs_hist = [frame0] * self.obs_horizon
        return sub

    def _make_demo_timestep_subenv(self, episode_id: int) -> _SubEnv:
        """Init at a sampled demo frame and run the configured horizon.

        The full sim state is the reference ``q_t`` (objects always clean); the
        To-frame obs window is the real demo history ``q_{t-To+1..t}`` with iid
        Gaussian noise on every hand-qpos frame. ``perturb_init_hand`` also
        offsets the physical initial hand by the current-frame noise. With
        ``single_step.enabled`` the horizon is one; with ``demo_start.enabled``
        the horizon may be increased by EnvWorker's curriculum.
        """
        sampler = self._demo_timestep_sampler
        ep, model = self._get_episode(episode_id)
        t = sampler.sample_timestep(ep.num_frames, horizon_steps=self.max_episode_steps)
        noise = sampler.sample_hand_noise(self.obs_horizon, self.spec.hand_dim)
        data = mujoco.MjData(model)
        data.qpos[:] = ep.qpos_demo[t]
        if sampler.perturb_init_hand:
            data.qpos[: self.spec.hand_dim] += noise[-1]
        data.qvel[:] = ep.qvel_demo[t]
        mujoco.mj_forward(model, data)
        substeps = max(1, round((1.0 / ep.frequency) / float(model.opt.timestep)))
        ep_len = min(self.max_episode_steps, ep.num_frames - 1 - t)
        sub = _SubEnv(
            episode=ep,
            model=model,
            data=data,
            substeps=substeps,
            ep_len=ep_len,
            start_frame=t,
        )
        sub.obs_hist = sampler.build_obs_history(
            ep, t, self.obs_horizon, self.need_tool_cloud, noise
        )
        return sub

    # ------------------------------------------------------------------- reset
    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = None,
    ):
        if not self.use_fixed_reset_state_ids:
            self.update_reset_state_ids()
        for i in range(self.num_envs):
            self.subenvs[i] = self._make_subenv(int(self.reset_state_ids[i]))
        infos: dict[str, Any] = {}
        return self._wrap_obs(), infos

    # -------------------------------------------------------------------- step
    def _step_one(self, idx: int, action: np.ndarray) -> tuple[float, dict]:
        """Physics + reward for one sub-env; runs inside the thread pool."""
        sub = self.subenvs[idx]
        if sub.done:
            return 0.0, {}
        sub.data.ctrl[:] = action
        for _ in range(sub.substeps):
            mujoco.mj_step(sub.model, sub.data)
        sub.steps += 1
        reward, info = self.reward_fn.compute(sub)
        obs_frame = sub.obs_frame(self.need_tool_cloud)
        if self._demo_timestep_sampler.enabled:
            obs_frame = self._demo_timestep_sampler.noise_obs_frame(
                obs_frame, self.spec.hand_dim
            )
        sub.obs_hist.append(obs_frame)
        # keep history bounded
        if len(sub.obs_hist) > self.obs_horizon:
            del sub.obs_hist[: -self.obs_horizon]
        sub.ret += reward
        # DeepMimic ET: failure termination (object escaped / lost tracking)
        et = self._get_early_termination(sub.episode, sub.model)
        if et is not None and not sub.done:
            terminate, _reason = et.check(
                sub.model, sub.data, sub.episode, sub.demo_frame, sub.steps
            )
            if terminate:
                sub.terminated = True
                sub.done = True
                self._record_final_errors(sub)
        if not sub.done and sub.steps >= sub.ep_len:
            sub.done = True
            self._record_final_errors(sub)
        return reward, info

    def _record_final_errors(self, sub: _SubEnv) -> None:
        sim = sub.data.qpos
        demo = sub.demo_qpos(sub.steps)
        tool, target, hand_dim = (
            self.spec.tool_obj_qpos,
            self.spec.target_obj_qpos,
            self.spec.hand_dim,
        )
        sub.final_tool_err = float(
            np.linalg.norm(
                sim[tool.start : tool.start + 3] - demo[tool.start : tool.start + 3]
            )
        )
        sub.final_target_err = float(
            np.linalg.norm(
                sim[target.start : target.start + 3]
                - demo[target.start : target.start + 3]
            )
        )
        sub.final_hand_err = float(np.abs(sim[:hand_dim] - demo[:hand_dim]).mean())

    def step(self, actions, build_obs: bool = True):
        """Execute one 44-dim qpos-target action per sub-env.

        Args:
            actions: (num_envs, 44) array of hand qpos targets.
            build_obs: skip the (relatively expensive) stacked-obs assembly for
                intermediate chunk steps; only the last step of a chunk needs it.
        """
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().float().numpy()
        actions = np.asarray(actions, dtype=np.float64)
        assert actions.shape == (self.num_envs, self.spec.hand_dim), (
            f"expected actions ({self.num_envs}, {self.spec.hand_dim}), "
            f"got {actions.shape}"
        )

        results = list(
            self._pool.map(
                self._step_one, range(self.num_envs), [actions[i] for i in range(self.num_envs)]
            )
        )
        rewards = torch.tensor([r for r, _ in results], dtype=torch.float32)

        terminations = torch.tensor(
            [sub.terminated for sub in self.subenvs], dtype=torch.bool
        )
        truncations = torch.tensor(
            [sub.done and not sub.terminated for sub in self.subenvs],
            dtype=torch.bool,
        )

        infos: dict[str, Any] = {}
        if self.record_metrics:
            infos["episode"] = self._episode_metrics()

        obs = self._wrap_obs() if build_obs else None
        return obs, rewards, terminations, truncations, infos

    def chunk_step(self, chunk_actions):
        """Execute a (num_envs, chunk, 44) action chunk; RLinf EnvWorker API."""
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.detach().cpu().float().numpy()
        chunk_actions = np.asarray(chunk_actions)
        assert chunk_actions.ndim == 3 and chunk_actions.shape[2] == self.spec.hand_dim, (
            f"expected (B, chunk, {self.spec.hand_dim}), got {chunk_actions.shape}"
        )
        chunk_size = chunk_actions.shape[1]

        obs_list: list = []
        infos_list: list = []
        chunk_rewards = []
        raw_truncations = []
        raw_terminations = []
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

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [B, chunk]
        raw_terminations = torch.stack(raw_terminations, dim=1)
        raw_truncations = torch.stack(raw_truncations, dim=1)

        past_terminations = raw_terminations.any(dim=1)
        past_truncations = raw_truncations.any(dim=1)

        chunk_terminations = torch.zeros_like(raw_terminations)
        chunk_terminations[:, -1] = past_terminations
        chunk_truncations = torch.zeros_like(raw_truncations)
        chunk_truncations[:, -1] = past_truncations
        return obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list

    # ----------------------------------------------------------------- obs/info
    def _wrap_obs(self) -> dict[str, torch.Tensor]:
        def stack(key: str) -> torch.Tensor:
            per_env = [
                np.stack([f[key] for f in sub.obs_hist[-self.obs_horizon :]])
                for sub in self.subenvs
            ]
            return torch.from_numpy(np.stack(per_env))

        obs = {
            "states": stack("qpos"),  # (B, To, 44)
            "pointcloud": stack("pointcloud"),  # (B, To, K, 3)
        }
        if self.need_tool_cloud:
            obs["tool_pointcloud"] = stack("tool_pointcloud")
        return obs

    def _episode_metrics(self) -> dict[str, torch.Tensor]:
        rets = torch.tensor([sub.ret for sub in self.subenvs], dtype=torch.float32)
        lens = torch.tensor(
            [max(sub.steps, 1) for sub in self.subenvs], dtype=torch.float32
        )
        tool_err = torch.tensor(
            [sub.final_tool_err for sub in self.subenvs], dtype=torch.float32
        )
        target_err = torch.tensor(
            [sub.final_target_err for sub in self.subenvs], dtype=torch.float32
        )
        hand_err = torch.tensor(
            [sub.final_hand_err for sub in self.subenvs], dtype=torch.float32
        )
        success = (tool_err < self.success_threshold_m) & torch.tensor(
            [sub.done and not sub.terminated for sub in self.subenvs]
        )
        metrics = {
            "return": rets,
            "episode_len": lens,
            "reward": rets / lens,
            "success_once": success,
            "tool_pos_err_final_m": tool_err,
            "target_pos_err_final_m": target_err,
            "hand_qpos_err_final": hand_err,
        }
        if self.et_enabled:
            metrics["terminated_early"] = torch.tensor(
                [float(sub.terminated) for sub in self.subenvs]
            )
        if self.rsi.enabled:
            metrics["rsi_start_frame"] = torch.tensor(
                [float(sub.start_frame) for sub in self.subenvs]
            )
        if self.demo_start.enabled:
            metrics["demo_start_frame"] = torch.tensor(
                [float(sub.start_frame) for sub in self.subenvs]
            )
        if self.single_step.enabled:
            metrics["single_step_frame"] = torch.tensor(
                [float(sub.start_frame) for sub in self.subenvs]
            )
        return metrics

    # ------------------------------------------------------------------ render
    def capture_image(self, infos=None) -> np.ndarray:
        """Tile the front-camera view of the first few sub-envs (RecordVideo)."""
        h, w = self._render_size
        n = min(self._video_num_envs, self.num_envs)
        frames = []
        for sub in self.subenvs[:n]:
            if sub is None:
                frames.append(np.zeros((h, w, 3), dtype=np.uint8))
                continue
            if sub.renderer is None:
                sub.renderer = mujoco.Renderer(sub.model, height=h, width=w)
            cam = "front" if mujoco.mj_name2id(
                sub.model, mujoco.mjtObj.mjOBJ_CAMERA, "front"
            ) >= 0 else 0
            sub.renderer.update_scene(sub.data, camera=cam)
            frames.append(sub.renderer.render())
        return np.concatenate(frames, axis=1)

    def close(self):
        for sub in self.subenvs:
            if sub is not None and sub.renderer is not None:
                sub.renderer.close()
        self._pool.shutdown(wait=False)
