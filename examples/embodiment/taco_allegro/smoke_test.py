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

"""Standalone (no-Ray) smoke test for the TACO flow-matching RL integration.

Checks, on a couple of real TACO episodes + a real flow-policy checkpoint:

1. env construction / reset / chunk_step with the RLinf embodied conventions;
2. rollout inference (noise-injected sampling) -> env stepping -> rewards;
3. ODE parity: in eval mode the RL wrapper's sampler must reproduce the
   original ``flow_policy.algos.FlowMatching.sample`` bit-for-bit (same seed);
4. PPO logprob parity: ``default_forward`` recomputed log-probs on the stored
   chains must match the rollout-time ``prev_logprobs``;
5. a backward pass through the PPO-style surrogate (gradients flow).

Run (inside an env with rlinf + flow_policy + mujoco installed):

    python examples/embodiment/taco_allegro/smoke_test.py \
        --ckpt /root/flow-policy/outputs/flow-policy/2026-06-10/18-12-14/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from omegaconf import OmegaConf


def build_env_cfg(args) -> OmegaConf:
    return OmegaConf.create(
        {
            "env_type": "taco",
            "dataset_root": args.dataset_root,
            "allegro_assets_root": args.allegro_assets,
            "scene_root": args.scene_root,
            "trajectory_file": args.trajectory_file,
            "episodes": None,
            "categories": None,
            "one_per_category": True,
            "max_episodes": args.num_envs,
            "obs_mode": args.obs_mode,
            "obs_horizon": args.obs_horizon,
            "num_points": args.num_points,
            "auto_reset": False,
            "ignore_terminations": False,
            "max_episode_steps": args.max_steps,
            "max_steps_per_rollout_epoch": args.max_steps,
            "success_threshold_m": 0.1,
            "seed": 0,
            "group_size": 1,
            "use_fixed_reset_state_ids": True,
            "num_threads": 0,
            "reward": {"type": "tracking"},
            "video_cfg": {"save_video": False, "info_on_video": False},
        }
    )


def build_model_cfg(args) -> OmegaConf:
    return OmegaConf.create(
        {
            "model_type": "flow_policy_taco",
            "model_path": args.ckpt,
            "precision": None,
            "num_action_chunks": args.num_action_chunks,
            "action_dim": 44,
            "is_lora": False,
            "num_denoise_steps": 0,
            "noise_method": "flow_sde",
            "noise_level": 0.5,
            "add_value_head": True,
            "detach_critic_input": True,
        }
    )


def flatten_forward_inputs(per_step: list[dict]) -> dict:
    """[T x {key: (B, ...)}] -> {key: (T*B, ...)} (training-batch layout)."""
    out = {}
    for key in per_step[0]:
        out[key] = torch.cat([step[key] for step in per_step], dim=0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="flow-policy .pt checkpoint")
    ap.add_argument("--dataset-root", default="/root/Data/taco-brush-allegro")
    ap.add_argument(
        "--allegro-assets",
        default="/root/Spider/spider/assets/robots/allegro/assets",
    )
    ap.add_argument("--scene-root", default="/tmp/taco_smoke_scene_root")
    ap.add_argument("--trajectory-file", default="trajectory_ctrl_30hz_im.npz")
    ap.add_argument("--num-envs", type=int, default=2)
    ap.add_argument("--num-chunk-steps", type=int, default=4)
    ap.add_argument("--num-action-chunks", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=32)
    ap.add_argument("--obs-mode", default=None, help="default: from ckpt shape_meta")
    ap.add_argument("--obs-horizon", type=int, default=None)
    ap.add_argument("--num-points", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # fill obs contract from the checkpoint so env and model always agree
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sm = state["shape_meta"]
    args.obs_mode = args.obs_mode or sm["obs_mode"]
    args.obs_horizon = args.obs_horizon or int(sm["obs_horizon"])
    args.num_points = args.num_points or int(sm["num_points"])
    del state
    print(
        f"[smoke] ckpt={args.ckpt}\n"
        f"[smoke] obs_mode={args.obs_mode} To={args.obs_horizon} "
        f"K={args.num_points} chunk={args.num_action_chunks}"
    )

    device = torch.device(args.device)

    # ----------------------------------------------------------------- model
    from rlinf.models import get_model

    model_cfg = build_model_cfg(args)
    model = get_model(model_cfg).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] model loaded: {n_params / 1e6:.2f}M params on {device}")

    # ------------------------------------------------------------------- env
    from rlinf.envs import get_env_cls

    env_cfg = build_env_cfg(args)
    env_cls = get_env_cls("taco", env_cfg)
    t0 = time.time()
    env = env_cls(
        cfg=env_cfg,
        num_envs=args.num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    obs, _ = env.reset()
    print(
        f"[smoke] env ready in {time.time() - t0:.1f}s | episodes: "
        f"{[sub.episode.name for sub in env.subenvs]}"
    )
    for k, v in obs.items():
        print(f"[smoke]   obs[{k}]: {tuple(v.shape)} {v.dtype}")

    # --------------------------------------------------- rollout (train mode)
    per_step_fi: list[dict] = []
    per_step_logp: list[torch.Tensor] = []
    reward_curve: list[float] = []
    t0 = time.time()
    for step in range(args.num_chunk_steps):
        with torch.no_grad():
            actions, result = model.predict_action_batch(env_obs=obs, mode="train")
        assert actions.shape == (args.num_envs, args.num_action_chunks, 44)
        assert result["prev_logprobs"].shape == (
            args.num_envs,
            args.num_action_chunks,
            44,
        )
        assert result["prev_values"].shape == (args.num_envs, 1)
        per_step_fi.append(result["forward_inputs"])
        per_step_logp.append(result["prev_logprobs"])

        obs_list, rewards, terms, truncs, infos = env.chunk_step(actions.numpy())
        obs = obs_list[-1]
        reward_curve.append(float(rewards.mean()))
        print(
            f"[smoke] chunk {step}: reward/step={rewards.mean():.3f} "
            f"trunc={truncs[:, -1].tolist()}"
        )
    dt = time.time() - t0
    sps = args.num_envs * args.num_chunk_steps * args.num_action_chunks / dt
    print(f"[smoke] rollout ok: {dt:.1f}s ({sps:.0f} env-steps/s)")
    assert all(r > 0 for r in reward_curve), "tracking reward should be positive"

    # ----------------------------------------------------- ODE (eval) parity
    from flow_policy.algos.flow_matching import FlowMatching
    from flow_policy.data.normalizer import Normalizer
    from flow_policy.models.transformer_policy import FlowMatchingTransformerPolicy

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mcfg = {k: v for k, v in state["cfg"]["model"].items() if k != "_target_"}
    ref_policy = FlowMatchingTransformerPolicy(shape_meta=sm, **mcfg)
    nz = Normalizer(
        sm["action_dim"], sm["obs_mode"], point_dim=sm["point_dim"], qpos_dim=sm["qpos_dim"]
    )
    ref = FlowMatching(
        model=ref_policy,
        normalizer=nz,
        num_sample_steps=state["cfg"]["algo"].get("num_sample_steps", 16),
        normalize_obs=state["cfg"]["algo"].get("normalize_obs", True),
    )
    ref.model.load_state_dict(state["model"])
    ref.normalizer.load_state_dict(state["normalizer"])
    ref = ref.to(device).eval()

    fp_obs = {
        "pointcloud": obs["pointcloud"].to(device),
        "qpos": obs["states"].to(device),
    }
    if args.obs_mode == "pc2_qpos":
        fp_obs["tool_pointcloud"] = obs["tool_pointcloud"].to(device)

    torch.manual_seed(123)
    with torch.no_grad():
        ref_chunk = ref.sample(fp_obs)  # (B, Tp, 44) raw
    torch.manual_seed(123)
    with torch.no_grad():
        _, eval_result = model.predict_action_batch(env_obs=obs, mode="eval")
    rl_chunk = eval_result["forward_inputs"]["model_action"].reshape(
        args.num_envs, -1, 44
    )
    ode_err = (ref_chunk.cpu() - rl_chunk).abs().max().item()
    print(f"[smoke] ODE parity (eval mode vs IL sampler): max|diff|={ode_err:.2e}")
    assert ode_err < 1e-4, "eval-mode sampler must match the IL flow ODE"

    # --------------------------------------------- PPO logprob parity + grads
    fi = flatten_forward_inputs(per_step_fi)
    prev_logp = torch.cat(per_step_logp, dim=0)  # (T*B, chunk, 44)

    model.train()
    out = model(
        forward_inputs=fi,
        compute_logprobs=True,
        compute_entropy=False,
        compute_values=True,
    )
    logp = out["logprobs"]
    assert logp.shape == prev_logp.shape, (logp.shape, prev_logp.shape)
    logp_err = (logp.detach().cpu().float() - prev_logp.float()).abs().max().item()
    print(f"[smoke] logprob parity (train forward vs rollout): max|diff|={logp_err:.2e}")
    assert logp_err < 1e-3, "recomputed logprobs must match rollout logprobs"
    assert out["values"].shape == (logp.shape[0],)

    # PPO-style surrogate: ratio==1 at parity, but gradients must flow.
    bsz = logp.shape[0]
    adv = torch.randn(bsz, device=logp.device)
    ratio = torch.exp(
        logp.reshape(bsz, -1).sum(-1) - prev_logp.to(logp.device).reshape(bsz, -1).sum(-1)
    )
    loss = -(adv * ratio).mean() + out["values"].pow(2).mean()
    loss.backward()
    grads = [
        p.grad.abs().sum().item()
        for p in model.parameters()
        if p.grad is not None
    ]
    print(
        f"[smoke] backward ok: loss={loss.item():.4f}, "
        f"{len(grads)} grad tensors, total |grad|={np.sum(grads):.3e}"
    )
    assert np.sum(grads) > 0, "no gradients reached the policy"

    env.close()
    print("[smoke] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
