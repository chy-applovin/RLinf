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

"""Standalone tests for single-step (bandit) RL + hand-obs-noise (no Ray/GPU):

    python examples/embodiment/taco_allegro/test_single_step.py
"""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

EP = "brush__brush__bowl__20230927_027"
TO = 16
HAND_DIM = 44


def make_cfg(**overrides):
    cfg = {
        "env_type": "taco",
        "sim_backend": "cpu",
        "dataset_root": "/root/Data/taco-brush-allegro",
        "allegro_assets_root": "/root/Spider/spider/assets/robots/allegro/assets",
        "scene_root": "/tmp/taco_single_step_test",
        "trajectory_file": "trajectory_ctrl_30hz_im.npz",
        "episodes": [EP],
        "categories": None,
        "one_per_category": False,
        "max_episodes": None,
        "obs_mode": "pc2_qpos",
        "obs_horizon": TO,
        "num_points": 64,
        "auto_reset": False,
        "ignore_terminations": False,
        "max_episode_steps": 1,
        "max_steps_per_rollout_epoch": 1,
        "seed": 0,
        "group_size": 1,
        "use_fixed_reset_state_ids": True,
        "num_threads": 0,
        "reward": {
            "type": "tracking",
            "tool_pos_weight": 1.0,
            "target_pos_weight": 1.0,
            "hand_qpos_weight": 0.0,  # objects only
            "contact_weight": 0.0,
        },
        "video_cfg": {"save_video": False, "info_on_video": False},
        "rsi": {"enabled": False},
        "early_termination": {"enabled": False},
        "single_step": {
            "enabled": True,
            "hand_obs_noise_std": 0.0,
            "perturb_init_hand": False,
            "min_frame": 0,
        },
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def build(cfg, n):
    from rlinf.envs import get_env_cls

    return get_env_cls("taco", cfg)(
        cfg=cfg, num_envs=n, seed_offset=0, total_num_processes=1
    )


def ss(**ss_over):
    base = {
        "enabled": True,
        "hand_obs_noise_std": 0.0,
        "perturb_init_hand": False,
        "min_frame": 0,
    }
    base.update(ss_over)
    return base


def test_sampling_and_alignment():
    """t is diverse & in range, ep_len==1, obs history matches the demo window."""
    n = 32
    env = build(make_cfg(single_step=ss()), n)
    env.reset()
    demo = env.subenvs[0].episode.qpos_demo
    T = demo.shape[0]
    starts = [s.start_frame for s in env.subenvs]
    assert len(set(starts)) > 8, f"timesteps not diverse: {starts}"
    assert min(starts) >= 0 and max(starts) <= T - 2, f"t out of range: {starts}"
    for s in env.subenvs:
        assert s.ep_len == 1, f"single-step ep_len must be 1, got {s.ep_len}"
        # clean init state (no perturb): qpos == demo[t]
        assert np.allclose(s.data.qpos, demo[s.start_frame], atol=1e-9)
        # obs history: hand qpos == demo frames t-To+1..t (edge-padded), no noise
        t = s.start_frame
        for k, frame in enumerate(s.obs_hist):
            f = int(np.clip(t - TO + 1 + k, 0, T - 1))
            assert np.allclose(frame["qpos"], demo[f, :HAND_DIM], atol=1e-5), (
                f"obs frame {k} hand != demo frame {f}"
            )
    env.close()
    print(f"[1] sampling+alignment: OK (t in [{min(starts)},{max(starts)}], ep_len=1)")


def test_hand_noise():
    """Noise hits ONLY the hand qpos; clouds untouched; std matches; init clean."""
    n = 64
    std = 0.02
    env = build(make_cfg(single_step=ss(hand_obs_noise_std=std)), n)
    env.reset()
    demo = env.subenvs[0].episode.qpos_demo
    T = demo.shape[0]
    devs = []
    for s in env.subenvs:
        t = s.start_frame
        # current-frame obs hand minus clean demo hand = the injected noise
        dev = s.obs_hist[-1]["qpos"] - demo[t, :HAND_DIM]
        devs.append(dev)
        # point cloud must be the clean demo re-pose (objects never noised)
        from rlinf.envs.taco.scene import synth_obs_frame

        clean = synth_obs_frame(demo[t], s.episode, True)
        assert np.allclose(s.obs_hist[-1]["pointcloud"], clean["pointcloud"], atol=1e-6)
        # perturb_init_hand=False -> actual hand stays clean
        assert np.allclose(s.data.qpos[:HAND_DIM], demo[t, :HAND_DIM], atol=1e-9)
    emp = float(np.std(np.concatenate(devs)))
    assert abs(emp - std) < 0.005, f"empirical hand-noise std {emp:.4f} != {std}"
    env.close()
    print(f"[2] hand-noise (obs-only): OK (empirical std {emp:.4f}, clouds clean)")


def test_perturb_init_hand():
    """perturb_init_hand=True: actual init hand == demo + current-frame noise."""
    n = 16
    env = build(
        make_cfg(single_step=ss(hand_obs_noise_std=0.02, perturb_init_hand=True)), n
    )
    env.reset()
    demo = env.subenvs[0].episode.qpos_demo
    for s in env.subenvs:
        t = s.start_frame
        inj = s.obs_hist[-1]["qpos"] - demo[t, :HAND_DIM]  # current-frame noise
        # actual init hand offset == current-frame obs noise (consistent)
        assert np.allclose(
            s.data.qpos[:HAND_DIM], demo[t, :HAND_DIM] + inj, atol=1e-6
        ), "init hand must be demo hand + current-frame noise"
        # objects are NOT perturbed
        assert np.allclose(s.data.qpos[HAND_DIM:], demo[t, HAND_DIM:], atol=1e-9)
    env.close()
    print("[3] perturb_init_hand: OK (init hand offset == current-frame obs noise)")


def test_single_step_done_and_reward():
    """One step -> truncation (not termination); GT action tracks objects well."""
    n = 24
    env = build(make_cfg(single_step=ss()), n)
    env.reset()
    demo = env.subenvs[0].episode.qpos_demo
    # GT action for env i = demo[t_i + 1][:44] (action_offset=1 next-frame target)
    acts = np.stack(
        [demo[None, min(s.start_frame + 1, demo.shape[0] - 1), :HAND_DIM] for s in env.subenvs]
    )  # (n, 1, 44)
    _, rew, term, trunc, infos = env.chunk_step(acts)
    assert rew.shape == (n, 1)
    assert trunc[:, -1].all(), "single step must truncate every env"
    assert not term.any(), "single step is a truncation, not a termination"
    ep = infos[-1]["episode"]
    assert "single_step_frame" in ep
    # objects should be tracked by the GT action (start on-manifold at q_t)
    r = rew[:, 0].numpy()
    tool = ep["tool_pos_err_final_m"].numpy()
    target = ep["target_pos_err_final_m"].numpy()
    assert r.mean() > 0.6, f"GT-action single-step reward too low: {r.mean():.3f}"
    env.close()
    print(
        f"[4] single-step done+reward: OK (GT-action r {r.mean():.3f}, "
        f"tool_err {tool.mean():.3f}m, target_err {target.mean():.3f}m)"
    )


def test_default_off_regression():
    """single_step.enabled=False -> original episode behavior (frame-0 repeat)."""
    env = build(make_cfg(max_episode_steps=160, single_step=ss(enabled=False)), 2)
    env.reset()
    demo = env.subenvs[0].episode.qpos_demo
    for s in env.subenvs:
        assert s.start_frame == 0 and s.ep_len == min(160, demo.shape[0] - 1)
        # history is frame-0 repeated To times
        for frame in s.obs_hist:
            assert np.allclose(frame["qpos"], demo[0, :HAND_DIM], atol=1e-5)
    assert "single_step_frame" not in env._episode_metrics()
    env.close()
    print("[5] default-off regression: OK (start_frame=0, frame-0 history)")


def calibrate_gt_reward():
    """Print GT-action reward for both perturb modes (noise std sweep)."""
    print("--- GT-action single-step reward calibration (n=64) ---")
    for std in (0.0, 0.01, 0.02, 0.04):
        for perturb in (False, True):
            env = build(
                make_cfg(
                    single_step=ss(hand_obs_noise_std=std, perturb_init_hand=perturb)
                ),
                64,
            )
            env.reset()
            demo = env.subenvs[0].episode.qpos_demo
            acts = np.stack(
                [
                    demo[None, min(s.start_frame + 1, demo.shape[0] - 1), :HAND_DIM]
                    for s in env.subenvs
                ]
            )
            _, rew, _, _, _ = env.chunk_step(acts)
            print(
                f"  std={std:.2f} perturb_init_hand={perturb!s:5}  "
                f"GT-action r/step = {rew[:,0].mean():.3f}"
            )
            env.close()


if __name__ == "__main__":
    test_sampling_and_alignment()
    test_hand_noise()
    test_perturb_init_hand()
    test_single_step_done_and_reward()
    test_default_off_regression()
    calibrate_gt_reward()
    print("ALL SINGLE-STEP TESTS PASSED")
