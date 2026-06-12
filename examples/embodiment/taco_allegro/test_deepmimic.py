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

"""Standalone functional tests for the DeepMimic aids (RSI + early termination).

No Ray / no GPU required:

    python examples/embodiment/taco_allegro/test_deepmimic.py
"""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf

EP = "brush__brush__bowl__20230927_027"


def make_cfg(**overrides):
    cfg = {
        "env_type": "taco",
        "sim_backend": "cpu",
        "dataset_root": "/root/Data/taco-brush-allegro",
        "allegro_assets_root": "/root/Spider/spider/assets/robots/allegro/assets",
        "scene_root": "/tmp/taco_dm_test",
        "trajectory_file": "trajectory_ctrl_30hz_im.npz",
        "episodes": [EP],
        "categories": None,
        "one_per_category": False,
        "max_episodes": None,
        "obs_mode": "pc2_qpos",
        "obs_horizon": 16,
        "num_points": 64,  # small clouds: obs content irrelevant here
        "auto_reset": False,
        "ignore_terminations": False,
        "max_episode_steps": 160,
        "max_steps_per_rollout_epoch": 160,
        "seed": 0,
        "group_size": 1,
        "use_fixed_reset_state_ids": True,
        "num_threads": 0,
        "reward": {"type": "tracking"},
        "video_cfg": {"save_video": False, "info_on_video": False},
        "rsi": {"enabled": False, "min_remaining_steps": 8},
        "early_termination": {"enabled": False},
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def build(cfg, n):
    from rlinf.envs import get_env_cls

    return get_env_cls("taco", cfg)(
        cfg=cfg, num_envs=n, seed_offset=0, total_num_processes=1
    )


def demo_actions(demo, frames, n):
    """(n, len(frames), 44) demo qpos targets for the given frame indices."""
    return np.repeat(demo[None, frames, :44], n, axis=0)


def test_default_off():
    env = build(make_cfg(), 2)
    env.reset()
    assert all(s.start_frame == 0 for s in env.subenvs)
    acts = np.repeat(env.subenvs[0].episode.qpos_demo[None, 1:5, :44], 2, axis=0)
    _, rew, term, trunc, infos = env.chunk_step(acts)
    assert not term.any() and not trunc.any()
    assert "terminated_early" not in infos[-1]["episode"]
    assert "rsi_start_frame" not in infos[-1]["episode"]
    env.close()
    print("[1] default-off regression: OK (start_frame=0, no ET fields)")


def test_rsi():
    n = 16
    env = build(make_cfg(rsi={"enabled": True, "min_remaining_steps": 8}), n)
    env.reset()
    demo = env.subenvs[0].episode.qpos_demo
    starts = [s.start_frame for s in env.subenvs]
    assert len(set(starts)) > 4, f"start frames not diverse: {starts}"
    assert max(starts) <= demo.shape[0] - 1 - 8
    # init state == demo[start_frame]
    for s in env.subenvs:
        assert np.allclose(s.data.qpos, demo[s.start_frame], atol=1e-9)
        assert s.ep_len == min(160, demo.shape[0] - 1 - s.start_frame)
    # Frame alignment under RSI: replay demo actions from each env's own
    # start frame. Right after init the sim state IS the aligned demo frame,
    # so chunk-0 reward must be high for EVERY start frame (alignment check).
    # Over longer horizons, open-loop replay from a cold mid-grasp init can
    # degrade (contact/dynamics state differs from a frame-0 rollout - the
    # same dynamical-infeasibility caveat as in DeepMimic; the closed-loop
    # policy corrects for it), so we only require early-start envs to sustain.
    per_env_r = []
    for k in range(8):
        acts = np.stack(
            [
                demo[
                    np.clip(
                        s.start_frame + 1 + 4 * k + np.arange(4), 0, demo.shape[0] - 1
                    ),
                    :44,
                ]
                for s in env.subenvs
            ]
        )
        _, rew, term, trunc, infos = env.chunk_step(acts)
        per_env_r.append(rew.mean(dim=1).numpy())
    per_env_r = np.stack(per_env_r)  # (chunks, n)
    assert per_env_r[0].min() > 0.5, (
        f"chunk-0 reward must be near-demo for all start frames: {per_env_r[0]}"
    )
    early = [i for i, s in enumerate(env.subenvs) if s.start_frame <= 10]
    if early:
        assert per_env_r[-1][early].min() > 0.7, (
            "early-start envs must sustain demo-replay reward"
        )
    ep = infos[-1]["episode"]
    assert "rsi_start_frame" in ep
    env.close()
    print(
        f"[2] RSI: OK (starts {sorted(starts)[:6]}..., chunk-0 replay reward "
        f"min {per_env_r[0].min():.3f})"
    )


def test_et_hand_object_distance():
    n = 4
    env = build(
        make_cfg(
            early_termination={
                "enabled": True,
                "criteria": ["hand_object_distance"],
                "min_steps": 8,
                "hand_object_dist_threshold": 0.10,
            }
        ),
        n,
    )
    env.reset()
    demo = env.subenvs[0].episode.qpos_demo
    # freeze the hand at its frame-0 pose: when the demo enters the grasp /
    # manipulation phase, the demo palm-object distance shrinks while the sim
    # hand stays away -> "object escaped relative to reference" must fire.
    frozen = np.repeat(demo[None, [0, 0, 0, 0], :44], n, axis=0)
    fired_at = None
    for k in range(40):
        _, rew, term, trunc, infos = env.chunk_step(frozen)
        if term.any():
            fired_at = (k + 1) * 4
            break
    assert fired_at is not None, "hand_object_distance ET never fired"
    assert term[:, -1].all(), "all frozen envs should terminate together"
    assert not trunc.any()
    ep = infos[-1]["episode"]
    assert ep["terminated_early"].sum() == n
    assert not ep["success_once"].any(), "terminated episodes must not be successes"
    # frozen envs stay frozen and rewards are zeroed afterwards
    _, rew2, term2, _, _ = env.chunk_step(frozen)
    assert float(rew2.abs().sum()) == 0.0
    env.close()
    print(f"[3] ET hand_object_distance: OK (fired at control step {fired_at})")


def test_et_object_tracking():
    n = 2
    env = build(
        make_cfg(
            early_termination={
                "enabled": True,
                "criteria": ["object_tracking"],
                "min_steps": 4,
                "object_err_threshold": 0.05,  # tight threshold to force firing
            }
        ),
        n,
    )
    env.reset()
    demo = env.subenvs[0].episode.qpos_demo
    frozen = np.repeat(demo[None, [0, 0, 0, 0], :44], n, axis=0)
    fired_at = None
    for k in range(40):
        _, _, term, _, infos = env.chunk_step(frozen)
        if term.any():
            fired_at = (k + 1) * 4
            break
    # untouched objects deviate from the demo by >5cm during manipulation
    assert fired_at is not None, "object_tracking ET never fired"
    env.close()
    print(f"[4] ET object_tracking: OK (fired at control step {fired_at})")


if __name__ == "__main__":
    test_default_off()
    test_rsi()
    test_et_hand_object_distance()
    test_et_object_tracking()
    print("ALL DEEPMIMIC TESTS PASSED")
