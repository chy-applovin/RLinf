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

"""Standalone tests for the anti-reward-hacking fixes (no Ray / no GPU):

1. contact-consistency reward term (``reward.contact_weight``);
2. tightened early-termination ``object_err_threshold`` (0.10).

Both target the hacking mode found in the step-1000 IL-finetune evals:
"hover near the brush, never grasp" outscored real manipulation.

    python examples/embodiment/taco_allegro/test_antihack.py
"""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

EP = "brush__brush__bowl__20230927_027"

W_POSE = 2.1  # tool 1.0 + target 1.0 + hand 0.1
W_CONTACT = 0.5
W_ALL = W_POSE + W_CONTACT


def make_cfg(**overrides):
    cfg = {
        "env_type": "taco",
        "sim_backend": "cpu",
        "dataset_root": "/root/Data/taco-brush-allegro",
        "allegro_assets_root": "/root/Spider/spider/assets/robots/allegro/assets",
        "scene_root": "/tmp/taco_antihack_test",
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
        "rsi": {"enabled": False},
        "early_termination": {"enabled": False},
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def build(cfg, n=1):
    from rlinf.envs import get_env_cls

    return get_env_cls("taco", cfg)(
        cfg=cfg, num_envs=n, seed_offset=0, total_num_processes=1
    )


def rollout_rewards(env, actions_fn, num_chunks=40):
    """Run 40 x 4-step chunks; return per-step rewards (160,) of env 0."""
    rews = []
    for k in range(num_chunks):
        acts = actions_fn(k)  # (n, 4, 44)
        _, rew, _, _, _ = env.chunk_step(acts)
        rews.append(rew[0].numpy())
    return np.concatenate(rews)


def contact_ref(env):
    return env.reward_fn._get_contact_ref(env.subenvs[0])


def test_contact_ref():
    """Demo contact precompute: a real manipulation window must exist."""
    env = build(make_cfg(reward={"type": "tracking", "contact_weight": W_CONTACT}))
    env.reset()
    ref = contact_ref(env)
    tool_frac = ref.demo_tool_contact.mean()
    target_frac = ref.demo_target_contact.mean()
    tool_frames = np.flatnonzero(ref.demo_tool_contact)
    assert tool_frames.size > 0, "demo never has right-hand<->tool contact?!"
    assert 0.05 < tool_frac < 0.95, f"implausible tool contact fraction {tool_frac}"
    env.close()
    print(
        f"[1] contact ref: OK (tool contact {tool_frac:.0%} of frames "
        f"[{tool_frames[0]}..{tool_frames[-1]}], target {target_frac:.0%})"
    )
    return ref


def test_contact_term(ref):
    """Hover earns ~0 contact term in the demo-contact window; replay earns it.

    Both envs run identical deterministic physics, so the implied contact
    match can be recovered from the two reward streams:
        r_contact = (W_POSE * r_pose + W_CONTACT * match) / W_ALL

    NOTE on the replay ceiling: open-loop demo replay only reaches ~0.35-0.4
    window match (not ~1.0) because the IK-retargeted demo's grip is
    dynamically marginal - the replay sim loses brush contact mid-window
    (the same dynamics gap seen in the RSI cold-start tests). The term is a
    *discriminator*: hover scores ~0, any real engagement scores >0, and a
    closed-loop policy that squeezes can push it toward 1.
    """
    demo = None

    def build_pair():
        nonlocal demo
        plain = build(make_cfg())  # pose-only tracking reward
        withc = build(
            make_cfg(reward={"type": "tracking", "contact_weight": W_CONTACT})
        )
        plain.reset()
        withc.reset()
        demo = plain.subenvs[0].episode.qpos_demo
        return plain, withc

    # frame k (1-based executed step) <-> reward index k-1
    dt = ref.demo_tool_contact[1:161]
    dg = ref.demo_target_contact[1:161]
    window = np.flatnonzero(dt | dg)
    # a policy that NEVER touches anything earns exactly the free half-scores
    # of the frames where only one pair is in demo contact:
    zero_touch = 0.5 * ((~dt).astype(float) + (~dg).astype(float))

    # --- hover: hand frozen at its frame-0 pose, objects never touched
    plain, withc = build_pair()
    frozen = demo[None, [0, 0, 0, 0], :44]
    r_plain = rollout_rewards(plain, lambda k: frozen)
    r_withc = rollout_rewards(withc, lambda k: frozen)
    match_hover = (W_ALL * r_withc - W_POSE * r_plain) / W_CONTACT
    plain.close(), withc.close()
    assert np.allclose(match_hover, zero_touch, atol=1e-4), (
        "hover must earn exactly the zero-touch baseline (per-pair score 0 "
        f"whenever the demo is in contact); max dev "
        f"{np.abs(match_hover - zero_touch).max():.4f}"
    )

    # --- open-loop demo replay: the hand actually grasps
    plain, withc = build_pair()

    def demo_acts(k):
        idx = np.clip(1 + 4 * k + np.arange(4), 0, demo.shape[0] - 1)
        return demo[None, idx, :44]

    r_plain = rollout_rewards(plain, demo_acts)
    r_withc = rollout_rewards(withc, demo_acts)
    match_replay = (W_ALL * r_withc - W_POSE * r_plain) / W_CONTACT
    plain.close(), withc.close()
    gain_replay = (match_replay - zero_touch)[window].mean()
    assert gain_replay > 0.1, (
        f"demo replay should earn contact above the zero-touch baseline, "
        f"got +{gain_replay:.3f}"
    )
    print(
        f"[2] contact term: OK (window match: hover {match_hover[window].mean():.3f}"
        f" = zero-touch baseline, replay {match_replay[window].mean():.3f} "
        f"(+{gain_replay:.3f} above baseline))"
    )


def test_et_010_kills_hover():
    """With object_err_threshold=0.10, the hover policy must terminate early."""
    env = build(
        make_cfg(
            early_termination={
                "enabled": True,
                "criteria": ["hand_object_distance", "object_tracking"],
                "min_steps": 8,
                "hand_object_dist_threshold": 0.10,
                "object_err_threshold": 0.10,
            }
        )
    )
    env.reset()
    demo = env.subenvs[0].episode.qpos_demo
    frozen = demo[None, [0, 0, 0, 0], :44]
    fired_at = None
    for k in range(40):
        _, _, term, _, infos = env.chunk_step(frozen)
        if term.any():
            fired_at = (k + 1) * 4
            break
    assert fired_at is not None, "ET@0.10 never fired on the hover policy"
    assert fired_at < 120, f"ET fired too late ({fired_at}) to spoil the hack"
    assert not infos[-1]["episode"]["success_once"].any()
    env.close()
    print(f"[3] ET object_err 0.10: OK (hover terminated at control step {fired_at})")


if __name__ == "__main__":
    ref = test_contact_ref()
    test_contact_term(ref)
    test_et_010_kills_hover()
    print("ALL ANTI-HACKING TESTS PASSED")
