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

"""Single-step (one-transition) RL sampling for the TACO env.

Episode-level RL of the IK-imitation flow policy is hard: errors compound over
the rollout and the closed-loop state drifts off the demo manifold. This module
implements a much simpler contextual-bandit alternative:

* sample one demo timestep ``t`` per env (uniformly over the trajectory);
* initialize the full sim state to the reference frame ``q_t`` (hand + objects),
  and build the To-frame observation window from the REAL demo frames
  ``q_{t-To+1..t}`` (edge-padded at the start), exactly the window the IL policy
  was trained on (Diffusion-Policy convention, ``action_offset=1``);
* run EXACTLY ONE control step; the reward is the next-frame OBJECT tracking
  error vs ``q_{t+1}`` (hands are not rewarded). The episode then ends, so the
  return is just that single-step reward (GAE bootstrap is masked by the done).

Robustness augmentation (the point of this scheme): independent Gaussian noise
``epsilon`` is added to the hand qpos of every observation frame
``q_{t-To+1..t}^{hand}``, so the policy must learn a mapping that, even when the
observed hand pose is perturbed, still emits an action that drives the OBJECT to
its next reference pose. Optionally (``perturb_init_hand``) the actual initial
hand state is offset by the current-frame noise too, so the offset is physical
and the policy must compensate rather than merely denoise its input.

Config-gated (default OFF) and isolated here + small guarded hooks in
``TacoEnv``; with the flag off the env behaves exactly as before. Mutually
exclusive with RSI / early termination (those are episode-level aids).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from rlinf.envs.taco.scene import HAND_DIM, synth_obs_frame

if TYPE_CHECKING:
    from rlinf.envs.taco.scene import EpisodeData

__all__ = ["SingleStepSampler"]


class SingleStepSampler:
    """Samples per-reset (timestep, hand-noise) and builds the demo-history obs."""

    def __init__(self, cfg: dict[str, Any] | None, seed: int):
        cfg = dict(cfg) if cfg else {}
        self.enabled = bool(cfg.get("enabled", False))
        # std of the iid Gaussian noise added to the hand qpos of EVERY obs frame
        self.hand_obs_noise_std = float(cfg.get("hand_obs_noise_std", 0.0))
        # also offset the *actual* initial hand state by the current-frame noise
        # (physical offset to compensate) vs pure observation noise (denoise)
        self.perturb_init_hand = bool(cfg.get("perturb_init_hand", False))
        # never sample below this demo frame (skip the static pre-motion frames)
        self.min_frame = int(cfg.get("min_frame", 0))
        self._rng = np.random.default_rng(seed)

    def sample_timestep(self, num_frames: int) -> int:
        """Uniform t in [min_frame, T-2] (need frame t+1 for the reward target)."""
        hi = num_frames - 2
        lo = min(max(self.min_frame, 0), max(hi, 0))
        return int(self._rng.integers(lo, hi + 1))

    def sample_hand_noise(self, obs_horizon: int) -> np.ndarray:
        """(To, HAND_DIM) iid Gaussian hand-qpos noise; zeros when std <= 0."""
        if self.hand_obs_noise_std <= 0.0:
            return np.zeros((obs_horizon, HAND_DIM), dtype=np.float64)
        return self._rng.normal(
            0.0, self.hand_obs_noise_std, size=(obs_horizon, HAND_DIM)
        )

    def build_obs_history(
        self,
        episode: "EpisodeData",
        t: int,
        obs_horizon: int,
        need_tool: bool,
        noise: np.ndarray,
    ) -> list[dict[str, np.ndarray]]:
        """To-frame demo-history obs ending at frame t (edge-padded), hand-noised.

        Frame k (k=0..To-1) is demo frame ``clip(t - To + 1 + k, 0, T-1)`` with
        ``noise[k]`` added to its hand qpos. Object clouds come from the demo's
        own object poses (objects are never noised).
        """
        hist: list[dict[str, np.ndarray]] = []
        for k in range(obs_horizon):
            f = int(np.clip(t - obs_horizon + 1 + k, 0, episode.num_frames - 1))
            frame = synth_obs_frame(episode.qpos_demo[f], episode, need_tool)
            frame["qpos"] = (frame["qpos"] + noise[k]).astype(np.float32)
            hist.append(frame)
        return hist
