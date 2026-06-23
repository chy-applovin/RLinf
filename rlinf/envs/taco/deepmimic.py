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

"""DeepMimic-style training aids for the TACO demo-tracking environment.

Two techniques from Peng et al., "DeepMimic: Example-Guided Deep
Reinforcement Learning of Physics-Based Character Skills" (SIGGRAPH 2018),
adapted to bimanual-Allegro object manipulation:

* **Reference State Initialization (RSI)** - instead of always starting an
  episode at demo frame 0, sample a random reference frame and initialize the
  full sim state (hand + object qpos/qvel) there. Without RSI the policy can
  only reach late-demo states by first mastering everything before them; with
  RSI every phase of the demo is visited from the very first rollout, and the
  critic sees the rewards of late states before the policy can reach them.

* **Early Termination (ET)** - end the episode as soon as the rollout enters
  an unrecoverable failure state (DeepMimic: character falls). Post-failure
  frames otherwise fill the batch with uninformative zero-ish-reward samples
  and (worse) teach the critic that mid-demo states have low value. Criteria
  implemented here:

  - ``hand_object_distance``: an object "escapes" its hand - the sim
    palm-to-object distance exceeds the demo's distance at the aligned frame
    by more than a threshold (catches drops while being phase-aware: in
    pre-grasp phases the demo distance is large too, so no false trigger);
  - ``object_tracking``: an object's position error vs the aligned demo frame
    exceeds a threshold (the direct DeepMimic analog of link deviation).

Both features are config-gated (default OFF) and live entirely in this module
plus small guarded hooks in ``TacoEnv``; with the flags off the env behaves
exactly as before.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from rlinf.envs.taco.scene import EpisodeData

__all__ = ["RSISampler", "EarlyTermination"]

_VALID_CRITERIA = ("hand_object_distance", "object_tracking")


class RSISampler:
    """Samples per-reset reference start frames (uniform over the demo)."""

    def __init__(self, cfg: dict[str, Any] | None, seed: int):
        cfg = dict(cfg) if cfg else {}
        self.enabled = bool(cfg.get("enabled", False))
        # never start so late that fewer than this many control steps remain
        self.min_remaining_steps = int(cfg.get("min_remaining_steps", 8))
        self._rng = np.random.default_rng(seed)

    def sample_start_frame(self, num_frames: int) -> int:
        """Uniform start frame in [0, T-1-min_remaining]; 0 when disabled."""
        if not self.enabled:
            return 0
        high = max(num_frames - 1 - self.min_remaining_steps, 1)
        return int(self._rng.integers(0, high))


class EarlyTermination:
    """Per-episode failure detector evaluated after every control step.

    One instance is built per (episode, MjModel) pair: it precomputes the
    demo's palm world positions and palm-to-object distances for every frame
    (via ``mj_kinematics`` on a scratch ``MjData``), so the per-step check is
    a handful of vector ops.
    """

    def __init__(
        self,
        cfg: dict[str, Any] | None,
        model: mujoco.MjModel,
        episode: EpisodeData,
    ):
        cfg = dict(cfg) if cfg else {}
        self._spec = episode.spec
        self.enabled = bool(cfg.get("enabled", False))
        self.criteria = list(cfg.get("criteria", ["hand_object_distance"]))
        unknown = [c for c in self.criteria if c not in _VALID_CRITERIA]
        if unknown:
            raise ValueError(
                f"Unknown early-termination criteria {unknown}; valid: {_VALID_CRITERIA}"
            )
        # grace period (control steps since episode start) before checking
        self.min_steps = int(cfg.get("min_steps", 8))
        self.hand_object_dist_threshold = float(
            cfg.get("hand_object_dist_threshold", 0.10)
        )
        self.object_err_threshold = float(cfg.get("object_err_threshold", 0.25))

        if not self.enabled:
            return

        self._right_palm = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "right_palm"
        )
        self._left_palm = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "left_palm"
        )
        assert self._right_palm >= 0 and self._left_palm >= 0, (
            "scene must contain right_palm/left_palm bodies for early termination"
        )

        # Precompute demo palm-to-object distances per frame (kinematics only).
        demo = episode.qpos_demo
        scratch = mujoco.MjData(model)
        n = demo.shape[0]
        self.demo_tool_dist = np.empty(n)
        self.demo_target_dist = np.empty(n)
        tool_q, target_q = self._spec.tool_obj_qpos, self._spec.target_obj_qpos
        for t in range(n):
            scratch.qpos[:] = demo[t]
            mujoco.mj_kinematics(model, scratch)
            self.demo_tool_dist[t] = np.linalg.norm(
                scratch.xpos[self._right_palm]
                - demo[t, tool_q.start : tool_q.start + 3]
            )
            self.demo_target_dist[t] = np.linalg.norm(
                scratch.xpos[self._left_palm]
                - demo[t, target_q.start : target_q.start + 3]
            )

    def check(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        episode: EpisodeData,
        demo_frame: int,
        steps_in_episode: int,
    ) -> tuple[bool, str]:
        """Return (terminate?, reason) for the current sim state.

        Args:
            model/data: live sub-env simulation.
            episode: episode data (for the aligned demo frame).
            demo_frame: frame-aligned demo index (start_frame + executed steps).
            steps_in_episode: executed control steps since (re)set.
        """
        if not self.enabled or steps_in_episode < self.min_steps:
            return False, ""
        demo_frame = min(demo_frame, episode.num_frames - 1)
        demo = episode.qpos_demo[demo_frame]
        qpos = data.qpos
        tool_q, target_q = self._spec.tool_obj_qpos, self._spec.target_obj_qpos

        if "object_tracking" in self.criteria:
            tool_err = np.linalg.norm(
                qpos[tool_q.start : tool_q.start + 3]
                - demo[tool_q.start : tool_q.start + 3]
            )
            target_err = np.linalg.norm(
                qpos[target_q.start : target_q.start + 3]
                - demo[target_q.start : target_q.start + 3]
            )
            if max(tool_err, target_err) > self.object_err_threshold:
                return True, "object_tracking"

        if "hand_object_distance" in self.criteria:
            # refresh kinematics so palm xpos matches the integrated qpos
            mujoco.mj_kinematics(model, data)
            sim_tool_dist = np.linalg.norm(
                data.xpos[self._right_palm]
                - qpos[tool_q.start : tool_q.start + 3]
            )
            sim_target_dist = np.linalg.norm(
                data.xpos[self._left_palm]
                - qpos[target_q.start : target_q.start + 3]
            )
            escape_tool = sim_tool_dist - self.demo_tool_dist[demo_frame]
            escape_target = sim_target_dist - self.demo_target_dist[demo_frame]
            if max(escape_tool, escape_target) > self.hand_object_dist_threshold:
                return True, "hand_object_distance"

        return False, ""

    @property
    def hand_dim(self) -> int:
        return self._spec.hand_dim
