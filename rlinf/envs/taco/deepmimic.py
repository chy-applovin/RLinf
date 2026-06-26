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

__all__ = ["compute_contact_frame", "RSISampler", "EarlyTermination"]

_VALID_CRITERIA = ("hand_object_distance", "object_tracking")


def compute_contact_frame(
    qpos_demo: np.ndarray,
    spec,
    threshold_m: float = 1e-3,
    which: str = "earliest",
) -> int:
    """First demo frame where a manipulated object's translation has moved
    more than ``threshold_m`` from its frame-0 position (= "object starts
    moving" = contact). ``which``: ``earliest`` (min of tool/target) | ``tool``
    | ``target``. Returns a frame in ``[1, T-1]``; returns ``T-1`` if no object
    ever moves (degenerate demo) so the sampler still has a valid range.
    """
    qpos_demo = np.asarray(qpos_demo)
    num_frames = qpos_demo.shape[0]

    def _first_move(obj_slice: slice) -> int:
        pos = qpos_demo[:, obj_slice.start : obj_slice.start + 3]
        disp = np.linalg.norm(pos - pos[0], axis=1)
        moved = np.nonzero(disp > threshold_m)[0]
        return int(moved[0]) if moved.size else num_frames - 1

    if which == "tool":
        cf = _first_move(spec.tool_obj_qpos)
    elif which == "target":
        cf = _first_move(spec.target_obj_qpos)
    elif which == "earliest":
        cf = min(_first_move(spec.tool_obj_qpos), _first_move(spec.target_obj_qpos))
    else:
        raise ValueError(f"unknown contact_object '{which}'")
    return int(np.clip(cf, 1, num_frames - 1))


class RSISampler:
    """Samples per-reset reference start frames from the demo's PRE-CONTACT
    window (uniform in [0, contact_frame))."""

    def __init__(self, cfg: dict[str, Any] | None, seed: int):
        cfg = dict(cfg) if cfg else {}
        self.enabled = bool(cfg.get("enabled", False))
        # When True, every env in a GRPO group shares ONE sampled start frame (the
        # env also already shares the group's episode), so a group is a set of
        # stochastic rollouts from one identical initial state. This makes the GRPO
        # group-mean a per-state baseline V(s_start) and cancels the RSI start-frame
        # difficulty bias in the advantage. When False (default) every env samples
        # its own start frame independently (legacy behavior).
        self.group_shared = bool(cfg.get("group_shared", False))
        # never start so late that fewer than this many control steps remain
        self.min_remaining_steps = int(cfg.get("min_remaining_steps", 8))
        # object "starts moving" (= contact) when displaced more than this (m)
        self.contact_threshold_m = float(cfg.get("contact_pos_threshold_m", 1e-3))
        self.contact_object = str(cfg.get("contact_object", "earliest"))
        self._rng = np.random.default_rng(seed)
        self._contact_cache: dict[str, int] = {}  # episode.name -> contact frame

    def _high(self, episode: EpisodeData, spec) -> int:
        """Exclusive upper bound for sampling: before contact AND leaving
        >= min_remaining_steps before the demo end."""
        cf = self._contact_cache.get(episode.name)
        if cf is None:
            cf = compute_contact_frame(
                episode.qpos_demo, spec, self.contact_threshold_m, self.contact_object
            )
            self._contact_cache[episode.name] = cf
        return max(min(cf, episode.num_frames - 1 - self.min_remaining_steps), 1)

    def sample_start_frame(self, episode: EpisodeData, spec) -> int:
        """Pre-contact start frame; 0 when disabled."""
        if not self.enabled:
            return 0
        return int(self._rng.integers(0, self._high(episode, spec)))

    def sample_start_frames(self, episode: EpisodeData, spec, n: int) -> np.ndarray:
        """``n`` independent pre-contact start frames (GPU per-world); zeros when
        disabled."""
        if not self.enabled:
            return np.zeros(n, dtype=np.int64)
        return self._rng.integers(0, self._high(episode, spec), size=n).astype(np.int64)


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
