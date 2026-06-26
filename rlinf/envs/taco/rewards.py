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

"""Pluggable per-step rewards for the TACO bimanual-Allegro environment.

The reward type is selected via ``env.*.reward.type`` in the yaml config:

* ``tracking``  (default) - dense demo-tracking reward computed from the live
  sim state vs the frame-aligned demonstration. Cheap, fully in-env, and a
  reasonable shaping signal while the task reward is still undecided.
* ``zero``      - no in-env reward. Use this when the reward comes entirely
  from an external reward model (e.g. VLM-as-judge through RLinf's
  ``EmbodiedRewardWorker``; see ``VLMJudgeReward`` below).
* ``vlm_judge`` - placeholder for an in-env VLM-as-judge reward. Intentionally
  NOT implemented yet (design TBD); see the class docstring for the two
  integration options.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

if TYPE_CHECKING:
    from rlinf.envs.taco.scene import EpisodeData
    from rlinf.envs.taco.taco_env import _SubEnv


class BaseTacoReward(ABC):
    """Per-step reward interface.

    ``compute`` is called once per control step for one sub-env, AFTER physics
    has been stepped, with the sub-env exposing:

    * ``sub.data.qpos``      - live sim state (58,)
    * ``sub.demo_qpos(k)``   - frame-aligned demo state for executed step k
    * ``sub.steps``          - number of executed control steps so far
    """

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg

    @abstractmethod
    def compute(self, sub: "_SubEnv") -> tuple[float, dict[str, float]]:
        """Return (reward, info_dict) for the current step of one sub-env."""


class ZeroReward(BaseTacoReward):
    """No in-env reward (external reward model provides the signal)."""

    def compute(self, sub: "_SubEnv") -> tuple[float, dict[str, float]]:
        return 0.0, {}


def _subtree_geom_mask(model: mujoco.MjModel, root_body: int) -> np.ndarray:
    """Boolean (ngeom,) mask of geoms on ``root_body`` or any descendant."""
    in_subtree = np.zeros(model.nbody, dtype=bool)
    for b in range(model.nbody):
        cur = b
        while cur > 0:
            if cur == root_body:
                in_subtree[b] = True
                break
            cur = int(model.body_parentid[cur])
    return in_subtree[model.geom_bodyid]


def _body_of_qposadr(model: mujoco.MjModel, qposadr: int) -> int:
    """Body carrying the (free) joint whose qpos starts at ``qposadr``."""
    for j in range(model.njnt):
        if int(model.jnt_qposadr[j]) == qposadr:
            return int(model.jnt_bodyid[j])
    raise ValueError(f"no joint with qposadr={qposadr} in model")


class _ContactRef:
    """Per-(episode, MjModel) reference for the contact-consistency term.

    Precomputes (once, via ``mj_forward`` on a scratch ``MjData``):

    * geom masks for the right/left hand subtrees (rooted at ``right_palm`` /
      ``left_palm``) and for the tool/target free-joint objects;
    * per-demo-frame booleans ``demo_tool_contact[t]`` / ``demo_target_contact[t]``:
      does the demo state at frame t have right-hand<->tool (resp.
      left-hand<->target) contact?

    Mirrors Spider's contact reward design (reward contact only where the
    reference is in contact), with a binary contact test instead of contact-
    point positions (the TACO demos ship no contact-site labels).
    """

    def __init__(
        self, model: mujoco.MjModel, episode: "EpisodeData", dist_tol: float
    ):
        self.dist_tol = float(dist_tol)
        right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_palm")
        left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_palm")
        assert right >= 0 and left >= 0, (
            "scene must contain right_palm/left_palm bodies for the contact reward"
        )
        self.right_hand = _subtree_geom_mask(model, right)
        self.left_hand = _subtree_geom_mask(model, left)
        self.tool = _subtree_geom_mask(
            model, _body_of_qposadr(model, episode.tool_obj_qpos.start)
        )
        self.target = _subtree_geom_mask(
            model, _body_of_qposadr(model, episode.target_obj_qpos.start)
        )

        scratch = mujoco.MjData(model)
        n = episode.num_frames
        self.demo_tool_contact = np.zeros(n, dtype=bool)
        self.demo_target_contact = np.zeros(n, dtype=bool)
        for t in range(n):
            scratch.qpos[:] = episode.qpos_demo[t]
            mujoco.mj_forward(model, scratch)
            tool_c, target_c = self.sim_contacts(scratch)
            self.demo_tool_contact[t] = tool_c
            self.demo_target_contact[t] = target_c

    def sim_contacts(self, data: mujoco.MjData) -> tuple[bool, bool]:
        """(right_hand<->tool, left_hand<->target) contact flags for ``data``."""
        n = int(data.ncon)
        if n == 0:
            return False, False
        geom = data.contact.geom[:n]
        near = data.contact.dist[:n] < self.dist_tol
        g0, g1 = geom[:, 0], geom[:, 1]

        def touching(a: np.ndarray, b: np.ndarray) -> bool:
            return bool((((a[g0] & b[g1]) | (b[g0] & a[g1])) & near).any())

        return touching(self.right_hand, self.tool), touching(
            self.left_hand, self.target
        )


class TrackingReward(BaseTacoReward):
    """Dense demo-tracking reward (+ optional contact-consistency term).

    r_t = [ w_tool    * exp(-|tool_pos   - demo_tool_pos|   / s_tool)
          + w_target  * exp(-|target_pos - demo_target_pos| / s_target)
          + w_hand    * exp(-mean|hand_qpos - demo_hand_qpos| / s_hand)
          + w_contact * contact_match_t ] / W

    where W = sum of the active weights when ``normalize_by_weights`` (default),
    so the per-step reward is bounded in (0, 1] and the episode return scales
    with how long the rollout stays on the demo trajectory. Executed step k
    (1-based) is compared against demo frame min(k, T-1), matching the
    dataset's action_offset=1 convention.

    **Contact-consistency term** (``contact_weight`` > 0, default 0 = off; cf.
    Spider's contact reward / DeepMimic-style reference matching): for each
    (hand, object) pair - (right hand, tool) and (left hand, target) - the pair
    scores 1 when the aligned demo frame is NOT in contact (nothing required),
    and scores ``sim_in_contact`` (0/1) when the demo IS in contact;
    ``contact_match_t`` is the mean of the two pair scores. This directly
    rewards the *fact of grasping* during the demo's manipulation window and
    makes "hover near the object without touching it" strictly unprofitable -
    the reward-hacking mode found in the step-1000 evals of the IL-finetune
    runs. Spurious extra contact outside the demo window is deliberately not
    penalized (IK-retargeted demos have noisy contact phase boundaries).
    """

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        self.w_tool = float(cfg.get("tool_pos_weight", 1.0))
        self.w_target = float(cfg.get("target_pos_weight", 1.0))
        self.w_hand = float(cfg.get("hand_qpos_weight", 0.1))
        self.w_contact = float(cfg.get("contact_weight", 0.0))
        self.s_tool = float(cfg.get("tool_pos_scale", 0.05))  # meters
        self.s_target = float(cfg.get("target_pos_scale", 0.05))  # meters
        self.s_hand = float(cfg.get("hand_qpos_scale", 0.5))  # radians
        # a MuJoCo contact counts as touching when its dist < this tol (m)
        self.contact_dist_tol = float(cfg.get("contact_dist_tol", 1.0e-3))
        self.norm = (
            self.w_tool + self.w_target + self.w_hand + self.w_contact
            if bool(cfg.get("normalize_by_weights", True))
            else 1.0
        )
        # contact references are built lazily per episode (compute() runs in
        # the env's thread pool -> guard the cache with a lock)
        self._contact_refs: dict[str, _ContactRef] = {}
        self._contact_lock = threading.Lock()

    def _get_contact_ref(self, sub: "_SubEnv") -> _ContactRef:
        key = sub.episode.name
        ref = self._contact_refs.get(key)
        if ref is None:
            with self._contact_lock:
                ref = self._contact_refs.get(key)
                if ref is None:
                    ref = _ContactRef(sub.model, sub.episode, self.contact_dist_tol)
                    self._contact_refs[key] = ref
        return ref

    def compute(self, sub: "_SubEnv") -> tuple[float, dict[str, float]]:
        sim = sub.data.qpos
        demo = sub.demo_qpos(sub.steps)
        tool_obj_qpos = sub.episode.tool_obj_qpos
        target_obj_qpos = sub.episode.target_obj_qpos
        hand_dim = sub.episode.hand_dim

        tool_err = float(
            np.linalg.norm(
                sim[tool_obj_qpos.start : tool_obj_qpos.start + 3]
                - demo[tool_obj_qpos.start : tool_obj_qpos.start + 3]
            )
        )
        target_err = float(
            np.linalg.norm(
                sim[target_obj_qpos.start : target_obj_qpos.start + 3]
                - demo[target_obj_qpos.start : target_obj_qpos.start + 3]
            )
        )
        hand_err = float(np.abs(sim[:hand_dim] - demo[:hand_dim]).mean())

        weighted_sum = (
            self.w_tool * np.exp(-tool_err / self.s_tool)
            + self.w_target * np.exp(-target_err / self.s_target)
            + self.w_hand * np.exp(-hand_err / self.s_hand)
        )
        info = {
            "tool_pos_err": tool_err,
            "target_pos_err": target_err,
            "hand_qpos_err": hand_err,
        }

        if self.w_contact > 0.0:
            ref = self._get_contact_ref(sub)
            t = sub.demo_frame
            sim_tool_c, sim_target_c = ref.sim_contacts(sub.data)
            tool_score = float(sim_tool_c) if ref.demo_tool_contact[t] else 1.0
            target_score = (
                float(sim_target_c) if ref.demo_target_contact[t] else 1.0
            )
            contact_match = 0.5 * (tool_score + target_score)
            weighted_sum += self.w_contact * contact_match
            info["contact_match"] = contact_match

        return float(weighted_sum / self.norm), info


class VLMJudgeReward(BaseTacoReward):
    """VLM-as-judge reward - design TBD, intentionally not implemented.

    Two integration options once the judging scheme is decided:

    1. **RLinf-native (recommended for GPU VLMs).** Keep ``reward.type: zero``
       in the env and enable RLinf's external reward pipeline instead::

           reward:
             use_reward_model: True
             reward_mode: per_step | terminal | history_buffer
             reward_weight: ...
             env_reward_weight: ...

       The ``EnvWorker`` then ships observations (e.g. rendered ``main_images``
       frames from ``TacoEnv.capture_image``) to an ``EmbodiedRewardWorker``
       hosting the VLM (see ``rlinf/models/embodiment/reward/vlm_reward_model.py``
       for the existing VLM reward-model scaffolding), and mixes the returned
       score with the env reward. This keeps the (large) VLM off the env
       workers and onto its own GPU placement.

    2. **In-env (only for cheap/remote-API judges).** Implement ``compute``
       here: render the ``front`` camera at a low frequency, accumulate frames
       in ``sub``, query the VLM every N steps or at episode end, and return
       the score as a (sparse) reward. Beware: a blocking API call inside
       ``chunk_step`` stalls the whole rollout pipeline.
    """

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        raise NotImplementedError(
            "VLM-as-judge reward is not implemented yet (design TBD). "
            "Use reward.type: tracking, or reward.type: zero + the RLinf "
            "external reward-model pipeline (reward.use_reward_model: True). "
            "See VLMJudgeReward docstring for the integration plan."
        )

    def compute(self, sub: "_SubEnv") -> tuple[float, dict[str, float]]:
        raise NotImplementedError


_REWARD_REGISTRY: dict[str, type[BaseTacoReward]] = {
    "tracking": TrackingReward,
    "zero": ZeroReward,
    "vlm_judge": VLMJudgeReward,
}


def build_reward(cfg: dict[str, Any] | None) -> BaseTacoReward:
    """Build a reward from the ``env.*.reward`` config dict."""
    cfg = dict(cfg) if cfg else {}
    reward_type = str(cfg.get("type", "tracking"))
    if reward_type not in _REWARD_REGISTRY:
        raise ValueError(
            f"Unknown TACO reward type '{reward_type}'. "
            f"Available: {sorted(_REWARD_REGISTRY)}"
        )
    return _REWARD_REGISTRY[reward_type](cfg)
