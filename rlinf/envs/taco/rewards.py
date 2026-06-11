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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
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


class TrackingReward(BaseTacoReward):
    """Dense demo-tracking reward.

    r_t = w_tool   * exp(-|tool_pos   - demo_tool_pos|   / s_tool)
        + w_target * exp(-|target_pos - demo_target_pos| / s_target)
        + w_hand   * exp(-mean|hand_qpos - demo_hand_qpos| / s_hand)

    Each term is bounded in (0, w], so the per-step reward is bounded and the
    episode return scales with how long the rollout stays on the demo
    trajectory. Executed step k (1-based) is compared against demo frame
    min(k, T-1), matching the dataset's action_offset=1 convention.
    """

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        self.w_tool = float(cfg.get("tool_pos_weight", 1.0))
        self.w_target = float(cfg.get("target_pos_weight", 1.0))
        self.w_hand = float(cfg.get("hand_qpos_weight", 0.1))
        self.s_tool = float(cfg.get("tool_pos_scale", 0.05))  # meters
        self.s_target = float(cfg.get("target_pos_scale", 0.05))  # meters
        self.s_hand = float(cfg.get("hand_qpos_scale", 0.5))  # radians

    def compute(self, sub: "_SubEnv") -> tuple[float, dict[str, float]]:
        from rlinf.envs.taco.scene import HAND_DIM, TARGET_OBJ_QPOS, TOOL_OBJ_QPOS

        sim = sub.data.qpos
        demo = sub.demo_qpos(sub.steps)

        tool_err = float(
            np.linalg.norm(
                sim[TOOL_OBJ_QPOS.start : TOOL_OBJ_QPOS.start + 3]
                - demo[TOOL_OBJ_QPOS.start : TOOL_OBJ_QPOS.start + 3]
            )
        )
        target_err = float(
            np.linalg.norm(
                sim[TARGET_OBJ_QPOS.start : TARGET_OBJ_QPOS.start + 3]
                - demo[TARGET_OBJ_QPOS.start : TARGET_OBJ_QPOS.start + 3]
            )
        )
        hand_err = float(np.abs(sim[:HAND_DIM] - demo[:HAND_DIM]).mean())

        reward = (
            self.w_tool * np.exp(-tool_err / self.s_tool)
            + self.w_target * np.exp(-target_err / self.s_target)
            + self.w_hand * np.exp(-hand_err / self.s_hand)
        )
        info = {
            "tool_pos_err": tool_err,
            "target_pos_err": target_err,
            "hand_qpos_err": hand_err,
        }
        return float(reward), info


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
