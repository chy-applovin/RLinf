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

"""Per-robot qpos layout + asset materialization specs for the TACO env.

The TACO bimanual scenes follow a fixed qpos convention (hand dofs first, then
two free-joint objects: tool, then target). Different hands have different dof
counts, so the env reads the layout from a ``RobotSpec`` instead of hardcoded
constants. Add a hand by adding one entry to ``ROBOT_SPECS``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["RobotSpec", "ROBOT_SPECS", "get_robot_spec"]


@dataclass(frozen=True)
class RobotSpec:
    """Everything the TACO env needs to know about a bimanual hand.

    Attributes:
        name: robot key (also the asset subdir under ``assets/robots/<name>/``).
        hand_dim: number of hand qpos dofs (== ``MjModel.nu`` position actuators).
        tool_obj_qpos: free-joint qpos slice of the right object (tool):
            pos(3) + quat wxyz(4).
        target_obj_qpos: free-joint qpos slice of the left object (target).
        mesh_alias: scene-expected mesh filename -> source filename in
            ``robot_assets_root``. Empty for robots whose scene mesh names match
            the on-disk files; non-empty when the dataset scenes reference a
            differently-named (e.g. baked/shared) mesh set.
    """

    name: str
    hand_dim: int
    tool_obj_qpos: slice
    target_obj_qpos: slice
    mesh_alias: dict = field(default_factory=dict)


ROBOT_SPECS: dict[str, RobotSpec] = {
    # 44 hand dofs + 2 free-joint objects (nq=58); matches the legacy
    # HAND_DIM/TOOL_OBJ_QPOS/TARGET_OBJ_QPOS constants in scene.py.
    "allegro": RobotSpec("allegro", 44, slice(44, 51), slice(51, 58)),
    # 56 hand dofs + 2 free-joint objects (nq=70). The dataset scenes reference
    # 6 fingertip meshes by a baked shared-geometry naming (DP_HB1_*/elastomer_
    # HB1_*) that the on-disk Spider mesh set stores under per-finger names;
    # left/right sources are byte-identical, so aliasing both sides to right_*
    # is faithful.
    "sharpa": RobotSpec(
        "sharpa",
        56,
        slice(56, 63),
        slice(63, 70),
        {
            "DP_HB1_4F.STL": "right_DP.STL",
            "DP_visual_HB1_4F.STL": "right_DP_visual.STL",
            "DP_HB1_TH.STL": "right_thumb_DP.STL",
            "DP_Visual_HB1_TH.STL": "right_thumb_DP_visual.STL",
            "elastomer_HB1_4F.STL": "elastomer.STL",
            "elastomer_HB1_TH.STL": "thumb_elastomer.STL",
        },
    ),
}


def get_robot_spec(name: str) -> RobotSpec:
    """Return the ``RobotSpec`` for ``name`` (raises on unknown robot)."""
    if name not in ROBOT_SPECS:
        raise ValueError(
            f"Unknown TACO robot '{name}'. Available: {sorted(ROBOT_SPECS)}"
        )
    return ROBOT_SPECS[name]
