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

"""Pure helpers for selecting and naming the best-reward trajectory video.

Kept free of MuJoCo / Ray / torch so the selection logic is unit-testable in
isolation. Used by ``EnvWorker`` (local pick) and ``EmbodiedRunner`` (global
pick).
"""

import math
import os
from typing import Optional, Sequence

import numpy as np


def _finite(value) -> Optional[float]:
    """Return ``value`` as a finite float, or ``None`` if missing/NaN/-inf."""
    if value is None:
        return None
    v = float(value)
    if math.isnan(v) or v == float("-inf"):
        return None
    return v


def pick_local_best(
    returns_per_stage: Sequence[Optional[Sequence[float]]],
) -> Optional[tuple[int, int, float]]:
    """Pick the max finite return across all stages.

    Args:
      returns_per_stage: one sequence of per-env returns per pipeline stage
          (``None`` or empty for stages with no data).

    Returns:
      ``(stage_id, env_idx, return_value)`` of the best finite return, or
      ``None`` when no finite candidate exists. Ties resolve to the lowest
      ``(stage_id, env_idx)``.
    """
    best: Optional[tuple[int, int, float]] = None
    for stage_id, rets in enumerate(returns_per_stage):
        if rets is None:
            continue
        for env_idx, raw in enumerate(rets):
            v = _finite(raw)
            if v is None:
                continue
            if best is None or v > best[2]:
                best = (stage_id, env_idx, v)
    return best


def select_global_best_rank(
    returns: Sequence[Optional[float]],
) -> Optional[int]:
    """Index of the max finite per-rank return (ties → lowest rank)."""
    best_rank: Optional[int] = None
    best_val = float("-inf")
    for rank, raw in enumerate(returns):
        v = _finite(raw)
        if v is None:
            continue
        if v > best_val:
            best_val = v
            best_rank = rank
    return best_rank


def build_best_artifact_path(
    base_dir: str, step: int, return_value: float, suffix: str = ".mp4"
) -> str:
    """Path ``<base_dir>/step_<step>_ret_<return:.3f><suffix>``.

    Used for both the saved replay data (``.npz``) and the offline-rendered
    video (``.mp4``), so a data file and its video share the same stem.
    """
    return os.path.join(base_dir, f"step_{step}_ret_{return_value:.3f}{suffix}")


def aligned_demo_indices(
    num_rollout_frames: int, start_frame: int, demo_len: int
) -> list[int]:
    """Demo frame index for each rollout frame, matching the reward alignment.

    Rollout frame ``i`` maps to demo frame ``min(start_frame + i, demo_len - 1)``
    -- the same rule the tracking reward uses (``_SubEnv.demo_qpos``). The clamp
    freezes the reference on its last frame when the rollout outlives the demo;
    requesting fewer frames than the demo simply truncates it.

    Args:
      num_rollout_frames: number of frames in the rollout trajectory.
      start_frame: RSI start frame of the episode (0 when RSI is off).
      demo_len: number of frames in the demonstration (``qpos_demo`` length).

    Returns:
      A list of length ``num_rollout_frames`` of demo frame indices.
    """
    last = demo_len - 1
    return [min(start_frame + i, last) for i in range(num_rollout_frames)]


def hstack_with_separator(
    left: np.ndarray,
    right: np.ndarray,
    sep_width: int = 2,
    sep_value: int = 255,
) -> np.ndarray:
    """Horizontally concat two equal-height RGB frames with a separator column.

    Left panel is the rollout, right panel is the reference demonstration. Pure
    numpy so it stays unit-testable without a GL backend.
    """
    h = left.shape[0]
    sep = np.full((h, sep_width, 3), sep_value, dtype=left.dtype)
    return np.concatenate([left, sep, right], axis=1)
