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
