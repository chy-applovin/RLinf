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

"""Scene / episode IO utilities for the TACO bimanual-Allegro dataset.

The logic here intentionally mirrors ``mujoco-spider-env/scripts/rollout_eval.py``
(the existing closed-loop eval harness of the flow-matching policy) so that the
RL observation distribution matches both the IL training data and the eval
pipeline:

* ``prepare_scene``        - episode ``scene.xml`` uses ``meshdir=../../../assets``,
  so we symlink the episode into a Spider "processed" asset layout first;
* ``load_canonical_cloud`` - the simulator never emits point clouds. We take the
  demo's frame-0 world cloud, express it in the object's frame-0 local frame
  (FPS-subsampled exactly like the dataset loader), and re-pose it with the
  live sim object pose at every control step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Default qpos layout of the bimanual Allegro TACO scenes (see Spider
# scripts/convert_kinematic_to_act.py): 44 hand dofs + 2 free-joint objects.
HAND_DIM = 44


def object_qpos_slices(hand_dim: int) -> tuple[slice, slice]:
    """Return (tool, target) free-joint qpos slices for a hand dimension."""
    hand_dim = int(hand_dim)
    return slice(hand_dim, hand_dim + 7), slice(hand_dim + 7, hand_dim + 14)


TOOL_OBJ_QPOS, TARGET_OBJ_QPOS = object_qpos_slices(HAND_DIM)


# --------------------------------------------------------------------- geometry
def quat2mat(q: np.ndarray) -> np.ndarray:
    """MuJoCo quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ]
    )


def obj_pose(qpos_obj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Free-joint qpos slice -> (position, rotation matrix)."""
    return qpos_obj[:3].copy(), quat2mat(qpos_obj[3:7])


def synth_obs_frame(
    qpos: np.ndarray, episode: "EpisodeData", need_tool: bool
) -> dict[str, np.ndarray]:
    """Synthesize one observation frame from an episode-layout qpos vector.

    Re-poses the canonical object-frame clouds with the object pose in ``qpos``
    and slices the hand qpos. Works for both the live sim state and any demo
    frame, so the closed-loop obs (``_SubEnv.obs_frame``) and the single-step
    demo-history obs share identical synthesis logic.
    """
    p, rot = obj_pose(qpos[episode.target_obj_qpos])
    frame = {
        "pointcloud": (episode.target_local @ rot.T + p).astype(np.float32),
        "qpos": qpos[: episode.hand_dim].astype(np.float32).copy(),
    }
    if need_tool:
        pr, rr = obj_pose(qpos[episode.tool_obj_qpos])
        frame["tool_pointcloud"] = (episode.tool_local @ rr.T + pr).astype(np.float32)
    return frame


def farthest_point_indices(
    points: np.ndarray, num_points: int, rng: np.random.Generator
) -> np.ndarray:
    """FPS indices for a single ``(N, C)`` cloud.

    Must stay numerically identical to
    ``flow_policy.data.dataset._farthest_point_indices`` so the synthesized RL
    observations match the IL training distribution.
    """
    n = points.shape[0]
    idx = np.empty(num_points, dtype=np.int64)
    start = int(rng.integers(n))
    idx[0] = start
    dist = np.sum((points - points[start]) ** 2, axis=1)
    for i in range(1, num_points):
        nxt = int(np.argmax(dist))
        idx[i] = nxt
        dist = np.minimum(dist, np.sum((points - points[nxt]) ** 2, axis=1))
    return idx


# ---------------------------------------------------------------------- episodes
def list_episodes(root: Path, trajectory_file: str) -> list[Path]:
    """All episode dirs under ``root`` that have a scene and a trajectory."""
    eps = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if (d / "scene.xml").exists() and (d / trajectory_file).exists():
            eps.append(d)
    return eps


def episode_category(name: str) -> str:
    """``<verb>__<tool>__<target>__<datetime>_<id>`` -> ``<verb>__<tool>__<target>``."""
    return name.rsplit("__", 1)[0]


def one_per_category(eps: list[Path]) -> list[Path]:
    """Keep the first (sorted) episode of each category, deterministically."""
    seen: dict[str, Path] = {}
    for ep in sorted(eps, key=lambda p: p.name):
        cat = episode_category(ep.name)
        if cat not in seen:
            seen[cat] = ep
    return [seen[c] for c in sorted(seen)]


def select_episodes(
    root: Path,
    trajectory_file: str,
    names: list[str] | None = None,
    categories: list[str] | None = None,
    use_one_per_category: bool = False,
    max_episodes: int | None = None,
) -> list[Path]:
    """Resolve the episode set used by one env worker (deterministic order)."""
    if names:
        eps = [root / e for e in names]
        missing = [str(e) for e in eps if not e.is_dir()]
        if missing:
            raise FileNotFoundError(f"episodes not found under {root}: {missing}")
    else:
        eps = list_episodes(root, trajectory_file)
    if categories:
        cats = set(categories)
        eps = [e for e in eps if episode_category(e.name) in cats]
    if use_one_per_category:
        eps = one_per_category(eps)
    if max_episodes is not None:
        eps = eps[: int(max_episodes)]
    if not eps:
        raise RuntimeError(
            f"No TACO episodes selected under {root} "
            f"(trajectory_file={trajectory_file}, names={names}, categories={categories})"
        )
    return eps


# ---------------------------------------------------------------------- scene io
def prepare_scene(
    episode_dir: Path,
    scene_root: Path,
    robot_assets: Path,
    robot_name: str = "allegro",
    robot_asset_glob: str = "*.stl",
    robot_asset_aliases: dict[str, str] | None = None,
) -> str:
    """Symlink the episode into a Spider processed layout; return scene.xml path.

    Layout (so ``meshdir=../../../assets/`` inside the episode scene resolves):

        {scene_root}/assets/robots/{robot_name}/assets/*
        {scene_root}/assets/objects/{tool_*,target_*}
        {scene_root}/{robot_name}/bimanual/{episode}/scene.xml
    """
    robot_name = str(robot_name)
    src = Path(robot_assets)
    rob = scene_root / "assets" / "robots" / robot_name / "assets"
    rob.mkdir(parents=True, exist_ok=True)
    for mesh in src.glob(str(robot_asset_glob)):
        link = rob / mesh.name
        if not link.exists():
            try:
                link.symlink_to(mesh)
            except FileExistsError:
                pass
    for alias, real in (robot_asset_aliases or {}).items():
        link = rob / str(alias)
        if not link.exists():
            try:
                link.symlink_to(src / str(real))
            except FileExistsError:
                pass
    objdst = scene_root / "assets" / "objects"
    objdst.mkdir(parents=True, exist_ok=True)
    for od in (episode_dir / "objects").iterdir():
        link = objdst / od.name
        if not link.exists():
            try:
                link.symlink_to(od)
            except FileExistsError:
                pass
    taskdir = scene_root / robot_name / "bimanual" / episode_dir.name
    taskdir.mkdir(parents=True, exist_ok=True)
    scene = taskdir / "scene.xml"
    if not scene.exists():
        try:
            scene.symlink_to(episode_dir / "scene.xml")
        except FileExistsError:
            pass
    return str(scene)


# ------------------------------------------------------------------ episode data
@dataclass
class EpisodeData:
    """Loaded, simulator-independent data of one TACO episode."""

    episode_dir: Path
    name: str
    category: str
    scene_xml: str
    qpos_demo: np.ndarray  # (T, 58) float64
    qvel_demo: np.ndarray  # (T, 56) float64
    frequency: float
    # canonical object-frame clouds (already FPS-subsampled to num_points)
    target_local: np.ndarray  # (K, 3) float64
    tool_local: np.ndarray | None  # (K, 3) float64, only for pc2_qpos
    hand_dim: int = HAND_DIM
    meta: dict = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return int(self.qpos_demo.shape[0])

    @property
    def tool_obj_qpos(self) -> slice:
        return object_qpos_slices(self.hand_dim)[0]

    @property
    def target_obj_qpos(self) -> slice:
        return object_qpos_slices(self.hand_dim)[1]


def _canonical_cloud(
    episode_dir: Path,
    pointcloud_dir: str,
    role: str,
    obj_slice: slice,
    qpos_demo: np.ndarray,
    pc_start: int,
    num_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Demo frame-0 world cloud expressed in the object's frame-0 local frame."""
    pc_root = episode_dir / pointcloud_dir
    files = sorted(pc_root.glob(f"{role}_*.npy"))
    if not files:
        raise FileNotFoundError(f"{episode_dir}: no {pointcloud_dir}/{role}_*.npy")
    w0 = np.load(files[0]).astype(np.float64)[pc_start]  # (K_raw, 3) world @ frame 0
    p, rot = obj_pose(qpos_demo[0, obj_slice])
    loc = (w0 - p) @ rot  # object-frame points
    idx = farthest_point_indices(loc, num_points, rng)
    return loc[idx]


def load_episode_data(
    episode_dir: Path,
    scene_root: Path,
    robot_assets: Path,
    trajectory_file: str,
    num_points: int,
    need_tool_cloud: bool,
    hand_dim: int | None = HAND_DIM,
    pointcloud_dir: str = "pointcloud",
    robot_name: str = "allegro",
    robot_asset_glob: str = "*.stl",
    robot_asset_aliases: dict[str, str] | None = None,
) -> EpisodeData:
    """Load everything needed to instantiate + observe one episode scene."""
    traj = np.load(episode_dir / trajectory_file, allow_pickle=True)
    qpos_demo = traj["qpos"].astype(np.float64)
    qvel_demo = traj["qvel"].astype(np.float64)
    if hand_dim is None:
        hand_dim = int(qpos_demo.shape[1] - 14)
    hand_dim = int(hand_dim)
    if hand_dim <= 0 or qpos_demo.shape[1] < hand_dim + 14:
        raise ValueError(
            f"{episode_dir.name}: invalid hand_dim={hand_dim} for "
            f"qpos shape {qpos_demo.shape}; expected at least hand_dim + 14"
        )
    tool_obj_qpos, target_obj_qpos = object_qpos_slices(hand_dim)
    pc_start = int(traj["pc_start"]) if "pc_start" in traj.files else 0
    freq = float(traj["frequency"]) if "frequency" in traj.files else 30.0

    # rng seed 0 matches rollout_eval.py so FPS picks identical points.
    rng = np.random.default_rng(0)
    target_local = _canonical_cloud(
        episode_dir, pointcloud_dir, "target", target_obj_qpos, qpos_demo, pc_start, num_points, rng
    )
    tool_local = None
    if need_tool_cloud:
        tool_local = _canonical_cloud(
            episode_dir, pointcloud_dir, "tool", tool_obj_qpos, qpos_demo, pc_start, num_points, rng
        )

    meta_path = episode_dir / pointcloud_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    scene_xml = prepare_scene(
        episode_dir,
        scene_root,
        robot_assets,
        robot_name=robot_name,
        robot_asset_glob=robot_asset_glob,
        robot_asset_aliases=robot_asset_aliases,
    )
    return EpisodeData(
        episode_dir=episode_dir,
        name=episode_dir.name,
        category=episode_category(episode_dir.name),
        scene_xml=scene_xml,
        qpos_demo=qpos_demo,
        qvel_demo=qvel_demo,
        frequency=freq,
        target_local=target_local,
        tool_local=tool_local,
        hand_dim=hand_dim,
        meta=meta,
    )
