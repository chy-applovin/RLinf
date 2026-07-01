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

from rlinf.envs.taco.robots import RobotSpec

# The TACO qpos layout (hand dofs first, then the tool and target free joints)
# is robot-parametrized via RobotSpec on both the CPU and GPU backends; see
# rlinf/envs/taco/robots.py.


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
    """Synthesize one observation frame from a single qpos vector.

    The qpos length is robot-dependent (e.g. 58 for Allegro, 70 for sharpa);
    the layout comes from ``episode.spec``. Re-poses the canonical object-frame
    clouds with the object pose in ``qpos`` and slices the hand qpos. Works for both the live sim state and any demo
    frame, so the closed-loop obs (``_SubEnv.obs_frame``) and the single-step
    demo-history obs share identical synthesis logic.
    """
    spec = episode.spec
    p, rot = obj_pose(qpos[spec.target_obj_qpos])
    frame = {
        "pointcloud": (episode.target_local @ rot.T + p).astype(np.float32),
        "qpos": qpos[: spec.hand_dim].astype(np.float32).copy(),
    }
    if need_tool:
        pr, rr = obj_pose(qpos[spec.tool_obj_qpos])
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
# Physics <option> injected into episode scenes that ship without one. The TACO
# dataset scenes omit <option>, so MuJoCo falls back to its heavy defaults:
# timestep=0.002 (=> 25 physics substeps per 20 Hz control step) and a Newton
# iteration cap of 100. Matching the Spider retargeting scenes
# (timestep=0.01 => 5 substeps, integrator=implicitfast) makes the physics
# ~4x cheaper per env at the same control rate. The iteration cap is left at 10
# for tidiness but is NOT the lever: MuJoCo's solver iterates to a tolerance and
# early-terminates, so 100 vs 10 measures the same. NOTE: the timestep change
# alters contact dynamics
# vs the 0.002 default -- validate RL training quality when enabling. Pass
# ``physics_option={}`` to prepare_scene/load_episode_data to keep the raw scene.
DEFAULT_PHYSICS_OPTION: dict[str, str] = {
    "timestep": "0.01",
    "iterations": "10",
    "ls_iterations": "50",
    "integrator": "implicitfast",
}


def _patch_scene_option(xml_text: str, option: dict[str, str] | None) -> str:
    """Return ``xml_text`` with a physics ``<option>`` set to ``option``.

    Inserts a new ``<option .../>`` right after the ``<mujoco>`` tag when the
    scene has none (the TACO case), or merges the attributes into an existing
    ``<option>``. ``option=None`` uses ``DEFAULT_PHYSICS_OPTION``; ``option={}``
    is an explicit opt-out (returns the text unchanged).
    """
    import re

    if option is None:
        option = DEFAULT_PHYSICS_OPTION
    if not option:
        return xml_text

    existing = re.search(r"<option\b[^>]*>", xml_text)
    if existing:
        tag = existing.group(0)
        self_close = tag.rstrip().endswith("/>")
        inner = tag.rstrip()[: -2 if self_close else -1]
        for k, v in option.items():
            if re.search(rf'\b{k}="[^"]*"', inner):
                inner = re.sub(rf'\b{k}="[^"]*"', f'{k}="{v}"', inner)
            else:
                inner = inner.rstrip() + f' {k}="{v}"'
        merged = inner + ("/>" if self_close else ">")
        return xml_text[: existing.start()] + merged + xml_text[existing.end() :]

    attrs = " ".join(f'{k}="{v}"' for k, v in option.items())
    return re.sub(
        r"(<mujoco\b[^>]*>)", rf"\1\n  <option {attrs}/>", xml_text, count=1
    )


def prepare_scene(
    episode_dir: Path,
    scene_root: Path,
    spec: RobotSpec,
    robot_assets: Path,
    physics_option: dict[str, str] | None = None,
) -> str:
    """Materialize the episode into a Spider processed layout; return scene.xml path.

    Layout (so ``meshdir=../../../assets/`` inside the episode scene resolves):

        {scene_root}/assets/robots/{spec.name}/assets/*.STL  (+ spec.mesh_alias)
        {scene_root}/assets/objects/{tool_*,target_*}
        {scene_root}/{spec.name}/bimanual/{episode}/scene.xml

    The episode ``scene.xml`` is written (not symlinked) with a physics
    ``<option>`` injected (see ``DEFAULT_PHYSICS_OPTION``); the written file lives
    at the same depth as the old symlink so ``meshdir=../../../assets/`` still
    resolves. Pass ``physics_option={}`` to keep the raw (slow-default) scene.
    """
    rob = scene_root / "assets" / "robots" / spec.name / "assets"
    rob.mkdir(parents=True, exist_ok=True)
    robot_assets = Path(robot_assets)
    for stl in list(robot_assets.glob("*.stl")) + list(robot_assets.glob("*.STL")):
        link = rob / stl.name
        if not link.exists():
            try:
                link.symlink_to(stl)
            except FileExistsError:
                pass
    # Bridge baked/shared scene mesh names to the on-disk source files.
    for alias, src in spec.mesh_alias.items():
        src_path = robot_assets / src
        if not src_path.exists():
            raise FileNotFoundError(
                f"robot '{spec.name}' mesh alias '{alias}' -> '{src}' not found "
                f"under robot_assets_root '{robot_assets}'."
            )
        link = rob / alias
        if not link.exists():
            try:
                link.symlink_to(src_path)
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
    taskdir = scene_root / spec.name / "bimanual" / episode_dir.name
    taskdir.mkdir(parents=True, exist_ok=True)
    scene = taskdir / "scene.xml"
    if not scene.exists():
        patched = _patch_scene_option(
            (episode_dir / "scene.xml").read_text(), physics_option
        )
        tmp = scene.with_suffix(".xml.tmp")
        tmp.write_text(patched)
        try:
            tmp.replace(scene)  # atomic; safe under concurrent env workers
        except FileExistsError:
            tmp.unlink(missing_ok=True)
    return str(scene)


# ------------------------------------------------------------------ episode data
@dataclass
class EpisodeData:
    """Loaded, simulator-independent data of one TACO episode."""

    episode_dir: Path
    name: str
    category: str
    scene_xml: str
    qpos_demo: np.ndarray  # (T, nq) float64; nq is robot-dependent (spec)
    qvel_demo: np.ndarray  # (T, nv) float64; nv is robot-dependent (spec)
    frequency: float
    # canonical object-frame clouds (already FPS-subsampled to num_points)
    target_local: np.ndarray  # (K, 3) float64
    tool_local: np.ndarray | None  # (K, 3) float64, only for pc2_qpos
    meta: dict = field(default_factory=dict)
    spec: RobotSpec | None = None  # robot qpos layout; set by load_episode_data

    @property
    def num_frames(self) -> int:
        return int(self.qpos_demo.shape[0])


def _canonical_cloud(
    episode_dir: Path,
    role: str,
    obj_slice: slice,
    qpos_demo: np.ndarray,
    pc_start: int,
    num_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Demo frame-0 world cloud expressed in the object's frame-0 local frame."""
    files = sorted((episode_dir / "pointcloud").glob(f"{role}_*.npy"))
    if not files:
        raise FileNotFoundError(f"{episode_dir}: no pointcloud/{role}_*.npy")
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
    spec: RobotSpec,
    physics_option: dict[str, str] | None = None,
) -> EpisodeData:
    """Load everything needed to instantiate + observe one episode scene.

    ``physics_option`` is forwarded to ``prepare_scene`` (see
    ``DEFAULT_PHYSICS_OPTION``): ``None`` injects the spider-matched physics
    ``<option>`` and ``{}`` keeps the raw dataset scene.
    """
    traj = np.load(episode_dir / trajectory_file, allow_pickle=True)
    qpos_demo = traj["qpos"].astype(np.float64)
    qvel_demo = traj["qvel"].astype(np.float64)
    pc_start = int(traj["pc_start"]) if "pc_start" in traj.files else 0
    freq = float(traj["frequency"]) if "frequency" in traj.files else 30.0

    # rng seed 0 matches rollout_eval.py so FPS picks identical points.
    rng = np.random.default_rng(0)
    target_local = _canonical_cloud(
        episode_dir, "target", spec.target_obj_qpos, qpos_demo, pc_start, num_points, rng
    )
    tool_local = None
    if need_tool_cloud:
        tool_local = _canonical_cloud(
            episode_dir, "tool", spec.tool_obj_qpos, qpos_demo, pc_start, num_points, rng
        )

    meta_path = episode_dir / "pointcloud" / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    scene_xml = prepare_scene(
        episode_dir, scene_root, spec, robot_assets, physics_option=physics_option
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
        meta=meta,
        spec=spec,
    )
