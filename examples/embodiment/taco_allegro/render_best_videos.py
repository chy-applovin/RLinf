#!/usr/bin/env python3
"""Render best-reward trajectory videos OFFLINE from saved replay data.

Training (run_sharpa.sh with SAVE_BEST_VIDEO=1, SIM_BACKEND=cpu) dumps, per
best episode, a tiny ``step_*_ret_*.npz`` (qpos trajectory + episode name) and,
once, the resolved ``env_train_cfg.yaml``. This script rebuilds each episode's
MuJoCo model from the dataset (via the saved config), replays the qpos frames,
and writes an mp4 of the same stem -- on any machine with a working GL backend
that also has the repo + TACO dataset.

Usage:
    # pick a GL backend that works on THIS machine:
    MUJOCO_GL=egl    python examples/embodiment/taco_allegro/render_best_videos.py <best_dir>
    MUJOCO_GL=osmesa python examples/embodiment/taco_allegro/render_best_videos.py <best_dir>
    MUJOCO_GL=glfw   python examples/embodiment/taco_allegro/render_best_videos.py <best_dir>   # needs a display

    # <best_dir> is env.train.video_cfg.best_reward_video_dir
    #   (default <log_path>/video/train/best).
    # Options:
    #   --fps 30          output frame rate
    #   --overwrite       re-render even if the mp4 already exists
    #   --no-reference    render rollout only (no side-by-side demo panel)
    #   --glob 'step_*'   filename pattern (without .npz)
    #   --cfg PATH        env config yaml (default <best_dir>/env_train_cfg.yaml)

Each mp4 shows, side-by-side, the single best rollout (LEFT) and the reference
demonstration it imitates (RIGHT), frame-aligned. Pass --no-reference for the
rollout only.
Requires EMBODIED_PATH (and any asset env vars) only if env_train_cfg.yaml was
saved unresolved; normally it is saved resolved and needs nothing extra.
"""

from __future__ import annotations

import argparse
import glob
import os

import imageio
import mujoco
import numpy as np
from omegaconf import OmegaConf

from rlinf.envs.taco.taco_env import TacoEnv
from rlinf.utils.best_video import aligned_demo_indices, hstack_with_separator

_ENV_CACHE: dict[str, TacoEnv] = {}


def _quat_qpos_slices(model: "mujoco.MjModel") -> list[tuple[int, int]]:
    """qpos index ranges that hold unit quaternions (free/ball joints).

    MuJoCo lays a free joint out as [3 translation, 4 quaternion (wxyz)] and a
    ball joint as [4 quaternion]. These must be slerp-ed, not linearly blended,
    or interpolated object orientations drift off the unit sphere and wobble.
    """
    slices: list[tuple[int, int]] = []
    for j in range(model.njnt):
        adr = int(model.jnt_qposadr[j])
        jt = model.jnt_type[j]
        if jt == mujoco.mjtJoint.mjJNT_FREE:
            slices.append((adr + 3, adr + 7))
        elif jt == mujoco.mjtJoint.mjJNT_BALL:
            slices.append((adr, adr + 4))
    return slices


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two wxyz quaternions."""
    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:  # take the shorter arc
        q1 = -q1
        dot = -dot
    if dot > 0.9995:  # nearly parallel -> linear + renormalize
        q = q0 + t * (q1 - q0)
        return q / (np.linalg.norm(q) + 1e-12)
    theta0 = np.arccos(dot)
    theta = theta0 * t
    s0 = np.sin(theta0 - theta) / np.sin(theta0)
    s1 = np.sin(theta) / np.sin(theta0)
    return s0 * q0 + s1 * q1


def _interpolate_qpos(qpos: np.ndarray, model: "mujoco.MjModel", k: int) -> np.ndarray:
    """Insert k-1 smooth sub-frames between each saved control-step pose.

    The replay npz stores ONE qpos per 5 Hz control step (100 physics substeps
    are collapsed into one frame), so straight playback shows only ~13 distinct
    poses and looks choppy. This linearly interpolates the non-quaternion dofs
    and slerps the free/ball-joint quaternions to fabricate intermediate frames
    -- a visual approximation of the discarded substep path, not the true one.
    """
    if k <= 1 or len(qpos) < 2:
        return qpos
    quat_slices = _quat_qpos_slices(model)
    out: list[np.ndarray] = []
    for a, b in zip(qpos[:-1], qpos[1:]):
        for i in range(k):
            t = i / k
            frame = a + t * (b - a)  # linear default for all dofs
            for s, e in quat_slices:  # override quaternion blocks with slerp
                frame[s:e] = _slerp(a[s:e], b[s:e], t)
            out.append(frame)
    out.append(qpos[-1].copy())  # keep the final pose exactly
    return np.asarray(out)


def _build_env_for_episode(cfg, episode: str) -> TacoEnv:
    """Build (and cache) a single-env TacoEnv pinned to one episode.

    Samplers are disabled so construction is simple; the recorded qpos fully
    determines each rendered frame, so the start state does not matter -- only
    the episode's model (geometry) does.
    """
    if episode in _ENV_CACHE:
        return _ENV_CACHE[episode]
    c = cfg.copy()
    OmegaConf.set_struct(c, False)
    c.episodes = [episode]
    c.group_size = 1
    c.use_fixed_reset_state_ids = True
    c.auto_reset = False
    for sampler in ("rsi", "early_termination", "single_step", "demo_start"):
        if c.get(sampler) is not None:
            c[sampler]["enabled"] = False
    env = TacoEnv(cfg=c, num_envs=1, seed_offset=0, total_num_processes=1)
    env.reset()
    _ENV_CACHE[episode] = env
    return env


def _render_qpos_sequence(env, sub, qpos: np.ndarray) -> list:
    """Render each qpos pose through the env's capture pipeline."""
    frames = []
    for frame_qpos in qpos:
        sub.data.qpos[:] = frame_qpos
        mujoco.mj_forward(sub.model, sub.data)
        frames.append(env.capture_image().copy())
    return frames


def render_npz(
    npz_path: str,
    cfg,
    fps: int,
    overwrite: bool,
    interp: int = 1,
    reference: bool = True,
) -> "str | None":
    """Render one replay npz to an mp4; return the output path (or None).

    With ``reference=True`` (default) the demonstration the policy imitates is
    rendered to the right of the rollout (left), frame-aligned via
    ``aligned_demo_indices``. ``start_frame`` defaults to 0 for older npz files
    that predate it (correct when RSI was off).
    """
    out_path = (
        npz_path[:-4] + ".mp4" if npz_path.endswith(".npz") else npz_path + ".mp4"
    )
    if os.path.exists(out_path) and not overwrite:
        print(f"[skip] {out_path} exists (use --overwrite)")
        return None

    data = np.load(npz_path, allow_pickle=False)
    qpos = data["qpos"]  # [T+1, nq]
    episode = str(data["episode"])
    start_frame = int(data["start_frame"]) if "start_frame" in data.files else 0

    env = _build_env_for_episode(cfg, episode)
    sub = env.subenvs[0]

    rollout_qpos = _interpolate_qpos(qpos, sub.model, interp)
    rollout_frames = _render_qpos_sequence(env, sub, rollout_qpos)

    frames = rollout_frames
    if reference:
        try:
            demo = sub.episode.qpos_demo  # [T, nq], same model/space as rollout
            idx = aligned_demo_indices(len(qpos), start_frame, len(demo))
            ref_qpos = _interpolate_qpos(demo[idx], sub.model, interp)
            ref_frames = _render_qpos_sequence(env, sub, ref_qpos)
            frames = [
                hstack_with_separator(roll, ref)
                for roll, ref in zip(rollout_frames, ref_frames)
            ]
        except Exception as exc:  # degrade to rollout-only on a demo-side failure
            print(
                f"[warn] {npz_path}: reference render failed "
                f"({type(exc).__name__}: {exc}); rollout-only."
            )
            frames = rollout_frames

    writer = imageio.get_writer(out_path, fps=fps)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()
    print(
        f"[ok]   {out_path}  ({len(frames)} frames, episode={episode}, "
        f"reference={'on' if reference and frames is not rollout_frames else 'off'})"
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "best_dir",
        help="Directory holding step_*_ret_*.npz and env_train_cfg.yaml.",
    )
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument(
        "--interp",
        type=int,
        default=1,
        help=(
            "Sub-frames per saved control step (slerp for quats + linear for "
            "the rest). 1 = raw ~13 keyframes (choppy); 10 = smooth. "
            "Approximates the discarded 100 physics substeps per control step."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-reference",
        dest="reference",
        action="store_false",
        help="Render the rollout only (default: rollout | reference side-by-side).",
    )
    parser.set_defaults(reference=True)
    parser.add_argument(
        "--glob",
        default="step_*_ret_*",
        help="Filename pattern (without .npz) to match.",
    )
    parser.add_argument(
        "--cfg",
        default=None,
        help="env config yaml (default: <best_dir>/env_train_cfg.yaml).",
    )
    args = parser.parse_args()

    cfg_path = args.cfg or os.path.join(args.best_dir, "env_train_cfg.yaml")
    if not os.path.exists(cfg_path):
        raise SystemExit(f"env config not found: {cfg_path} (pass --cfg)")
    cfg = OmegaConf.load(cfg_path)

    npz_files = sorted(glob.glob(os.path.join(args.best_dir, f"{args.glob}.npz")))
    if not npz_files:
        print(f"No replay files matched {args.glob}.npz in {args.best_dir}")
        return

    backend = os.environ.get("MUJOCO_GL", "<default>")
    print(f"Rendering {len(npz_files)} file(s) with MUJOCO_GL={backend}")
    rendered = 0
    for npz_path in npz_files:
        try:
            if render_npz(
                npz_path, cfg, args.fps, args.overwrite, args.interp, args.reference
            ):
                rendered += 1
        except Exception as exc:  # keep going on a single bad file
            print(f"[fail] {npz_path}: {type(exc).__name__}: {exc}")
    print(f"Done: {rendered}/{len(npz_files)} rendered.")


if __name__ == "__main__":
    main()
