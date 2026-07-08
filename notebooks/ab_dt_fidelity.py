#!/usr/bin/env python
"""dt (physics timestep) A/B fidelity study for the TACO sharpa GPU sim.

For each case (backend, dt, substeps) roll the SAME open-loop action sequence
(demo qpos targets at 20 Hz, optionally + per-env seeded Gaussian noise) and
record per-control-step tracking rewards / errors and env-0 qpos trajectories.

CPU MuJoCo @ dt=0.002 is the reference (that is what the running CPU RL
experiments and the IL eval pipeline use).

Note on substeps: the 20 Hz control period is 0.05 s. dt=0.002/0.0025/0.005
divide it exactly (25/20/10 substeps); dt=0.004 does NOT (12.5) - the env
would round to 12 substeps making sim time drift 4% per control step vs the
demo frame clock, so it is included here only to quantify that misalignment.

Outputs one npz per case under --out (default /root/RLinf/logs/ab_dt/):
  rewards (B, T), tool_err (B, T), target_err (B, T), hand_err (B, T),
  returns (B,), qpos0 (T+1, nq), nan_frac (T,), ms_per_ctrl_step (scalar)

Usage:
    .venv/bin/python notebooks/ab_dt_fidelity.py run --noise 0.0
    .venv/bin/python notebooks/ab_dt_fidelity.py run --noise 0.01
    xvfb-run -a env MUJOCO_GL=glfw .venv/bin/python \
        notebooks/ab_dt_fidelity.py render
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

SCENE = "/tmp/taco_scene_root/proc_1/sharpa/bimanual/brush__brush__helmet__20231027_011/scene.xml"
TRAJ = "/root/Data/taco-brush-sharpa-20hz-both/brush__brush__helmet__20231027_011/trajectory_mpc_20hz.npz"
OUT = Path("/root/RLinf/logs/ab_dt")

HAND = 56
TOOL = slice(56, 63)
TARGET = slice(63, 70)

# reward config of the running experiments (taco_sharpa20hz_tp4_raft_*)
W_TOOL, W_TARGET, W_HAND = 1.0, 1.0, 1.0
S_TOOL, S_TARGET, S_HAND = 0.05, 0.05, 0.35
NORM = W_TOOL + W_TARGET + W_HAND

CASES = {
    # name: (backend, dt, substeps)
    "cpu_dt2": ("cpu", 0.002, 25),
    "gpu_dt2": ("gpu", 0.002, 25),
    "gpu_dt25": ("gpu", 0.0025, 20),
    "gpu_dt4": ("gpu", 0.004, 12),   # 0.05/0.004=12.5 -> drifts vs demo clock
    "gpu_dt5": ("gpu", 0.005, 10),
}


def tracking_reward(qpos, demo):
    """qpos (B, nq) or (nq,), demo (nq,) -> reward, tool_err, target_err, hand_err."""
    qpos = np.atleast_2d(qpos)
    tool_err = np.linalg.norm(qpos[:, TOOL][:, :3] - demo[TOOL][:3], axis=1)
    target_err = np.linalg.norm(qpos[:, TARGET][:, :3] - demo[TARGET][:3], axis=1)
    hand_err = np.abs(qpos[:, :HAND] - demo[:HAND]).mean(axis=1)
    r = (
        W_TOOL * np.exp(-tool_err / S_TOOL)
        + W_TARGET * np.exp(-target_err / S_TARGET)
        + W_HAND * np.exp(-hand_err / S_HAND)
    ) / NORM
    return r, tool_err, target_err, hand_err


def make_actions(qpos_demo, num_envs, steps, noise, seed=0):
    """(steps, B, HAND) demo qpos targets + per-env seeded noise (same across cases)."""
    acts = np.empty((steps, num_envs, HAND), dtype=np.float64)
    for t in range(steps):
        acts[t] = qpos_demo[min(t + 1, len(qpos_demo) - 1), :HAND]
    if noise > 0:
        rng = np.random.default_rng(seed)
        acts += rng.normal(0.0, noise, acts.shape)
    return acts


def run_cpu(dt, substeps, actions, qpos_demo, qvel_demo):
    num_envs = actions.shape[1]
    m = mujoco.MjModel.from_xml_path(SCENE)
    m.opt.timestep = dt
    T = actions.shape[0]
    rewards = np.zeros((num_envs, T))
    errs = np.zeros((3, num_envs, T))
    qpos0 = np.zeros((T + 1, m.nq))
    t_wall = 0.0
    for b in range(num_envs):
        d = mujoco.MjData(m)
        d.qpos[:] = qpos_demo[0]
        d.qvel[:] = qvel_demo[0]
        mujoco.mj_forward(m, d)
        if b == 0:
            qpos0[0] = d.qpos
        t0 = time.perf_counter()
        for t in range(T):
            d.ctrl[:] = actions[t, b]
            for _ in range(substeps):
                mujoco.mj_step(m, d)
            demo = qpos_demo[min(t + 1, len(qpos_demo) - 1)]
            r, te, ge, he = tracking_reward(d.qpos, demo)
            rewards[b, t], errs[0, b, t], errs[1, b, t], errs[2, b, t] = r[0], te[0], ge[0], he[0]
            if b == 0:
                qpos0[t + 1] = d.qpos
        t_wall += time.perf_counter() - t0
    nan_frac = np.zeros(T)
    ms = t_wall / (num_envs * T) * 1e3
    return rewards, errs, qpos0, nan_frac, ms


def run_gpu(dt, substeps, actions, qpos_demo, qvel_demo,
            iterations=10, ls_iterations=20, njmax=350, nconmax=100):
    import warp as wp

    import mujoco_warp as mjwarp

    num_envs = actions.shape[1]
    m_cpu = mujoco.MjModel.from_xml_path(SCENE)
    m_cpu.opt.timestep = dt
    m_cpu.opt.iterations = iterations
    m_cpu.opt.ls_iterations = ls_iterations
    d_cpu = mujoco.MjData(m_cpu)
    d_cpu.qpos[:] = qpos_demo[0]
    d_cpu.qvel[:] = qvel_demo[0]
    mujoco.mj_forward(m_cpu, d_cpu)

    wp.init()
    m = mjwarp.put_model(m_cpu)
    if not wp.is_conditional_graph_supported():
        m.opt.graph_conditional = False
    d = mjwarp.put_data(m_cpu, d_cpu, nworld=num_envs, nconmax=nconmax, njmax=njmax)
    mjwarp.step(m, d)
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        mjwarp.step(m, d)
    graph = capture.graph
    wp.synchronize()

    # reset state after warmup/capture steps
    wp.copy(d.qpos, wp.array(np.tile(qpos_demo[0], (num_envs, 1)), dtype=float))
    wp.copy(d.qvel, wp.array(np.tile(qvel_demo[0], (num_envs, 1)), dtype=float))
    wp.copy(d.qacc_warmstart, wp.zeros((num_envs, m_cpu.nv), dtype=float))
    wp.copy(d.time, wp.zeros(num_envs, dtype=float))
    mjwarp.forward(m, d)
    wp.synchronize()

    T = actions.shape[0]
    rewards = np.zeros((num_envs, T))
    errs = np.zeros((3, num_envs, T))
    qpos0 = np.zeros((T + 1, m_cpu.nq))
    nan_frac = np.zeros(T)
    peak_nefc = 0
    peak_nacon_per_env = 0.0
    qpos0[0] = d.qpos.numpy()[0]
    t0 = time.perf_counter()
    for t in range(T):
        wp.copy(d.ctrl, wp.array(actions[t], dtype=float))
        for _ in range(substeps):
            wp.capture_launch(graph)
        wp.synchronize()
        qpos = d.qpos.numpy()
        demo = qpos_demo[min(t + 1, len(qpos_demo) - 1)]
        r, te, ge, he = tracking_reward(qpos, demo)
        rewards[:, t], errs[0, :, t], errs[1, :, t], errs[2, :, t] = r, te, ge, he
        nan_frac[t] = np.isnan(qpos).any(axis=1).mean()
        peak_nefc = max(peak_nefc, int(d.nefc.numpy().max()))
        peak_nacon_per_env = max(peak_nacon_per_env, float(d.nacon.numpy()[0]) / num_envs)
        qpos0[t + 1] = qpos[0]
    t_wall = time.perf_counter() - t0
    ms = t_wall / T * 1e3
    print(f"    peak nefc/world {peak_nefc} (njmax={njmax}) | "
          f"peak nacon/world {peak_nacon_per_env:.0f} (nconmax={nconmax})")
    return rewards, errs, qpos0, nan_frac, ms


def cmd_run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    traj = np.load(TRAJ, allow_pickle=True)
    qpos_demo = traj["qpos"].astype(np.float64)
    qvel_demo = traj["qvel"].astype(np.float64)
    T = min(98, len(qpos_demo) - 1)

    tag = "noise0" if args.noise == 0 else f"noise{args.noise:g}"
    tag += args.tag
    for name in args.cases.split(","):
        backend, dt, substeps = CASES[name]
        n = args.cpu_envs if backend == "cpu" else args.num_envs
        actions = make_actions(qpos_demo, n, T, args.noise, seed=0)
        if backend == "cpu":
            rewards, errs, qpos0, nan_frac, ms = run_cpu(
                dt, substeps, actions, qpos_demo, qvel_demo
            )
        else:
            rewards, errs, qpos0, nan_frac, ms = run_gpu(
                dt, substeps, actions, qpos_demo, qvel_demo, njmax=args.njmax,
                iterations=args.iterations, ls_iterations=args.ls_iterations,
            )
        returns = np.nansum(rewards, axis=1)
        path = OUT / f"{name}_{tag}.npz"
        np.savez_compressed(
            path, rewards=rewards, tool_err=errs[0], target_err=errs[1],
            hand_err=errs[2], returns=returns, qpos0=qpos0, nan_frac=nan_frac,
            ms_per_ctrl_step=ms, dt=dt, substeps=substeps,
        )
        print(f"[{name} {tag}] T={T} envs={rewards.shape[0]} "
              f"{ms:7.1f} ms/ctrl-step | reward mean {np.nanmean(rewards):.4f} "
              f"final tool_err {np.nanmean(errs[0][:, -1]):.4f} m | "
              f"nan_frac end {nan_frac[-1]:.4f} -> {path.name}")


def cmd_render(args):
    import imageio.v2 as imageio

    m = mujoco.MjModel.from_xml_path(SCENE)
    h, w = 480, 640
    renderer = mujoco.Renderer(m, h, w)
    cam = "front" if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "front") >= 0 else 0
    d = mujoco.MjData(m)

    traj = np.load(TRAJ, allow_pickle=True)
    demo_qpos = traj["qpos"].astype(np.float64)

    cases = args.cases.split(",")
    tag = "noise0" if args.noise == 0 else f"noise{args.noise:g}"
    trajs = {"demo": demo_qpos}
    for name in cases:
        trajs[name] = np.load(OUT / f"{name}_{tag}.npz")["qpos0"]

    T = min(len(t) for t in trajs.values())
    frames = []
    for t in range(0, T, args.stride):
        row = []
        for name, qp in trajs.items():
            d.qpos[:] = np.nan_to_num(qp[t])
            mujoco.mj_forward(m, d)
            renderer.update_scene(d, camera=cam)
            img = renderer.render().copy()
            img[:26] = img[:26] // 2
            row.append(img)
        frames.append(np.concatenate(row, axis=1))
    out = OUT / f"dt_ab_{tag}.mp4"
    imageio.mimsave(out, frames, fps=max(1, 20 // args.stride), macro_block_size=1)
    print(f"wrote {out} panels={list(trajs)} frames={len(frames)}")
    # keyframes for the notebook
    for frac in (0.25, 0.5, 0.99):
        idx = int(frac * (len(frames) - 1))
        p = OUT / f"dt_ab_{tag}_f{int(frac * 100):02d}.png"
        imageio.imwrite(p, frames[idx])
        print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cases", default=",".join(CASES))
    r.add_argument("--num-envs", type=int, default=256)
    r.add_argument("--cpu-envs", type=int, default=1)
    r.add_argument("--noise", type=float, default=0.0)
    r.add_argument("--tag", default="", help="suffix for repeat runs, e.g. _rep")
    r.add_argument("--njmax", type=int, default=800)
    r.add_argument("--iterations", type=int, default=10)
    r.add_argument("--ls-iterations", type=int, default=20)
    r.set_defaults(func=cmd_run)
    v = sub.add_parser("render")
    v.add_argument("--cases", default="cpu_dt2,gpu_dt2,gpu_dt25,gpu_dt5")
    v.add_argument("--noise", type=float, default=0.0)
    v.add_argument("--stride", type=int, default=2)
    v.set_defaults(func=cmd_render)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
