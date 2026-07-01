#!/usr/bin/env python
"""Standalone mjwarp benchmark for the TACO sharpa scene.

Quantifies the GPU-sim rollout cost factors independently of RLinf:

  1. no CUDA graph (current fallback: graph capture fails on driver < 12.4)
  2. graph_conditional=False + CUDA graph, default solver iterations (100)
  3. graph + tuned iterations
  4. Newton vs CG solver
  5. nconmax semantics bug (per-world vs total)
  6. capturing all substeps of one control step in a single graph

Usage:
    python bench_mjwarp_sharpa.py --num-envs 512 --device cuda:0
"""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

SCENE = "/tmp/taco_scene_root/proc_1/sharpa/bimanual/brush__brush__helmet__20231027_011/scene.xml"
TRAJ = "/root/Data/taco-brush-sharpa-20hz-both/brush__brush__helmet__20231027_011/trajectory_mpc_20hz.npz"


def build(
    num_envs: int,
    solver: str,
    iterations: int,
    ls_iterations: int,
    nconmax_per_env: int,
    njmax: int,
    broken_nconmax: bool,
    graph_conditional: bool,
    tolerance: float | None = None,
    timestep: float | None = None,
):
    import mujoco_warp as mjwarp

    mjm = mujoco.MjModel.from_xml_path(SCENE)
    if solver == "cg":
        mjm.opt.solver = mujoco.mjtSolver.mjSOL_CG
    elif solver == "newton":
        mjm.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    mjm.opt.iterations = iterations
    mjm.opt.ls_iterations = ls_iterations
    if tolerance is not None:
        mjm.opt.tolerance = tolerance
    if timestep is not None:
        mjm.opt.timestep = timestep

    traj = np.load(TRAJ, allow_pickle=True)
    qpos0, qvel0 = traj["qpos"][0], traj["qvel"][0]

    mjd = mujoco.MjData(mjm)
    mjd.qpos[:] = qpos0
    mjd.qvel[:] = qvel0
    mujoco.mj_forward(mjm, mjd)

    m = mjwarp.put_model(mjm)
    m.opt.graph_conditional = graph_conditional
    nconmax = nconmax_per_env * num_envs if broken_nconmax else nconmax_per_env
    d = mjwarp.put_data(mjm, mjd, nworld=num_envs, nconmax=nconmax, njmax=njmax)
    return mjwarp, mjm, m, d


def bench(name: str, num_envs: int, control_steps: int, substeps: int, use_graph: bool,
          graph_substeps: bool = False, **build_kwargs):
    import warp as wp

    t0 = time.perf_counter()
    mjwarp, mjm, m, d = build(num_envs, **build_kwargs)
    wp.synchronize()
    t_build = time.perf_counter() - t0

    # warm-up (module load) outside capture
    t0 = time.perf_counter()
    mjwarp.step(m, d)
    wp.synchronize()
    t_warm = time.perf_counter() - t0

    graph = None
    t_capture = 0.0
    if use_graph:
        t0 = time.perf_counter()
        try:
            with wp.ScopedCapture() as capture:
                if graph_substeps:
                    for _ in range(substeps):
                        mjwarp.step(m, d)
                else:
                    mjwarp.step(m, d)
            graph = capture.graph
            wp.synchronize()
        except RuntimeError as e:
            print(f"  [{name}] graph capture FAILED: {e}")
        t_capture = time.perf_counter() - t0

    # timed rollout: control_steps x substeps, driven by demo hand qpos targets
    traj = np.load(TRAJ, allow_pickle=True)
    demo_ctrl = traj["qpos"][:, : mjm.nu].astype(np.float32)
    ctrl_host = np.tile(demo_ctrl[0], (num_envs, 1))
    peak_ncon_per_env = 0
    peak_nefc_per_env = 0
    wp.synchronize()
    t0 = time.perf_counter()
    for step_i in range(control_steps):
        ctrl_host[:] = demo_ctrl[min(step_i + 1, len(demo_ctrl) - 1)]
        wp.copy(d.ctrl, wp.array(ctrl_host, dtype=float, device=d.ctrl.device))
        if graph is not None:
            if graph_substeps:
                wp.capture_launch(graph)
            else:
                for _ in range(substeps):
                    wp.capture_launch(graph)
        else:
            for _ in range(substeps):
                mjwarp.step(m, d)
        wp.synchronize()
        peak_ncon_per_env = max(peak_ncon_per_env, int(d.nacon.numpy()[0]) / num_envs)
        peak_nefc_per_env = max(peak_nefc_per_env, int(d.nefc.numpy().max()))
    t_roll = time.perf_counter() - t0

    niter = d.solver_niter.numpy()
    qpos = d.qpos.numpy()
    nan_frac = float(np.isnan(qpos).mean())
    ms_per_ctrl = t_roll / control_steps * 1e3
    est_rollout_s = ms_per_ctrl / 1e3 * 98  # h98 episode
    print(
        f"  [{name}] build {t_build:.1f}s warm {t_warm:.1f}s capture {t_capture:.1f}s | "
        f"{ms_per_ctrl:8.1f} ms/ctrl-step | est h98 rollout {est_rollout_s:7.1f}s | "
        f"solver_niter mean {niter.mean():.1f} max {niter.max()} | "
        f"ncon/env {peak_ncon_per_env:.0f} nefc max {peak_nefc_per_env} | nan {nan_frac:.3f}"
    )
    # free
    del d, m
    wp.synchronize()
    return ms_per_ctrl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--control-steps", type=int, default=20)
    ap.add_argument("--substeps", type=int, default=25)
    ap.add_argument("--cases", type=str, default="all")
    args = ap.parse_args()

    import warp as wp

    wp.init()
    print(f"conditional graph supported: {wp.is_conditional_graph_supported()}")
    print(f"num_envs={args.num_envs} control_steps={args.control_steps} substeps={args.substeps}")

    cases = args.cases.split(",") if args.cases != "all" else [
        "fallback", "graph_it100", "graph_it10", "graph_it5",
        "cg_it10", "cg_it25", "graph_it10_sub",
    ]

    common = dict(
        num_envs=args.num_envs,
        control_steps=args.control_steps,
        substeps=args.substeps,
        nconmax_per_env=100,
        njmax=350,
        broken_nconmax=False,
    )

    if "broken_nconmax" in cases:
        # current TacoEnvGPU behavior: nconmax = 100 * num_envs interpreted per world
        bench("broken_nconmax_it10_graph", use_graph=True,
              **{**common, "broken_nconmax": True},
              solver="newton", iterations=10, ls_iterations=20, graph_conditional=False)

    if "fallback" in cases:
        # current TacoEnvGPU behavior on this driver: no graph, graph_conditional=True
        # (capture_while host-syncs every solver iteration)
        bench("fallback_no_graph_newton_it100", use_graph=False, **common,
              solver="newton", iterations=100, ls_iterations=50, graph_conditional=True)

    if "fallback_gc_off" in cases:
        bench("no_graph_newton_it100_gcFalse", use_graph=False, **common,
              solver="newton", iterations=100, ls_iterations=50, graph_conditional=False)

    if "graph_it100" in cases:
        bench("graph_newton_it100", use_graph=True, **common,
              solver="newton", iterations=100, ls_iterations=50, graph_conditional=False)

    if "graph_it10" in cases:
        bench("graph_newton_it10_ls20", use_graph=True, **common,
              solver="newton", iterations=10, ls_iterations=20, graph_conditional=False)

    if "graph_it5" in cases:
        bench("graph_newton_it5_ls15", use_graph=True, **common,
              solver="newton", iterations=5, ls_iterations=15, graph_conditional=False)

    if "cg_it10" in cases:
        bench("graph_cg_it10", use_graph=True, **common,
              solver="cg", iterations=10, ls_iterations=20, graph_conditional=False)

    if "cg_it25" in cases:
        bench("graph_cg_it25", use_graph=True, **common,
              solver="cg", iterations=25, ls_iterations=20, graph_conditional=False)

    if "graph_it10_sub" in cases:
        bench("graph_newton_it10_wholechunk", use_graph=True, graph_substeps=True, **common,
              solver="newton", iterations=10, ls_iterations=20, graph_conditional=False)

    if "dt4" in cases:
        # timestep 0.002 -> 0.004: 13 substeps per 20 Hz control step instead of 25
        bench("graph_newton_it10_dt0.004", use_graph=True,
              **{**common, "substeps": 13},
              solver="newton", iterations=10, ls_iterations=20,
              graph_conditional=False, timestep=0.004)


if __name__ == "__main__":
    main()
