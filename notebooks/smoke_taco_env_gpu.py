#!/usr/bin/env python
"""End-to-end smoke of the fixed TacoEnvGPU via the real experiment config."""

import argparse
import os
import time

os.environ.setdefault("EMBODIED_PATH", "/root/RLinf/examples/embodiment")

import numpy as np
import torch
from hydra import compose, initialize_config_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--chunk-steps", type=int, default=25)
    ap.add_argument("--config", type=str,
                    default="taco_sharpa20hz_tp4_raft_episode_k10_helmet_step0_h98_env1024")
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args()

    with initialize_config_dir(
        config_dir="/root/RLinf/examples/embodiment/config", version_base=None
    ):
        cfg = compose(config_name=args.config,
                      overrides=["env.train.sim_backend=gpu"] + args.override)

    from rlinf.envs.taco.taco_env_gpu import TacoEnvGPU

    t0 = time.perf_counter()
    env = TacoEnvGPU(
        cfg=cfg.env.train,
        num_envs=args.num_envs,
        seed_offset=0,
        total_num_processes=1,
    )
    print(f"init: {time.perf_counter() - t0:.1f}s | substeps={env.substeps} "
          f"graph={'OK' if env._graph is not None else 'NONE'} "
          f"solver_it={env.model_cpu.opt.iterations} ls_it={env.model_cpu.opt.ls_iterations}")

    t0 = time.perf_counter()
    obs, _ = env.reset()
    print(f"reset: {time.perf_counter() - t0:.2f}s | obs keys "
          f"{ {k: tuple(v.shape) for k, v in obs.items()} }")

    chunk = int(cfg.actor.model.num_action_chunks)
    rng = np.random.default_rng(0)
    demo = env.episode.qpos_demo[:, : env.hand_dim].astype(np.float32)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    total_ctrl_steps = 0
    for i in range(args.chunk_steps):
        base = demo[min((i + 1) * chunk, len(demo) - 1)]
        actions = np.tile(base, (args.num_envs, chunk, 1))
        if args.noise > 0:
            actions += rng.normal(0, args.noise, actions.shape).astype(np.float32)
        obs_list, rewards, terms, truncs, infos = env.chunk_step(torch.from_numpy(actions))
        total_ctrl_steps += chunk
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"rollout: {args.chunk_steps} chunk steps ({total_ctrl_steps} ctrl steps) "
          f"in {dt:.1f}s -> {dt / total_ctrl_steps * 1e3:.1f} ms/ctrl-step")
    print(f"rewards last chunk mean {rewards.float().mean():.4f} | "
          f"episode return mean {env._returns.mean().item():.3f} | steps {env.steps}")
    qpos = env._qpos()
    print(f"qpos nan frac {torch.isnan(qpos).float().mean().item():.4f}")


if __name__ == "__main__":
    main()
