#!/usr/bin/env python
"""Build (and optionally execute) the GPU-sim rollout benchmark notebook.

Usage:
    .venv/bin/python notebooks/_build_gpusim_bench_nb.py
    .venv/bin/jupyter nbconvert --to notebook --execute \
        notebooks/gpusim_bench.ipynb --output gpusim_bench.ipynb \
        --ExecutePreprocessor.timeout=1200
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# TACO sharpa GPU-sim (mjwarp) rollout benchmark — 2026-07-01\n"
    "\n"
    "Root-cause audit of slow GPU rollouts in RLinf "
    "(`taco_sharpa20hz_tp4_raft_*` experiments).\n"
    "\n"
    "Findings quantified here:\n"
    "1. Driver 535 (CUDA 12.2) lacks conditional CUDA graph nodes → mjwarp "
    "graph capture fails → fallback host-syncs every solver iteration.\n"
    "2. `put_data(nconmax=...)` is per-world in mjwarp 3.7 → passing "
    "`100*num_envs` allocates `num_envs^2` contacts (OOM at 512 envs).\n"
    "3. Scene XML has no `<option>` → Newton iterations=100/ls=50 defaults "
    "get fully unrolled into the CUDA graph.\n"
    "4. Newton vs CG comparison (colleague hypothesis): Newton converges in "
    "~2.6 iters vs CG ~6.1 and is faster end-to-end — keep Newton.\n"
))

cells.append(nbf.v4.new_code_cell(
    "import subprocess, sys\n"
    "\n"
    "PY = '/root/RLinf/.venv/bin/python'\n"
    "BENCH = '/root/RLinf/notebooks/bench_mjwarp_sharpa.py'\n"
    "\n"
    "def run(args):\n"
    "    out = subprocess.run([PY, BENCH] + args, capture_output=True, text=True,\n"
    "                         cwd='/root/RLinf')\n"
    "    for line in out.stdout.splitlines() + out.stderr.splitlines():\n"
    "        if line.startswith('  [') or line.startswith('num_envs') \\\n"
    "           or 'conditional graph' in line or 'Failed to allocate' in line:\n"
    "            print(line)\n"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 1. Current fallback path (no CUDA graph) vs fixed graph path\n"
    "`fallback_no_graph_newton_it100` reproduces what `TacoEnvGPU` does today "
    "on this driver; the graph cases require `opt.graph_conditional=False`."
))
cells.append(nbf.v4.new_code_cell(
    "run(['--num-envs', '512', '--control-steps', '3', '--cases', 'fallback'])"
))
cells.append(nbf.v4.new_code_cell(
    "run(['--num-envs', '512', '--control-steps', '10',\n"
    "     '--cases', 'graph_it100,graph_it10,graph_it5'])"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 2. Newton vs CG (colleague hypothesis 1)\n"
    "CG needs ~2.3x more iterations to converge on this scene and is slower "
    "end-to-end. The 'CG is faster' folklore comes from MJX/JAX, which cannot "
    "run Newton's conditional loop efficiently — it does not apply to mjwarp."
))
cells.append(nbf.v4.new_code_cell(
    "run(['--num-envs', '512', '--control-steps', '10', '--cases', 'cg_it10,cg_it25'])"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 3. nconmax per-world semantics bug\n"
    "`broken_nconmax` reproduces the old `nconmax=100*num_envs` call. At 512 "
    "envs it OOMs (~57 GB single allocation); at 128 envs it 'only' costs "
    "~1.5x speed."
))
cells.append(nbf.v4.new_code_cell(
    "run(['--num-envs', '128', '--control-steps', '10',\n"
    "     '--cases', 'broken_nconmax,graph_it10'])"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 4. Scaling + timestep headroom\n"
    "dt=0.004 halves substeps (25→13) — physics change, needs A/B validation "
    "before adoption."
))
cells.append(nbf.v4.new_code_cell(
    "run(['--num-envs', '1024', '--control-steps', '10', '--cases', 'graph_it10'])"
))
cells.append(nbf.v4.new_code_cell(
    "run(['--num-envs', '512', '--control-steps', '10', '--cases', 'dt4'])"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 5. Fixed `TacoEnvGPU` end-to-end (obs synthesis + reward included)\n"
    "Uses the real experiment config with `sim_backend=gpu`. Also reports the "
    "NaN-world fraction near episode end (mjwarp has no BADQACC auto-reset; "
    "rewards are now nan-guarded and logged as `sim_nan_frac`)."
))
cells.append(nbf.v4.new_code_cell(
    "out = subprocess.run([PY, '/root/RLinf/notebooks/smoke_taco_env_gpu.py',\n"
    "                      '--num-envs', '512', '--chunk-steps', '25'],\n"
    "                     capture_output=True, text=True, cwd='/root/RLinf')\n"
    "for line in out.stdout.splitlines():\n"
    "    if any(k in line for k in ('init:', 'reset:', 'rollout:', 'rewards', 'nan')):\n"
    "        print(line)"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 6. RL loop timer comparison (from live runs)\n"
    "| setup | env_interact_step | generate_rollouts | global step |\n"
    "|---|---|---|---|\n"
    "| CPU sim, 512 envs / 5 workers (live RAFT run) | ~161 s | ~195 s | ~260 s |\n"
    "| GPU sim fixed, 256 envs / 1 GPU (smoke-gpusim-fixed-20260701) | ~15 s | ~31 s | ~67 s |\n"
))

nb["cells"] = cells
path = "/root/RLinf/notebooks/gpusim_bench.ipynb"
nbf.write(nb, path)
print("wrote", path)
