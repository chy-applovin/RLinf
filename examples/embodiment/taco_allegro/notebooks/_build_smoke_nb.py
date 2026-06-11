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

"""Build + execute the TACO flow-RL smoke notebook.

Programmatically generates ``taco_rl_smoke.ipynb`` (nbformat) and executes it
in place, so the committed notebook shows real results on real episodes +
checkpoint. Re-point ``--ckpt`` at a different checkpoint to get a comparable
A/B notebook.

    .venv/bin/python examples/embodiment/taco_allegro/notebooks/_build_smoke_nb.py \
        --ckpt /root/flow-policy/outputs/flow-policy/2026-06-10/18-12-14/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent


def build(ckpt: str, out: Path) -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# TACO flow-matching RL: smoke check (env + policy + PPO plumbing)\n\n"
            "Validates the `feat/taco-allegro-flow-rl` integration on **real** TACO "
            "episodes and a **real** flow-policy checkpoint:\n\n"
            "1. `TacoEnv` reset / chunk_step (point clouds re-posed from live sim state);\n"
            "2. noise-injected rollout (`flow_sde`, piRL-style) + tracking reward;\n"
            "3. **ODE parity**: eval-mode sampler must equal the original IL sampler;\n"
            "4. **PPO log-prob parity**: training forward on stored chains must match "
            "rollout log-probs;\n"
            "5. backward pass through a PPO-style surrogate.\n\n"
            f"Checkpoint: `{ckpt}`"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "import sys, time\n"
            "sys.path.insert(0, '/root/RLinf')\n"
            "sys.path.insert(0, '/root/RLinf/examples/embodiment/taco_allegro')\n"
            "import numpy as np, torch\n"
            "import matplotlib.pyplot as plt\n"
            "from types import SimpleNamespace\n"
            "from smoke_test import build_env_cfg, build_model_cfg\n\n"
            f"CKPT = {ckpt!r}\n"
            "state = torch.load(CKPT, map_location='cpu', weights_only=False)\n"
            "sm = state['shape_meta']\n"
            "args = SimpleNamespace(\n"
            "    ckpt=CKPT,\n"
            "    dataset_root='/root/Data/taco-brush-allegro',\n"
            "    allegro_assets='/root/Spider/spider/assets/robots/allegro/assets',\n"
            "    scene_root='/tmp/taco_smoke_nb_scene_root',\n"
            "    trajectory_file='trajectory_ctrl_30hz_im.npz',\n"
            "    num_envs=4, num_chunk_steps=8, num_action_chunks=4, max_steps=32,\n"
            "    obs_mode=sm['obs_mode'], obs_horizon=int(sm['obs_horizon']),\n"
            "    num_points=int(sm['num_points']),\n"
            ")\n"
            "del state\n"
            "print('shape_meta:', dict(sm))"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "from rlinf.models import get_model\n"
            "from rlinf.envs import get_env_cls\n\n"
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
            "model = get_model(build_model_cfg(args)).to(device).eval()\n"
            "print(f'model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params on {device}')\n\n"
            "env_cfg = build_env_cfg(args)\n"
            "env = get_env_cls('taco', env_cfg)(cfg=env_cfg, num_envs=args.num_envs,\n"
            "                                   seed_offset=0, total_num_processes=1, worker_info=None)\n"
            "obs, _ = env.reset()\n"
            "print('episodes:', [s.episode.name for s in env.subenvs])\n"
            "for k, v in obs.items():\n"
            "    print(f'  obs[{k}]: {tuple(v.shape)} {v.dtype}')"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## Rollout (train mode, noise-injected) + tracking reward"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "per_step_fi, per_step_logp = [], []\n"
            "step_rewards = []  # (T, B)\n"
            "t0 = time.time()\n"
            "for step in range(args.num_chunk_steps):\n"
            "    with torch.no_grad():\n"
            "        actions, result = model.predict_action_batch(env_obs=obs, mode='train')\n"
            "    per_step_fi.append(result['forward_inputs'])\n"
            "    per_step_logp.append(result['prev_logprobs'])\n"
            "    obs_list, rewards, terms, truncs, infos = env.chunk_step(actions.numpy())\n"
            "    obs = obs_list[-1]\n"
            "    step_rewards.append(rewards.numpy())  # (B, chunk)\n"
            "dt = time.time() - t0\n"
            "step_rewards = np.concatenate(step_rewards, axis=1)  # (B, T*chunk)\n"
            "print(f'rollout: {dt:.1f}s '\n"
            "      f'({args.num_envs*args.num_chunk_steps*args.num_action_chunks/dt:.0f} env-steps/s)')\n"
            "ep = infos[-1]['episode']\n"
            "print('final tool_pos_err_final_m:', ep['tool_pos_err_final_m'].numpy().round(3))\n"
            "print('success (<0.1m):', ep['success_once'].numpy())\n\n"
            "fig, ax = plt.subplots(figsize=(7, 3))\n"
            "for i in range(args.num_envs):\n"
            "    ax.plot(step_rewards[i], label=env.subenvs[i].episode.category)\n"
            "ax.set_xlabel('control step'); ax.set_ylabel('tracking reward (0,1]')\n"
            "ax.set_title('per-step tracking reward under flow_sde exploration noise')\n"
            "ax.legend(fontsize=7); fig.tight_layout(); plt.show()"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## Parity check 1 - eval-mode sampler == original IL flow ODE"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "from flow_policy.algos.flow_matching import FlowMatching\n"
            "from flow_policy.data.normalizer import Normalizer\n"
            "from flow_policy.models.transformer_policy import FlowMatchingTransformerPolicy\n\n"
            "state = torch.load(CKPT, map_location='cpu', weights_only=False)\n"
            "mcfg = {k: v for k, v in state['cfg']['model'].items() if k != '_target_'}\n"
            "ref = FlowMatching(\n"
            "    model=FlowMatchingTransformerPolicy(shape_meta=sm, **mcfg),\n"
            "    normalizer=Normalizer(sm['action_dim'], sm['obs_mode'],\n"
            "                          point_dim=sm['point_dim'], qpos_dim=sm['qpos_dim']),\n"
            "    num_sample_steps=state['cfg']['algo'].get('num_sample_steps', 16),\n"
            "    normalize_obs=state['cfg']['algo'].get('normalize_obs', True),\n"
            ")\n"
            "ref.model.load_state_dict(state['model']); ref.normalizer.load_state_dict(state['normalizer'])\n"
            "ref = ref.to(device).eval()\n\n"
            "fp_obs = {'pointcloud': obs['pointcloud'].to(device), 'qpos': obs['states'].to(device)}\n"
            "if args.obs_mode == 'pc2_qpos':\n"
            "    fp_obs['tool_pointcloud'] = obs['tool_pointcloud'].to(device)\n"
            "torch.manual_seed(123)\n"
            "with torch.no_grad():\n"
            "    ref_chunk = ref.sample(fp_obs)\n"
            "torch.manual_seed(123)\n"
            "with torch.no_grad():\n"
            "    _, eval_result = model.predict_action_batch(env_obs=obs, mode='eval')\n"
            "rl_chunk = eval_result['forward_inputs']['model_action'].reshape(args.num_envs, -1, 44)\n"
            "ode_err = (ref_chunk.cpu() - rl_chunk).abs().max().item()\n"
            "print(f'ODE parity max|diff| = {ode_err:.2e}')\n"
            "assert ode_err < 1e-4"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## Parity check 2 - PPO log-prob recompute + backward pass"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "fi = {k: torch.cat([s[k] for s in per_step_fi], dim=0) for k in per_step_fi[0]}\n"
            "prev_logp = torch.cat(per_step_logp, dim=0)\n"
            "model.train()\n"
            "out = model(forward_inputs=fi, compute_logprobs=True, compute_values=True)\n"
            "logp = out['logprobs']\n"
            "logp_err = (logp.detach().cpu().float() - prev_logp.float()).abs().max().item()\n"
            "print(f'logprob parity max|diff| = {logp_err:.2e}')\n"
            "assert logp_err < 1e-3\n\n"
            "bsz = logp.shape[0]\n"
            "adv = torch.randn(bsz, device=logp.device)\n"
            "ratio = torch.exp(logp.reshape(bsz, -1).sum(-1)\n"
            "                  - prev_logp.to(logp.device).reshape(bsz, -1).sum(-1))\n"
            "loss = -(adv * ratio).mean() + out['values'].pow(2).mean()\n"
            "loss.backward()\n"
            "gsum = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)\n"
            "print(f'ratio at parity: mean={ratio.mean().item():.4f} (should be ~1)')\n"
            "print(f'backward ok: total |grad| = {gsum:.3e}')\n"
            "assert gsum > 0\n"
            "env.close()\n"
            "print('ALL CHECKS PASSED')"
        )
    )

    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nbf.write(nb, out)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        default="/root/flow-policy/outputs/flow-policy/2026-06-10/18-12-14/checkpoints/best.pt",
    )
    ap.add_argument("--out", default=str(HERE / "taco_rl_smoke.ipynb"))
    ap.add_argument("--no-execute", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    build(args.ckpt, out)

    if not args.no_execute:
        import nbclient

        nb = nbf.read(out, as_version=4)
        client = nbclient.NotebookClient(nb, timeout=1200, kernel_name="python3")
        client.execute()
        nbf.write(nb, out)
        print(f"executed {out}")


if __name__ == "__main__":
    main()
