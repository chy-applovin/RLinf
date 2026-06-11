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

"""Export an RLinf actor checkpoint back to the flow-policy .pt format.

The RL wrapper stores the flow transformer under ``policy.*`` and the
normalizer under ``normalizer.*`` in the actor's full state dict. This script
re-packages them into the {model, normalizer, shape_meta, cfg} layout that
``flow_policy`` tooling (e.g. mujoco-spider-env/scripts/rollout_eval.py)
loads, so RL-trained policies can be evaluated with the exact same closed-loop
IL eval harness (deterministic flow ODE, videos, tracking metrics).

    python export_flow_ckpt.py \
        --rl-ckpt <run>/checkpoints/global_step_300/actor/model_state_dict/full_weights.pt \
        --arch-ckpt /root/flow-policy/outputs/.../checkpoints/best.pt \
        --out /tmp/rl_step300.pt
"""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl-ckpt", required=True, help="RLinf actor full_weights.pt")
    ap.add_argument(
        "--arch-ckpt",
        required=True,
        help="original flow-policy .pt providing shape_meta/cfg (architecture spec)",
    )
    ap.add_argument("--out", required=True, help="output flow-policy-format .pt")
    args = ap.parse_args()

    rl_sd = torch.load(args.rl_ckpt, map_location="cpu", weights_only=False)
    arch = torch.load(args.arch_ckpt, map_location="cpu", weights_only=False)

    model_sd = {
        k[len("policy.") :]: v.float() for k, v in rl_sd.items() if k.startswith("policy.")
    }
    normalizer_sd = {
        k[len("normalizer.") :]: v.float()
        for k, v in rl_sd.items()
        if k.startswith("normalizer.")
    }
    assert model_sd, "no policy.* keys found in the RL checkpoint"
    assert normalizer_sd, "no normalizer.* keys found in the RL checkpoint"

    out_state = {
        "model": model_sd,
        "normalizer": normalizer_sd,
        "shape_meta": arch["shape_meta"],
        "cfg": arch["cfg"],
        "source_rl_ckpt": args.rl_ckpt,
    }
    torch.save(out_state, args.out)
    n_params = sum(v.numel() for v in model_sd.values())
    print(f"exported {len(model_sd)} model tensors ({n_params / 1e6:.2f}M params)")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
