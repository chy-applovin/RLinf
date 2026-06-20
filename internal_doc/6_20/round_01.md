# Round 01 - A4 Analysis and R01 Launch Plan

Date: 2026-06-20 UTC

## Goal

Use the previous A4 step0 curriculum run as the baseline evidence, diagnose the current gap to robust closed-loop dexterous manipulation, then launch four isolated follow-up experiments on GPUs 0-7.

Target behavior: from demo frame 0, the policy should maintain in-distribution bimanual Allegro hand motion while tracking both tool and target objects through the full 200-frame episode.

## Previous Run Analyzed

- Run name: `a4-step0-curriculum-object-hand-h4to200-chunk4-gbs128-1ep-brush_bowl`
- W&B: https://wandb.ai/hanyang-chen-app-applovin/taco-allegro-flow-rl/runs/ue0o814q
- Training log: `logs/20260619-2124-a4-step0-curriculum-h4to200-chunk4-gbs128/`
- Latest checkpoint: `global_step_5000`
- Exported eval checkpoint: `internal_doc/6_20/round_01/assets/a4_gs5000_flow.pt`

## W&B Evidence

Local artifact: `internal_doc/6_20/round_01/assets/a4_wandb_query.json`

| step | horizon | reward | success_once | hand_err | tool_err_m | target_err_m | approx_kl | clip_fraction | train batches |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 4 | 0.942 | 1.000 | 0.044 | 0.0037 | 0.0023 | 0.00003 | 0.000 | 1 |
| 504 | 8 | 0.938 | 1.000 | 0.081 | 0.0036 | 0.0023 | 0.0718 | 0.068 | 2 |
| 1514 | 16 | 0.913 | 1.000 | 0.167 | 0.0022 | 0.0024 | 0.349 | 1.776 | 4 |
| 2001 | 32 | 0.894 | 1.000 | 0.217 | 0.0079 | 0.0044 | 0.506 | 2.402 | 8 |
| 2997 | 64 | 0.640 | 0.000 | 0.442 | 0.1998 | 0.2209 | 1.020 | 6.878 | 16 |
| 3997 | 128 | 0.478 | 0.992 | 0.438 | 0.0347 | 0.0248 | 1.249 | 8.743 | 32 |
| 4994 | 200 | 0.469 | 0.672 | 0.527 | 0.3291 | 0.1411 | 1.674 | 11.878 | 50 |
| summary | 200 | 0.450 | 0.750 | 0.520 | 0.2914 | 0.3159 | 1.976 | 12.107 | 50 |

Interpretation: before horizon 32 the object errors are small, but hand error is already growing. At horizon 64 and beyond, reward and success become unstable while PPO KL/clip fraction grow sharply. Because A4 used `update_epoch=4`, horizon 200 means `50 train batches * 4 = 200 optimizer updates` per global step.

## Closed-Loop Eval and Video Artifacts

Policy eval command used `rollout_eval.py` with `--trajectory-file trajectory_ctrl_30hz_im.npz`, `--action-horizon 4`, and `--render-every 2`.

Saved artifacts:

- Policy video: `internal_doc/6_20/round_01/assets/eval_a4_gs5000_policy/brush__brush__bowl__20230927_027/rollout.mp4`
- Demo-control video: `internal_doc/6_20/round_01/assets/eval_demo_trackability/brush__brush__bowl__20230927_027/rollout.mp4`
- Montage: `internal_doc/6_20/round_01/assets/a4_eval_montage.jpg`
- Error curves: `internal_doc/6_20/round_01/assets/a4_eval_error_curves.png`
- Error summary: `internal_doc/6_20/round_01/assets/a4_eval_error_summary.json`

Policy closed-loop metrics:

```text
tool_pos_err_mean_m   = 0.0799
tool_pos_err_final_m  = 0.0570
target_pos_err_mean_m = 0.0262
target_pos_err_final_m= 0.0161
hand_qpos_err_mean    = 0.4505
hand_qpos_err_final   = 0.6317
```

Demo-control baseline:

```text
tool_pos_err_final_m  = 0.0800
target_pos_err_final_m= 0.0950
hand_qpos_err_mean    = 0.0233
```

Video/trajectory diagnosis:

- The demo-control rollout is trackable in the simulator, so the eval setup itself is not the main failure.
- Policy object tracking can look successful in this one closed-loop episode, but the hand trajectory leaves the demo distribution almost immediately.
- Hand error crosses 0.1 at frame 1, 0.2 at frame 3, and 0.4 at frame 12.
- Target object error never exceeds 0.1m in the policy eval, while hand qpos reaches much larger ranges than the demo (`sim_hand_max=5.37` vs `demo_hand_max=3.59`).
- This matches the concern that object tracking alone, or weak hand regularization under aggressive PPO, can produce out-of-distribution hand actions that happen to move the object but are fragile for longer closed-loop rollouts.

## Diagnosis

Main suspected issues:

1. PPO update intensity explodes as horizon increases. A4 goes from 4 updates/global-step at horizon 4 to 200 updates/global-step at horizon 200.
2. The policy learns object tracking while allowing hand qpos to drift far outside the demo distribution.
3. Large `approx_kl` and `clip_fraction` indicate the policy updates are too aggressive for the on-policy data distribution at long horizons.

## Implemented Change

Added a default-off curriculum parameter:

```yaml
horizon_curriculum:
  max_train_global_batches_per_step: null
```

When set to integer `M`, actor still computes advantage/return on the full rollout pool, but PPO training samples at most `M * actor.global_batch_size` transitions from that pool. This leaves non-curriculum behavior unchanged and does not alter rollout collection.

Code/config docs updated:

- `rlinf/workers/actor/fsdp_actor_worker.py`
- `examples/embodiment/config/env/taco_brush_allegro.yaml`
- `internal_doc/taco_curriculum_feature.md`
- `internal_doc/taco_rl_training_parameters.md`

Validation:

- `python -m py_compile rlinf/workers/actor/fsdp_actor_worker.py`
- Hydra `--cfg job --resolve` passed for all four R01 configs.
- One-global-step smoke passed with `taco_allegro_ppo_flow_r01_cap8_ue1`, TensorBoard-only logging, and `runner.max_epochs=1`. The metric table included `curriculum_max_train_global_batches_per_step`, confirming the new cap path is active.

## R01 Experiments

| ID | config | GPUs | hypothesis |
| --- | --- | --- | --- |
| E1 | `taco_allegro_ppo_flow_r01_cap8_ue1` | 0,1 | Cap to 8 train batches and `update_epoch=1` should suppress KL/clip growth from excessive PPO reuse. |
| E2 | `taco_allegro_ppo_flow_r01_cap16_ue1` | 2,3 | Cap 16 tests whether cap8 underuses long-horizon data. |
| E3 | `taco_allegro_ppo_flow_r01_cap8_ue1_lowlr` | 4,5 | Lower LR should further reduce policy drift if cap alone is insufficient. |
| E4 | `taco_allegro_ppo_flow_r01_cap8_ue1_hand1` | 6,7 | Stronger hand tracking should reduce out-of-distribution hand motion while keeping object tracking. |

All four use step0 reset, `num_action_chunks=4`, horizon curriculum 4->200, `global_batch_size=128`, W&B logging, and `max_epochs=5000`.

## Launch Commands

Commands are recorded before launch; run-specific PIDs/W&B URLs will be appended after startup.

```bash
env EMBODIED_PATH=/root/RLinf/examples/embodiment /root/RLinf/.venv/bin/python examples/embodiment/train_embodied_agent.py --config-path /root/RLinf/examples/embodiment/config --config-name taco_allegro_ppo_flow_r01_cap8_ue1 runner.logger.log_path=/root/RLinf/logs/20260620-r01-cap8-ue1
```

```bash
env EMBODIED_PATH=/root/RLinf/examples/embodiment /root/RLinf/.venv/bin/python examples/embodiment/train_embodied_agent.py --config-path /root/RLinf/examples/embodiment/config --config-name taco_allegro_ppo_flow_r01_cap16_ue1 runner.logger.log_path=/root/RLinf/logs/20260620-r01-cap16-ue1
```

```bash
env EMBODIED_PATH=/root/RLinf/examples/embodiment /root/RLinf/.venv/bin/python examples/embodiment/train_embodied_agent.py --config-path /root/RLinf/examples/embodiment/config --config-name taco_allegro_ppo_flow_r01_cap8_ue1_lowlr runner.logger.log_path=/root/RLinf/logs/20260620-r01-cap8-ue1-lowlr
```

```bash
env EMBODIED_PATH=/root/RLinf/examples/embodiment /root/RLinf/.venv/bin/python examples/embodiment/train_embodied_agent.py --config-path /root/RLinf/examples/embodiment/config --config-name taco_allegro_ppo_flow_r01_cap8_ue1_hand1 runner.logger.log_path=/root/RLinf/logs/20260620-r01-cap8-ue1-hand1
```
