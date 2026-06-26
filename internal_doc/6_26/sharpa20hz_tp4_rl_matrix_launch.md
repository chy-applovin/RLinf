# Sharpa 20Hz Tp=4 RL Matrix Launch

Date: 2026-06-26

## Goal

Launch four RL runs from the IL checkpoint `/root/flow-policy/outputs/flow-policy/sharpa20hz_tp4/checkpoints/epoch_4100.pt`, crossing two single-episode environments with two PPO-style algorithms.

All runs reset from original demo frame 0. `single_step`, `demo_start`, RSI, and early termination are disabled.

## Shared Settings

- Dataset: `/root/Data/taco-brush-sharpa-20hz-both`
- Robot mesh root: `/root/Data/sharpa_mesh/meshes`
- Trajectory: `trajectory_mpc_20hz.npz`
- Point cloud directory: `20hz_mpc_pointcloud`
- Obs/action contract: `obs_horizon=4`, `num_action_chunks=4`, `hand_dim=56`, `action_dim=56`
- Reward: tool object + target object + hand qpos tracking, weights `1/1/1`
- Batch: `total_num_envs=128`, `global_batch_size=128`, `micro_batch_size=16`, `update_epoch=1`
- W&B project: `taco-allegro-flow-rl`

## Required Code Support

The older TACO env assumed a 44-DoF Allegro hand and `pointcloud/`. This launch adds config-controlled CPU env support for Sharpa: `hand_dim`, `robot_name`, `robot_asset_glob`, `robot_asset_aliases`, and `pointcloud_dir`. Defaults preserve Allegro behavior.

## Run Matrix

| Config | Episode | Algo | Horizon | GPUs | W&B run name |
| --- | --- | --- | ---: | --- | --- |
| `taco_sharpa20hz_tp4_ppo_helmet_step0.yaml` | `brush__brush__helmet__20231027_011` | `PPO` | 96 | 0-1 | `sharpa20hz-tp4-helmet-ppo-step0-h96-gbs128` |
| `taco_sharpa20hz_tp4_return_as_adv_helmet_step0.yaml` | `brush__brush__helmet__20231027_011` | `PPO_with_return_as_adv` | 96 | 2-3 | `sharpa20hz-tp4-helmet-retadv-step0-h96-gbs128` |
| `taco_sharpa20hz_tp4_ppo_box_step0.yaml` | `brush__brush__box__20230930_034` | `PPO` | 68 | 4-5 | `sharpa20hz-tp4-box-ppo-step0-h68-gbs128` |
| `taco_sharpa20hz_tp4_return_as_adv_box_step0.yaml` | `brush__brush__box__20230930_034` | `PPO_with_return_as_adv` | 68 | 6-7 | `sharpa20hz-tp4-box-retadv-step0-h68-gbs128` |

## Launch Commands

### sharpa20hz-tp4-helmet-ppo-step0-h96-gbs128

```bash
EMBODIED_PATH=/root/RLinf/examples/embodiment /root/RLinf/.venv/bin/python /root/RLinf/examples/embodiment/train_embodied_agent.py --config-path /root/RLinf/examples/embodiment/config --config-name taco_sharpa20hz_tp4_ppo_helmet_step0.yaml
```

### sharpa20hz-tp4-helmet-retadv-step0-h96-gbs128

```bash
EMBODIED_PATH=/root/RLinf/examples/embodiment /root/RLinf/.venv/bin/python /root/RLinf/examples/embodiment/train_embodied_agent.py --config-path /root/RLinf/examples/embodiment/config --config-name taco_sharpa20hz_tp4_return_as_adv_helmet_step0.yaml
```

### sharpa20hz-tp4-box-ppo-step0-h68-gbs128

```bash
EMBODIED_PATH=/root/RLinf/examples/embodiment /root/RLinf/.venv/bin/python /root/RLinf/examples/embodiment/train_embodied_agent.py --config-path /root/RLinf/examples/embodiment/config --config-name taco_sharpa20hz_tp4_ppo_box_step0.yaml
```

### sharpa20hz-tp4-box-retadv-step0-h68-gbs128

```bash
EMBODIED_PATH=/root/RLinf/examples/embodiment /root/RLinf/.venv/bin/python /root/RLinf/examples/embodiment/train_embodied_agent.py --config-path /root/RLinf/examples/embodiment/config --config-name taco_sharpa20hz_tp4_return_as_adv_box_step0.yaml
```

## Validation Notes

- Dataset metadata read before config generation: helmet has 99 frames, box has 72 frames; both are 20Hz and have qpos `(T, 70)`, ctrl `(T, 56)`.
- Sharpa scene/data smoke passed before launch: both selected episodes load with `model.nu=56`, `nq=70`, target/tool clouds `(512, 3)`.
- Hydra resolve passed for all four configs; resolved snapshots are archived in `internal_doc/6_26/resolved_sharpa20hz_tp4_rl_matrix/`.
- PPO smoke passed with `8` envs, `4` control steps, `max_epochs=1`: return `3.890`, reward `0.973`, final target/tool errors `0.00145m/0.00034m`.
- PPO_with_return_as_adv smoke passed with the same small setup: return `3.888`, reward `0.972`, no critic metrics logged.

## Active Launch Status

Launched from commit `44562e8194975c4a51c78055a068d48e19f1b6b4` and pushed to `fork/feat/ppo-return-as-adv`.

| Effective run | PID | GPUs | W&B URL | Status at check |
| --- | ---: | --- | --- | --- |
| `sharpa20hz-tp4-helmet-ppo-step0-h96-gbs128` | 2666976 | 0-1 | https://wandb.ai/hanyang-chen-app-applovin/taco-allegro-flow-rl/runs/wprfe2ti | alive, reached global step 8 |
| `sharpa20hz-tp4-helmet-retadv-step0-h96-gbs128` | 2666977 | 2-3 | https://wandb.ai/hanyang-chen-app-applovin/taco-allegro-flow-rl/runs/mhupnmps | alive, reached global step 8 |
| `sharpa20hz-tp4-box-ppo-step0-h68-gbs128-retry1` | 2749851 | 4-5 | https://wandb.ai/hanyang-chen-app-applovin/taco-allegro-flow-rl/runs/bhpskg7o | alive, reached global step 3 |
| `sharpa20hz-tp4-box-retadv-step0-h68-gbs128` | 2666979 | 6-7 | https://wandb.ai/hanyang-chen-app-applovin/taco-allegro-flow-rl/runs/r4yzvb2f | alive, reached global step 11 |

The first `box PPO` launch (`sharpa20hz-tp4-box-ppo-step0-h68-gbs128`, PID 2666978, W&B run `gcs2myon`) hit a Ray/Gloo collective worker abort immediately after its first rollout and was terminated. It was restarted as `retry1` after GPUs 4-5 were free; the retry entered normal training.
