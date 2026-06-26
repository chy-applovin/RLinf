# Sharpa 20Hz Tp=4 Fixed Frame-20 Horizon-35 Eta3 Launch

Date: 2026-06-26

## Goal

Launch four additional RL runs matching the existing fixed-frame/horizon batch, changing only the Flow SDE sampling noise level from `0.5` to `3.0`.

## Shared Settings

- Dataset: `/root/Data/taco-brush-sharpa-20hz-both`
- Initial IL checkpoint: `/root/flow-policy/outputs/flow-policy/sharpa20hz_tp4/checkpoints/epoch_4100.pt`
- Init frame: `demo_start.fixed_frame=20`
- Initial observation history: demo frames `[17, 18, 19, 20]`
- True env episode cap: `max_episode_steps=35`
- Chunk rollout slots: `max_steps_per_rollout_epoch=36`, `num_action_chunks=4`
- Reward: tool object + target object + hand qpos tracking, weights `1/1/1`
- Changed parameter: `actor.model.noise_method=flow_sde`, `actor.model.noise_level=3.0`
- Previous matched batch used `actor.model.noise_level=0.5`

Hydra resolve passed for all four configs. Resolved snapshots are archived in `internal_doc/6_26/resolved_sharpa20hz_tp4_f20_h35_eta3/`.

## Run Matrix

| Config | Episode | Algo | Init frame | Env cap | Eta/noise | GPUs | W&B run name |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `taco_sharpa20hz_tp4_ppo_helmet_f20_h35_eta3.yaml` | `brush__brush__helmet__20231027_011` | PPO | 20 | 35 | 3.0 | 0-1 | `sharpa20hz-tp4-helmet-ppo-f20-h35-eta3-gbs128` |
| `taco_sharpa20hz_tp4_return_as_adv_helmet_f20_h35_eta3.yaml` | `brush__brush__helmet__20231027_011` | PPO_with_return_as_adv | 20 | 35 | 3.0 | 2-3 | `sharpa20hz-tp4-helmet-retadv-f20-h35-eta3-gbs128` |
| `taco_sharpa20hz_tp4_ppo_box_f20_h35_eta3.yaml` | `brush__brush__box__20230930_034` | PPO | 20 | 35 | 3.0 | 4-5 | `sharpa20hz-tp4-box-ppo-f20-h35-eta3-gbs128` |
| `taco_sharpa20hz_tp4_return_as_adv_box_f20_h35_eta3.yaml` | `brush__brush__box__20230930_034` | PPO_with_return_as_adv | 20 | 35 | 3.0 | 6-7 | `sharpa20hz-tp4-box-retadv-f20-h35-eta3-gbs128` |

## Launch Commands

```bash
EMBODIED_PATH=/root/RLinf/examples/embodiment /root/RLinf/.venv/bin/python /root/RLinf/examples/embodiment/train_embodied_agent.py --config-path /root/RLinf/examples/embodiment/config --config-name taco_sharpa20hz_tp4_ppo_helmet_f20_h35_eta3.yaml
```

```bash
EMBODIED_PATH=/root/RLinf/examples/embodiment /root/RLinf/.venv/bin/python /root/RLinf/examples/embodiment/train_embodied_agent.py --config-path /root/RLinf/examples/embodiment/config --config-name taco_sharpa20hz_tp4_return_as_adv_helmet_f20_h35_eta3.yaml
```

```bash
EMBODIED_PATH=/root/RLinf/examples/embodiment /root/RLinf/.venv/bin/python /root/RLinf/examples/embodiment/train_embodied_agent.py --config-path /root/RLinf/examples/embodiment/config --config-name taco_sharpa20hz_tp4_ppo_box_f20_h35_eta3.yaml
```

```bash
EMBODIED_PATH=/root/RLinf/examples/embodiment /root/RLinf/.venv/bin/python /root/RLinf/examples/embodiment/train_embodied_agent.py --config-path /root/RLinf/examples/embodiment/config --config-name taco_sharpa20hz_tp4_return_as_adv_box_f20_h35_eta3.yaml
```


## Active Launch Status

Launched from commit `0cc9e290` and pushed to `fork/feat/ppo-return-as-adv` before training start. Effective launches use `setsid -f` so they survive the Codex command session.

| Run | PID | GPUs | W&B URL | Status at check |
| --- | ---: | --- | --- | --- |
| `sharpa20hz-tp4-helmet-ppo-f20-h35-eta3-gbs128` | 4089974 | 0-1 | https://wandb.ai/hanyang-chen-app-applovin/taco-allegro-flow-rl/runs/j5emmbre | alive, reached global step 10 |
| `sharpa20hz-tp4-helmet-retadv-f20-h35-eta3-gbs128` | 4095644 | 2-3 | https://wandb.ai/hanyang-chen-app-applovin/taco-allegro-flow-rl/runs/kwxezbjj | alive, reached global step 8 |
| `sharpa20hz-tp4-box-ppo-f20-h35-eta3-gbs128` | 4102536 | 4-5 | https://wandb.ai/hanyang-chen-app-applovin/taco-allegro-flow-rl/runs/n0nwtdwa | alive, reached global step 6 |
| `sharpa20hz-tp4-box-retadv-f20-h35-eta3-gbs128` | 4109116 | 6-7 | https://wandb.ai/hanyang-chen-app-applovin/taco-allegro-flow-rl/runs/1g3zhioq | alive, reached global step 4 |
