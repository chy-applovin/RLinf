# Sharpa 20Hz Tp=4 Fixed Frame-20 Horizon-35 Launch

Date: 2026-06-26

## Goal

Launch four additional RL runs while keeping the existing step0 matrix alive. These runs use the same two single-episode tasks and two PPO-style algorithms, but reset from demo frame 20 and train only the local 35-control-step segment.

## Code Support

`DemoTimestepSampler` now supports `demo_start.fixed_frame`. When this key is unset, behavior is unchanged: `demo_start` samples uniformly from `[min_frame, max_allowed_start]`. When set, reset deterministically uses that frame and validates that enough future demo frames remain for `max_episode_steps`.

For this launch:

- `demo_start.enabled=True`
- `demo_start.fixed_frame=20`
- `demo_start.min_frame=20`
- `max_episode_steps=35`
- `max_steps_per_rollout_epoch=36`
- `actor.model.num_action_chunks=4`

The true environment cap is 35 control steps. The rollout worker needs an integer number of 4-step chunks, so it requests 9 chunk slots (`36` control-step slots). The final slot after env done is not executed by the MuJoCo env; this preserves the current chunked PPO machinery without changing existing runs.

## Initialization Smoke

CPU reset smoke passed for both episodes:

- `helmet`: `start_frame=20`, `ep_len=35`, history frames `[17, 18, 19, 20]`, max history qpos error `0.000e+00`, obs shape `(1, 4, 56)`.
- `box`: `start_frame=20`, `ep_len=35`, history frames `[17, 18, 19, 20]`, max history qpos error `0.000e+00`, obs shape `(1, 4, 56)`.

Hydra resolve passed for all four configs. Resolved snapshots are archived in `internal_doc/6_26/resolved_sharpa20hz_tp4_f20_h35/`.

## Run Matrix

| Config | Episode | Algo | Init frame | Env cap | Rollout slots | GPUs | W&B run name |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `taco_sharpa20hz_tp4_ppo_helmet_f20_h35.yaml` | `brush__brush__helmet__20231027_011` | PPO | 20 | 35 | 36 | 0-1 | `sharpa20hz-tp4-helmet-ppo-f20-h35-gbs128` |
| `taco_sharpa20hz_tp4_return_as_adv_helmet_f20_h35.yaml` | `brush__brush__helmet__20231027_011` | PPO_with_return_as_adv | 20 | 35 | 36 | 2-3 | `sharpa20hz-tp4-helmet-retadv-f20-h35-gbs128` |
| `taco_sharpa20hz_tp4_ppo_box_f20_h35.yaml` | `brush__brush__box__20230930_034` | PPO | 20 | 35 | 36 | 4-5 | `sharpa20hz-tp4-box-ppo-f20-h35-gbs128` |
| `taco_sharpa20hz_tp4_return_as_adv_box_f20_h35.yaml` | `brush__brush__box__20230930_034` | PPO_with_return_as_adv | 20 | 35 | 36 | 6-7 | `sharpa20hz-tp4-box-retadv-f20-h35-gbs128` |

## Launch Commands

```bash
EMBODIED_PATH=/root/RLinf/examples/embodiment /root/RLinf/.venv/bin/python /root/RLinf/examples/embodiment/train_embodied_agent.py --config-path /root/RLinf/examples/embodiment/config --config-name taco_sharpa20hz_tp4_ppo_helmet_f20_h35.yaml
```

```bash
EMBODIED_PATH=/root/RLinf/examples/embodiment /root/RLinf/.venv/bin/python /root/RLinf/examples/embodiment/train_embodied_agent.py --config-path /root/RLinf/examples/embodiment/config --config-name taco_sharpa20hz_tp4_return_as_adv_helmet_f20_h35.yaml
```

```bash
EMBODIED_PATH=/root/RLinf/examples/embodiment /root/RLinf/.venv/bin/python /root/RLinf/examples/embodiment/train_embodied_agent.py --config-path /root/RLinf/examples/embodiment/config --config-name taco_sharpa20hz_tp4_ppo_box_f20_h35.yaml
```

```bash
EMBODIED_PATH=/root/RLinf/examples/embodiment /root/RLinf/.venv/bin/python /root/RLinf/examples/embodiment/train_embodied_agent.py --config-path /root/RLinf/examples/embodiment/config --config-name taco_sharpa20hz_tp4_return_as_adv_box_f20_h35.yaml
```


## Active Launch Status

Launched from commit `55250c72` and pushed to `fork/feat/ppo-return-as-adv` before training start. The first ordinary `nohup &` attempt did not survive this Codex execution environment and emitted empty logs; the effective launches below use `setsid -f` and are alive.

| Run | PID | GPUs | W&B URL | Status at check |
| --- | ---: | --- | --- | --- |
| `sharpa20hz-tp4-helmet-ppo-f20-h35-gbs128` | 3913399 | 0-1 | https://wandb.ai/hanyang-chen-app-applovin/taco-allegro-flow-rl/runs/np467w5c | alive, reached global step 8 |
| `sharpa20hz-tp4-helmet-retadv-f20-h35-gbs128` | 3917603 | 2-3 | https://wandb.ai/hanyang-chen-app-applovin/taco-allegro-flow-rl/runs/1jz1mr17 | alive, reached global step 6 |
| `sharpa20hz-tp4-box-ppo-f20-h35-gbs128` | 3923687 | 4-5 | https://wandb.ai/hanyang-chen-app-applovin/taco-allegro-flow-rl/runs/bfkrqqut | alive, reached global step 4 |
| `sharpa20hz-tp4-box-retadv-f20-h35-gbs128` | 3930124 | 6-7 | https://wandb.ai/hanyang-chen-app-applovin/taco-allegro-flow-rl/runs/l0f3jxdy | alive, reached global step 1 |
