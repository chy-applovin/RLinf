# RL Fine-tuning of the TACO Flow-Matching Policy (bimanual Allegro)

RL post-training for a point-cloud **flow-matching policy** (the external
[`flow_policy`](/root/flow-policy) package) that controls a bimanual Allegro
hand (44-dim qpos targets) in MuJoCo, on scenes from the TACO dataset
(`/root/Data/taco-brush-allegro`, one episode folder = one scene instance).

## What is wired together

| Piece | Implementation |
|---|---|
| Algorithm | πRL-style **PPO with flow-matching noise injection** (`flow_sde`): one random denoise step is stochastic during rollout, its Gaussian log-prob drives the PPO ratio. Same recipe as RLinf's OpenPi0 integration. |
| Policy | `model_type: flow_policy_taco` → `rlinf/models/embodiment/flow_policy_taco/`. Loads a flow-policy `.pt` checkpoint (model + normalizer + shape_meta) and adds a value head (GAE critic). Eval mode = the original deterministic flow ODE. |
| Environment | `env_type: taco` → `rlinf/envs/taco/`. Vectorized CPU MuJoCo; each sub-env is one TACO episode (own `scene.xml` + objects). Point-cloud obs are synthesized exactly like the IL eval harness (`mujoco-spider-env/scripts/rollout_eval.py`): canonical frame-0 object cloud re-posed by the live sim object pose. |
| Reward | Pluggable (`rlinf/envs/taco/rewards.py`): dense demo-**tracking** reward (default), `zero`, and a **VLM-as-judge stub** (design TBD, see below). |
| Training stack | Standard RLinf embodied pipeline: `EnvWorker` ↔ `MultiStepRolloutWorker` ↔ `EmbodiedFSDPActor` (`loss_type: actor_critic`, `adv_type: gae`). |

## Observation / action contract

Must match the flow-policy checkpoint's `shape_meta` (asserted at load time):

| Field | Shape | Notes |
|---|---|---|
| `states` | (B, To, 44) | hand qpos history (To = `obs_horizon`, e.g. 16) |
| `pointcloud` | (B, To, K, 3) | target-object cloud (K = `num_points`, e.g. 512) |
| `tool_pointcloud` | (B, To, K, 3) | only for `obs_mode: pc2_qpos` |
| action | (B, chunk, 44) | first `num_action_chunks` (= IL `action_horizon`, e.g. 4) steps of the denoised (Tp, 44) chunk; position targets at 30 Hz |

## Setup

```bash
cd /root/RLinf
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python -e . -e /root/flow-policy \
    "mujoco==3.7.0" gymnasium "imageio[ffmpeg]" transformers matplotlib
```

(`flow_policy` only needs torch/numpy/einops; Spider is NOT required — the env
only needs its Allegro STL meshes, pointed to by `allegro_assets_root`.)

## Smoke test (run before burning GPU time)

```bash
cd /root/RLinf
.venv/bin/python examples/embodiment/taco_allegro/smoke_test.py \
    --ckpt /root/flow-policy/outputs/flow-policy/2026-06-10/18-12-14/checkpoints/best.pt
```

Verifies: env reset/chunk_step on real episodes → noise-injected rollout →
**ODE parity** (eval sampler ≡ IL sampler) → **PPO log-prob parity** (training
forward ≡ rollout log-probs on stored chains) → backward pass.

## Launch training

```bash
cd /root/RLinf
source .venv/bin/activate
bash examples/embodiment/run_embodiment.sh taco_allegro_ppo_flow
```

Key config: `examples/embodiment/config/taco_allegro_ppo_flow.yaml`
(+ `config/env/taco_brush_allegro.yaml`, `config/model/flow_policy_taco.yaml`).

Knobs you will most likely touch:

| Knob | Where | Default |
|---|---|---|
| flow-policy checkpoint | `actor.model.model_path` | imIK `best.pt` |
| exploration noise | `actor.model.noise_level`, `noise_method` | 0.5, `flow_sde` |
| executed chunk length | `actor.model.num_action_chunks` | 4 |
| episode subset | `env.train.{episodes,categories,one_per_category,max_episodes}` | all 411 |
| reward | `env.train.reward.*` | tracking |
| PPO ratio granularity | `algorithm.logprob_type` | `chunk_level` (try `action_level` if ratios saturate: chunk-level sums 4×44 Gaussian terms) |
| parallel envs / rollout length | `env.train.total_num_envs`, `max_steps_per_rollout_epoch` | 32, 160 |
| wandb | `runner.logger.logger_backends` | tensorboard only |

Constraint chain to keep in mind when scaling:
`total_num_envs % (num_gpus * pipeline_stage_num) == 0`,
`max_steps_per_rollout_epoch % num_action_chunks == 0`, and
`actor.global_batch_size = total_num_envs × max_steps_per_rollout_epoch / num_action_chunks`
(per update epoch) with `global_batch_size % (micro_batch_size × num_gpus) == 0`.

## Reward: current state & VLM-as-judge plan

The default `tracking` reward is a dense, bounded demo-tracking signal
(tool/target object position + hand qpos vs the frame-aligned demo). Episode
**success** is logged as `tool_pos_err_final_m < 0.1` — the same metric as the
IL closed-loop eval — regardless of which reward is used.

VLM-as-judge is **deliberately left unimplemented** (design TBD). Two ready
integration points:

1. **RLinf-native (recommended)**: set `env.*.reward.type: zero` and
   `reward.use_reward_model: True`; host the VLM in an `EmbodiedRewardWorker`
   on its own GPU placement (`rlinf/models/embodiment/reward/vlm_reward_model.py`
   already provides scaffolding). The env's `capture_image()` can supply
   `main_images` frames once cameras are routed into the obs.
2. **In-env**: implement `VLMJudgeReward.compute` in
   `rlinf/envs/taco/rewards.py` (only sensible for cheap/remote API judges —
   blocking calls stall the rollout loop).

## Known limitations (v0)

- `auto_reset=False` only: fixed-length rollout epochs, episodes truncate at
  `min(max_episode_steps, T_demo - 1)`; post-done steps are loss-masked by the
  standard RLinf machinery.
- Train-time videos require a working GL backend (`MUJOCO_GL=glfw` + `xvfb-run`
  on this machine); `video_cfg.save_video` is off by default.
- Physics is CPU MuJoCo (threaded). Scale rollout throughput with more env
  workers (`cluster.component_placement`) before reaching for MJWarp.
