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
| PPO ratio granularity | `algorithm.logprob_type` | `token_level` (per Gaussian element; `chunk_level` sums 4×44 terms and explodes the ratio) |
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

## Experiment record

### 2026-06-11 - from-scratch PPO sanity run (`taco_allegro_ppo_flow_scratch.yaml`)

Random-init weights (`actor.model.random_init: True`; ckpt used only as
architecture/IO spec + normalizer stats), single real scene
`brush__brush__bowl__20230927_027`, dense tracking reward vs
`trajectory_ctrl_30hz_im.npz`, PPO 300 steps x 64 envs x 160 steps, 8xH100,
~9.7 s/step (~50 min). wandb:
[taco-allegro-flow-rl/48v2zqaf](https://wandb.ai/hanyang-chen-app-applovin/taco-allegro-flow-rl/runs/48v2zqaf).

| metric | step ~0 | step 299 |
|---|---|---|
| train `env/success_once` (tool err < 0.1 m) | 0.19 | **1.00** |
| train `env/tool_pos_err_final_m` | 0.76 | **0.033** |
| eval (deterministic ODE) `success_once` | 0.50 @49 | **1.00** @199-299 |
| eval `tool_pos_err_final_m` | 0.48 @49 | **0.033** |
| `rollout/rewards` per step | 0.39 | 0.46 |

Conclusion: the full RL loop (noise-injected flow sampling -> PPO -> weight
sync) trains a randomly initialized flow policy from scratch and improves the
tracking return. Checkpoints (DCP + model_state_dict) every 50 steps under
`<log_path>/scratch-ppo-tracking-1ep-brush_bowl/checkpoints/global_step_*/actor/`.

**Qualitative follow-up** (closed-loop videos via `export_flow_ckpt.py` +
`mujoco-spider-env/scripts/rollout_eval.py`, step 50 vs step 300): neither
checkpoint grasps the brush - the sim tool trajectories of the two are
bit-identical (the hand never touches it). PPO instead converged to a
"don't disturb anything" local optimum: step 50 still knocks the bowl away
(target_err_final 1.07 m), step 300 keeps both objects undisturbed. Per-step
reward decomposition: doing nothing earns ~0.80/step outside the manipulation
window and ~0.20/step inside it (the unearned tool term ~= 23 return units),
so the reward does prefer real manipulation, but exp(-err/0.05) is nearly
flat beyond ~15 cm and exploration never finds grasping from scratch.

Two lessons baked into the metric/reward design backlog:

1. `success_tool<0.1m` at the FINAL frame is trivially satisfiable on
   episodes whose demo returns objects near their start pose (here the
   brush's net displacement at frame 160 is only 0.034 m) - success should be
   measured during the peak-displacement window or via mean tracking error;
2. from-scratch RL cannot discover dexterous grasping under a pure tracking
   reward - the intended usage (RL fine-tuning FROM the IL checkpoint,
   random_init: False) starts inside the reward basin instead.

### 2026-06-11 - hand-only tracking reward, from scratch (PAUSED at step 311, diagnosed)

`taco_allegro_ppo_flow_scratch_hand.yaml`, wandb run `o0oo4y92`. Reward
declined 0.456 -> 0.40 instead of learning; paused for diagnosis. Findings
(step-200 ckpt closed-loop eval + demo-obs probes + wandb):

1. **Phase is unobservable from scratch.** Tracking a time-varying 44-dof
   demo requires knowing "which demo frame is now". The IL checkpoint reads
   phase from its obs (demo-obs probe: output-vs-phase corr **0.985**)
   because on-distribution qpos history IS the phase; the random-init/RL
   policies are phase-blind (corr 0.23 / **0.11**) - off-distribution own
   states and static object clouds carry no time signal. Without phase, the
   reachable policy class is ~static poses, ceiling r~=0.67 (demo-time-mean
   pose baseline).
2. **PPO drifted BELOW even that static ceiling (0.40 < 0.52 frame-0-pose <
   0.67 mean-pose).** Critic EV ~0.43: more than half the return variance is
   uncontrollable (SDE exploration + contact chaos), so advantages mostly
   reinforce noise-correlated base motion. The first update was huge
   (approx_kl 6.9 at step 0, lr 3e-4 on a random policy) and damaged the
   decent initial prior (random-init flow ~= dataset marginal distribution,
   r=0.456). Closed-loop result: the bimanual base flails out of the
   workspace and launches both objects meters away (target_pos_err_final
   6.1 m) - nothing penalizes collisions in the hand-only reward.
3. **Reward shape dilutes the signal.** mean|dq| over 44 dims mixes base
   meters with finger radians: base error dominates (1.09 vs 0.23), and the
   per-dof gradient at err~0.5 is ~0.02/rad - ~500x weaker than the object
   terms near their basins in the previous run.

Fix directions (not yet applied): add phase/goal conditioning to the obs
(DeepMimic-style), per-group error scales (base-pos/base-rot/fingers),
small nonzero object terms to penalize collisions, KL-guarded warmup (lower
initial lr / critic warmup), and/or RL from the IL checkpoint (phase-aware
by construction).

## DeepMimic training aids (config-gated, default off)

Two techniques from DeepMimic (Peng et al. 2018), implemented in
`rlinf/envs/taco/deepmimic.py` with guarded hooks in the CPU env (flags off =>
behavior identical to before; GPU backend asserts them off):

* **RSI** (`env.*.rsi.enabled`): episodes start at a uniformly sampled demo
  frame (full hand+object qpos/qvel from the reference), so late-demo phases
  are visited from the first rollout instead of being gated on mastering
  everything before them. Reward/termination frame-alignment, episode caps
  and metrics are all start-frame aware (`rsi_start_frame` logged).
  Caveat (same as DeepMimic): cold-starting mid-grasp contact states is
  dynamically imperfect - open-loop demo replay from mid-grasp frames slowly
  loses the object; the closed-loop policy is expected to correct.
* **Early termination** (`env.*.early_termination.enabled`): episodes end at
  the first unrecoverable failure so post-failure frames don't pollute the
  batch or the critic's value targets. Criteria (combinable):
  `hand_object_distance` - sim palm-to-object distance exceeds the demo's
  distance at the aligned frame by a threshold (phase-aware drop detection);
  `object_tracking` - object position error vs the aligned demo frame above a
  threshold (the direct DeepMimic analog). Termination flows through RLinf's
  existing terminations -> loss-mask/GAE machinery (requires
  `ignore_terminations: False`); terminated episodes are never counted as
  successes (`terminated_early` logged).

Tests: `examples/embodiment/taco_allegro/test_deepmimic.py` (no Ray needed) -
default-off regression, RSI distribution/alignment, both ET criteria firing.
Recipe with both enabled: `taco_allegro_ppo_flow_ilft_dm.yaml` (GPUs 4-7).

NOTE (operational): RLinf joins an existing Ray cluster (`address="auto"`,
fixed namespace and channel/actor-group names), so two trainings cannot run
concurrently on one machine even on disjoint GPUs - launch the DM run after
the current one finishes.

## Simulator backends: CPU (default) vs GPU (MuJoCo Warp)

`env.*.sim_backend: cpu | gpu` selects between the original threaded
CPU-MuJoCo env (`taco_env.py`, untouched) and a MuJoCo-Warp batched env
(`taco_env_gpu.py`: physics, point-cloud synthesis and tracking reward all
batched in torch on one CUDA device). GPU v1 constraints: exactly one episode
per worker (`episodes: [name]`), reward `tracking`/`zero`, no video.

Benchmark on this machine (H100, CUDA driver 12.2, single episode,
demo-action chunks, env stepping + obs + reward only):

| num_envs | CPU (16 threads) | GPU (mjwarp) |
|---|---|---|
| 64 | **4445 steps/s** | 1567 steps/s |
| 256 | **4158 steps/s** | 589 steps/s |
| 1024 | - | OOM (convex narrowphase workspace ~100 GB) |

Physics parity is excellent (open-loop demo replay: identical hand error to
4 decimals), but the GPU backend is currently SLOWER here because (a) the
CUDA driver is 12.2 while mjwarp's CUDA-graph capture needs >=12.4
(conditional graph nodes), so every physics step pays hundreds of Python
kernel launches, and (b) the 8-convex-hull objects x dense hand contact pairs
make the GPU narrowphase workspace scale badly with worlds. Also note env
physics is only ~5% of the training step time at 64 envs (the bottleneck is
the policy's 16-step denoising inference). Default stays `cpu`; revisit `gpu`
after a driver upgrade (>=12.4) or for far larger world counts.

## Known limitations (v0)

- `auto_reset=False` only: fixed-length rollout epochs, episodes truncate at
  `min(max_episode_steps, T_demo - 1)`; post-done steps are loss-masked by the
  standard RLinf machinery.
- Train-time videos require a working GL backend (`MUJOCO_GL=glfw` + `xvfb-run`
  on this machine); `video_cfg.save_video` is off by default.
- Physics is CPU MuJoCo (threaded). Scale rollout throughput with more env
  workers (`cluster.component_placement`) before reaching for MJWarp.
