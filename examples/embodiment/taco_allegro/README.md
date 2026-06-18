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
(tool/target object position + hand qpos vs the frame-aligned demo), plus an
optional **contact-consistency term** (`reward.contact_weight`, default 0):
when the frame-aligned demo has hand-object contact, the sim earns that term
only by actually touching (right hand↔tool, left hand↔target) — added after
the step-1000 evals exposed a "hover without grasping" reward hack (see the
experiment record). Episode **success** is logged as
`tool_pos_err_final_m < 0.1` — the same metric as the IL closed-loop eval —
regardless of which reward is used.

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

### 2026-06-11/12 - IL-finetune runs (plain + DeepMimic), step-1000 eval

Runs `58erpa3a` (ilft) and `73hk83x4` (ilft+RSI/ET), both 1000 steps from
epoch_800.pt, object-dominant tracking reward. Training reward improved
0.435 -> 0.511 (ilft); closed-loop eval of BOTH step-1000 checkpoints:

| | IL epoch_800 | RL ilft | RL ilft+DM |
|---|---|---|---|
| tool_err mean / final (m) | 0.109 / 0.085 | 0.080 / 0.033 | 0.080 / 0.033 |
| brush displacement (max) | 0.082 m | **0.001 m** | **0.001 m** |
| reward-equiv / step | 0.442 | 0.512 | 0.515 |

**Both RL policies un-learned the grasp**: tool trajectories are bit-identical
passive physics (the hand hovers near the brush but never closes), yet the
reward IMPROVED - because on this episode an untouched brush (manipulation-
window err 0.158, final 0.033) outscores IL's imperfect brushing (0.191 /
0.085). PPO correctly optimized a mis-specified reward; the DeepMimic aids
did not prevent it because (a) `object_err_threshold: 0.25` exceeds the
demo's own max brush displacement (0.178 m), so "never grasp" never trips
`object_tracking`, and (b) hovering keeps the palm within the
`hand_object_distance` margin without grasping.

Fix directions: tighten `object_err_threshold` to ~0.10 (an untouched brush
hits 0.158 mid-window -> non-grasping episodes get terminated and the hack
becomes unprofitable), and/or add a contact-consistency reward term (reward
hand-object contact when the demo is in contact, cf. Spider's contact
reward). **Both implemented in the v2 recipe below.**

### 2026-06-12 - anti-reward-hacking v2 (`taco_allegro_ppo_flow_ilft_dm_v2.yaml`)

Implements exactly the two fixes diagnosed above; everything else (IL init
epoch_800.pt, object-dominant weights, RSI + both ET criteria, single
episode, PPO hyperparams, 1000 steps, ckpt every 200, 4 GPUs) is unchanged
from `taco_allegro_ppo_flow_ilft_dm.yaml`:

1. **ET `object_err_threshold` 0.25 -> 0.10 m** (config-only). The threshold
   must lie BELOW the demo's own peak brush displacement (0.178 m) to be able
   to catch "never grasp" (untouched brush errs ~0.158 m mid-window).
   Verified: the hover policy now terminates at control step ~52 instead of
   surviving all 160 steps.
2. **NEW contact-consistency reward term**
   (`reward.contact_weight: 0.5`, `rlinf/envs/taco/rewards.py`). Per
   (hand, object) pair - (right palm subtree, tool) and (left palm subtree,
   target) - the pair scores 1 when the frame-aligned demo is NOT in contact,
   and scores sim-contact (0/1, MuJoCo contact list with `dist <
   contact_dist_tol`) when the demo IS in contact; the term is the mean of
   the two pair scores. Demo contact flags are precomputed per episode by
   replaying demo qpos through `mj_forward` (brush_027: tool contact 64% of
   frames [12..152], target 46%). Direct analog of Spider's contact reward
   with binary contact instead of contact-site positions (TACO demos carry no
   contact labels). Spurious extra contact is deliberately not penalized
   (IK-retargeted contact phase boundaries are noisy). With weights
   tool 1.0 / target 1.0 / hand 0.1 / contact 0.5 (norm 2.6), hovering now
   forfeits ~0.19 r/step across the contact window - an order of magnitude
   more than the ~0.07 r/step the hack used to gain. CPU backend only
   (`TacoEnvGPU` asserts `contact_weight == 0`).

Calibration caveat (measured, in `test_antihack.py`): open-loop demo replay
only reaches ~0.39 window contact-match (not 1.0) because the IK-retargeted
grip is dynamically marginal - the replay sim loses brush contact mid-window.
The term is a *discriminator* (hover scores exactly the zero-touch baseline,
real engagement scores above it), not a tracking target; a closed-loop policy
that actually squeezes can push it toward 1.

Tests: `examples/embodiment/taco_allegro/test_antihack.py` (no Ray) - demo
contact window sanity, hover == zero-touch baseline + replay strictly above
it (+0.24), ET@0.10 kills the hover policy at step 52. Plus
`test_deepmimic.py` regression (all pass).

wandb: same project `taco-allegro-flow-rl`, run
`ilft-ppo-dm-v2-contact-et010-1ep-brush_bowl`, with `wandb_notes`/`wandb_tags`
(new optional `runner.logger` keys wired into `MetricLogger`) documenting the
full setup in the run page.

### 2026-06-18 - single-step (contextual-bandit) RL + hand-obs noise (`taco_allegro_ppo_flow_single_step.yaml`)

Episode-level RL of the IK-IL policy keeps failing (compounding error +
off-manifold drift; every recipe above eventually hacks or stalls). New
approach: drop episode-level RL entirely (NO RSI / NO early termination) and
optimize a **one-transition** objective instead - see "Single-step
(contextual-bandit) RL mode" below for the mechanism.

- init from the IK-IL checkpoint `epoch_800.pt`; each of 512 envs samples one
  demo timestep `t`, inits the full state to `q_t`, builds the To=16 obs window
  from the **real** demo history `q_{t-15..t}` (the exact IL window), runs ONE
  control step; **reward = next-frame OBJECT tracking vs `q_{t+1}`** (tool 1.0 +
  target 1.0, normalized; **hand weight 0** -> objects only). The episode ends,
  so the PPO return is exactly that single-step reward (verified: `reward` ==
  `return` in the logs; GAE bootstrap is masked by the done).
- **robustness augmentation**: iid Gaussian noise (std 0.02) is added to the
  hand qpos of every obs frame; `perturb_init_hand: True` also offsets the
  actual initial hand by the current-frame noise, so the hand genuinely starts
  displaced and the policy must emit an action that still tracks the object
  (not merely denoise its input).

Calibration (`test_single_step.py`, GT-action = demo `q_{t+1}` as the action):
the single-step setup is sound - replaying the GT action from `q_t` tracks the
objects at **r 0.906** (tool/target err ~0.6 cm). Pure obs noise leaves the
GT-action reward unchanged (0.912 @ std 0.02 - reward reads the true state, not
the noised obs), while `perturb_init_hand=True` lowers the GT ceiling slightly
(0.906 -> 0.878 @ std 0.02): the offset is real (it costs object tracking) yet
recoverable (still 0.88), so std 0.02 is a meaningful-but-not-destructive
perturbation. RL optimizes the gap between the IL policy's noised-obs behavior
and this ceiling.

Pipeline smoke (3 steps, 16 envs, 4 GPUs): clean; `reward`==`return`,
`single_step_frame` logged, `approx_kl` ~8e-4. Full run: 512 envs/step,
lr 1e-5 / value_lr 1e-4 (critic warmup 20), `num_action_chunks=1`, 1000 steps,
ckpt every 200, GPUs 0-3. wandb run
`single-step-rl-handnoise-1ep-brush_bowl` (notes/tags document the setup).

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
the current one finishes. (Workaround used here: `RAY_ADDRESS=local` forces an
isolated cluster per run.)

## Single-step (contextual-bandit) RL mode (config-gated, default off)

An alternative to episode-level RL for the IK-IL policy, implemented in
`rlinf/envs/taco/single_step.py` with guarded hooks in the CPU env (flag off =>
behavior identical to before; GPU backend asserts it off). Motivation:
episode-level RL of this policy is hard - errors compound over the closed-loop
rollout and the state drifts off the demo manifold, and the tracking reward is
easy to hack (see the experiment record). Single-step RL sidesteps all of that
by learning a one-transition map.

Mechanism (`env.*.single_step.enabled`, use with `max_episode_steps: 1`,
`max_steps_per_rollout_epoch: 1`, `actor.model.num_action_chunks: 1`; mutually
exclusive with `rsi` / `early_termination`):

* **reset** samples one demo timestep `t` (uniform in `[min_frame, T-2]`),
  initializes the full sim state to the reference `q_t` (hand + objects), and
  builds the To-frame observation window from the **real** demo frames
  `q_{t-To+1..t}` (edge-padded) - the exact Diffusion-Policy window the IL
  policy trained on (`action_offset=1`);
* **one control step**, then the episode ends; the reward is the next-frame
  tracking error vs `q_{t+1}`. Pair with an **object-only** tracking reward
  (`reward.hand_qpos_weight: 0`) so only the objects are rewarded. Because the
  episode is length 1, the PPO return equals that single reward - the value
  head learns `E[reward | s]` as a pure baseline (no bootstrap; GAE's
  `~dones[t+1]` masks the future).

Hand-observation-noise robustness (`single_step.hand_obs_noise_std`): iid
Gaussian noise is added to the hand qpos of **every** observation frame, so the
policy learns a mapping that emits the object-correct action even when the
observed hand pose is perturbed. `single_step.perturb_init_hand: True`
additionally offsets the actual initial hand state by the current-frame noise
(physical displacement to compensate, vs pure input noise to denoise).
`single_step_frame` (sampled `t`) is logged per rollout.

Tests: `examples/embodiment/taco_allegro/test_single_step.py` (no Ray) -
timestep sampling/range, demo-window obs alignment, hand-noise (obs-only and
init-perturb), single-step truncation + GT-action reward, default-off
regression, and a GT-action reward calibration sweep.

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
