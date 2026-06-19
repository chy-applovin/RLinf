# TACO Demo-Start Horizon Curriculum Feature

本文单独说明 TACO demo-start curriculum 的功能目标、参数含义、旧实现的问题，以及当前 efficient curriculum sampling 的执行顺序。

相关文件：

- `rlinf/workers/env/env_worker.py`
- `rlinf/workers/rollout/hf/huggingface_worker.py`
- `rlinf/workers/actor/fsdp_actor_worker.py`
- `rlinf/envs/taco/taco_env.py`
- `rlinf/envs/taco/single_step.py`
- `examples/embodiment/config/env/taco_brush_allegro.yaml`
- `examples/embodiment/config/taco_allegro_ppo_flow_a2_curriculum_object.yaml`
- `examples/embodiment/config/taco_allegro_ppo_flow_a3_curriculum_hand.yaml`

## 1. 功能目标

episode-level RL 对 IK imitation learned policy 很难，因为初始误差会在长 horizon 内快速累积。single-step RL 稳定，但只用 object reward 时会出现 reward hacking：策略可能输出训练集中不存在的 hand action，只要单步让 object 到目标位置即可；这种 hand action 在闭环测试时容易使系统发散。

Curriculum 的目标是折中：

1. 每次 reset 仍然从 demo 中随机采样 timestep `t`，初始化到 `q_t`。
2. 训练初期只交互 1 步，避免长 horizon 的困难 credit assignment。
3. 随训练推进，逐渐增加 rollout horizon，例如 `1 -> 2 -> 4 -> 8 -> 16 -> 32`。
4. 让策略逐步面对更长的闭环误差积累。

## 2. 配置入口

```yaml
demo_start:
  enabled: True
  hand_obs_noise_std: 0.02
  perturb_init_hand: True
  min_frame: 0

horizon_curriculum:
  enabled: True
  collect_current_horizon_only: True
  resample_valid_transitions: True
  sample_to_global_batch_multiple: False
  start_horizon: 1
  end_horizon: 32
  milestones:
    - step: 0
      horizon: 1
    - step: 1000
      horizon: 2
    - step: 2500
      horizon: 4
    - step: 4500
      horizon: 8
    - step: 7000
      horizon: 16
    - step: 9000
      horizon: 32
```

## 3. 参数含义

### demo_start.enabled

开启 demo timestep reset。每个 env reset 时：

1. 从当前 episode demo trajectory 中采样 timestep `t`。
2. sim full state 初始化到 demo state `q_t`。
3. observation history 使用 demo 历史窗口 `q_{t-To+1} ... q_t`。
4. 后续 rollout 从 `q_t` 开始闭环执行 policy action。

`demo_start` 与 `single_step`、RSI、early termination 互斥。

### hand_obs_noise_std

对 observation history 中每一帧 hand qpos 添加 Gaussian noise。该噪声只作用于 hand qpos，不扰动 object pose / point cloud。

目标是学习 robust mapping：当输入手部位置有偏移时，策略仍能输出能正确 track object 的 action。

### perturb_init_hand

若为 true，当前物理初始 hand qpos 也加上 observation 当前帧对应的 noise。这样训练的是“真实手部状态偏移后如何补偿”。

若为 false，则只做 observation noise，更像 denoising。

### horizon_curriculum.enabled

开启 horizon schedule。当前 global step 对应的 horizon 由 milestones 或线性 ramp 决定。

### horizon_curriculum.collect_current_horizon_only

控制 rollout 阶段是否只收集当前 curriculum horizon。该开关只影响 env/rollout 的时间长度，不决定 actor 是否重采样。

若为 true：

- env episode cap 被设成当前 horizon。
- rollout worker 只请求 `ceil(current_horizon / num_action_chunks)` 次 policy inference。
- 前期短 horizon 不会继续请求 max horizon 后面的无效 action/logprob/value。

若为 false：

- 复现旧 fixed-envelope 行为。
- env episode cap 可以变短，但 rollout worker 仍按 `max_steps_per_rollout_epoch` 请求满 rollout。
- done 后 transition 通过 loss mask 排除，但 rollout/training forward 仍可能浪费。

为了兼容旧配置，若没有显式写该字段，它默认跟随 `resample_valid_transitions`：旧 A2/A3 的 `resample_valid_transitions=True` 会自动进入动态收集路径；旧的 `resample_valid_transitions=False` 配置仍保持 fixed-envelope。

### horizon_curriculum.resample_valid_transitions

控制 actor 在 advantage/return 计算之后是否把有效 transition pool 重采样成一个固定 `actor.global_batch_size`。

若为 true：

- actor 从有效 transition pool 中采样到固定 `actor.global_batch_size`。
- 每个 global step 通常只训练一个 global batch，因此 optimizer updates/global step 通常等于 `update_epoch`。
- pool 不足时会 replacement sampling，并记录 reuse ratio。

若为 false：

- actor 不强制压成一个 train global batch。
- 若 `sample_to_global_batch_multiple=True`，actor 会把有效 transition pool resize 到最近的 `actor.global_batch_size` 整数倍。
- 若 `sample_to_global_batch_multiple=False`，actor 保留全部收集 transition，此时 pool 必须已经能被 `global_batch_size` 整除。

### horizon_curriculum.sample_to_global_batch_multiple

仅在 `resample_valid_transitions=False` 时使用。它解决的是 PPO dataloader 的形状约束，而不是 rollout 阶段的 horizon 问题。

执行逻辑：

1. rollout 按当前 curriculum horizon 收集真实 transition pool。
2. actor 先在原始 pool 上计算 advantage/return。
3. 统计有效 transition 数 `pool_size`。
4. 找到距离 `pool_size` 最近的 `actor.global_batch_size` 整数倍。
5. 若目标小于 pool，则无放回下采样；若目标大于 pool，则有放回补齐。
6. actor 按 `global_batch_size` 切分为一个或多个 train global batches。

因此每个 global step 的 optimizer updates 约为：

```text
updates_per_global_step = update_epoch * resized_pool_size / global_batch_size
```

这比“每个 global step 必须 rollout 出恰好一个 global_batch_size”更自然；rollout 只由当前 curriculum horizon 决定。

### start_horizon / end_horizon

curriculum 的最小和最大 horizon。`end_horizon` 通常等于 `env.train.max_episode_steps`。

### milestones

分段常数 schedule。每个 item 表示：

```yaml
- step: <global_step>
  horizon: <effective_horizon>
```

若当前 `global_step >= step`，则使用对应 horizon。多个 milestone 中取最后一个满足条件的值。

## 4. Horizon 与 Chunk 的关系

`horizon` 是 env control step 数。

`num_action_chunks` 是一次 policy inference 输出多少个连续 action。

代码中动态 rollout chunk 数为：

```text
train_chunk_steps = ceil(current_horizon / num_action_chunks)
```

当前 TACO 实验中：

```yaml
actor.model.num_action_chunks: 1
```

所以：

```text
train_chunk_steps = current_horizon
```

如果以后 `num_action_chunks=4`：

```text
horizon=10 -> train_chunk_steps=ceil(10/4)=3
```

## 5. 旧 Fixed-Envelope Curriculum 的问题

旧 A2/A3 中：

```yaml
max_episode_steps: 32
max_steps_per_rollout_epoch: 32
```

即使当前 curriculum horizon 是 1 或 2，rollout worker 仍然按 32 个 chunk 请求 policy inference。

实际行为：

1. 前 `k` 步：env 正常推进 MuJoCo。
2. 第 `k` 步后：sub-env 已经 `done`。
3. 之后 rollout worker 仍然请求 action/logprob/value。
4. Env 收到 action 后，因为 `sub.done=True`，`TacoEnv._step_one()` 直接返回 `0 reward`，不再执行 `mujoco.mj_step()`。
5. Actor training 中这些 done 后位置通过 `loss_mask` 排除。

因此旧实现浪费两类计算：

- rollout 阶段：done 后仍然做 policy inference。
- training 阶段：padding transition 也会参与 model forward，只是 loss 被 mask。

## 6. Efficient Curriculum Sampling 的执行顺序

当前 `resample_valid_transitions=True` 后，一次 global step 的 curriculum 路径如下。

### Step 1: Runner 设置 global step

Runner 每个 global step 开始时调用：

```text
actor.set_global_step(global_step)
rollout.set_global_step(global_step)
env.set_global_step(global_step)
```

env worker 和 rollout worker 用这个 global step 计算当前 curriculum horizon。

### Step 2: 计算当前 horizon

Env worker 和 rollout worker 都根据同一份 config 计算：

```text
current_horizon = schedule(global_step)
```

例如：

```text
global_step 0-999    -> horizon 1
global_step 1000-2499 -> horizon 2
global_step 2500-4499 -> horizon 4
```

### Step 3: Env 设置 episode cap

Env worker 把每个 TACO env 的：

```text
env.max_episode_steps = current_horizon
```

因此每次 reset 时，demo-start sub-env 的 episode length 变为：

```text
ep_len = min(current_horizon, remaining_demo_frames)
```

### Step 4: Rollout worker 只请求当前 horizon 对应的 chunk 数

Rollout worker 不再固定循环 `max_steps_per_rollout_epoch`，而是：

```text
train_chunk_steps = ceil(current_horizon / num_action_chunks)
```

只请求这些 chunk 的 action/logprob/value。

当前 `num_action_chunks=1`，所以 horizon 2 时只请求 2 次 policy inference。

### Step 5: Env 真实执行 action

Env 对每个 action chunk 调用 `chunk_step()`。当前 chunk size 为 1，所以每个 action 对应一个 control step。

Env 在第 `current_horizon` 步到达 truncation，并记录 episode metrics。

### Step 6: Actor 先计算所有有效 rollout 的 advantage/return

收集到的 trajectory 发送给 actor 后，actor 先按真实 k-step rollout 计算：

```text
advantages
returns
loss_mask
loss_mask_sum
```

注意：重采样发生在 advantage/return 计算之后。这样每个 transition 的 advantage 是在真实短 horizon trajectory 上计算出来的。

### Step 7: 从有效 transition pool 中采样固定 global batch

Actor 根据 `loss_mask` 找到有效 transition pool：

```text
valid_pool = all transitions where loss_mask is true
```

然后采样到固定大小：

```text
target_size_per_rank = actor.global_batch_size / actor_world_size
```

若 pool size 足够，则无放回采样；若 pool size 不足，则有放回采样。

采样后，actor 的 batch shape 被整理成等价于一个 train global batch：

```text
[1, global_batch_size_per_rank, ...]
```

bootstrap 类字段如 `dones`、`truncations`、`terminations`、`prev_values` 会保留当前 transition 和 next transition 两个位置，保证 GAE 和 value loss 的结构一致。

### Step 8: PPO training

Actor 使用采样后的 fixed-size batch 执行 PPO 更新。

当前配置下：

```text
update_epoch = 4
global_batch_size = 4096
micro_batch_size = 64
actor_world_size = 2
```

所以每个 global step：

```text
optimizer updates = 4
micro-batch forward/backward per GPU = 4 * (4096 / 2 / 64) = 128
```

## 7. 新旧行为对比

以 A2/A3 的 horizon 2 阶段为例：

```text
total_num_envs = 128
max_steps_per_rollout_epoch = 32
global_batch_size = 4096
num_action_chunks = 1
```

旧 fixed-envelope 行为：

```text
真实 MuJoCo transition: 128 * 2 = 256
rollout policy inference: 128 * 32 = 4096
training tensor envelope: 4096 transitions, 大部分 invalid
optimizer updates/global step: 4
```

新 efficient sampling 行为：

```text
真实 MuJoCo transition: 128 * 2 = 256
rollout policy inference: 128 * 2 = 256
valid transition pool: 256
sampled training batch: 4096, pool 不足时 replacement
optimizer updates/global step: 4
```

新行为减少 rollout 阶段无效 inference，并保证 training forward 都来自有效 transition。若 `global_batch_size` 仍为 4096，则 training 阶段 forward/backward 计算量仍保持 4096 samples/update。

## 8. Logged Metrics

开启 curriculum 后，env metrics 会记录：

- `env/curriculum_horizon_steps`：当前 effective horizon。
- `env/curriculum_rollout_chunk_steps`：当前 rollout 请求的 chunk 数。
- `env/curriculum_transition_pool_size`：当前理论有效 transition pool size，约为 `current_horizon * total_num_envs`。

开启 efficient sampling 后，rollout metrics 会记录：

- `rollout/curriculum_transition_pool_size`：actor 端看到的全局有效 pool size。
- `rollout/curriculum_transition_pool_size_per_rank`：每个 actor rank 上的 pool size。
- `rollout/curriculum_sampled_batch_size`：采样后的全局 batch size。
- `rollout/curriculum_sampled_batch_size_per_rank`：每个 actor rank 采样后的 batch size。
- `rollout/curriculum_sample_with_replacement`：pool 不足时为 1，否则为 0。
- `rollout/curriculum_sample_reuse_ratio`：`target_size_per_rank / pool_size_per_rank`。

Smoke test 示例：

```text
curriculum_horizon_steps=1
curriculum_rollout_chunk_steps=1
curriculum_transition_pool_size=8
curriculum_sampled_batch_size=64
curriculum_sample_with_replacement=1
curriculum_sample_reuse_ratio=8
```

这表示只 rollout 1 步，从 8 个有效 transition 中有放回采样到 global batch 64。

## 9. 使用建议

### 默认推荐

对于 demo-start curriculum RL，推荐：

```yaml
horizon_curriculum:
  enabled: True
  resample_valid_transitions: True
```

这样可以避免前期 horizon 很短时的大量无效 rollout inference。

### 何时关闭 resample_valid_transitions

只有在需要复现实验中旧 fixed-envelope 行为，或需要严格保持旧 rollout tensor 长度时，才建议：

```yaml
resample_valid_transitions: False
```

### global_batch_size 的选择

如果 early curriculum pool 很小，但 `global_batch_size` 很大，会发生高 reuse ratio 的 replacement sampling。

例如：

```text
pool = 128
global_batch_size = 4096
reuse ratio = 32
```

这会降低样本多样性。若需要更公平或更高效的 ablation，可以考虑：

- 降低 early-stage global batch size，但这需要框架支持动态 batch。
- 增加 `total_num_envs`，扩大 pool。
- 接受 replacement sampling，但在结果解释中报告 reuse ratio。

当前实现选择固定 `global_batch_size`，以满足 actor training 的固定 batch 约束。

## 10. 与 A1/A2/A3 的关系

- A1：`single_step.enabled=True`，不使用 curriculum。
- A2：`demo_start.enabled=True`，`horizon_curriculum.enabled=True`，object-only reward。
- A3：`demo_start.enabled=True`，`horizon_curriculum.enabled=True`，object + hand reward。

若已经有旧代码启动的 A2/A3 进程，修改代码不会影响正在运行的进程。要使用 efficient curriculum sampling，需要重启训练或从最新 checkpoint resume。


## 11. Step0 Full-Pool Curriculum (A4)

A4 不是 demo-start：`single_step.enabled=False`、`demo_start.enabled=False`、`rsi.enabled=False`、`early_termination.enabled=False`。在 TACO env 中 RSI 关闭时，reset sampler 返回 frame 0，因此每个 rollout 都从 demo 的 step0 初始状态开始。

A4 使用的 curriculum 组合是：

```yaml
horizon_curriculum:
  enabled: True
  collect_current_horizon_only: True
  resample_valid_transitions: False
  sample_to_global_batch_multiple: True
  start_horizon: 4
  end_horizon: 200
```

执行顺序：

1. 根据 global step 计算当前 horizon，例如 `4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 200`。
2. Env worker 把每个 env 的 `max_episode_steps` 改成当前 horizon。
3. Rollout worker 请求 `ceil(horizon / num_action_chunks)` 个 action chunks；A4 中 `num_action_chunks=4`。
4. Env 从 step0 开始真实执行到当前 horizon，不从中间 demo timestep 初始化。
5. Actor 对所有收集到的 chunks 先计算 advantage/return。
6. 因为 `resample_valid_transitions=False` 且 `sample_to_global_batch_multiple=True`，actor 不采样成一个固定 batch，而是把有效 chunks resize 到最近的 `actor.global_batch_size` 整数倍。
7. Actor 按 `actor.global_batch_size` 分成一个或多个 global batches。每个 global step 的 optimizer update 次数约为：

```text
updates_per_global_step = (resized_chunks / global_batch_size) * update_epoch
```

例如 A4 默认：

```text
total_num_envs = 128
num_action_chunks = 4
global_batch_size = 128
update_epoch = 4
horizon = 64
collected_chunks = 128 * ceil(64 / 4) = 2048
resized_chunks = 2048  # already a global_batch_size multiple
updates_per_global_step = (2048 / 128) * 4 = 64
```

该模式避免了 fixed-envelope rollout 浪费，同时不会像 A2/A3 一样把 pool 重采样成单个固定 batch；代价是 horizon 变长后每个 global step 的 update 次数会显著增加。
