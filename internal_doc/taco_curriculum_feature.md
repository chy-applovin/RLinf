# TACO Horizon Curriculum Feature

本文单独说明 TACO horizon curriculum 的功能目标、参数含义、执行顺序，以及当前推荐的 step0/full-pool 训练语义。

相关文件：

- `rlinf/workers/env/env_worker.py`
- `rlinf/workers/rollout/hf/huggingface_worker.py`
- `rlinf/workers/actor/fsdp_actor_worker.py`
- `rlinf/envs/taco/taco_env.py`
- `examples/embodiment/config/env/taco_brush_allegro.yaml`
- `examples/embodiment/config/taco_allegro_ppo_flow_a4_step0_curriculum_hand.yaml`

## 1. 功能目标

Curriculum 的目标是让 rollout horizon 随训练逐渐变长，但不要把 rollout 收集到的样本数和 actor 的 `global_batch_size` 耦合起来。

合理语义是：

1. 当前 global step 根据 schedule 得到 `current_horizon`。
2. Env/rollout 只收集当前 horizon 对应的真实 transition pool。
3. Actor 先在这个真实 pool 上计算 advantage/return。
4. PPO 更新前，只为了满足 dataloader 切分，把有效 pool resize 到最近的 `actor.global_batch_size` 整数倍。
5. Actor 按 `global_batch_size` 切成一个或多个 train global batches。

被删除的旧语义是：从有效 pool 中强行采样到等于一个 `global_batch_size`。这会把“rollout size”和“batch size”重新耦合，不再作为支持路径。

## 2. 配置入口

```yaml
horizon_curriculum:
  enabled: True
  collect_current_horizon_only: True
  sample_to_global_batch_multiple: True
  start_horizon: 4
  end_horizon: 200
  milestones:
    - step: 0
      horizon: 4
    - step: 500
      horizon: 8
    - step: 1000
      horizon: 16
    - step: 1800
      horizon: 32
    - step: 2800
      horizon: 64
    - step: 3800
      horizon: 128
    - step: 4500
      horizon: 200
```

## 3. 参数含义

### horizon_curriculum.enabled

Curriculum 总开关。若为 false，env 和 rollout 使用 `max_episode_steps` / `max_steps_per_rollout_epoch` 的固定最大长度。

若为 true，当前 global step 对应的 horizon 由 `milestones` 或线性 ramp 计算。

### horizon_curriculum.collect_current_horizon_only

控制 rollout 阶段是否只收集当前 curriculum horizon。该开关只影响 env/rollout 的时间长度，不决定 actor batch size。

若为 true：

- env episode cap 被设成当前 horizon。
- rollout worker 只请求 `ceil(current_horizon / num_action_chunks)` 次 policy inference。
- 前期短 horizon 不会继续请求 max horizon 后面的无效 action/logprob/value。

若为 false：

- 复现 fixed-envelope 行为。
- env episode cap 可以变短，但 rollout worker 仍按 `max_steps_per_rollout_epoch` 请求满 rollout。
- done 后 transition 通过 loss mask 排除，但 rollout/training forward 仍可能浪费。

A4 使用 true。

### horizon_curriculum.sample_to_global_batch_multiple

控制 actor 端是否在 advantage/return 计算之后，把有效 transition pool resize 到最近的 `actor.global_batch_size` 整数倍。

它解决的是 PPO dataloader 的形状约束，而不是 rollout 阶段的 horizon 问题。开启后：

1. 统计有效 transition 数 `pool_size`。
2. 找到距离 `pool_size` 最近的 `actor.global_batch_size` 整数倍。
3. 若目标小于 pool，则无放回下采样。
4. 若目标大于 pool，则有放回补齐。
5. Actor 按 `global_batch_size` 切分成一个或多个 train global batches。

因此每个 global step 的 optimizer updates 约为：

```text
updates_per_global_step = update_epoch * resized_pool_size / global_batch_size
```

这比“每个 global step 必须 rollout 出恰好一个 global_batch_size”更自然；rollout 只由当前 curriculum horizon 决定。

### start_horizon / end_horizon

Curriculum 的最小和最大 horizon。`end_horizon` 通常等于 `env.train.max_episode_steps`。

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

动态 rollout chunk 数为：

```text
train_chunk_steps = ceil(current_horizon / num_action_chunks)
```

A4 使用：

```yaml
actor.model.num_action_chunks: 4
```

所以：

```text
horizon=4   -> train_chunk_steps=1
horizon=64  -> train_chunk_steps=16
horizon=200 -> train_chunk_steps=50
```

## 5. 执行顺序

一次 global step 的 curriculum 路径如下。

1. Runner 把当前 `global_step` 设置给 actor / rollout / env worker。
2. Env worker 和 rollout worker 根据同一份 schedule 计算当前 horizon。
3. Env worker 设置每个 TACO env 的 `max_episode_steps = current_horizon`。
4. Rollout worker 只循环 `ceil(current_horizon / num_action_chunks)` 次，向 policy 请求 action/logprob/value。
5. Env 执行 action chunk，形成真实 rollout pool。
6. Actor 先基于完整 pool 计算 advantage/return。
7. 若 `sample_to_global_batch_multiple=True`，actor resize 有效 pool 到最近的 `global_batch_size` 整数倍。
8. Actor 按 `global_batch_size` 切成 train global batches，并对每个 batch 执行 `update_epoch` 轮 PPO 更新。

## 6. Step0 Curriculum (A4)

A4 不是 demo-start：

```yaml
single_step.enabled: False
demo_start.enabled: False
rsi.enabled: False
early_termination.enabled: False
```

在 TACO env 中，RSI 关闭时 reset sampler 返回 frame 0，因此每个 rollout 都从 demo 的 step0 初始状态开始。

A4 默认：

```text
total_num_envs = 128
num_action_chunks = 4
global_batch_size = 128
update_epoch = 4
```

收集 chunks 数：

```text
collected_chunks = total_num_envs * ceil(current_horizon / num_action_chunks)
```

例子：

```text
horizon 4   -> collected chunks 128  -> 4 optimizer updates / global step
horizon 32  -> collected chunks 1024 -> 32 optimizer updates / global step
horizon 64  -> collected chunks 2048 -> 64 optimizer updates / global step
horizon 200 -> collected chunks 6400 -> 200 optimizer updates / global step
```

当前 A4 参数天然使 `collected_chunks` 都是 `global_batch_size=128` 的整数倍，所以 `sample_to_global_batch_multiple=True` 多数时候是 no-op；它主要用于未来改 `total_num_envs`、`global_batch_size`、horizon 或 valid mask 后仍能满足 PPO batch 切分。

## 7. Logged Metrics

开启 curriculum 后，env metrics 会记录：

- `env/curriculum_horizon_steps`：当前 effective horizon。
- `env/curriculum_rollout_chunk_steps`：当前 rollout 请求的 chunk 数。
- `env/curriculum_transition_pool_size`：当前理论有效 transition pool size，约为 `current_horizon * total_num_envs`。

开启 `sample_to_global_batch_multiple` 后，rollout metrics 会记录：

- `rollout/curriculum_transition_pool_size`：actor 端看到的全局有效 pool size。
- `rollout/curriculum_transition_pool_size_per_rank`：每个 actor rank 上的 pool size。
- `rollout/curriculum_sampled_batch_size`：resize 后的全局 batch pool size。
- `rollout/curriculum_sampled_batch_size_per_rank`：每个 actor rank resize 后的 batch pool size。
- `rollout/curriculum_sample_with_replacement`：pool 不足、需要有放回补齐时为 1，否则为 0。
- `rollout/curriculum_sample_reuse_ratio`：`target_size_per_rank / pool_size_per_rank`。
- `rollout/curriculum_sample_global_batch_multiple`：resize 后包含多少个 train global batches。

## 8. 使用建议

推荐路径：

```yaml
horizon_curriculum:
  enabled: True
  collect_current_horizon_only: True
  sample_to_global_batch_multiple: True
```

这样 rollout horizon 和 actor batch size 解耦：horizon 决定收集多少 on-policy samples，`global_batch_size` 只决定 PPO 每次 optimizer update 消耗多少 samples。
