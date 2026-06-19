# TACO Flow-Policy RL 训练参数与整体工作原理

本文面向接下来继续使用或修改 TACO brush Allegro flow-policy RL 框架的人，解释当前实现中的训练循环、rollout 数据形状、PPO 更新次数，以及关键配置参数的含义。

相关入口：

- 训练配置：`examples/embodiment/config/taco_allegro_ppo_flow_a*.yaml`
- 环境默认配置：`examples/embodiment/config/env/taco_brush_allegro.yaml`
- Runner：`rlinf/runners/embodied_runner.py`
- Env worker：`rlinf/workers/env/env_worker.py`
- Rollout worker：`rlinf/workers/rollout/hf/huggingface_worker.py`
- Actor worker：`rlinf/workers/actor/fsdp_actor_worker.py`
- TACO env：`rlinf/envs/taco/taco_env.py`

## 1. 总体训练循环

当前 TACO 实验使用 `EmbodiedRunner`。一次外层 `global step` 的顺序是：

1. `actor` 同步权重到 `rollout` worker。
2. `env` reset 或继续准备当前 rollout 的初始 observation。
3. `rollout` worker 根据 env observation 做 policy inference，输出 action、old logprob、value 和训练时需要的 forward inputs。
4. `env` 执行动作，得到 reward、done、truncation、termination 和下一帧 observation。
5. rollout 数据发送到 `actor`。
6. `actor.compute_advantages_and_returns()` 计算 advantage 和 return。
7. `actor.run_training()` 用 PPO loss 更新模型。
8. runner 执行 `global_step += 1`，并按 `save_interval` 保存 checkpoint。

因此日志中的：

```text
Global Step: 2492/10000
```

表示已经完成了 2492 次“采样 rollout + 计算 advantage + PPO 训练”的外层迭代，不是 mini-batch update 次数。

在当前 `EmbodiedRunner` 中：

```text
max_steps = runner.max_epochs
```

所以 `runner.max_epochs: 10000` 对应 10000 个 global steps。

## 2. Action Chunk 与 Rollout Step

RLinf 中 policy 一次 inference 不一定只输出一个环境 action。它可以输出一个 action chunk。

关键参数：

```yaml
actor:
  model:
    num_action_chunks: 1
```

含义：

- `num_action_chunks = C`：一次 policy inference 输出连续 `C` 个 action。
- env 的 `chunk_step()` 会一次接收 shape 为 `[num_envs, C, action_dim]` 的动作，并执行这 `C` 个 control steps。
- `max_steps_per_rollout_epoch` 表示每个 rollout epoch 最多收集多少个 control steps。
- 代码内部的 `n_train_chunk_steps` 是：

```text
n_train_chunk_steps = max_steps_per_rollout_epoch / num_action_chunks
```

当前 A1/A2/A3 都设置 `num_action_chunks = 1`，所以：

```text
1 chunk = 1 policy inference = 1 env control step
```

如果以后设 `num_action_chunks = 4`，那么：

```text
1 chunk = 1 policy inference = 4 env control steps
```

## 3. Global Step / Update Epoch / Global Batch / Micro Batch

### global step

`global step` 是 runner 外层 RL iteration。一次 global step 包含 rollout、advantage/return 计算和 actor training。

Checkpoint 路径中的：

```text
checkpoints/global_step_2000
```

表示保存于第 2000 个外层 global step 后。

### update_epoch

配置：

```yaml
algorithm:
  update_epoch: 4
```

含义：对同一批 rollout data 重复训练几轮 PPO。当前为 4，即同一批数据会被训练 4 遍。

### global_batch_size

配置：

```yaml
actor:
  global_batch_size: 4096
```

含义：一次 optimizer update 使用的全局 batch size，跨所有 actor GPU 统计。

如果 actor world size 是 2，那么每张 actor GPU 每次 optimizer update 处理：

```text
global_batch_size / actor_world_size = 4096 / 2 = 2048 samples
```

### micro_batch_size

配置：

```yaml
actor:
  micro_batch_size: 64
```

含义：每张 GPU 单次 forward/backward 的小 batch。多个 micro batch 通过 gradient accumulation 累积成一次 optimizer update。

每张 GPU 每次 optimizer update 需要的 micro batch 数是：

```text
global_batch_size / actor_world_size / micro_batch_size
```

例如 A2/A3：

```text
4096 / 2 / 64 = 32 micro batches per GPU per optimizer update
```

### 每个 global step 到底更新几次？

一般公式：

```text
optimizer update 次数 / global step
= update_epoch * ceil_or_split(rollout_train_samples / global_batch_size)
```

当前 TACO configs 和 efficient curriculum sampling 路径会让 actor 每个 global step 收到刚好一个 `global_batch_size` 的训练 batch。因此：

```text
optimizer update 次数 / global step = update_epoch = 4
```

但每次 optimizer update 内部仍然包含多个 micro batch 的 forward/backward。

以 A2/A3 为例：

```text
update_epoch = 4
global_batch_size = 4096
actor_world_size = 2
micro_batch_size = 64
```

所以：

```text
optimizer step / global step = 4
micro batch forward-backward / GPU / global step = 4 * (4096 / 2 / 64) = 128
```

## 4. TACO 环境相关参数

### episode / dataset

```yaml
dataset_root: /root/Data/taco-brush-allegro
trajectory_file: trajectory_ctrl_30hz_im.npz
episodes: ["brush__brush__bowl__20230927_027"]
```

- `dataset_root`：TACO episode 根目录。
- `trajectory_file`：用于初始化 sim、构造 observation 和计算 tracking reward 的 demo trajectory。
- `episodes`：训练使用的 episode 列表。当前实验只训练固定 episode。

### observation

```yaml
obs_mode: pc2_qpos
obs_horizon: 16
num_points: 512
```

- `obs_mode = pc2_qpos`：observation 包含 target point cloud、tool point cloud 和 hand qpos。
- `obs_horizon = 16`：输入给 policy 的历史帧数。
- `num_points = 512`：每个 point cloud 的采样点数。

### rollout / episode length

```yaml
max_episode_steps: 32
max_steps_per_rollout_epoch: 32
auto_reset: False
ignore_terminations: False
```

- `max_episode_steps`：env 内单个 episode 的最多 control steps。
- `max_steps_per_rollout_epoch`：一个 rollout epoch 最多收集多少 control steps。
- `auto_reset=False`：TACO env 不在 worker 内自动 reset；每个 rollout epoch 由 worker 管理 reset。
- `ignore_terminations=False`：done/truncation 会进入 loss mask 和 GAE。

在 curriculum 未开启时，通常应保持：

```text
max_episode_steps == max_steps_per_rollout_epoch
```

在 curriculum 开启时，这两个值表示最大 horizon。当前有效 horizon 由 `horizon_curriculum` 动态决定。

### reward

```yaml
reward:
  type: tracking
  tool_pos_weight: 1.0
  target_pos_weight: 1.0
  hand_qpos_weight: 0.5
  tool_pos_scale: 0.05
  target_pos_scale: 0.05
  hand_qpos_scale: 0.5
  contact_weight: 0.0
```

- `type=tracking`：使用 demo tracking reward。
- `tool_pos_weight`：tool object 位置 tracking 权重。
- `target_pos_weight`：target object 位置 tracking 权重。
- `hand_qpos_weight`：hand qpos tracking 权重。
- `*_scale`：reward 中误差归一化尺度。尺度越小，同样误差惩罚越强。
- `contact_weight`：contact consistency 额外项，当前实验关闭。

A1/A3 使用 object + hand tracking，A2 使用 object-only tracking。

## 5. Single-Step 与 Demo-Start

### single_step

```yaml
single_step:
  enabled: True
  hand_obs_noise_std: 0.02
  perturb_init_hand: True
  min_frame: 0
```

语义：

1. 从 demo 中随机采样 timestep `t`。
2. sim 初始化到 `q_t`。
3. observation 使用 demo 历史窗口。
4. 只 rollout 1 个 control step。
5. reward 对比下一帧 demo state。

约束：

```text
single_step.enabled=True 时 max_episode_steps 必须为 1
```

### demo_start

```yaml
demo_start:
  enabled: True
  hand_obs_noise_std: 0.02
  perturb_init_hand: True
  min_frame: 0
```

语义与 `single_step` 类似，也是从 demo timestep `t` 初始化，但允许 rollout 多步，通常与 `horizon_curriculum` 搭配。

### hand_obs_noise_std / perturb_init_hand

- `hand_obs_noise_std`：对输入 observation 历史窗口中的 hand qpos 加 iid Gaussian noise。
- `perturb_init_hand=True`：同时把当前物理初始 hand qpos 也加上当前帧 noise，使策略学习“手部有偏移时如何补偿”。
- object state 不加噪声。

## 6. 三个实验配置的语义

### A1

文件：

```text
examples/embodiment/config/taco_allegro_ppo_flow_a1_single_step_hand.yaml
```

语义：

- single-step RL
- reward = object tracking + hand tracking
- no curriculum

关键值：

```text
total_num_envs = 512
max_episode_steps = 1
max_steps_per_rollout_epoch = 1
global_batch_size = 512
update_epoch = 4
```

### A2

文件：

```text
examples/embodiment/config/taco_allegro_ppo_flow_a2_curriculum_object.yaml
```

语义：

- demo-start curriculum RL
- reward = object-only tracking
- efficient curriculum sampling enabled

关键值：

```text
total_num_envs = 128
max_episode_steps = 32
max_steps_per_rollout_epoch = 32
global_batch_size = 4096
update_epoch = 4
```

### A3

文件：

```text
examples/embodiment/config/taco_allegro_ppo_flow_a3_curriculum_hand.yaml
```

语义：

- demo-start curriculum RL
- reward = object tracking + hand tracking
- efficient curriculum sampling enabled

关键值与 A2 相同，但 `hand_qpos_weight=0.5`。

## 7. 常见误解

### `max_epochs` 不是 optimizer update 次数

在当前 EmbodiedRunner 中，`max_epochs=10000` 是 10000 个 global steps。若 `update_epoch=4` 且每个 global step 只有一个 train global batch，则大约有：

```text
10000 * 4 = 40000 optimizer updates
```

### `global_batch_size` 不决定每个 global step 的 update 次数

`global_batch_size` 决定一次 optimizer update 用多少样本。每个 global step 有几次 optimizer update 主要由：

```text
update_epoch
rollout_train_samples / global_batch_size
```

共同决定。

### micro batch 不是 optimizer update

micro batch 是 gradient accumulation 的切分单位。多个 micro batch 累积后才执行一次 `optimizer.step()`。

### action chunk 不是 global batch

action chunk 属于 rollout/env 时间维度，global batch 属于 actor training 样本维度。二者独立。
