# TACO GPU simulation (mjwarp) 加速与最佳实践 — 2026-07-01

本文记录 GPU rollout 慢的根因排查、dt / placement 两组 A/B 实验的结论，以及已经固化为默认值的设置。证据见两份已执行的 notebook：

- `notebooks/gpusim_bench.ipynb` — 根因量化（graph capture / nconmax / Newton vs CG / iterations）
- `notebooks/gpusim_dt_placement_ab.ipynb` — dt 保真 A/B + placement 矩阵
- 复现脚本：`notebooks/bench_mjwarp_sharpa.py`、`notebooks/ab_dt_fidelity.py`、`notebooks/smoke_taco_env_gpu.py`

## 1. 根因（已修复，均在 `rlinf/envs/taco/taco_env_gpu.py`）

| # | 问题 | 后果 | 修复 |
|---|---|---|---|
| 1 | 本机驱动 535（CUDA 12.2）不支持 conditional CUDA graph node，mjwarp 默认 `graph_conditional=True` 的 `capture_while` 让 graph capture 直接抛错 | 永久回退到裸 `mjwarp.step`，且 fallback 路径**每次 solver 迭代都做一次 device→host 同步**：1176 ms/控制步 | capture 前检测 `wp.is_conditional_graph_supported()`，不支持则 `opt.graph_conditional=False`（solver 循环 unroll 进 graph，kernel 内仍按 world 提前退出）：158 ms/控制步 |
| 2 | mjwarp 3.7 的 `put_data(nconmax=…)` 是 **per-world** 语义，旧代码传了 `100*num_envs` | 总 contact 分配 = `100*num_envs²`，512 env 时单块分配 57 GB 直接 OOM | 传 per-world 值 `nconmax=100`（实测峰值 ~55/world） |
| 3 | scene.xml 没有 `<option>`，MuJoCo 默认 Newton iterations=100 / ls_iterations=50 被整体 unroll 进 graph | 每个 substep 空跑 ~97 次迭代的 kernel launch：293 ms/控制步 | 新配置键 `gpu_solver_iterations: 10`、`gpu_ls_iterations: 20`（实测收敛只需均值 2.6、最大 7 次） |
| 4 | `njmax=350`/world，但 sharpa 场景峰值 nefc ≈ 610/world；mjwarp 对超出 njmax 的约束行**静默丢弃** | 部分接触约束丢失（保真损失，无报错） | 默认 `gpu_njmax_per_env: 800` |
| 5 | mjwarp 没有 CPU MuJoCo 的 BADQACC 自动重置，episode 末尾少数 world 发散成 NaN | reward/return 变 NaN，会毒化 RAFT 的 top-k episode 筛选 | reward `nan_to_num(0)` + 新增 `env/sim_nan_frac` 指标（正常 ≤1%，飙升说明物理不稳定） |
| 6 | 分支 `feat/taco-allegro-flow-rl` 缺 sharpa 版 `scene.py`/`taco_env.py` 等 5 个文件（半截 cherry-pick） | 新起任何 sharpa 实验直接 TypeError | 已从 `feat/ppo-return-as-adv` checkout 回来一并提交 |

同事两个猜想的核对：

- **"Newton 慢、应换 CG"：不成立。** 同场景 Newton 收敛 ~2.6 次迭代（158 ms/控制步），CG 需要 ~6 次（174–204 ms）。"CG 快"的经验来自 MJX/JAX（跑不好 Newton 的动态循环），不适用 mjwarp。保留 Newton（可用 `gpu_solver: cg` 切换做对照）。
- **"graph 每个 rollout 后被清除、下个 step 重新 compile"：不成立。** env 与 graph 在 `EnvWorker.init_worker` 只建一次；真实情况是 graph 从未 capture 成功（见 #1），launch 时的 "compile 失败" 提示即 capture 抛错的 warning。

## 2. dt A/B：**保持 dt=0.002（25 substeps/20 Hz 控制步）**

方法（`notebooks/ab_dt_fidelity.py`）：256 env 开环回放 demo 目标 + 每 env 固定种子噪声，**各 dt case 动作逐 env 完全相同**；另跑同配置两次作为 run-to-run null（mjwarp 原子操作导致接触顺序非确定，本身就有随机性）。指标：tracking reward 曲线、tool 误差、blow-up 率、**per-env return 的 Spearman 相关与 top-10% 选择重叠**（对应 RAFT episode 筛选）。渲染对比视频：`logs/ab_dt/dt_ab_noise0.mp4`。

| case | ms/控制步 | Spearman vs dt2 | top-10% overlap |
|---|---|---|---|
| dt=2 ms 重跑（**null**）| 140 | 0.953 | 0.56 |
| dt=2.5 ms（20 substeps）| 106 | 0.929 | 0.52 |
| dt=4 ms（12.5 substeps，**不整除 50 ms**）| 64 | 0.860 | 0.16 |
| dt=5 ms（10 substeps）| 56 | 0.874 | 0.40 |

结论：

- dt=2.5 ms 与 run-to-run null 基本重合（物理上无损），但端到端只省 ~3%（sim 在修复后只占 global step 的 ~20%），不值得改默认。
- dt≥4 ms 的 ranking 一致性明显跌破 null → 真实的物理漂移；且 4 ms 不整除 20 Hz 控制周期（substeps 取整后 sim 时钟每步漂 4%）。闭环 smoke 里 dt 造成的差异也淹没在 3-step 方差内，不足以翻案。
- 开环中位数轨迹各 dt 几乎重合，差异集中在 13–21% 的 blow-up 尾部 env（GPU 后端固有，dt 无关，CPU 无此现象——这是 mjwarp vs CPU 的保真差异，靠 `sim_nan_frac`/reward 曲线监控）。
- `env.*.gpu_timestep` 键保留（默认 `null` = 场景默认 0.002），要试更粗 dt 时用它做显式 A/B。

## 3. Placement A/B：**保持 collocated，多 GPU 直接扩 `actor,env,rollout`**

512 envs、gbs 512、dt=2 ms、3 个 global step（`smoke_gpusim_p*.yaml`）：

| 布局 | env_interact | generate_rollouts | actor train | global step |
|---|---|---|---|---|
| 1 GPU collocated | 20.6 s | 48.6 s | 65.8 s | **116 s** |
| 2 GPU collocated | 15.9 s | 34.2 s | 36.5 s | **73 s** |
| 4 GPU collocated | 14.9 s | 27.1 s | 24.0 s | **55 s** |
| 4 GPU，env 集中在 2 个（`actor,rollout: 0-3` + `env: 0-1`）| 14.8 s | 32.6 s | 26.3 s | **63 s** |

结论：mjwarp 吞吐对 batch 亚线性（512→1024 env 仅 158→192 ms/步），所以 env worker 数量本身不是杠杆——1 worker×512 env 和 4 worker×128 env 的 sim 时间只差 ~30%。多 GPU 的收益主要来自 actor 训练和 rollout 推理的并行；把 env 挤到 GPU 子集的异构布局反而因为与 rollout 争抢共享 GPU 而更慢。**最佳实践：保持简单的 `actor,env,rollout: <该实验拥有的全部 GPU>`**，几个实验并行时按老习惯一实验一 GPU 即可。

## 4. 已固化的默认值

- `examples/embodiment/config/env/taco_brush_allegro.yaml`（所有 TACO 实验继承）：
  `gpu_nconmax_per_env: 100`、`gpu_njmax_per_env: 800`、`gpu_solver: newton`、`gpu_solver_iterations: 10`、`gpu_ls_iterations: 20`、`gpu_timestep: null`、`gpu_cuda_graph: True`。
- 8 个 `taco_sharpa20hz_tp4_raft_*.yaml`：`env.train.sim_backend: gpu`（eval 保持 `cpu`，保留视频渲染能力；GPU 后端不支持渲染）。
- 端到端 smoke：`smoke_gpusim_fixed.yaml`（256 env、单 GPU、3 epoch，~4 min）。起大实验前先跑它确认 `graph=OK`、`sim_nan_frac≈0`。

效果（512 env 同 pipeline）：env 交互 **161 s → 15–21 s**，global step **260 s → 55–116 s**（取决于 GPU 数）。

## 5. 遗留事项

- **驱动升级到 ≥550（CUDA 12.4+）**：conditional graph 可用后 solver 真正早退，预计再省 20–30%；升级后 `taco_env_gpu.py` 会自动走回 `graph_conditional=True` 路径，无需改代码。
- GPU 后端 13–21% 的尾部 blow-up env（CPU 无）是 mjwarp 数值行为，与 dt/iterations/njmax 均无关（已逐一排除）；长训练如发现 `sim_nan_frac` 或 reward 分布异常，优先怀疑这里。
- mjwarp 本身非逐位确定（原子加的接触顺序），同配置两次 top-10% overlap 只有 ~0.56：解读 RAFT 筛选结果时把这当作噪声底。
