# RLAS 方法口径与代码结构

> 本项目研究的是：在 200 demos 等较大演示集条件下，如何通过更聪明的训练数据分布，在更少训练代价下实现更快收敛。

## 1. RLAS 到底在解决什么

在 200 demos 这种数据规模下，baseline 的 episode-uniform 训练会浪费很多更新步数：

- 训练单位还是 episode-level
- 每次只是在 episode 内随机选 `start_ts`
- 大量片段重复、低价值或对当前阶段帮助很小

因此，RLAS 的目标不是“再发明一个训练框架”，而是把训练分布从粗粒度 episode 采样，改成可控的 anchor-level 采样，并在训练过程中动态重排这些 anchor 的重要性。

## 2. 当前统一方法口径

| 名称 | 训练单位 | 权重 | 当前用途 |
|---|---|---|---|
| Baseline | episode-level | 隐式均匀抽 episode，episode 内随机 `start_ts` | 主对照 |
| RLAS-Anchor Trick | anchor-level | 静态权重 | RLAS 静态基座 / 附录 ablation |
| RLAS | anchor-level | reducible-loss 动态权重 | 主方法 |

这里的 ATI、`anchorS1_noneK4`、`anchorS1_stageU_K4_clip5_eps0p01` 都被视为 `RLAS-Anchor Trick` 的静态实例。

## 3. 为什么这条路线适合“多训练样本下的快速收敛”

逻辑很简单：

1. 数据量扩大到 200 demos 后，训练可选片段明显更多。
2. 一旦把训练单位改成 anchor-level，就能枚举并索引片段。
3. 有了 anchor-level 索引后，才可能做：
   - 静态 plan
   - 动态 loss-aware weighting
   - 多 budget 可复现比较
4. 因此，“更快收敛”的真正杠杆是训练分布，而不是换模型主干。

## 4. 当前真实实现链路

| 顺序 | 文件 | 作用 |
|---|---|---|
| 1 | `policy/ACT/scripts/build_sampling_plan.py` | 枚举 anchors，生成离线 sampling plan |
| 2 | `policy/ACT/sampling_plan.py` | 定义 plan 协议、校验和读取逻辑 |
| 3 | `policy/ACT/utils.py` | `load_data_with_rlas_support()` 读取 plan 并创建 sampler |
| 4 | `policy/ACT/imitate_episodes_rlas.py` | RLAS 训练入口，按 `target_updates` 保存中间 ckpt |
| 5 | `policy/ACT/rlas/train_integration.py` | 定期触发 anchor scoring 和权重更新 |
| 6 | `policy/ACT/rlas/anchor_scorer.py` | 计算所有 anchor 的当前 loss |
| 7 | `policy/ACT/rlas/rlas_core.py` | reducible-loss、softmax、epsilon mixing、EMA、快照 |

## 5. 关键运行时产物

| 目录 | 产物 |
|---|---|
| `policy/ACT/processed_data/...` | 训练数据 |
| `policy/ACT/sampling_plans/...` | `plan_meta.json`、`plan_arrays.npz` |
| `policy/ACT/act_ckpt/...` | 模型 checkpoint、`dataset_stats.pkl` |
| `policy/ACT/logs/...` | train log、eval log、sweep `summary.csv` |
| `eval_result/...` | 闭环评测结果与视频 |
| `policy/ACT/act_ckpt/.../rlas_snapshots/` | 权重快照、baseline losses |
| `policy/ACT/act_ckpt/.../rlas_viz/` | RLAS 可视化图和统计 |

## 6. 当前代码边界

- RLAS 当前只接在 `ACT` 训练路线上。
- `imitate_episodes_rlas.py` 依赖 `sampling_plan_path` 才能启用 RLAS。
- `load_data_with_rlas_support()` 在读取 plan 后，会把训练样本从 episode-level 切成 anchor-level。
- `RLAS_WARMUP` 的作用不是装饰项，而是避免模型在过早阶段用噪声 loss 做动态重排。
- `rlas_core.py` 里动态更新的本质是：
  - `current_loss - baseline_loss`
  - 只强调正的 reducible loss
  - softmax 转为权重
  - 再和均匀分布做 epsilon mixing

## 7. 当前已知限制

- 动态权重的有效幅度经常不算很大，因此不是所有任务都出现极强重排。
- `stack_blocks_two` 的 RLAS 目录里有大量空 `summary.csv`，说明实验卫生还要补。
- `stack_blocks_three` 目前没有 `sweeps_rlas/.../summary.csv`，所以还不能说“RLAS 已在四任务主线里闭环”。
- `run_rlas_experiments.sh` 更像便捷脚本，不是 source-of-truth；真实行为仍应回到三个 sweep 脚本核对。

## 8. 合作者需要掌握的稳定词汇

| 词汇 | 含义 |
|---|---|
| `task_name` | 任务名，如 `open_laptop` |
| `task_config` | 训练/评测配置，本项目主线固定 `demo_clean` |
| `expert_data_num=200` | 当前主线数据规模 |
| `sampling_plan_path` | 某个 sampling plan 目录 |
| `target_updates` | 本次训练的最大优化步数 |
| `PLAN_*` | 静态 anchor / sampling plan 环境变量 |
| `RLAS_*` | 动态权重更新环境变量 |
| `summary.csv` | 单次 sweep 的总入口 |
| `rlas_W*_I*_T*_E*` | RLAS 配置命名 |

## 9. 给合作者的最短理解

如果只记一句话，那就是：

> 这版 RLAS 的核心不是“新模型”，而是“把 ACT 的训练分布从 episode-uniform 改成可索引、可静态控制、可动态更新的 anchor-level 分布”，并用它去解决 200-demo 场景下的快速收敛问题。
