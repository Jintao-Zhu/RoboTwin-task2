# RLAS 合作者交接总入口

> 文档状态：当前入口
>
> 本项目研究的是：在 200 demos 等较大演示集条件下，如何通过更聪明的训练数据分布，在更少训练代价下实现更快收敛。
>
> - 仓库 canonical 链接：<https://github.com/duanyhui/RoboTwin_distillation>
> - 官方 ACT 安装入口：<https://robotwin-platform.github.io/doc/usage/ACT.html>
> - 当前整理口径：基于仓库中截至 2026-04-04 已存在的代码、脚本、日志与 `summary.csv`

## 1. 这套交接包解决什么

这套 `docs_new` 不是旧 `docs` 的索引，而是一套可以直接交给合作者接手的 handoff 包。  
默认受众是“懂 RoboTwin / ACT，但不熟这版 RLAS”的协作者。

合作者拿到下面两样东西就应该能继续工作：

- `policy/ACT/docs_new/`
- 仓库代码本身，或 GitHub 链接 <https://github.com/duanyhui/RoboTwin_distillation>

## 2. 先看顺序

1. [RLAS_SETUP_AND_REPRO.md](./RLAS_SETUP_AND_REPRO.md)
2. [RLAS_METHOD_AND_CODEBASE.md](./RLAS_METHOD_AND_CODEBASE.md)
3. [RLAS_RESULTS_AND_EVIDENCE.md](./RLAS_RESULTS_AND_EVIDENCE.md)
4. [RLAS_EXPERIMENT_BACKLOG.md](./RLAS_EXPERIMENT_BACKLOG.md)
5. [RLAS_RESEARCH_TRAJECTORY_AND_EXPLORATIONS.md](./RLAS_RESEARCH_TRAJECTORY_AND_EXPLORATIONS.md)
6. [RLAS_DOC_STATUS.md](./RLAS_DOC_STATUS.md)

## 3. 当前统一口径

### 3.1 研究主线

- 默认数据规模固定为 `expert_data_num=200`
- 默认训练配置固定为 `task_config=demo_clean`
- 主线预算固定为 `10000,30000,60000,120000`
- 核心命题不是“换一个方法名”，而是“在多训练样本下更快收敛”

### 3.2 方法命名

| 名称 | 当前定义 | 论文中的角色 |
|---|---|---|
| Baseline | episode-level uniform sampling，episode 内随机 `start_ts` | 主对照 |
| RLAS-Anchor Trick | anchor-level 训练单位 + 静态权重 | RLAS 的静态基座 |
| RLAS | anchor-level + reducible-loss 动态更新权重 | 主方法 |

ATI 不再作为论文主文里的独立方法名。  
如果需要保留 ATI，只作为 `RLAS-Anchor Trick` 的静态实例或附录 ablation。

## 4. 当前任务版图

| 任务 | 当前定位 |
|---|---|
| `beat_block_hammer` | 主结果任务 |
| `open_laptop` | 主结果任务 |
| `stack_blocks_two` | 主结果任务 |
| `stack_blocks_three` | 论文主线预留任务，当前仍需补齐 RLAS 主结果 |
| `adjust_bottle` | sanity check，不进入主结果主线 |

## 5. Source of Truth 代码链路

下面这条链路是当前仓库里最应该被当作真实实现来源的路径：

| 环节 | 文件 | 当前职责 |
|---|---|---|
| Plan 生成入口 | `policy/ACT/scripts/build_sampling_plan.py` | 从 `processed_data` 生成 anchor-level sampling plan |
| Plan 协议 | `policy/ACT/sampling_plan.py` | 定义 `plan_meta.json + plan_arrays.npz` 及 sha256 校验 |
| 数据加载 | `policy/ACT/utils.py` | `load_data_with_rlas_support()` 读取 plan、过滤 train split、创建 sampler |
| RLAS 训练入口 | `policy/ACT/imitate_episodes_rlas.py` | 基于 `target_updates` 做 RLAS 训练 |
| 动态更新 hook | `policy/ACT/rlas/train_integration.py` | 定期触发 scorer、更新 sampler、落快照 |
| Anchor scoring | `policy/ACT/rlas/anchor_scorer.py` | 计算 anchor 的当前 loss |
| Reducible-loss 核心 | `policy/ACT/rlas/rlas_core.py` | `current - baseline`、softmax、epsilon mixing、EMA baseline |

## 6. `docs_new` 文档地图

| 文档 | 作用 |
|---|---|
| [RLAS_SETUP_AND_REPRO.md](./RLAS_SETUP_AND_REPRO.md) | 官方 ACT 安装入口、本项目环境差异、200-demo 复现链路、CLI 与产物规则 |
| [RLAS_METHOD_AND_CODEBASE.md](./RLAS_METHOD_AND_CODEBASE.md) | RLAS 方法口径、代码结构、产物目录、已知边界 |
| [RLAS_RESULTS_AND_EVIDENCE.md](./RLAS_RESULTS_AND_EVIDENCE.md) | 当前已完成实验、代表性结果路径、可信结论、证据缺口 |
| [RLAS_EXPERIMENT_BACKLOG.md](./RLAS_EXPERIMENT_BACKLOG.md) | 接下来要补的实验清单，精确到 task/method/seed/budget/config/script |
| [RLAS_EXPERIMENT_RECORDING_CHECKLIST.md](./RLAS_EXPERIMENT_RECORDING_CHECKLIST.md) | 实验做完后必须记录的结果、路径、失败信息与归档规范 |
| [RLAS_RESEARCH_TRAJECTORY_AND_EXPLORATIONS.md](./RLAS_RESEARCH_TRAJECTORY_AND_EXPLORATIONS.md) | 你已经做过的探索、阶段性结论、哪些路线不必重复试 |
| [RLAS_DOC_STATUS.md](./RLAS_DOC_STATUS.md) | 这套 handoff 包的阅读顺序、硬规则和来源说明 |

## 7. 协作时的硬规则

- 任何文档如果和真实代码、脚本、日志、`summary.csv` 冲突，以仓库中的事实文件为准。
- 官方 ACT 页只负责基础安装与基础 ACT 流程；本项目的 `200 demos + budget sweep + RLAS` 差异项只看 [RLAS_SETUP_AND_REPRO.md](./RLAS_SETUP_AND_REPRO.md)。
- 当前实验结论分为两类：
  - 现成单 run 样例：用于快速定位路径和复现
  - best-over-runs 汇总：用于总结当前上界与论文缺口
- 空 `summary.csv` 一律视为未完成或中断，不视为负结果。
- `stack_blocks_three` 当前只有 baseline 和静态 anchor 证据，还没有进入 RLAS 主表资格。
