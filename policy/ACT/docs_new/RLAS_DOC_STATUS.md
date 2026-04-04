# RLAS 文档状态索引

> 文档状态：当前入口索引
>
> 本项目研究的是：在 200 demos 等较大演示集条件下，如何通过更聪明的训练数据分布，在更少训练代价下实现更快收敛。

## 1. 推荐阅读顺序

1. [RLAS_COLLABORATOR_HANDOFF.md](./RLAS_COLLABORATOR_HANDOFF.md)
2. [RLAS_SETUP_AND_REPRO.md](./RLAS_SETUP_AND_REPRO.md)
3. [RLAS_METHOD_AND_CODEBASE.md](./RLAS_METHOD_AND_CODEBASE.md)
4. [RLAS_RESULTS_AND_EVIDENCE.md](./RLAS_RESULTS_AND_EVIDENCE.md)
5. [RLAS_EXPERIMENT_BACKLOG.md](./RLAS_EXPERIMENT_BACKLOG.md)
6. [RLAS_EXPERIMENT_RECORDING_CHECKLIST.md](./RLAS_EXPERIMENT_RECORDING_CHECKLIST.md)
7. [RLAS_RESEARCH_TRAJECTORY_AND_EXPLORATIONS.md](./RLAS_RESEARCH_TRAJECTORY_AND_EXPLORATIONS.md)

## 2. `docs_new` 文档角色

| 文档 | 角色 |
|---|---|
| [RLAS_COLLABORATOR_HANDOFF.md](./RLAS_COLLABORATOR_HANDOFF.md) | 合作者接手总入口 |
| [RLAS_SETUP_AND_REPRO.md](./RLAS_SETUP_AND_REPRO.md) | 官方安装入口、本项目差异项、最短复现链路 |
| [RLAS_METHOD_AND_CODEBASE.md](./RLAS_METHOD_AND_CODEBASE.md) | 方法口径、source-of-truth 代码链路、运行时边界 |
| [RLAS_RESULTS_AND_EVIDENCE.md](./RLAS_RESULTS_AND_EVIDENCE.md) | 当前已完成实验与可信结论 |
| [RLAS_EXPERIMENT_BACKLOG.md](./RLAS_EXPERIMENT_BACKLOG.md) | 接下来要补的实验任务面板 |
| [RLAS_EXPERIMENT_RECORDING_CHECKLIST.md](./RLAS_EXPERIMENT_RECORDING_CHECKLIST.md) | 每次实验结束后要保留哪些路径、指标、失败记录和论文归档信息 |
| [RLAS_RESEARCH_TRAJECTORY_AND_EXPLORATIONS.md](./RLAS_RESEARCH_TRAJECTORY_AND_EXPLORATIONS.md) | 研究历程、已做探索、哪些路线不必重复争论 |

## 3. 读文档时的硬规则

- `docs_new` 是 handoff 主链路；旧 `policy/ACT/docs/` 不是默认阅读入口。
- 官方安装只看 RoboTwin ACT 页面：<https://robotwin-platform.github.io/doc/usage/ACT.html>
- 任何数字、路径或结论如果和仓库里的真实 `summary.csv`、脚本、日志冲突，以仓库事实为准。
- 当前论文主线固定为四个任务：
  - `beat_block_hammer`
  - `open_laptop`
  - `stack_blocks_two`
  - `stack_blocks_three`
- `adjust_bottle` 只保留为 sanity check。
- ATI 在这里不作为论文主文的独立方法名，而是 `RLAS-Anchor Trick` 的静态实例。

## 4. 为什么旧 `docs` 还保留

旧目录 `policy/ACT/docs/` 仍然保留，因为它记录了：

- 早期 200-demo 现象发现
- 静态 anchor 的工程落地过程
- 动态采样文献调研
- 理论包装和审稿人向写法
- 参数与性能调试过程

但这些材料已经被浓缩进 `docs_new`。  
协作者不需要把旧 `docs` 当成主阅读链路。

## 5. Provenance

下面这张表只说明 `docs_new` 的来源，不是“继续阅读链接”。

| 新文档 | 主要浓缩来源 |
|---|---|
| `RLAS_COLLABORATOR_HANDOFF.md` | `policy/ACT/docs/RLAS_README.md`、`policy/ACT/docs/RLAS_RESEARCH_TRAJECTORY.md`、现有日志路径 |
| `RLAS_SETUP_AND_REPRO.md` | `policy/ACT/docs/RLAS_README.md`、`policy/ACT/docs/RLAS_PARAMETERS.md`、`policy/ACT/docs/RLAS_PERFORMANCE.md`、官方 ACT 页 |
| `RLAS_METHOD_AND_CODEBASE.md` | `policy/ACT/docs/RLAS_METHOD_DEEP_DIVE_AND_PLAIN_EXPLANATION.md`、`policy/ACT/docs/rlas_implementation_details.md`、真实代码链路 |
| `RLAS_RESULTS_AND_EVIDENCE.md` | `policy/ACT/docs/figures/rlas_aggregated_stats.json`、真实 `summary.csv`、`policy/ACT/docs/ACT_200demos_fast_convergence_status.md` |
| `RLAS_EXPERIMENT_BACKLOG.md` | `policy/ACT/docs/RLAS_PAPER_PLAN_AND_NEXT_EXPERIMENTS.md`、`policy/ACT/docs/RLAS_NEXT_EXPERIMENTS_EXECUTION_GUIDE.md` |
| `RLAS_EXPERIMENT_RECORDING_CHECKLIST.md` | 当前 sweep 脚本的真实输出字段、`summary.csv` 结构、`rlas_viz/` 与 `rlas_snapshots/` 的实际产物 |
| `RLAS_RESEARCH_TRAJECTORY_AND_EXPLORATIONS.md` | `policy/ACT/docs/ACT_200demos_fast_convergence_status.md`、`policy/ACT/docs/ACT_what_made_60k_better.md`、`policy/ACT/docs/ACT_dynamic_sampling_literature_and_next_steps.md`、`policy/ACT/docs/survey/` 下理论文档 |
