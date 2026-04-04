# RLAS 接下来要补的实验清单

> 本项目研究的是：在 200 demos 等较大演示集条件下，如何通过更聪明的训练数据分布，在更少训练代价下实现更快收敛。

## 1. backlog 的使用规则

- 这份文档不是 brainstorming，而是可执行任务面板。
- 主表口径必须优先采用“固定 paper config + 多 seed”，而不是“每个 budget 各选一组最优参数”。
- 所有任务默认使用：
  - `task_config=demo_clean`
  - `expert_data_num=200`
  - `budgets=10000,30000,60000,120000`

## 2. 固定的 paper config 默认值

| 任务 | paper config | 当前理由 |
|---|---|---|
| `beat_block_hammer` | `W200 I5000 T1.0 E0.1` | 当前信号最稳定，早期和末期都强 |
| `open_laptop` | `W200 I5000 T1.0 E0.1` | 末期较稳，便于多 seed 固定口径 |
| `stack_blocks_two` | `W200 I2000 T1.0 E0.1` | 当前 60k 提升最明显 |
| `stack_blocks_three` | 待选 | 先做 seed0 扫描再固定 |

## 3. 验收标准

每条实验至少满足下面条件才算完成：

- 生成非空 `summary.csv`
- `summary.csv` 中包含四个主线 budget
- 每行都能解析出 `success_percent`
- RLAS run 额外要求能看到：
  - `rlas_snapshots/`
  - `rlas_viz/`
- 空 `summary.csv` 只记入失败记录，不写进主表

## 4. P0：前三个主线任务的多 seed 主结果

| Priority | Task | Method | Seed | Budget | Config | Script | Expected artifact |
|---|---|---|---|---|---|---|---|
| P0 | `beat_block_hammer` | Baseline | `0,1,2` | `10000,30000,60000,120000` | baseline 默认 | `scripts/baseline_budget_sweep.sh` | `logs/beat_block_hammer/sweeps/.../summary.csv` |
| P0 | `beat_block_hammer` | RLAS | `0,1,2` | `10000,30000,60000,120000` | `W200 I5000 T1.0 E0.1` | `scripts/rlas_budget_sweep.sh` | `logs/beat_block_hammer/sweeps_rlas/.../summary.csv` |
| P0 | `open_laptop` | Baseline | `0,1,2` | `10000,30000,60000,120000` | baseline 默认 | `scripts/baseline_budget_sweep.sh` | `logs/open_laptop/sweeps/.../summary.csv` |
| P0 | `open_laptop` | RLAS | `0,1,2` | `10000,30000,60000,120000` | `W200 I5000 T1.0 E0.1` | `scripts/rlas_budget_sweep.sh` | `logs/open_laptop/sweeps_rlas/.../summary.csv` |
| P0 | `stack_blocks_two` | Baseline | `0,1,2` | `10000,30000,60000,120000` | baseline 默认 | `scripts/baseline_budget_sweep.sh` | `logs/stack_blocks_two/sweeps/.../summary.csv` |
| P0 | `stack_blocks_two` | RLAS | `0,1,2` | `10000,30000,60000,120000` | `W200 I2000 T1.0 E0.1` | `scripts/rlas_budget_sweep.sh` | `logs/stack_blocks_two/sweeps_rlas/.../summary.csv` |

P0 完成后的主表读法：

- 先比较固定 paper config 的多 seed 平均与方差
- 不再继续用 best-over-runs 充当最终主表
- 每次实验结束后，按 [RLAS_EXPERIMENT_RECORDING_CHECKLIST.md](./RLAS_EXPERIMENT_RECORDING_CHECKLIST.md) 归档结果

## 5. P0.5：让 `stack_blocks_three` 进入主线

### 5.1 先做 seed0 RLAS 入口扫描

| Priority | Task | Method | Seed | Budget | Config | Script | Expected artifact |
|---|---|---|---|---|---|---|---|
| P0.5 | `stack_blocks_three` | RLAS | `0` | `10000,30000,60000,120000` | `W200 I5000 T1.0 E0.1` | `scripts/rlas_budget_sweep.sh` | 非空 `summary.csv` + `rlas_viz/` + `rlas_snapshots/` |
| P0.5 | `stack_blocks_three` | RLAS | `0` | `10000,30000,60000,120000` | `W5000 I5000 T1.0 E0.1` | `scripts/rlas_budget_sweep.sh` | 同上 |
| P0.5 | `stack_blocks_three` | RLAS | `0` | `10000,30000,60000,120000` | `W5000 I5000 T0.5 E0.1` | `scripts/rlas_budget_sweep.sh` | 同上 |
| P0.5 | `stack_blocks_three` | RLAS | `0` | `10000,30000,60000,120000` | `W5000 I10000 T1.0 E0.1` | `scripts/rlas_budget_sweep.sh` | 同上 |
| P0.5 | `stack_blocks_three` | RLAS | `0` | `10000,30000,60000,120000` | `W200 I2000 T1.0 E0.1` | `scripts/rlas_budget_sweep.sh` | 同上 |

### 5.2 固定 `stack_blocks_three` 的 paper config

按下面规则选，不允许临场重定口径：

1. 优先选择 `120000` 成功率最高的 config
2. 若并列，选 `60000` 更高的 config
3. 若仍并列，优先 `W200`
4. 若仍并列，优先 `I5000`

### 5.3 再补 seed1/2

| Priority | Task | Method | Seed | Budget | Config | Script | Expected artifact |
|---|---|---|---|---|---|---|---|
| P0.5 | `stack_blocks_three` | Baseline | `1,2` | `10000,30000,60000,120000` | baseline 默认 | `scripts/baseline_budget_sweep.sh` | `logs/stack_blocks_three/sweeps/.../summary.csv` |
| P0.5 | `stack_blocks_three` | RLAS | `1,2` | `10000,30000,60000,120000` | 按上面规则选出的单一 config | `scripts/rlas_budget_sweep.sh` | `logs/stack_blocks_three/sweeps_rlas/.../summary.csv` |

## 6. P1：参数规律与机制证据

P1 只在 seed0 上做，目标是解释为什么有效，不是补主表。

| Priority | Task | Seed | Sweep | Fixed defaults | Script | Expected artifact |
|---|---|---|---|---|---|---|
| P1 | `beat_block_hammer, open_laptop, stack_blocks_two` | `0` | `W in {200,1000,5000,10000}` | `I=task default, T=1.0, E=0.1, beta=0.5, alpha=1.0` | `scripts/rlas_budget_sweep.sh` | 非空 `summary.csv` + `rlas_viz/rlas_summary.json` |
| P1 | 同上 | `0` | `I in {2000,5000,10000}` | `W=task default, T=1.0, E=0.1, beta=0.5, alpha=1.0` | `scripts/rlas_budget_sweep.sh` | 同上 |
| P1 | 同上 | `0` | `T in {0.5,1.0,2.0}` | `W/I=task default, E=0.1` | `scripts/rlas_budget_sweep.sh` | 同上 |
| P1 | 同上 | `0` | `E in {0.01,0.05,0.1,0.2}` | `W/I/T=task default` | `scripts/rlas_budget_sweep.sh` | 同上 |

P1 输出要回答的问题：

- warmup 大小是否影响早期收益
- interval 大小是否影响稳定性
- 温度和 epsilon 是否只是在“更尖锐”和“更均匀”之间做平衡
- 哪些任务的动态重排强，哪些任务其实更像静态 gain

## 7. P2：效率账本与论文补强

| Priority | 目标 | 内容 | 预期产物 |
|---|---|---|---|
| P2 | 训练效率账本 | 记录每类方法的训练时间、评测时间、空 run 比例 | 单独表格或附录 |
| P2 | 机制证据 | 从 `rlas_snapshots/` 和 `rlas_viz/` 中抽取权重熵、正 reducible 比例、权重分布演化 | 附录图 |
| P2 | 稳健性 | 在固定 paper config 下做更多 seed 或有限扩展任务 | 主文补充或附录 |

## 8. 本轮不做什么

- 不再把 ATI 当成论文主文里的独立方法名重新讲一遍。
- 不在主表里继续混用“每个 budget 各挑不同最优配置”的 best-over-runs 口径。
- 不把空 `summary.csv` 直接计入性能统计。
