# RLAS 当前实验现状与证据读取方式

> 本项目研究的是：在 200 demos 等较大演示集条件下，如何通过更聪明的训练数据分布，在更少训练代价下实现更快收敛。

## 1. 读结果前先锁定三条规则

- 只把非空 `summary.csv` 当有效结果。
- 空 `summary.csv` 记作中断、失败或未完成，不计入性能比较。
- 当前结果要分清两类：
  - 现成单 run 样例：方便定位路径和快速复现
  - best-over-runs：反映当前仓库里已经跑出的最好曲线，但还不能替代多 seed 主表

## 2. 第一次接手时先看的三份样例

| 方法 | 路径 | 作用 |
|---|---|---|
| Baseline | `policy/ACT/logs/open_laptop/sweeps/demo_clean-200-seed0/20260115_105058/summary.csv` | baseline 样例 |
| ATI / 静态 anchor | `policy/ACT/logs/open_laptop/sweeps_plan/demo_clean-200-seed0/anchorS1_noneK4/20260116_220824/summary.csv` | 静态 anchor 样例 |
| RLAS | `policy/ACT/logs/open_laptop/sweeps_rlas/demo_clean-200-seed0/rlas_W200_I5000_T1p0_E0p1/20260202_105628/summary.csv` | 动态 RLAS 样例 |

这三份文件都能直接回溯：

- `ckpt_dir`
- `train_log`
- `eval_log`
- `result_path`
- 各 budget 的 `success_percent`

## 3. 当前主线任务状态

### 3.1 `beat_block_hammer`

- baseline 代表路径：
  `policy/ACT/logs/beat_block_hammer/sweeps/demo_clean-200-seed0/20260129_195310/summary.csv`
- 静态 anchor 代表路径：
  `policy/ACT/logs/beat_block_hammer/sweeps_plan/demo_clean-200-seed0/anchorS1_noneK4/20260116_220811/summary.csv`
- RLAS 代表路径：
  `policy/ACT/logs/beat_block_hammer/sweeps_rlas/demo_clean-200-seed0/rlas_W200_I5000_T1p0_E0p1/20260201_001636/summary.csv`

当前可直接复核的信号：

- baseline 代表 run：`10k=16, 30k=77, 60k=87, 120k=90`
- 静态 anchor none：`10k=84, 30k=75, 60k=93, 120k=86`
- RLAS 代表 run：`10k=76, 30k=89, 60k=95, 120k=97`

当前结论：

- 这是最明显体现“早期快速收敛”的任务之一。
- RLAS 在 10k 和 60k/120k 都有强信号。
- 当前已经具备进入主线的基础证据。

### 3.2 `open_laptop`

- baseline 代表路径：
  `policy/ACT/logs/open_laptop/sweeps/demo_clean-200-seed0/20260115_105058/summary.csv`
- 静态 anchor 代表路径：
  `policy/ACT/logs/open_laptop/sweeps_plan/demo_clean-200-seed0/anchorS1_noneK4/20260116_220824/summary.csv`
- RLAS 代表路径：
  `policy/ACT/logs/open_laptop/sweeps_rlas/demo_clean-200-seed0/rlas_W200_I5000_T1p0_E0p1/20260202_105628/summary.csv`

当前可直接复核的信号：

- baseline 代表 run：`10k=81, 30k=42, 60k=52, 120k=56`
- 静态 anchor none：`10k=75, 30k=85, 60k=88, 120k=91`
- RLAS 代表 run：`10k=69, 30k=76, 60k=66, 120k=86`

当前结论：

- baseline 的 run-to-run 波动明显偏大。
- 静态 anchor 和 RLAS 都显示出收益，但当前更像“高方差 baseline 上的稳定化/提升”。
- 写论文时要明确区分“代表 run”与“best-over-runs”。

### 3.3 `stack_blocks_two`

- baseline 代表路径：
  `policy/ACT/logs/stack_blocks_two/sweeps/demo_clean-200-seed0/20260114_010450/summary.csv`
- 静态 anchor none：
  `policy/ACT/logs/stack_blocks_two/sweeps_plan/demo_clean-200-seed0/anchorS1_noneK4/20260118_221131/summary.csv`
- 静态 anchor stageU：
  `policy/ACT/logs/stack_blocks_two/sweeps_plan/demo_clean-200-seed0/anchorS1_stageU_K4_clip5_eps0p01/20260117_194643/summary.csv`
- RLAS 当前最好中期曲线对应配置：
  `policy/ACT/logs/stack_blocks_two/sweeps_rlas/demo_clean-200-seed0/rlas_W200_I2000_T1p0_E0p1/20260205_214927/summary.csv`

当前可直接复核的信号：

- baseline 代表 run：`10k=11, 30k=42, 60k=48, 120k=89`
- 静态 anchor none：`10k=2, 30k=55, 60k=81, 120k=92`
- 静态 anchor stageU：`10k=4, 30k=42, 60k=80, 120k=89`
- best-over-runs 口径下，RLAS 已跑到：`10k=20, 30k=56, 60k=85, 120k=90`

当前结论：

- 这是“中期 budget 提升最强”的任务。
- 同时它也是当前实验卫生最差的任务，空 `summary.csv` 最多。
- 如果不先清理空 run 与重试记录，后面的论文主表会很脆弱。

### 3.4 `stack_blocks_three`

- baseline 路径：
  `policy/ACT/logs/stack_blocks_three/sweeps/demo_clean-200-seed0/20260115_105033/summary.csv`
- 静态 anchor none：
  `policy/ACT/logs/stack_blocks_three/sweeps_plan/demo_clean-200-seed0/anchorS1_noneK4/20260120_103626/summary.csv`
- 静态 anchor stageU：
  `policy/ACT/logs/stack_blocks_three/sweeps_plan/demo_clean-200-seed0/anchorS1_stageU_K4_clip5_eps0p01/20260120_103551/summary.csv`

当前可直接复核的信号：

- baseline：`10k=0, 30k=30, 60k=20, 120k=54`
- 静态 anchor none：`10k=0, 30k=1, 60k=4, 120k=59`
- 静态 anchor stageU：`10k=0, 30k=4, 60k=18, 120k=39`

当前结论：

- `stack_blocks_three` 已有 baseline 和静态 anchor 资产。
- `anchorS1_noneK4` 在 120k 上高于 baseline，说明静态 anchor 仍有价值。
- 但截至当前仓库状态，`policy/ACT/logs/stack_blocks_three/` 下没有任何 `sweeps_rlas/.../summary.csv`。
- 因此它是“主线预留任务”，不是“已有完整 RLAS 证据的任务”。

### 3.5 `adjust_bottle`

- baseline 路径：
  `policy/ACT/logs/adjust_bottle/sweeps/demo_clean-200-seed0/20260113_005538/summary.csv`

当前定位：

- 只保留为 sanity check。
- 不进入主结果主线。

## 4. 当前 best-over-runs 主结论

下面这张表总结的是当前仓库里已经跑出来的 best-over-runs，不是固定 paper config 的多 seed 结果。

| 任务 | Budget | Baseline-best | RLAS-best | 当前读法 |
|---|---:|---:|---:|---|
| `beat_block_hammer` | 10000 | 16 | 76 | RLAS 早期提升非常强 |
| `beat_block_hammer` | 30000 | 92 | 90 | 中期并非所有配置都胜出 |
| `beat_block_hammer` | 60000 | 92 | 95 | 后期仍有提升 |
| `beat_block_hammer` | 120000 | 93 | 97 | 最终点也有提升 |
| `open_laptop` | 10000 | 81 | 85 | 早期小幅提升 |
| `open_laptop` | 30000 | 50 | 79 | 对高方差 baseline 有明显优势 |
| `open_laptop` | 60000 | 52 | 79 | 中期优势较稳定 |
| `open_laptop` | 120000 | 73 | 86 | 末期仍优于 baseline-best |
| `stack_blocks_two` | 10000 | 11 | 20 | 早期提升有限但存在 |
| `stack_blocks_two` | 30000 | 42 | 56 | 中期开始明显拉开 |
| `stack_blocks_two` | 60000 | 48 | 85 | 当前最强中期收益点 |
| `stack_blocks_two` | 120000 | 89 | 90 | 最终点只略高于 baseline |

## 5. 当前 RLAS 资产完整性

按当前仓库盘点，三任务 RLAS 目录里共有：

- `28` 个 `summary.csv` 目录
- `12` 个非空 `summary.csv`
- `16` 个空 `summary.csv`

其中最需要注意的是：

- `open_laptop` 有少量空 run
- `stack_blocks_two` 空 run 最多
- `beat_block_hammer` 相对最干净

## 6. 当前论文层面的可信说法

可以说：

- 在 200-demo 设定下，训练分布是影响收敛速度的重要杠杆。
- 静态 anchor 已经能在多个任务上改变收敛曲线。
- RLAS 在 `beat_block_hammer`、`open_laptop`、`stack_blocks_two` 上都已经出现正向信号。
- `stack_blocks_two` 的中期收益尤其强。

还不能说：

- 四任务主线都已经被 RLAS 完整覆盖。
- 当前 best-over-runs 结果已经足够替代固定 config、多 seed 的主表。
- 空 `summary.csv` 可以被当成“方法失败”的证据。

## 7. 合作者接手时应该优先做什么

如果合作者只做三件事，优先顺序应该是：

1. 打开 `open_laptop` 的 baseline / ATI / RLAS 三份样例 `summary.csv`
2. 看 [RLAS_EXPERIMENT_BACKLOG.md](./RLAS_EXPERIMENT_BACKLOG.md) 里的 P0 和 P0.5
3. 明确 `stack_blocks_three` 现在还缺 RLAS 资产，不要误以为它已经完成
