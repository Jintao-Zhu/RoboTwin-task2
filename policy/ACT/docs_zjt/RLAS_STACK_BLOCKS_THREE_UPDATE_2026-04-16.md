# RLAS `stack_blocks_three` 专题更新（2026-04-16）

> 本文档用于把 `stack_blocks_three` 的既有证据与 2026-04-16 新完成的 RLAS 实验结果合并整理，口径与 `docs/RLAS_RESULTS_AND_EVIDENCE.md` 保持一致。

## 1. 读结果口径（与主文档一致）

- 只把非空 `summary.csv` 当有效结果。
- 空 `summary.csv` 记作中断/失败/未完成，不计入性能比较。
- 当前记录仍以“代表 run 证据”为主，不替代多 seed 主表。

## 2. `stack_blocks_three` 既有资产（实验前状态）

- baseline 路径：
  `policy/ACT/logs/stack_blocks_three/sweeps/demo_clean-200-seed0/20260115_105033/summary.csv`
- 静态 anchor none 路径：
  `policy/ACT/logs/stack_blocks_three/sweeps_plan/demo_clean-200-seed0/anchorS1_noneK4/20260120_103626/summary.csv`
- 静态 anchor stageU 路径：
  `policy/ACT/logs/stack_blocks_three/sweeps_plan/demo_clean-200-seed0/anchorS1_stageU_K4_clip5_eps0p01/20260120_103551/summary.csv`

既有可复核信号：

- baseline：`10k=0, 30k=30, 60k=20, 120k=54`
- 静态 anchor none：`10k=0, 30k=1, 60k=4, 120k=59`
- 静态 anchor stageU：`10k=0, 30k=4, 60k=18, 120k=39`

## 3. 本次新增 RLAS 实验（已完成）

### 3.1 结果入口

- 有效 `summary.csv`：
  `policy/ACT/logs/stack_blocks_three/sweeps_rlas/demo_clean-200-seed0/rlas_W200_I5000_T1p0_E0p1/20260414_194553/summary.csv`
- 空 `summary.csv`（不纳入比较）：
  `policy/ACT/logs/stack_blocks_three/sweeps_rlas/demo_clean-200-seed0/rlas_W200_I5000_T1p0_E0p1/20260414_194524/summary.csv`

### 3.2 本次 RLAS 代表 run 信号

- RLAS（`W200_I5000_T1p0_E0p1`）：`10k=0, 30k=23, 60k=73, 120k=80`

可回溯路径（来自 `summary.csv`）：

- `ckpt_dir`：
  `policy/ACT/act_ckpt/act-stack_blocks_three/demo_clean-u120000-rlas_rlas_W200_I5000_T1p0_E0p1-200`
- `train_log`：
  `policy/ACT/logs/stack_blocks_three/demo_clean-u120000-rlas_rlas_W200_I5000_T1p0_E0p1-200/train_20260414_194553.log`
- `eval_log`（按 budget）：
  - `policy/ACT/logs/stack_blocks_three/demo_clean-u10000-rlas_rlas_W200_I5000_T1p0_E0p1-200/eval_20260414_194553.log`
  - `policy/ACT/logs/stack_blocks_three/demo_clean-u30000-rlas_rlas_W200_I5000_T1p0_E0p1-200/eval_20260414_194553.log`
  - `policy/ACT/logs/stack_blocks_three/demo_clean-u60000-rlas_rlas_W200_I5000_T1p0_E0p1-200/eval_20260414_194553.log`
  - `policy/ACT/logs/stack_blocks_three/demo_clean-u120000-rlas_rlas_W200_I5000_T1p0_E0p1-200/eval_20260414_194553.log`
- `result_path`（`_result.txt`）：
  - `eval_result/stack_blocks_three/ACT/demo_clean/demo_clean-u10000-rlas_rlas_W200_I5000_T1p0_E0p1/2026-04-15 16:51:42/_result.txt`
  - `eval_result/stack_blocks_three/ACT/demo_clean/demo_clean-u30000-rlas_rlas_W200_I5000_T1p0_E0p1/2026-04-15 22:17:20/_result.txt`
  - `eval_result/stack_blocks_three/ACT/demo_clean/demo_clean-u60000-rlas_rlas_W200_I5000_T1p0_E0p1/2026-04-16 04:18:20/_result.txt`
  - `eval_result/stack_blocks_three/ACT/demo_clean/demo_clean-u120000-rlas_rlas_W200_I5000_T1p0_E0p1/2026-04-16 07:35:59/_result.txt`

## 4. 同口径对比（既有 vs 本次 RLAS）

| 方法 | 10k | 30k | 60k | 120k |
|---|---:|---:|---:|---:|
| Baseline（既有） | 0 | 30 | 20 | 54 |
| 静态 Anchor none（既有） | 0 | 1 | 4 | 59 |
| 静态 Anchor stageU（既有） | 0 | 4 | 18 | 39 |
| RLAS `W200_I5000_T1p0_E0p1`（本次） | 0 | 23 | 73 | 80 |

相对 baseline 的增益（RLAS - Baseline）：

- `10k: +0`
- `30k: -7`
- `60k: +53`
- `120k: +26`

## 5. 机制侧可复核信号（本次 run）

- 训练日志显示 RLAS sampler 正常启用，warmup 后进入动态更新。
- `RLAS updates` 在训练末达到 `24`。
- `rlas_snapshots/weight_update_history.json` 已落盘，共 `23` 次权重统计（`update 5200` 到 `115200`），说明不是“挂名启用”。

对应路径：

- `policy/ACT/logs/stack_blocks_three/demo_clean-u120000-rlas_rlas_W200_I5000_T1p0_E0p1-200/train_20260414_194553.log`
- `policy/ACT/act_ckpt/act-stack_blocks_three/demo_clean-u120000-rlas_rlas_W200_I5000_T1p0_E0p1-200/rlas_snapshots/weight_update_history.json`
- `policy/ACT/act_ckpt/act-stack_blocks_three/demo_clean-u120000-rlas_rlas_W200_I5000_T1p0_E0p1-200/rlas_viz/`

## 6. 当前阶段结论（`stack_blocks_three`）

- 该任务已不再是“缺 RLAS 资产”的预留状态；现在已有可复核 RLAS `summary.csv` 与完整训练/评测回溯链路。
- 本次信号最强的是中后期：`60k` 与 `120k` 显著高于既有 baseline 与静态 anchor。
- 早期 `30k` 仍低于 baseline，说明该任务的“早期快速收敛”特征还不稳定，更像“中后期追赶并反超”。

## 7. 建议写回主文档时的替换口径

- 将“`stack_blocks_three` 没有 `sweeps_rlas/.../summary.csv`”改为“已补齐一条有效 RLAS 证据 run（20260414_194553）”。
- 将该任务结论更新为：
  - “RLAS 在 `stack_blocks_three` 上已出现明确中后期收益（60k/120k）”；
  - “早期收益仍需多 seed 验证”。
- 在主结论里把“RLAS 覆盖三任务”更新为“已覆盖四任务，但 `stack_blocks_three` 当前仍以单 seed 证据为主”。
