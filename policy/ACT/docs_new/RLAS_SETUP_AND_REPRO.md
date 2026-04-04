# RLAS 环境、安装入口与最短复现链路

> 本项目研究的是：在 200 demos 等较大演示集条件下，如何通过更聪明的训练数据分布，在更少训练代价下实现更快收敛。

## 1. 官方 ACT 安装入口

本项目不再维护一套平行的 ACT 安装教程。  
环境搭建与基础 ACT 流程统一以 RoboTwin 官方页面为准：

- <https://robotwin-platform.github.io/doc/usage/ACT.html>

截至 2026-04-04，该页面按下面四部分组织：

1. `Install`
2. `Prepare Training Data`
3. `Train Policy`
4. `Eval Policy`

因此，本项目在 `docs_new` 里只补“相对官方 ACT 的差异项”。

## 2. 本仓库相对官方 ACT 的差异项

### 2.1 研究设定差异

官方 ACT 页的示例是 50 条数据，例如：

```bash
bash process_data.sh beat_block_hammer demo_clean 50
```

本项目默认切换为：

- `task_config=demo_clean`
- `expert_data_num=200`
- 主线预算 `10000,30000,60000,120000`

这不是随手改动，而是当前论文命题的一部分。  
我们的核心问题是：在 200 demos 这类较大演示集下，能否用更聪明的采样策略实现更快收敛。

### 2.2 训练流程差异

官方 ACT 页更偏向单次 `train.sh` / `eval.sh`。  
本项目的主线工作流改成：

1. 数据处理
2. 一次训练到最大 budget
3. 复用中间 checkpoint 做多 budget 评测
4. 用 `summary.csv` 汇总结果

也就是：

- Baseline 用 `scripts/baseline_budget_sweep.sh`
- 静态 anchor 用 `scripts/plan_budget_sweep.sh`
- RLAS 用 `scripts/rlas_budget_sweep.sh`

### 2.3 环境说明的写法

- `policy/ACT/conda_env.yaml` 可以视作仓库内依赖快照。
- 历史日志中出现过 `RoboTwin_ACT` 等环境名，但环境名本身不是 source of truth。
- 对外协作时，只要求依赖等价，不要求 conda 环境名称一致。

## 3. 当前稳定暴露的 CLI 接口

| 入口 | 作用 | 关键输入 | 实际关键输出 |
|---|---|---|---|
| `process_data.sh` | 处理原始 demo | `task_name task_config expert_data_num` | `processed_data/sim-<task>/<task_config>-<expert_data_num>/` |
| `train.sh` | baseline 单次训练 | `task_name task_config demos seed gpu [target_updates]` | `act_ckpt/act-<task>/...` |
| `eval.sh` | 单 ckpt 评测 | `task_name task_config ckpt_setting demos seed gpu` | `eval_result/...` |
| `scripts/baseline_budget_sweep.sh` | baseline 一次训练、多 budget 评测 | `task task_config demos seed gpu "budgets"` | `logs/<task>/sweeps/.../summary.csv` |
| `scripts/plan_budget_sweep.sh` | 静态 anchor / ATI 一次训练、多 budget 评测 | 同上 + `PLAN_*` | `logs/<task>/sweeps_plan/.../summary.csv` |
| `scripts/rlas_budget_sweep.sh` | RLAS 一次训练、多 budget 评测 | 同上 + `RLAS_*` | `logs/<task>/sweeps_rlas/.../summary.csv` |
| `scripts/run_rlas_experiments.sh` | 顺序跑 Baseline + ATI + RLAS | `task gpu "seeds"` | 三类 sweep 目录 |

补充说明：

- `run_rlas_experiments.sh` 的注释和打印里写了 `sweeps_baseline`，但 baseline 的真实汇总目录仍然是 `logs/<task>/sweeps/...`。协作时以脚本实际行为为准。
- `summary.csv` 是协作时最重要的结果入口，因为它已经记录了 ckpt、train log、eval log 和 result path。

## 4. 最短复现链路

以下命令按“同代码、不同环境”的交接场景组织，优先选 `open_laptop / demo_clean / 200 demos`。

### 4.1 预处理数据

如果本地已经有 `policy/ACT/processed_data/sim-open_laptop/demo_clean-200/`，这一步可以跳过。

```bash
cd /home/dyh/docker/dyh_RoboTwin/data/code/RoboTwin_distillation/policy/ACT
bash process_data.sh open_laptop demo_clean 200
```

### 4.2 Baseline 最短复现

```bash
bash scripts/baseline_budget_sweep.sh \
  open_laptop demo_clean 200 0 0 "10000 30000 60000 120000"
```

### 4.3 RLAS-Anchor Trick / ATI 最短复现

```bash
PLAN_TAG=anchorS1_noneK4 \
PLAN_STAGE_BALANCE=none \
bash scripts/plan_budget_sweep.sh \
  open_laptop demo_clean 200 0 0 "10000 30000 60000 120000"
```

### 4.4 RLAS 最短复现

```bash
RLAS_WARMUP=200 \
RLAS_INTERVAL=5000 \
RLAS_TEMP=1.0 \
RLAS_EPS=0.1 \
RLAS_BETA=0.5 \
RLAS_ALPHA=1.0 \
bash scripts/rlas_budget_sweep.sh \
  open_laptop demo_clean 200 0 0 "10000 30000 60000 120000"
```

## 5. 命名与产物规则

### 5.1 核心路径

- 处理后数据：`policy/ACT/processed_data/sim-<task>/<task_config>-<demos>/`
- plan：`policy/ACT/sampling_plans/<task>/<task_config>-<demos>/<plan_tag>/`
- ckpt：`policy/ACT/act_ckpt/act-<task>/...`
- logs：`policy/ACT/logs/<task>/...`
- eval：`eval_result/<task>/ACT/...`

### 5.2 命名约定

- `sampling_plan_path` 指向某个 plan 目录，内部至少有：
  - `plan_meta.json`
  - `plan_arrays.npz`
- RLAS plan 常见命名是：
  - `rlas_W200_I5000_T1p0_E0p1`
  - `rlas_W5000_I5000_T0p5_E0p1`
- 静态 anchor 常见命名是：
  - `anchorS1_noneK4`
  - `anchorS1_stageU_K4_clip5_eps0p01`
- `target_updates` 决定这次训练的最大 budget。

### 5.3 `summary.csv` 字段

| 方法 | 关键字段 |
|---|---|
| Baseline | `budget, ckpt_setting, ckpt_dir, train_log, eval_log, success_fraction, success_percent, result_path, train_seconds, eval_seconds` |
| Plan / ATI | Baseline 字段 + `plan_dir, plan_sha256` |
| RLAS | Baseline 字段 + `plan_dir, rlas_warmup, rlas_interval, rlas_temp, rlas_eps` |

## 6. 现成样例路径

下面三份 `summary.csv` 是最适合第一次接手时直接打开的样例：

- Baseline：
  `policy/ACT/logs/open_laptop/sweeps/demo_clean-200-seed0/20260115_105058/summary.csv`
- ATI / 静态 anchor：
  `policy/ACT/logs/open_laptop/sweeps_plan/demo_clean-200-seed0/anchorS1_noneK4/20260116_220824/summary.csv`
- RLAS：
  `policy/ACT/logs/open_laptop/sweeps_rlas/demo_clean-200-seed0/rlas_W200_I5000_T1p0_E0p1/20260202_105628/summary.csv`

这三份文件已经足够让协作者定位：

- ckpt 目录
- train log
- eval log
- 对应 budget 的成功率
- 最终结果文件路径

实验结束后要进一步整理哪些路径、指标和失败信息，统一按 [RLAS_EXPERIMENT_RECORDING_CHECKLIST.md](./RLAS_EXPERIMENT_RECORDING_CHECKLIST.md) 执行。

## 7. 复现时最容易踩的坑

- `budget` 必须能被 `SAVE_EVERY_UPDATES` 整除，否则中间 checkpoint 不完整。
- 空 `summary.csv` 不是负结果，而是中断或失败记录。
- `RLAS` 训练必须带 `--sampling_plan_path`，所以 RLAS 工作流默认先生成 plan。
- `expert_data_num` 在本项目主线里固定看作 `200`，不要用官方 ACT 页的 50-demo 示例直接替代。
