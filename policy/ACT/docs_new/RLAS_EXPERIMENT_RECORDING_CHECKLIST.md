# RLAS 实验记录清单

> 本项目研究的是：在 200 demos 等较大演示集条件下，如何通过更聪明的训练数据分布，在更少训练代价下实现更快收敛。

## 1. 为什么需要这份清单

当前仓库已经能自动产出很多结果文件，但如果实验结束后只记“成功率大概是多少”，后面会出现三个问题：

- 合作者无法准确复现同一组 run
- 写主表时分不清哪些 run 是有效结果、哪些只是中断记录
- 论文整理时找不到对应的 train log、eval log、可视化图和快照

因此，每次实验跑完后，至少要按这份清单记录一次。

## 2. 先记目录，不要先记结论

### 2.1 所有方法都必须记录

| 项目 | 必须记录的路径 |
|---|---|
| sweep 总入口 | `policy/ACT/logs/<task>/sweeps.../summary.csv` |
| 最大 budget ckpt 目录 | `policy/ACT/act_ckpt/act-<task>/<ckpt_setting>-<demos>/` |
| train log | `summary.csv` 里的 `train_log` |
| eval log | `summary.csv` 里的 `eval_log` |
| 闭环结果文件 | `summary.csv` 里的 `result_path` |

### 2.2 静态 anchor / ATI 额外记录

| 项目 | 必须记录的路径 |
|---|---|
| sampling plan 目录 | `policy/ACT/sampling_plans/<task>/<task_config>-<demos>/<plan_tag>/` |
| plan 校验信息 | `summary.csv` 里的 `plan_sha256` |

### 2.3 RLAS 额外记录

| 项目 | 必须记录的路径 |
|---|---|
| sampling plan 目录 | `summary.csv` 里的 `plan_dir` |
| 权重快照目录 | `policy/ACT/act_ckpt/.../rlas_snapshots/` |
| 可视化目录 | `policy/ACT/act_ckpt/.../rlas_viz/` |
| 可视化统计文件 | `policy/ACT/act_ckpt/.../rlas_viz/rlas_summary.json` |
| 权重更新历史 | `policy/ACT/act_ckpt/.../rlas_snapshots/weight_update_history.json` |

## 3. 每次实验结束后必须记录哪些元信息

下面这些字段不一定都在 `summary.csv` 里，所以需要人工补一行实验备注。

| 字段 | 说明 |
|---|---|
| `task` | 任务名，例如 `open_laptop` |
| `method` | `Baseline` / `RLAS-Anchor Trick` / `RLAS` |
| `task_config` | 本项目主线固定 `demo_clean` |
| `expert_data_num` | 本项目主线固定 `200` |
| `seed` | 当前 run 的 seed |
| `budgets` | 通常固定 `10000,30000,60000,120000` |
| `script` | 实际使用的入口脚本 |
| `sweep_tag` | 这次 run 的目录标签 |
| `plan_tag` | 如果有 sampling plan，需要记录 |
| `git commit` | 推荐记录当前 commit SHA；如果工作树有未提交改动，要记 `dirty` |
| `gpu / 机器` | 至少记 GPU id 或机器名，便于排查性能差异 |

## 4. 每次实验结束后必须记录哪些指标

### 4.1 主指标

直接从 `summary.csv` 抽取下面几列：

- `budget`
- `success_fraction`
- `success_percent`
- `train_seconds`
- `eval_seconds`

### 4.2 方法特定指标

静态 anchor / ATI 额外记录：

- `plan_dir`
- `plan_sha256`

RLAS 额外记录：

- `rlas_warmup`
- `rlas_interval`
- `rlas_temp`
- `rlas_eps`

### 4.3 论文归档标签

每次 run 最后要手工标一类：

- `main_table_candidate`
- `appendix_only`
- `debug_run`
- `invalid_empty_summary`

没有这一步，后面整理主结果时会很混乱。

## 5. 可视化结果和日志结果存放位置

### 5.1 Baseline / 静态 anchor / RLAS 共通

| 类型 | 存放位置 |
|---|---|
| train log | `policy/ACT/logs/<task>/<ckpt_setting>-<demos>/train_<tag>.log` |
| eval log | `policy/ACT/logs/<task>/<ckpt_setting>-<demos>/eval_<tag>.log` |
| sweep 汇总 | `policy/ACT/logs/<task>/sweeps.../<run_tag>/summary.csv` |
| 评测结果 | `eval_result/<task>/ACT/<task_config>/<ckpt_setting>/.../_result.txt` |

### 5.2 RLAS 专属

| 类型 | 存放位置 |
|---|---|
| 权重快照 | `policy/ACT/act_ckpt/.../rlas_snapshots/weights_update_*.npy` |
| baseline 快照 | `policy/ACT/act_ckpt/.../rlas_snapshots/baseline_update_*.npy` |
| 快照历史 | `policy/ACT/act_ckpt/.../rlas_snapshots/weight_update_history.json` |
| 可视化图 | `policy/ACT/act_ckpt/.../rlas_viz/*.png` |
| 可视化统计 | `policy/ACT/act_ckpt/.../rlas_viz/rlas_summary.json` |

## 6. 失败 run 也必须记录

下面这些情况不能直接删掉，必须记入失败记录：

- `summary.csv` 为空
- 中间 checkpoint 缺失
- 训练被中断
- OOM
- eval 阶段没有生成结果文件

失败 run 至少要记录：

- 任务
- 方法
- seed
- config
- 失败目录
- 最后的 train/eval log 路径
- 失败现象一句话

对 `stack_blocks_two` 这类空 run 很多的任务，这一步尤其重要。

## 7. 最小实验记录模板

建议每跑完一个 sweep，就补一条这样的记录：

```md
## Run Record

- task: open_laptop
- method: RLAS
- task_config: demo_clean
- expert_data_num: 200
- seed: 0
- budgets: 10000,30000,60000,120000
- script: scripts/rlas_budget_sweep.sh
- plan_tag: rlas_W200_I5000_T1p0_E0p1
- git_commit: <commit_sha_or_dirty>
- summary_csv: policy/ACT/logs/open_laptop/sweeps_rlas/demo_clean-200-seed0/rlas_W200_I5000_T1p0_E0p1/<run_tag>/summary.csv
- ckpt_dir: policy/ACT/act_ckpt/act-open_laptop/demo_clean-u120000-rlas_rlas_W200_I5000_T1p0_E0p1-200
- train_log: <from summary.csv>
- representative_eval_log: <from summary.csv>
- rlas_snapshots_dir: policy/ACT/act_ckpt/.../rlas_snapshots
- rlas_viz_dir: policy/ACT/act_ckpt/.../rlas_viz
- main metrics:
  - 10000: 69.0%
  - 30000: 76.0%
  - 60000: 66.0%
  - 120000: 86.0%
- label: main_table_candidate
- note: baseline 高方差，RLAS 末期更稳
```

## 8. 最后只保留一条原则

实验记录时，优先保存“可回溯路径 + 原始汇总文件 + 失败信息”，然后再写自己的解释。  
只写结论、不留路径，后面基本等于没跑过。
