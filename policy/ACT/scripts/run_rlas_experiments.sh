#!/usr/bin/env bash
# ============================================================================
# RLAS 完整实验脚本
# ============================================================================
# 运行 Baseline、ATI (静态 anchor)、RLAS (动态权重) 三种方法的对比实验
#
# 用法：
#   bash scripts/run_rlas_experiments.sh <task_name> <gpu_id> [seeds]
#
# 示例：
#   # 单任务单 GPU
#   bash scripts/run_rlas_experiments.sh stack_blocks_two 0 "0 1 2"
#
#   # 多任务并行（需要多 GPU）
#   bash scripts/run_rlas_experiments.sh stack_blocks_two 0 "0" &
#   bash scripts/run_rlas_experiments.sh open_laptop 1 "0" &
#   wait
#
# 输出：
#   - logs/<task>/sweeps_baseline/: Baseline 实验结果
#   - logs/<task>/sweeps_plan/: ATI (静态权重) 实验结果  
#   - logs/<task>/sweeps_rlas/: RLAS (动态权重) 实验结果
# ============================================================================

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "用法: bash scripts/run_rlas_experiments.sh <task_name> <gpu_id> [seeds]"
  echo "示例: bash scripts/run_rlas_experiments.sh stack_blocks_two 0 \"0 1 2\""
  exit 1
fi

task_name="$1"
gpu_id="$2"
seeds="${3:-0}"  # 默认只跑 seed 0

# 实验配置
task_config="demo_clean"
expert_data_num="200"
budgets="10000 30000 60000 120000"

# RLAS 超参数
export RLAS_WARMUP=10000
export RLAS_INTERVAL=10000
export RLAS_TEMP=1.0
export RLAS_EPS=0.1
export RLAS_BETA=0.5

# Plan 参数
export PLAN_STRIDE=1
export PLAN_STAGE_BALANCE=none

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
act_dir="$(cd "${script_dir}/.." && pwd)"

cd "${act_dir}"

echo "============================================"
echo "RLAS 完整实验"
echo "============================================"
echo "任务：${task_name}"
echo "数据：${expert_data_num} demos"
echo "Seeds：${seeds}"
echo "Budgets：${budgets}"
echo "GPU：${gpu_id}"
echo "============================================"

# 检查数据是否存在
dataset_dir="${act_dir}/processed_data/sim-${task_name}/${task_config}-${expert_data_num}"
if [[ ! -d "${dataset_dir}" ]]; then
  echo "[准备] 处理数据..."
  bash process_data.sh "${task_name}" "${task_config}" "${expert_data_num}"
fi

for seed in ${seeds}; do
  echo ""
  echo "========== Seed ${seed} =========="
  
  # --- 1. Baseline（原始 ACT）---
  echo ""
  echo "[1/3] Baseline (原始 ACT)..."
  SKIP_TRAIN=0 bash scripts/baseline_budget_sweep.sh \
    "${task_name}" "${task_config}" "${expert_data_num}" "${seed}" "${gpu_id}" "${budgets}" \
    || echo "[Baseline] 失败，继续下一个实验"
  
  # --- 2. ATI（静态 anchor 均匀权重）---
  echo ""
  echo "[2/3] ATI (静态 anchor 均匀权重)..."
  PLAN_TAG="ati_uniform_s${seed}" \
  SKIP_TRAIN=0 \
  bash scripts/plan_budget_sweep.sh \
    "${task_name}" "${task_config}" "${expert_data_num}" "${seed}" "${gpu_id}" "${budgets}" \
    || echo "[ATI] 失败，继续下一个实验"
  
  # --- 3. RLAS（动态权重）---
  echo ""
  echo "[3/3] RLAS (动态权重)..."
  PLAN_TAG="rlas_s${seed}" \
  SKIP_TRAIN=0 \
  bash scripts/rlas_budget_sweep.sh \
    "${task_name}" "${task_config}" "${expert_data_num}" "${seed}" "${gpu_id}" "${budgets}" \
    || echo "[RLAS] 失败"
  
done

echo ""
echo "============================================"
echo "实验完成！"
echo "结果位置："
echo "  - Baseline: logs/${task_name}/sweeps_baseline/"
echo "  - ATI:      logs/${task_name}/sweeps_plan/"
echo "  - RLAS:     logs/${task_name}/sweeps_rlas/"
echo "============================================"
