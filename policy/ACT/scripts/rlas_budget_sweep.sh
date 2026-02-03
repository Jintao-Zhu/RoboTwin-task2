#!/usr/bin/env bash
# ============================================================================
# RLAS Budget Sweep 脚本
# ============================================================================
# 基于 RLAS (Reducible Loss for Anchor Selection) 的训练+评测一体化脚本
# 
# 功能：
#   1. 离线生成 Sampling Plan（anchors + 初始权重）
#   2. 使用 RLAS 动态权重训练一次（跑到 max(budgets)）
#   3. 评测多个 budget 检查点，汇总成 summary.csv
#
# 用法：
#   bash scripts/rlas_budget_sweep.sh \
#       <task_name> <task_config> <expert_data_num> <seed> <gpu_id> "<budgets>"
#
# 示例：
#   # stack_blocks_two / 200 demos / seed0 / GPU0 / RLAS 动态权重
#   RLAS_WARMUP=10000 RLAS_INTERVAL=10000 RLAS_TEMP=1.0 \
#   bash scripts/rlas_budget_sweep.sh \
#       beat_block_hammer demo_clean 200 0 0 "5000 10000 30000 60000 120000"
#
# 环境变量：
#   # 通用
#   SWEEP_TAG              默认：YYYYMMDD_HHMMSS
#   SAVE_EVERY_UPDATES     默认：5000
#   PYTHON                 默认：python3
#   SKIP_PLAN              默认：0
#   SKIP_TRAIN             默认：0
#
#   # Plan 参数
#   PLAN_TAG               默认：rlas_W<warmup>_I<interval>_T<temp>
#   PLAN_STRIDE            默认：1
#   PLAN_STAGE_BALANCE     默认：none
#   PLAN_SAMPLER_SEED      默认：0
#
#   # RLAS 参数
#   RLAS_WARMUP            默认：10000
#   RLAS_INTERVAL          默认：10000
#   RLAS_TEMP              默认：1.0
#   RLAS_EPS               默认：0.1
#   RLAS_BETA              默认：0.5
#   RLAS_ALPHA             默认：1.0
#   RLAS_BATCH_SIZE        默认：64
# ============================================================================

set -euo pipefail

usage() {
  cat <<'USAGE'
RLAS Budget Sweep - 动态采样权重训练 + 多预算评测

用法：
  bash scripts/rlas_budget_sweep.sh \
    <task_name> <task_config> <expert_data_num> <seed> <gpu_id> "<budgets>"

示例：
  RLAS_WARMUP=10000 RLAS_INTERVAL=10000 \
  bash scripts/rlas_budget_sweep.sh \
    stack_blocks_two demo_clean 200 0 0 "10000 30000 60000 120000"
USAGE
}

if [[ $# -lt 6 ]]; then
  usage
  exit 1
fi

task_name="$1"
task_config="$2"
expert_data_num="$3"
seed="$4"
gpu_id="$5"
budgets_raw="$6"

# === 基础配置 ===
py="${PYTHON:-python3}"
save_every_updates="${SAVE_EVERY_UPDATES:-5000}"
skip_plan="${SKIP_PLAN:-0}"
skip_train="${SKIP_TRAIN:-0}"
sweep_tag="${SWEEP_TAG:-$(date +%Y%m%d_%H%M%S)}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

# === RLAS 参数 ===
rlas_warmup="${RLAS_WARMUP:-10000}"
rlas_interval="${RLAS_INTERVAL:-10000}"
rlas_temp="${RLAS_TEMP:-1.0}"
rlas_eps="${RLAS_EPS:-0.1}"
rlas_beta="${RLAS_BETA:-0.5}"
rlas_alpha="${RLAS_ALPHA:-1.0}"
rlas_batch_size="${RLAS_BATCH_SIZE:-64}"

# === Plan 参数 ===
plan_stride="${PLAN_STRIDE:-1}"
plan_num_stages="${PLAN_NUM_STAGES:-4}"
plan_stage_balance="${PLAN_STAGE_BALANCE:-none}"
plan_seed="${PLAN_SEED:-0}"
plan_sampler_seed="${PLAN_SAMPLER_SEED:-0}"
plan_num_samples="${PLAN_NUM_SAMPLES:-0}"
plan_no_replacement="${PLAN_NO_REPLACEMENT:-0}"

# 自动生成 plan_tag
slug_num() { echo "$1" | sed 's/\./p/g'; }
if [[ -z "${PLAN_TAG:-}" ]]; then
  PLAN_TAG="rlas_W${rlas_warmup}_I${rlas_interval}_T$(slug_num "${rlas_temp}")_E$(slug_num "${rlas_eps}")"
fi
plan_tag="${PLAN_TAG}"

# === 路径设置 ===
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
act_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${act_dir}/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONUNBUFFERED=1

# 解析 budgets
mapfile -t budgets < <(echo "${budgets_raw}" | tr ',' ' ' | xargs -n1 | grep -v '^$' | sort -n -u)
if [[ ${#budgets[@]} -eq 0 ]]; then
  echo "无法解析 budgets：${budgets_raw}"
  exit 1
fi

max_updates="${budgets[$(( ${#budgets[@]} - 1 ))]}"

for b in "${budgets[@]}"; do
  if (( b <= 0 )); then
    echo "budget 非法：${b}"
    exit 1
  fi
  if (( b % save_every_updates != 0 )); then
    echo "budget ${b} 不能被 SAVE_EVERY_UPDATES=${save_every_updates} 整除"
    exit 1
  fi
done

cd "${act_dir}"

# === 检查数据 ===
dataset_dir="${act_dir}/processed_data/sim-${task_name}/${task_config}-${expert_data_num}"
if [[ ! -d "${dataset_dir}" ]]; then
  echo "未找到数据目录：${dataset_dir}"
  echo "请先运行：bash process_data.sh ${task_name} ${task_config} ${expert_data_num}"
  exit 1
fi

# === Plan 路径 ===
plan_path="${PLAN_PATH:-${act_dir}/sampling_plans/${task_name}/${task_config}-${expert_data_num}/${plan_tag}}"

# === 输出目录 ===
run_summary_dir="${act_dir}/logs/${task_name}/sweeps_rlas/${task_config}-${expert_data_num}-seed${seed}/${plan_tag}/${sweep_tag}"
mkdir -p "${run_summary_dir}"
summary_csv="${run_summary_dir}/summary.csv"
echo "budget,ckpt_setting,ckpt_dir,plan_dir,train_log,eval_log,success_fraction,success_percent,result_path,train_seconds,eval_seconds,rlas_warmup,rlas_interval,rlas_temp,rlas_eps" > "${summary_csv}"

base_ckpt_setting="${task_config}-u${max_updates}-rlas_${plan_tag}"
base_ckpt_dir="${act_dir}/act_ckpt/act-${task_name}/${base_ckpt_setting}-${expert_data_num}"
base_log_dir="${act_dir}/logs/${task_name}/${base_ckpt_setting}-${expert_data_num}"
mkdir -p "${base_log_dir}"
train_log="${base_log_dir}/train_${sweep_tag}.log"

# === Step 1: 生成 Plan ===
if [[ "${skip_plan}" != "1" ]]; then
  echo "[plan] 生成 sampling plan：${plan_path}"
  plan_args=(
    --dataset_dir "${dataset_dir}"
    --out_dir "${plan_path}"
    --stride "${plan_stride}"
    --plan_seed "${plan_seed}"
    --sampler_seed "${plan_sampler_seed}"
    --num_stages "${plan_num_stages}"
    --stage_balance "${plan_stage_balance}"
  )
  "${py}" scripts/build_sampling_plan.py "${plan_args[@]}" 2>&1 | tee "${run_summary_dir}/build_plan.log"
else
  echo "[plan] SKIP_PLAN=1：跳过 plan 生成"
fi

if [[ ! -f "${plan_path}/plan_meta.json" ]]; then
  echo "plan 文件缺失：${plan_path}/plan_meta.json"
  exit 1
fi

# === Step 2: RLAS 训练 ===
train_seconds=""
if [[ "${skip_train}" != "1" ]]; then
  echo "[train] RLAS 训练：target_updates=${max_updates}"
  echo "[train] RLAS 参数：warmup=${rlas_warmup}, interval=${rlas_interval}, temp=${rlas_temp}, eps=${rlas_eps}"
  
  start_ts="$(date +%s)"
  
  train_args=(
    --task_name "sim-${task_name}-${task_config}-${expert_data_num}"
    --ckpt_dir "${base_ckpt_dir}"
    --policy_class ACT
    --kl_weight 10
    --chunk_size 50
    --hidden_dim 512
    --batch_size 8
    --dim_feedforward 3200
    --num_epochs 6000
    --lr 1e-5
    --state_dim 14
    --seed "${seed}"
    --target_updates "${max_updates}"
    --save_every_updates "${save_every_updates}"
    # Sampling plan
    --sampling_plan_path "${plan_path}"
    --plan_sampler_seed "${plan_sampler_seed}"
    --plan_num_samples "${plan_num_samples}"
    # RLAS 参数
    --enable_rlas
    --rlas_warmup_updates "${rlas_warmup}"
    --rlas_update_interval "${rlas_interval}"
    --rlas_temperature "${rlas_temp}"
    --rlas_epsilon_mix "${rlas_eps}"
    --rlas_ema_beta "${rlas_beta}"
    --rlas_alpha "${rlas_alpha}"
    --rlas_scoring_batch_size "${rlas_batch_size}"
  )
  
  if [[ "${plan_no_replacement}" == "1" ]]; then
    train_args+=(--plan_no_replacement)
  fi
  
  "${py}" imitate_episodes_rlas.py "${train_args[@]}" 2>&1 | tee "${train_log}"
  
  end_ts="$(date +%s)"
  train_seconds="$(( end_ts - start_ts ))"
else
  echo "[train] SKIP_TRAIN=1：跳过训练"
  existing_log="$(ls -1t "${base_log_dir}"/train_*.log 2>/dev/null | head -n 1 || true)"
  if [[ -n "${existing_log}" ]]; then
    train_log="${existing_log}"
  else
    train_log=""
  fi
fi

if [[ ! -d "${base_ckpt_dir}" ]]; then
  echo "未找到 ckpt 目录：${base_ckpt_dir}"
  exit 1
fi

base_stats="${base_ckpt_dir}/dataset_stats.pkl"
if [[ ! -f "${base_stats}" ]]; then
  echo "缺少 dataset_stats.pkl：${base_stats}"
  exit 1
fi

# === Step 3: 评测各 budget ===
strip_ansi() { sed -r 's/\x1B\[[0-9;]*[mK]//g'; }
parse_eval_metrics() {
  local eval_log_file="$1"
  local last_sr
  last_sr="$(tr '\r' '\n' < "${eval_log_file}" | strip_ansi | grep "Success rate:" | tail -n 1 || true)"
  local frac
  frac="$(echo "${last_sr}" | sed -n 's/.*Success rate: *\([0-9]\+\/[0-9]\+\).*/\1/p')"
  local pct
  pct="$(echo "${last_sr}" | sed -n 's/.*=> *\([0-9.]\+%\).*/\1/p')"
  local result_path
  result_path="$(tr '\r' '\n' < "${eval_log_file}" | strip_ansi | grep "Data has been saved to " | tail -n 1 | sed 's/.*Data has been saved to //')"
  echo "${frac},${pct},${result_path}"
}

for b in "${budgets[@]}"; do
  ckpt_setting="${task_config}-u${b}-rlas_${plan_tag}"
  ckpt_dir="${act_dir}/act_ckpt/act-${task_name}/${ckpt_setting}-${expert_data_num}"
  log_dir="${act_dir}/logs/${task_name}/${ckpt_setting}-${expert_data_num}"
  mkdir -p "${log_dir}"

  # 链接训练日志
  if (( b != max_updates )) && [[ -n "${train_log}" ]] && [[ -f "${train_log}" ]]; then
    ln -sf "$(realpath "${train_log}")" "${log_dir}/train_${sweep_tag}.log"
  fi

  # 创建 budget 对应的 ckpt 目录（软链接中间 checkpoint）
  if (( b != max_updates )); then
    update_ckpt="${base_ckpt_dir}/policy_update_${b}_seed_${seed}.ckpt"
    if [[ ! -f "${update_ckpt}" ]]; then
      echo "缺少 checkpoint：${update_ckpt}"
      exit 1
    fi
    mkdir -p "${ckpt_dir}"
    ln -sf "$(realpath "${base_stats}")" "${ckpt_dir}/dataset_stats.pkl"
    ln -sf "$(realpath "${update_ckpt}")" "${ckpt_dir}/policy_last.ckpt"
  else
    if [[ ! -f "${ckpt_dir}/dataset_stats.pkl" ]]; then
      ln -sf "$(realpath "${base_stats}")" "${ckpt_dir}/dataset_stats.pkl"
    fi
  fi

  eval_log="${log_dir}/eval_${sweep_tag}.log"
  echo "[eval] budget=${b}：${ckpt_setting}"
  eval_start_ts="$(date +%s)"
  
  bash eval.sh "${task_name}" "${task_config}" "${ckpt_setting}" "${expert_data_num}" "${seed}" "${gpu_id}" \
    2>&1 | tee "${eval_log}"
  
  eval_end_ts="$(date +%s)"
  eval_seconds="$(( eval_end_ts - eval_start_ts ))"

  IFS=',' read -r success_fraction success_percent result_path < <(parse_eval_metrics "${eval_log}")
  echo "${b},${ckpt_setting},${ckpt_dir},${plan_path},${train_log},${eval_log},${success_fraction},${success_percent},${result_path},${train_seconds},${eval_seconds},${rlas_warmup},${rlas_interval},${rlas_temp},${rlas_eps}" >> "${summary_csv}"
done

echo ""
echo "============================================"
echo "[RLAS Sweep] 完成！"
echo "汇总文件：${summary_csv}"
echo "============================================"
