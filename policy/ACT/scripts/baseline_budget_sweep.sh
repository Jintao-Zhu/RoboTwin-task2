#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
只训练 1 次（跑到最大预算），并通过“软链接中间 checkpoint”的方式，
评测多个训练预算（budgets），同时把每个预算对应的 ckpt 目录整理得更直观可读。

用法：
  bash policy/ACT/scripts/baseline_budget_sweep.sh \
    <task_name> <task_config> <expert_data_num> <seed> <gpu_id> "<budgets>"

示例：
  bash policy/ACT/scripts/baseline_budget_sweep.sh \
    beat_block_hammer demo_clean 200 0 0 "5000 10000 30000 60000 120000"

说明：
  - budgets 的单位是“optimizer updates”（优化器更新次数），不是 epoch。
  - budgets 必须是 SAVE_EVERY_UPDATES 的整数倍（脚本需要用到中间 checkpoint）。
  - 脚本只会训练到 max(budgets) 一次，然后复用中间 checkpoint 依次评测所有 budgets。

可选环境变量：
  SWEEP_TAG              默认：YYYYMMDD_HHMMSS（本次 sweep 的标签，用于日志文件名/目录）
  SAVE_EVERY_UPDATES     默认：5000（必须能整除所有 budgets）
  HDF5_USE_FILE_LOCKING  默认：FALSE（避免 h5py 文件锁导致 BlockingIOError）
  SKIP_TRAIN             默认：0（=1 时跳过训练，只做“链接+评测”）
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

save_every_updates="${SAVE_EVERY_UPDATES:-5000}"
skip_train="${SKIP_TRAIN:-0}"
sweep_tag="${SWEEP_TAG:-$(date +%Y%m%d_%H%M%S)}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
act_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${act_dir}/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

# 解析 budgets：允许空格/逗号分隔，去重并排序
mapfile -t budgets < <(echo "${budgets_raw}" | tr ',' ' ' | xargs -n1 | rg -v '^$' | sort -n -u)
if [[ ${#budgets[@]} -eq 0 ]]; then
  echo "无法从 budgets 字符串解析到任何数字：${budgets_raw}"
  exit 1
fi

max_updates="${budgets[$(( ${#budgets[@]} - 1 ))]}"

for b in "${budgets[@]}"; do
  if (( b <= 0 )); then
    echo "budget 非法（必须 > 0）：${b}"
    exit 1
  fi
  if (( b % save_every_updates != 0 )); then
    echo "budget ${b} 不能被 SAVE_EVERY_UPDATES=${save_every_updates} 整除。"
    echo "请改用类似 5000,10000,15000,... 的 budgets，或调整 SAVE_EVERY_UPDATES。"
    exit 1
  fi
done

cd "${act_dir}"

dataset_dir="${act_dir}/processed_data/sim-${task_name}/${task_config}-${expert_data_num}"
if [[ ! -d "${dataset_dir}" ]]; then
  echo "未找到处理后的数据目录：${dataset_dir}"
  echo "提示：请先处理数据（示例）："
  echo "  (cd ${act_dir} && bash process_data.sh ${task_name} ${task_config} ${expert_data_num})"
  exit 1
fi

base_ckpt_setting="${task_config}-u${max_updates}"
base_ckpt_dir="${act_dir}/act_ckpt/act-${task_name}/${base_ckpt_setting}-${expert_data_num}"
base_log_dir="${act_dir}/logs/${task_name}/${base_ckpt_setting}-${expert_data_num}"
mkdir -p "${base_log_dir}"

run_summary_dir="${act_dir}/logs/${task_name}/sweeps/${task_config}-${expert_data_num}-seed${seed}/${sweep_tag}"
mkdir -p "${run_summary_dir}"
summary_csv="${run_summary_dir}/summary.csv"

echo "budget,ckpt_setting,ckpt_dir,train_log,eval_log,success_fraction,success_percent,result_path,train_seconds,eval_seconds" > "${summary_csv}"

train_log="${base_log_dir}/train_${sweep_tag}.log"
train_seconds=""
if [[ "${skip_train}" != "1" ]]; then
  echo "[sweep] 只训练 1 次（最大预算）：target_updates=${max_updates}（save_every_updates=${save_every_updates}）"
  start_ts="$(date +%s)"
  # 直接调用 imitate_episodes.py，便于显式控制 --save_every_updates（从而生成中间 checkpoint）
  python3 imitate_episodes.py \
    --task_name "sim-${task_name}-${task_config}-${expert_data_num}" \
    --ckpt_dir "./act_ckpt/act-${task_name}/${base_ckpt_setting}-${expert_data_num}" \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 50 \
    --hidden_dim 512 \
    --batch_size 8 \
    --dim_feedforward 3200 \
    --num_epochs 6000 \
    --lr 1e-5 \
    --save_freq 2000 \
    --state_dim 14 \
    --seed "${seed}" \
    --target_updates "${max_updates}" \
    --save_every_updates "${save_every_updates}" \
    2>&1 | tee "${train_log}"
  end_ts="$(date +%s)"
  train_seconds="$(( end_ts - start_ts ))"
else
  echo "[sweep] SKIP_TRAIN=1：跳过训练，只做“链接+评测”。"
  # 尝试复用 base 目录下最近一次训练日志（如果没有，就留空）
  existing_train_log="$(ls -1t "${base_log_dir}"/train_*.log 2>/dev/null | head -n 1 || true)"
  if [[ -n "${existing_train_log}" ]]; then
    train_log="${existing_train_log}"
  else
    train_log=""
  fi
fi

if [[ ! -d "${base_ckpt_dir}" ]]; then
  echo "未找到 base ckpt 目录：${base_ckpt_dir}"
  exit 1
fi

base_stats="${base_ckpt_dir}/dataset_stats.pkl"
if [[ ! -f "${base_stats}" ]]; then
  echo "缺少 dataset_stats.pkl：${base_stats}"
  exit 1
fi

strip_ansi() {
  sed -r 's/\x1B\[[0-9;]*[mK]//g'
}

parse_eval_metrics() {
  local eval_log_file="$1"
  local last_sr
  last_sr="$(tr '\r' '\n' < "${eval_log_file}" | strip_ansi | rg "Success rate:" | tail -n 1 || true)"
  local frac
  frac="$(echo "${last_sr}" | sed -n 's/.*Success rate: *\([0-9]\+\/[0-9]\+\).*/\1/p')"
  local pct
  pct="$(echo "${last_sr}" | sed -n 's/.*=> *\([0-9.]\+%\).*/\1/p')"

  local result_path
  result_path="$(tr '\r' '\n' < "${eval_log_file}" | strip_ansi | rg "Data has been saved to " | tail -n 1 | sed 's/.*Data has been saved to //')"

  echo "${frac},${pct},${result_path}"
}

for b in "${budgets[@]}"; do
  ckpt_setting="${task_config}-u${b}"
  ckpt_dir="${act_dir}/act_ckpt/act-${task_name}/${ckpt_setting}-${expert_data_num}"
  log_dir="${act_dir}/logs/${task_name}/${ckpt_setting}-${expert_data_num}"
  mkdir -p "${log_dir}"
  if (( b != max_updates )) && [[ -n "${train_log}" ]] && [[ -f "${train_log}" ]]; then
    ln -sf "$(realpath "${train_log}")" "${log_dir}/train_${sweep_tag}.log"
  fi

  if (( b != max_updates )); then
    update_ckpt="${base_ckpt_dir}/policy_update_${b}_seed_${seed}.ckpt"
    if [[ ! -f "${update_ckpt}" ]]; then
      echo "缺少 budget=${b} 对应的中间 checkpoint：${update_ckpt}"
      echo "提示：可以把 SAVE_EVERY_UPDATES 设得更小（同时保证 budgets 能被它整除）后重跑。"
      exit 1
    fi

    mkdir -p "${ckpt_dir}"
    ln -sf "$(realpath "${base_stats}")" "${ckpt_dir}/dataset_stats.pkl"
    ln -sf "$(realpath "${update_ckpt}")" "${ckpt_dir}/policy_last.ckpt"
  else
    # 确保 base 目录里也有这两个文件（便于统一评测逻辑）
    if [[ ! -f "${ckpt_dir}/dataset_stats.pkl" ]]; then
      ln -sf "$(realpath "${base_stats}")" "${ckpt_dir}/dataset_stats.pkl"
    fi
    if [[ ! -f "${ckpt_dir}/policy_last.ckpt" ]]; then
      echo "base ckpt 目录缺少 policy_last.ckpt：${ckpt_dir}/policy_last.ckpt"
      exit 1
    fi
  fi

  eval_log="${log_dir}/eval_${sweep_tag}.log"
  echo "[sweep] 开始评测 budget=${b}：${ckpt_setting}"
  eval_start_ts="$(date +%s)"
  bash eval.sh "${task_name}" "${task_config}" "${ckpt_setting}" "${expert_data_num}" "${seed}" "${gpu_id}" \
    2>&1 | tee "${eval_log}"
  eval_end_ts="$(date +%s)"
  eval_seconds="$(( eval_end_ts - eval_start_ts ))"

  IFS=',' read -r success_fraction success_percent result_path < <(parse_eval_metrics "${eval_log}")
  echo "${b},${ckpt_setting},${ckpt_dir},${train_log},${eval_log},${success_fraction},${success_percent},${result_path},${train_seconds},${eval_seconds}" >> "${summary_csv}"
done

echo "[sweep] 完成。汇总文件：${summary_csv}"
