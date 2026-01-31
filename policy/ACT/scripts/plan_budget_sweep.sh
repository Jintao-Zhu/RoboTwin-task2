#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
基于 Sampling Plan（采样计划）做 budget sweep（训练预算扫描）：
- 先离线生成 plan（anchors + weights）
- 再只训练 1 次（跑到 max(budgets)）
- 最后通过“软链接中间 checkpoint”的方式，评测多个 budgets，并汇总成 summary.csv

用法：
  bash policy/ACT/scripts/plan_budget_sweep.sh \
    <task_name> <task_config> <expert_data_num> <seed> <gpu_id> "<budgets>"

示例：
  # open_laptop / 200 demos / seed0 / GPU0
  PLAN_STRIDE=2 PLAN_STAGE_BALANCE=none PLAN_TAG=S2_uniform \
  bash policy/ACT/scripts/plan_budget_sweep.sh \
    open_laptop demo_clean 200 0 0 "5000 10000 30000 60000 120000"

说明（关键概念的通俗解释）：
  - budget：optimizer updates（优化器更新次数），不是 epoch（训练轮数）。
  - Sampling Plan：一份“可枚举片段清单 + 每个片段权重”的文件；训练时按它抽样。
  - anchor：一个训练片段的锚点，用 (episode_id, start_ts) 表示（某条 demo 的某个起点帧）。

可选环境变量（推荐先只用这些，别把脚本搞复杂）：
  # 通用
  SWEEP_TAG              默认：YYYYMMDD_HHMMSS（本次 sweep 标签，用于日志/目录）
  SAVE_EVERY_UPDATES     默认：5000（必须能整除所有 budgets，才能链接中间 checkpoint）
  PYTHON                 默认：python3（如需指定 conda python：PYTHON=/path/to/python）
  SKIP_PLAN              默认：0（=1 时跳过 plan 生成，直接用 PLAN_PATH 指定的 plan）
  SKIP_TRAIN             默认：0（=1 时跳过训练，只做“链接+评测”）
  HDF5_USE_FILE_LOCKING  默认：FALSE（避免 h5py 文件锁导致 BlockingIOError）

  # plan 路径与命名（可读性来自这里）
  PLAN_TAG               默认：自动从 plan 参数拼出来（建议你显式给一个短 tag）
  PLAN_PATH              默认：空（为空则自动生成 plan 到 sampling_plans/ 下；非空则直接使用该 plan）
  PLAN_OUT_DIR           默认：sampling_plans/<task>/<task_config>-<demos>/<PLAN_TAG>

  # plan 生成参数（调用 scripts/build_sampling_plan.py）
  PLAN_STRIDE            默认：1
  PLAN_NUM_STAGES        默认：4
  PLAN_STAGE_BALANCE     默认：none（可选：none/uniform_over_stage/inv_freq）
  PLAN_INV_FREQ_POWER    默认：1.0
  PLAN_CLIP_RATIO        默认：5.0
  PLAN_EPSILON_MIX        默认：0.01
  PLAN_SEED              默认：0（影响 stride offset）
  PLAN_SAMPLER_SEED      默认：0（写进 meta；训练 sampler 也默认用它）
  PLAN_NO_REPLACEMENT    默认：0（=1 时关闭有放回采样）

  # 训练侧 plan 采样参数（传给 imitate_episodes.py）
  PLAN_NUM_SAMPLES        默认：0（<=0 表示默认=训练集 episode 数量，尽量对齐 baseline 的 steps/epoch）
  ACT_NUM_WORKERS         默认：1（如遇到 PermissionError，可设为 0 关闭多进程 DataLoader）

输出（可读性保证）：
  - ckpt_dir：policy/ACT/act_ckpt/act-<task>/<task_config>-u<budget>-plan_<PLAN_TAG>-<demos>/
  - log_dir ：policy/ACT/logs/<task>/<task_config>-u<budget>-plan_<PLAN_TAG>-<demos>/
  - sweep 汇总：policy/ACT/logs/<task>/sweeps_plan/<task_config>-<demos>-seed<seed>/<PLAN_TAG>/<SWEEP_TAG>/summary.csv
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

py="${PYTHON:-python3}"
save_every_updates="${SAVE_EVERY_UPDATES:-5000}"
skip_plan="${SKIP_PLAN:-0}"
skip_train="${SKIP_TRAIN:-0}"
sweep_tag="${SWEEP_TAG:-$(date +%Y%m%d_%H%M%S)}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

# === plan 参数（尽量用 env var，避免脚本解析太复杂）===
plan_stride="${PLAN_STRIDE:-1}"
plan_num_stages="${PLAN_NUM_STAGES:-4}"
plan_stage_balance="${PLAN_STAGE_BALANCE:-none}"
plan_inv_freq_power="${PLAN_INV_FREQ_POWER:-1.0}"
plan_clip_ratio="${PLAN_CLIP_RATIO:-5.0}"
plan_epsilon_mix="${PLAN_EPSILON_MIX:-0.01}"
plan_seed="${PLAN_SEED:-0}"
plan_sampler_seed="${PLAN_SAMPLER_SEED:-0}"
plan_no_replacement="${PLAN_NO_REPLACEMENT:-0}"

# 训练侧采样参数：每个 epoch 抽多少个 anchor（默认对齐 episode 数）
plan_num_samples="${PLAN_NUM_SAMPLES:-0}"

# 自动生成一个可读的 plan_tag（你也可以自己显式设置 PLAN_TAG）
slug_num() { echo "$1" | sed 's/\./p/g'; }
if [[ -z "${PLAN_TAG:-}" ]]; then
  # 例：S2_none_K4_clip5p0_eps0p01_p0_s0_rep
  tag="S${plan_stride}_${plan_stage_balance}_K${plan_num_stages}_clip$(slug_num "${plan_clip_ratio}")_eps$(slug_num "${plan_epsilon_mix}")_p${plan_seed}_s${plan_sampler_seed}"
  if [[ "${plan_no_replacement}" == "1" ]]; then
    tag="${tag}_norepl"
  fi
  if [[ "${plan_num_samples}" != "0" ]]; then
    tag="${tag}_n${plan_num_samples}"
  fi
  PLAN_TAG="${tag}"
fi
plan_tag="${PLAN_TAG}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
act_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${act_dir}/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONUNBUFFERED=1

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

plan_path="${PLAN_PATH:-}"
if [[ -z "${plan_path}" ]]; then
  plan_out_dir="${PLAN_OUT_DIR:-${act_dir}/sampling_plans/${task_name}/${task_config}-${expert_data_num}/${plan_tag}}"
  plan_path="${plan_out_dir}"
fi

run_summary_dir="${act_dir}/logs/${task_name}/sweeps_plan/${task_config}-${expert_data_num}-seed${seed}/${plan_tag}/${sweep_tag}"
mkdir -p "${run_summary_dir}"
summary_csv="${run_summary_dir}/summary.csv"
echo "budget,ckpt_setting,ckpt_dir,plan_dir,plan_sha256,train_log,eval_log,success_fraction,success_percent,result_path,train_seconds,eval_seconds" > "${summary_csv}"

base_ckpt_setting="${task_config}-u${max_updates}-plan_${plan_tag}"
base_ckpt_dir="${act_dir}/act_ckpt/act-${task_name}/${base_ckpt_setting}-${expert_data_num}"
base_log_dir="${act_dir}/logs/${task_name}/${base_ckpt_setting}-${expert_data_num}"
mkdir -p "${base_log_dir}"
train_log="${base_log_dir}/train_${sweep_tag}.log"

if [[ "${skip_plan}" != "1" ]]; then
  echo "[plan] 生成 sampling plan：${plan_path}"
  plan_args=(--dataset_dir "${dataset_dir}" --out_dir "${plan_path}" --stride "${plan_stride}" --plan_seed "${plan_seed}" --sampler_seed "${plan_sampler_seed}" --num_stages "${plan_num_stages}" --stage_balance "${plan_stage_balance}" --inv_freq_power "${plan_inv_freq_power}" --clip_ratio "${plan_clip_ratio}" --epsilon_mix "${plan_epsilon_mix}")
  if [[ "${plan_no_replacement}" == "1" ]]; then
    plan_args+=(--no_replacement)
  fi
  "${py}" scripts/build_sampling_plan.py "${plan_args[@]}" 2>&1 | tee "${run_summary_dir}/build_plan_${sweep_tag}.log"
else
  echo "[plan] SKIP_PLAN=1：跳过 plan 生成，直接使用：${plan_path}"
fi

if [[ ! -f "${plan_path}/plan_meta.json" ]] || [[ ! -f "${plan_path}/plan_arrays.npz" ]]; then
  echo "plan 文件缺失：${plan_path}/plan_meta.json 或 ${plan_path}/plan_arrays.npz"
  exit 1
fi

plan_sha256="$(
  "${py}" -c 'import json, sys; from pathlib import Path; p=Path(sys.argv[1])/"plan_meta.json"; meta=json.loads(p.read_text(encoding="utf-8")); print(meta.get("plan_sha256",""))' \
    "${plan_path}"
)"
echo "[plan] plan_sha256=${plan_sha256}"
cp -f "${plan_path}/plan_meta.json" "${run_summary_dir}/plan_meta.json"
cp -f "${plan_path}/plan_arrays.npz" "${run_summary_dir}/plan_arrays.npz"

train_seconds=""
if [[ "${skip_train}" != "1" ]]; then
  echo "[train] 只训练 1 次（最大预算）：target_updates=${max_updates}（save_every_updates=${save_every_updates}）"
  start_ts="$(date +%s)"
  "${py}" imitate_episodes.py \
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
    --sampling_plan_path "${plan_path}" \
    --plan_sampler_seed "${plan_sampler_seed}" \
    --plan_num_samples "${plan_num_samples}" \
    $( [[ "${plan_no_replacement}" == "1" ]] && echo "--plan_no_replacement" ) \
    2>&1 | tee "${train_log}"
  end_ts="$(date +%s)"
  train_seconds="$(( end_ts - start_ts ))"
else
  echo "[train] SKIP_TRAIN=1：跳过训练，只做“链接+评测”。"
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

# 把 plan 文件拷贝进 ckpt_dir，保证 ckpt 本身就是“自包含可复现”的
plan_ckpt_dir="${base_ckpt_dir}/sampling_plan"
mkdir -p "${plan_ckpt_dir}"
cp -f "${plan_path}/plan_meta.json" "${plan_ckpt_dir}/plan_meta.json"
cp -f "${plan_path}/plan_arrays.npz" "${plan_ckpt_dir}/plan_arrays.npz"
if [[ -f "${plan_path}/plan_stats.json" ]]; then
  cp -f "${plan_path}/plan_stats.json" "${plan_ckpt_dir}/plan_stats.json"
fi

strip_ansi() { sed -r 's/\x1B\[[0-9;]*[mK]//g'; }
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
  ckpt_setting="${task_config}-u${b}-plan_${plan_tag}"
  ckpt_dir="${act_dir}/act_ckpt/act-${task_name}/${ckpt_setting}-${expert_data_num}"
  log_dir="${act_dir}/logs/${task_name}/${ckpt_setting}-${expert_data_num}"
  mkdir -p "${log_dir}"

  # 让每个 budget 的 log_dir 都能看到同一份训练日志（便于对照）
  if (( b != max_updates )) && [[ -n "${train_log}" ]] && [[ -f "${train_log}" ]]; then
    ln -sf "$(realpath "${train_log}")" "${log_dir}/train_${sweep_tag}.log"
  fi

  # === budget 目录组织：dataset_stats + policy_last.ckpt + sampling_plan ===
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
    ln -sf "$(realpath "${plan_ckpt_dir}")" "${ckpt_dir}/sampling_plan"
  else
    # base 目录（max budget）本身就是 ckpt_dir
    if [[ ! -f "${ckpt_dir}/dataset_stats.pkl" ]]; then
      ln -sf "$(realpath "${base_stats}")" "${ckpt_dir}/dataset_stats.pkl"
    fi
    if [[ ! -f "${ckpt_dir}/policy_last.ckpt" ]]; then
      echo "base ckpt 目录缺少 policy_last.ckpt：${ckpt_dir}/policy_last.ckpt"
      exit 1
    fi
  fi

  eval_log="${log_dir}/eval_${sweep_tag}.log"
  echo "[eval] 开始评测 budget=${b}：${ckpt_setting}"
  eval_start_ts="$(date +%s)"
  bash eval.sh "${task_name}" "${task_config}" "${ckpt_setting}" "${expert_data_num}" "${seed}" "${gpu_id}" \
    2>&1 | tee "${eval_log}"
  eval_end_ts="$(date +%s)"
  eval_seconds="$(( eval_end_ts - eval_start_ts ))"

  IFS=',' read -r success_fraction success_percent result_path < <(parse_eval_metrics "${eval_log}")
  echo "${b},${ckpt_setting},${ckpt_dir},${plan_path},${plan_sha256},${train_log},${eval_log},${success_fraction},${success_percent},${result_path},${train_seconds},${eval_seconds}" >> "${summary_csv}"
done

echo "[sweep] 完成。汇总文件：${summary_csv}"
