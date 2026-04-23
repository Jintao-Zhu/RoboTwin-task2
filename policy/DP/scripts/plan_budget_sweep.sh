#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
用法：
  bash policy/DP/scripts/plan_budget_sweep.sh \
    <task_name> <task_config> <expert_data_num> <seed> <gpu_id> "<budgets>"

示例：
  bash policy/DP/scripts/plan_budget_sweep.sh \
    open_laptop demo_clean 200 0 0 "10000 30000 60000 120000"

可选环境变量：
  SAVE_EVERY_UPDATES   默认 30000
  SWEEP_TAG            默认 YYYYMMDD_HHMMSS
  SKIP_TRAIN           默认 0（=1 时跳过训练，仅做阶段链接 + 评测 + summary）
  HEAD_CAMERA_TYPE     默认 D435
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

save_every_updates="${SAVE_EVERY_UPDATES:-30000}"
skip_train="${SKIP_TRAIN:-0}"
sweep_tag="${SWEEP_TAG:-$(date +%Y%m%d_%H%M%S)}"
head_camera_type="${HEAD_CAMERA_TYPE:-D435}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dp_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${dp_dir}/../.." && pwd)"

if ! [[ "${save_every_updates}" =~ ^[0-9]+$ ]] || (( save_every_updates <= 0 )); then
  echo "SAVE_EVERY_UPDATES 必须是正整数，当前值：${save_every_updates}"
  exit 1
fi

mapfile -t budgets < <(
  echo "${budgets_raw}" \
    | tr ',' ' ' \
    | awk '{for (i=1; i<=NF; i++) print $i}' \
    | grep -E '^[0-9]+$' \
    | sort -n \
    | awk '!seen[$0]++'
)

if [[ ${#budgets[@]} -eq 0 ]]; then
  echo "budgets 解析失败：${budgets_raw}"
  exit 1
fi

max_budget="${budgets[$(( ${#budgets[@]} - 1 ))]}"

for b in "${budgets[@]}"; do
  if (( b < 0 )); then
    echo "budget 非法（必须 >= 0）：${b}"
    exit 1
  fi
  if (( b > 0 )) && (( b % save_every_updates != 0 )); then
    echo "budget ${b} 不能被 SAVE_EVERY_UPDATES=${save_every_updates} 整除。"
    exit 1
  fi
done

cd "${dp_dir}"

dataset_path="data/${task_name}-${task_config}-${expert_data_num}.zarr"
if [[ ! -d "${dataset_path}" ]]; then
  echo "未找到数据集 ${dataset_path}，开始自动处理数据。"
  bash process_data.sh "${task_name}" "${task_config}" "${expert_data_num}"
fi

if [[ ! -d "${dataset_path}" ]]; then
  echo "数据处理后仍未找到 ${dataset_path}"
  exit 1
fi

action_dim="$(python - "${dataset_path}" <<'PY'
import sys
import zarr

path = sys.argv[1]
root = zarr.open(path, mode='r')
keys = ("data/action", "action", "data/state", "state")
for key in keys:
    try:
        arr = root[key]
        print(int(arr.shape[-1]))
        raise SystemExit(0)
    except Exception:
        pass
raise SystemExit(1)
PY
)"

if [[ "${action_dim}" == "14" ]]; then
  config_name="robot_dp_14"
elif [[ "${action_dim}" == "16" ]]; then
  config_name="robot_dp_16"
else
  echo "仅支持 action_dim=14/16，当前推断值：${action_dim}"
  exit 1
fi

read -r horizon pad_before pad_after dataset_seed dataset_val_ratio dataset_max_train_episodes < <(
  python - "${dp_dir}/diffusion_policy/config/${config_name}.yaml" "${dp_dir}/diffusion_policy/config/task/default_task_${action_dim}.yaml" <<'PY'
import sys
import yaml

config_path = sys.argv[1]
task_path = sys.argv[2]
with open(config_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
with open(task_path, 'r', encoding='utf-8') as f:
    task_cfg = yaml.safe_load(f)

horizon = int(cfg['horizon'])
pad_before = int(cfg['n_obs_steps']) - 1
pad_after = int(cfg['n_action_steps']) - 1
seed = int(task_cfg['dataset']['seed'])
val_ratio = float(task_cfg['dataset']['val_ratio'])
max_train_episodes = task_cfg['dataset']['max_train_episodes']
max_train_episodes = 'null' if max_train_episodes is None else str(int(max_train_episodes))
print(horizon, pad_before, pad_after, seed, val_ratio, max_train_episodes)
PY
)

if [[ "${dataset_max_train_episodes}" == "null" ]]; then
  max_train_arg=(--max_train_episodes)
  max_train_value=(null)
else
  max_train_arg=(--max_train_episodes)
  max_train_value=("${dataset_max_train_episodes}")
fi

plan_tag="static-amix"
setting_tag="${task_config}-${plan_tag}"

plan_dir="plans/${task_name}/${setting_tag}-${expert_data_num}-seed${seed}"
mkdir -p "${plan_dir}"

base_ckpt_setting="${setting_tag}-u${max_budget}"
base_ckpt_dir="checkpoints/${task_name}/${base_ckpt_setting}-${expert_data_num}-seed${seed}"
base_log_dir="logs/${task_name}/${base_ckpt_setting}-${expert_data_num}-seed${seed}"
mkdir -p "${base_log_dir}"

train_log="${base_log_dir}/train_${sweep_tag}.log"
hydra_run_dir="${base_log_dir}/hydra_${sweep_tag}"
logs_json_path="${hydra_run_dir}/logs.json.txt"

summary_dir="logs/${task_name}/sweeps/${setting_tag}-${expert_data_num}-seed${seed}/${sweep_tag}"
mkdir -p "${summary_dir}"
summary_csv="${summary_dir}/summary.csv"
summary_json="${summary_dir}/summary.json"

build_plan_log="${summary_dir}/build_plan_${sweep_tag}.log"

echo "budget,ckpt_setting,ckpt_dir,ckpt_path,plan_dir,train_log,eval_log,success_fraction,success_percent,result_path,train_seconds,eval_seconds" > "${summary_csv}"

if [[ ! -f "${plan_dir}/plan_meta.json" || ! -f "${plan_dir}/plan_arrays.npz" ]]; then
  echo "[plan] 构建 action-motion + uniform mix 静态采样计划：${plan_dir}"
  if [[ "${dataset_max_train_episodes}" == "null" ]]; then
    python scripts/build_sampling_plan.py \
      --zarr_path "${dataset_path}" \
      --out_dir "${plan_dir}" \
      --horizon "${horizon}" \
      --pad_before "${pad_before}" \
      --pad_after "${pad_after}" \
      --seed "${dataset_seed}" \
      --val_ratio "${dataset_val_ratio}" \
      2>&1 | tee "${build_plan_log}"
  else
    python scripts/build_sampling_plan.py \
      --zarr_path "${dataset_path}" \
      --out_dir "${plan_dir}" \
      --horizon "${horizon}" \
      --pad_before "${pad_before}" \
      --pad_after "${pad_after}" \
      --seed "${dataset_seed}" \
      --val_ratio "${dataset_val_ratio}" \
      --max_train_episodes "${dataset_max_train_episodes}" \
      2>&1 | tee "${build_plan_log}"
  fi
else
  echo "[plan] 复用已有采样计划：${plan_dir}"
fi

if [[ ! -f "${plan_dir}/plan_meta.json" || ! -f "${plan_dir}/plan_arrays.npz" ]]; then
  echo "缺少 plan 文件：${plan_dir}/plan_meta.json 或 plan_arrays.npz"
  exit 1
fi

train_seconds=""

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONUNBUFFERED=1

if [[ "${skip_train}" != "1" ]]; then
  echo "[sweep] 使用 plan 训练一次到最大 budget=${max_budget}（save_every_updates=${save_every_updates}）"
  train_start_ts="$(date +%s)"
  if [[ "${dataset_max_train_episodes}" == "null" ]]; then
    python train.py \
      --config-name="${config_name}.yaml" \
      task.name="${task_name}" \
      task.dataset.zarr_path="${dataset_path}" \
      task.dataset.seed="${dataset_seed}" \
      task.dataset.val_ratio="${dataset_val_ratio}" \
      task.dataset.sampling_plan_path="${plan_dir}" \
      task.dataset.sampling_plan_verify_sha256=True \
      training.debug=False \
      training.seed="${seed}" \
      training.device="cuda:0" \
      training.target_updates="${max_budget}" \
      training.save_every_updates="${save_every_updates}" \
      training.save_init_checkpoint=True \
      training.resume=False \
      setting="${setting_tag}" \
      expert_data_num="${expert_data_num}" \
      head_camera_type="${head_camera_type}" \
      hydra.run.dir="${hydra_run_dir}" \
      2>&1 | tee "${train_log}"
  else
    python train.py \
      --config-name="${config_name}.yaml" \
      task.name="${task_name}" \
      task.dataset.zarr_path="${dataset_path}" \
      task.dataset.seed="${dataset_seed}" \
      task.dataset.val_ratio="${dataset_val_ratio}" \
      task.dataset.max_train_episodes="${dataset_max_train_episodes}" \
      task.dataset.sampling_plan_path="${plan_dir}" \
      task.dataset.sampling_plan_verify_sha256=True \
      training.debug=False \
      training.seed="${seed}" \
      training.device="cuda:0" \
      training.target_updates="${max_budget}" \
      training.save_every_updates="${save_every_updates}" \
      training.save_init_checkpoint=True \
      training.resume=False \
      setting="${setting_tag}" \
      expert_data_num="${expert_data_num}" \
      head_camera_type="${head_camera_type}" \
      hydra.run.dir="${hydra_run_dir}" \
      2>&1 | tee "${train_log}"
  fi
  train_end_ts="$(date +%s)"
  train_seconds="$(( train_end_ts - train_start_ts ))"
else
  echo "[sweep] SKIP_TRAIN=1，跳过训练，仅做阶段链接 + 评测 + 汇总。"
fi

if [[ ! -d "${base_ckpt_dir}" ]]; then
  echo "未找到 base checkpoint 目录：${base_ckpt_dir}"
  exit 1
fi

if [[ ! -f "${base_ckpt_dir}/policy_last.ckpt" ]]; then
  echo "缺少 ${base_ckpt_dir}/policy_last.ckpt"
  exit 1
fi

strip_ansi() {
  sed -E 's/\x1B\[[0-9;]*[mK]//g'
}

parse_eval_metrics() {
  local eval_log_file="$1"
  local last_sr
  local frac
  local pct
  local result_path

  last_sr="$(tr '\r' '\n' < "${eval_log_file}" | strip_ansi | grep -E 'Success rate:' | tail -n 1 || true)"
  frac="$(echo "${last_sr}" | sed -n 's/.*Success rate: *\([0-9]\+\/[0-9]\+\).*/\1/p')"
  pct="$(echo "${last_sr}" | sed -n 's/.*=> *\([0-9.]\+%\).*/\1/p')"
  result_path="$(tr '\r' '\n' < "${eval_log_file}" | strip_ansi | grep -E 'Data has been saved to ' | tail -n 1 | sed 's/.*Data has been saved to //' || true)"

  echo "${frac},${pct},${result_path}"
}

for b in "${budgets[@]}"; do
  ckpt_setting="${setting_tag}-u${b}"
  stage_ckpt_dir="checkpoints/${task_name}/${ckpt_setting}-${expert_data_num}-seed${seed}"
  log_dir="logs/${task_name}/${ckpt_setting}-${expert_data_num}-seed${seed}"
  mkdir -p "${log_dir}"

  if (( b == max_budget )); then
    ckpt_path="$(realpath "${base_ckpt_dir}/policy_last.ckpt")"
  else
    src_ckpt="${base_ckpt_dir}/policy_update_${b}_seed_${seed}.ckpt"
    if [[ ! -f "${src_ckpt}" ]]; then
      echo "缺少 budget=${b} 对应 checkpoint：${src_ckpt}"
      exit 1
    fi
    mkdir -p "${stage_ckpt_dir}"
    ln -sf "$(realpath "${src_ckpt}")" "${stage_ckpt_dir}/policy_last.ckpt"
    ckpt_path="$(realpath "${stage_ckpt_dir}/policy_last.ckpt")"
  fi

  eval_log="${log_dir}/eval_${sweep_tag}.log"

  echo "[sweep] 开始评测 budget=${b}"
  eval_start_ts="$(date +%s)"
  bash eval.sh "${task_name}" "${task_config}" "${ckpt_setting}" "${expert_data_num}" "${seed}" "${gpu_id}" "${ckpt_path}" \
    2>&1 | tee "${eval_log}"
  eval_end_ts="$(date +%s)"
  eval_seconds="$(( eval_end_ts - eval_start_ts ))"

  IFS=',' read -r success_fraction success_percent result_path < <(parse_eval_metrics "${eval_log}")

  echo "${b},${ckpt_setting},${stage_ckpt_dir},${ckpt_path},${plan_dir},${train_log},${eval_log},${success_fraction},${success_percent},${result_path},${train_seconds},${eval_seconds}" >> "${summary_csv}"
done

python - "${summary_csv}" "${summary_json}" "${task_name}" "${task_config}" "${expert_data_num}" "${seed}" "${max_budget}" "${save_every_updates}" "${plan_dir}" "${train_log}" "${logs_json_path}" "${budgets[@]}" <<'PY'
import csv
import json
import sys

summary_csv = sys.argv[1]
summary_json = sys.argv[2]
task_name = sys.argv[3]
task_config = sys.argv[4]
expert_data_num = int(sys.argv[5])
seed = int(sys.argv[6])
max_budget = int(sys.argv[7])
save_every_updates = int(sys.argv[8])
plan_dir = sys.argv[9]
train_log = sys.argv[10]
logs_json_path = sys.argv[11]
budgets = [int(x) for x in sys.argv[12:]]

records = []
with open(summary_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        records.append(row)

payload = {
    "task_name": task_name,
    "task_config": task_config,
    "expert_data_num": expert_data_num,
    "seed": seed,
    "max_budget": max_budget,
    "save_every_updates": save_every_updates,
    "plan_dir": plan_dir,
    "budgets": budgets,
    "train_log": train_log,
    "logs_json_path": logs_json_path,
    "records": records,
}

with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
PY

echo "[sweep] 完成"
echo "[sweep] plan_dir: ${plan_dir}"
echo "[sweep] train_log: ${train_log}"
echo "[sweep] logs.json.txt: ${logs_json_path}"
echo "[sweep] summary.csv: ${summary_csv}"
echo "[sweep] summary.json: ${summary_json}"
