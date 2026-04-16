#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash policy/DP/scripts/baseline_budget_sweep.sh \
    <task_name> <task_config> <expert_data_num> <seed> <gpu_id> "<budgets>"

Example:
  bash policy/DP/scripts/baseline_budget_sweep.sh \
    open_laptop demo_clean 200 0 0 "0 30000 60000 120000"

Optional environment variables:
  SAVE_EVERY_UPDATES   default: 30000
  SWEEP_TAG            default: YYYYMMDD_HHMMSS
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
sweep_tag="${SWEEP_TAG:-$(date +%Y%m%d_%H%M%S)}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dp_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${dp_dir}/../.." && pwd)"

mapfile -t budgets < <(
  echo "${budgets_raw}" \
    | tr ',' ' ' \
    | awk '{for (i=1; i<=NF; ++i) print $i}' \
    | rg '^[0-9]+$' \
    | sort -n -u
)

if [[ ${#budgets[@]} -eq 0 ]]; then
  echo "No valid budgets parsed from: ${budgets_raw}"
  exit 1
fi

if ! [[ "${save_every_updates}" =~ ^[0-9]+$ ]] || (( save_every_updates <= 0 )); then
  echo "SAVE_EVERY_UPDATES must be a positive integer, got: ${save_every_updates}"
  exit 1
fi

max_budget="${budgets[$(( ${#budgets[@]} - 1 ))]}"
for b in "${budgets[@]}"; do
  if (( b < 0 )); then
    echo "Invalid budget (must be >= 0): ${b}"
    exit 1
  fi
  if (( b > 0 && b % save_every_updates != 0 )); then
    echo "Budget ${b} is not divisible by SAVE_EVERY_UPDATES=${save_every_updates}."
    exit 1
  fi
done

cd "${dp_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

dataset_rel="data/${task_name}-${task_config}-${expert_data_num}.zarr"
if [[ ! -d "${dataset_rel}" ]]; then
  bash process_data.sh "${task_name}" "${task_config}" "${expert_data_num}"
fi

action_dim="$(
python3 - "${dataset_rel}" <<'PY'
import sys

path = sys.argv[1]
dim = None
try:
    import zarr
    root = zarr.open(path, mode="r")

    for key in ("action", "state", "data/action", "data/state"):
        try:
            arr = root[key]
            if hasattr(arr, "shape") and len(arr.shape) > 0:
                dim = int(arr.shape[-1])
                break
        except Exception:
            pass

    if dim is None:
        def walk(group, prefix=""):
            for key in group.keys():
                item = group[key]
                name = f"{prefix}{key}"
                if hasattr(item, "shape"):
                    yield name, item
                else:
                    try:
                        yield from walk(item, prefix=f"{name}/")
                    except Exception:
                        pass

        for name, item in walk(root):
            lname = name.lower()
            if "action" in lname or lname.endswith("state"):
                if len(item.shape) > 0:
                    dim = int(item.shape[-1])
                    break
except Exception:
    pass

if dim is None:
    print("")
else:
    print(dim)
PY
)"

if [[ "${action_dim}" != "14" && "${action_dim}" != "16" ]]; then
  echo "Failed to infer valid action_dim (14/16) from ${dataset_rel}; got: ${action_dim:-<empty>}"
  exit 1
fi

config_name="robot_dp_${action_dim}.yaml"
train_ckpt_setting="${task_config}-u${max_budget}"
train_ckpt_dir="${dp_dir}/checkpoints/${task_name}/${train_ckpt_setting}-${expert_data_num}-seed${seed}"

train_log_dir="${dp_dir}/logs/${task_name}/${train_ckpt_setting}-${expert_data_num}-seed${seed}"
mkdir -p "${train_log_dir}"
train_log="${train_log_dir}/train_${sweep_tag}.log"

summary_dir="${dp_dir}/logs/${task_name}/sweeps/${task_config}-${expert_data_num}-seed${seed}/${sweep_tag}"
mkdir -p "${summary_dir}"
summary_csv="${summary_dir}/summary.csv"
summary_json="${summary_dir}/summary.json"
records_tsv="${summary_dir}/records.tsv"

echo "budget,ckpt_setting,ckpt_dir,ckpt_path,train_log,eval_log,success_fraction,success_percent,result_path,train_seconds,eval_seconds" > "${summary_csv}"
: > "${records_tsv}"

train_start_ts="$(date +%s)"
python3 train.py --config-name="${config_name}" \
  task.name="${task_name}" \
  task.dataset.zarr_path="${dataset_rel}" \
  training.debug=False \
  training.seed="${seed}" \
  training.device="cuda:0" \
  exp_name="${task_name}-robot_dp-baseline-sweep" \
  logging.mode=online \
  setting="${task_config}" \
  expert_data_num="${expert_data_num}" \
  head_camera_type=D435 \
  training.target_updates="${max_budget}" \
  training.save_every_updates="${save_every_updates}" \
  training.resume=False \
  task.dataset.use_expert_action=True \
  task.dataset.mix_expert_action=False \
  task.dataset.add_expert_noise=False \
  2>&1 | tee "${train_log}"
train_end_ts="$(date +%s)"
train_seconds="$(( train_end_ts - train_start_ts ))"

if [[ ! -d "${train_ckpt_dir}" ]]; then
  echo "Training checkpoint directory not found: ${train_ckpt_dir}"
  exit 1
fi

strip_ansi() {
  sed -r 's/\x1B\[[0-9;]*[mK]//g'
}

parse_eval_metrics() {
  local eval_log_file="$1"
  local sr_line
  sr_line="$(tr '\r' '\n' < "${eval_log_file}" | strip_ansi | rg "Success rate:" | tail -n 1 || true)"
  local frac
  frac="$(echo "${sr_line}" | sed -n 's/.*Success rate: *\([0-9]\+\/[0-9]\+\).*/\1/p')"
  local pct
  pct="$(echo "${sr_line}" | sed -n 's/.*=> *\([0-9.]\+%\).*/\1/p')"
  local result_path
  result_path="$(tr '\r' '\n' < "${eval_log_file}" | strip_ansi | rg "Data has been saved to " | tail -n 1 | sed 's/.*Data has been saved to //')"
  echo -e "${frac}\t${pct}\t${result_path}"
}

for b in "${budgets[@]}"; do
  ckpt_setting="${task_config}-u${b}"
  stage_ckpt_dir="${dp_dir}/checkpoints/${task_name}/${ckpt_setting}-${expert_data_num}-seed${seed}"
  mkdir -p "${stage_ckpt_dir}"

  stage_src_ckpt="${train_ckpt_dir}/policy_update_${b}_seed_${seed}.ckpt"
  if [[ ! -f "${stage_src_ckpt}" ]]; then
    echo "Missing checkpoint for budget ${b}: ${stage_src_ckpt}"
    exit 1
  fi

  stage_ckpt_path="${stage_ckpt_dir}/policy_last.ckpt"
  ln -sfn "$(realpath "${stage_src_ckpt}")" "${stage_ckpt_path}"

  eval_log_dir="${dp_dir}/logs/${task_name}/${ckpt_setting}-${expert_data_num}-seed${seed}"
  mkdir -p "${eval_log_dir}"
  eval_log="${eval_log_dir}/eval_${sweep_tag}.log"

  eval_start_ts="$(date +%s)"
  bash eval.sh "${task_name}" "${task_config}" "${ckpt_setting}" "${expert_data_num}" "${seed}" "${gpu_id}" "${stage_ckpt_path}" \
    2>&1 | tee "${eval_log}"
  eval_end_ts="$(date +%s)"
  eval_seconds="$(( eval_end_ts - eval_start_ts ))"

  IFS=$'\t' read -r success_fraction success_percent result_path < <(parse_eval_metrics "${eval_log}")

  echo "${b},${ckpt_setting},${stage_ckpt_dir},${stage_ckpt_path},${train_log},${eval_log},${success_fraction},${success_percent},${result_path},${train_seconds},${eval_seconds}" >> "${summary_csv}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${b}" "${ckpt_setting}" "${stage_ckpt_dir}" "${stage_ckpt_path}" "${train_log}" "${eval_log}" \
    "${success_fraction}" "${success_percent}" "${result_path}" "${train_seconds}" "${eval_seconds}" >> "${records_tsv}"
done

budgets_joined="$(IFS=,; echo "${budgets[*]}")"
python3 - "${summary_json}" "${task_name}" "${task_config}" "${expert_data_num}" "${seed}" "${max_budget}" "${save_every_updates}" "${budgets_joined}" "${records_tsv}" <<'PY'
import json
import sys

summary_json = sys.argv[1]
task_name = sys.argv[2]
task_config = sys.argv[3]
expert_data_num = int(sys.argv[4])
seed = int(sys.argv[5])
max_budget = int(sys.argv[6])
save_every_updates = int(sys.argv[7])
budgets = [int(x) for x in sys.argv[8].split(",") if x]
records_tsv = sys.argv[9]

records = []
with open(records_tsv, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        (budget, ckpt_setting, ckpt_dir, ckpt_path, train_log, eval_log, success_fraction, success_percent,
         result_path, train_seconds, eval_seconds) = line.split("\t")
        records.append({
            "budget": int(budget),
            "ckpt_setting": ckpt_setting,
            "ckpt_dir": ckpt_dir,
            "ckpt_path": ckpt_path,
            "train_log": train_log,
            "eval_log": eval_log,
            "success_fraction": success_fraction,
            "success_percent": success_percent,
            "result_path": result_path,
            "train_seconds": int(train_seconds) if train_seconds else None,
            "eval_seconds": int(eval_seconds) if eval_seconds else None,
        })

data = {
    "task_name": task_name,
    "task_config": task_config,
    "expert_data_num": expert_data_num,
    "seed": seed,
    "max_budget": max_budget,
    "save_every_updates": save_every_updates,
    "budgets": budgets,
    "records": records,
}

with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
PY

echo "[baseline sweep] done"
echo "summary.csv: ${summary_csv}"
echo "summary.json: ${summary_json}"
