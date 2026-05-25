#!/usr/bin/env bash
set -euo pipefail

task_name=${1:?task_name required}
task_config=${2:?task_config required}
train_config_name=${3:?train_config_name required}
expert_data_num=${4:?expert_data_num required}
seed=${5:?seed required}
gpu_id=${6:?gpu_id required}
budgets_raw=${7:?budgets required}

export CUDA_VISIBLE_DEVICES=${gpu_id}

cd "$(dirname "$0")/.."

echo "[env] using python: $(which python)"
python -c "import sys; print('[env] executable:', sys.executable)"

repo_id="${task_name}-${task_config}-${expert_data_num}"

# budgets 支持空格或逗号
budgets=$(echo "${budgets_raw}" | tr ',' ' ' | xargs -n1 | sort -n | uniq | xargs)
max_budget=$(echo "${budgets}" | xargs -n1 | sort -n | tail -1)

save_every=${SAVE_EVERY_UPDATES:-10000}
sweep_tag=${SWEEP_TAG:-$(date +%Y%m%d_%H%M%S)}

method_tag="rlas"
model_name="${task_config}-${method_tag}-u${max_budget}-${expert_data_num}-seed${seed}"

plan_dir="sampling_plans/${task_name}/${task_config}-${method_tag}-anchors-${expert_data_num}-seed${seed}"

train_log_dir="logs/${task_name}/${model_name}"
mkdir -p "${train_log_dir}"

summary_dir="logs/${task_name}/sweeps/${task_config}-${method_tag}-${expert_data_num}-seed${seed}/${sweep_tag}"
mkdir -p "${summary_dir}"

summary_csv="${summary_dir}/summary.csv"
summary_json="${summary_dir}/summary.json"

echo "budget,success_rate,eval_log" > "${summary_csv}"
echo "[]" > "${summary_json}"

echo "[RLAS] repo_id=${repo_id}"
echo "[RLAS] train_config_name=${train_config_name}"
echo "[RLAS] model_name=${model_name}"
echo "[RLAS] budgets=${budgets}"
echo "[RLAS] max_budget=${max_budget}"
echo "[RLAS] save_every=${save_every}"

if [[ ! -d "${plan_dir}" ]]; then
  echo "[RLAS] build anchor sampling plan: ${plan_dir}"

  # pi0 action_horizon 默认 50，所以 tail_keep=49。
  # 这里 stage K=1，weights 不参与 RLAS，只用 anchor_indices 固定训练单位。
  python scripts/build_sampling_plan.py \
    --repo-id "${repo_id}" \
    --train-config-name "${train_config_name}" \
    --num-episodes-keep "${expert_data_num}" \
    --episode-selection-seed "${seed}" \
    --plan-seed "${seed}" \
    --sampler-seed "${seed}" \
    --replacement \
    --stride-S 1 \
    --valid-start-tail-keep 49 \
    --stage-strategy time_quantile \
    --num-stages-K 1 \
    --stage-balance uniform_over_stage \
    --clip-ratio 5.0 \
    --epsilon-mix 0.0 \
    --out-dir "${plan_dir}"
else
  echo "[RLAS] reuse existing plan: ${plan_dir}"
fi

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  echo "[RLAS] train once to max_budget=${max_budget}"

  XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45} \
  python scripts/train.py "${train_config_name}" \
    --exp-name "${model_name}" \
    --overwrite \
    --seed "${seed}" \
    --data.repo-id "${repo_id}" \
    --num-train-steps "${max_budget}" \
    --save-interval "${save_every}" \
    --keep-period "${save_every}" \
    --sampling-plan-path "${plan_dir}" \
    --rlas.enabled \
    --rlas.warmup-steps 10000 \
    --rlas.update-interval 10000 \
    --rlas.temperature 1.0 \
    --rlas.epsilon-mix 0.1 \
    --rlas.ema-beta 0.5 \
    --rlas.alpha 1.0 \
    --rlas.scoring-seed 12345 \
    --wandb-enabled false \
    2>&1 | tee "${train_log_dir}/train_${sweep_tag}.log"
else
  echo "[RLAS] SKIP_TRAIN=1, skip training"
fi

for budget in ${budgets}; do
  eval_log_dir="logs/${task_name}/${task_config}-${method_tag}-u${budget}-${expert_data_num}-seed${seed}"
  mkdir -p "${eval_log_dir}"
  eval_log="${eval_log_dir}/eval_${sweep_tag}.log"

  echo "[RLAS] eval checkpoint=${budget}"

  bash eval.sh \
    "${task_name}" \
    "${task_config}" \
    "${train_config_name}" \
    "${model_name}" \
    "${seed}" \
    "${gpu_id}" \
    "${budget}" \
    2>&1 | tee "${eval_log}"

  success_rate=$(grep -E "Success rate:" "${eval_log}" | tail -1 | awk '{print $NF}' || true)
  if [[ -z "${success_rate}" ]]; then
    success_rate="NA"
  fi

  echo "${budget},${success_rate},${eval_log}" >> "${summary_csv}"

  python - << EOF
import json
from pathlib import Path

path = Path("${summary_json}")
data = json.loads(path.read_text())
data.append({
    "budget": int("${budget}"),
    "success_rate": "${success_rate}",
    "eval_log": "${eval_log}",
})
path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
EOF

done

echo "[RLAS] summary_csv=${summary_csv}"
echo "[RLAS] summary_json=${summary_json}"
