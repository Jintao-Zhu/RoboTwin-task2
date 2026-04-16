#!/bin/bash

# == keep unchanged ==
policy_name=DP
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
ckpt_path=${7:-}
DEBUG=False

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..

eval_args=(
    --config "policy/$policy_name/deploy_policy.yml"
    --overrides
    --task_name "${task_name}"
    --task_config "${task_config}"
    --ckpt_setting "${ckpt_setting}"
    --expert_data_num "${expert_data_num}"
    --seed "${seed}"
)

if [[ -n "${ckpt_path}" ]]; then
    eval_args+=(--ckpt_path "${ckpt_path}")
fi

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py "${eval_args[@]}"
