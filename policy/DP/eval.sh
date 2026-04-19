#!/bin/bash

# == keep unchanged ==
policy_name=DP
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dp_dir="${script_dir}"
repo_root="$(cd "${dp_dir}/../.." && pwd)"

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

if [[ -n "${ckpt_path}" ]]; then
    if [[ "${ckpt_path}" = /* ]]; then
        : # already absolute
    elif [[ -f "${PWD}/${ckpt_path}" ]]; then
        ckpt_path="$(realpath "${PWD}/${ckpt_path}")"
    elif [[ -f "${dp_dir}/${ckpt_path}" ]]; then
        ckpt_path="$(realpath "${dp_dir}/${ckpt_path}")"
    elif [[ -f "${repo_root}/${ckpt_path}" ]]; then
        ckpt_path="$(realpath "${repo_root}/${ckpt_path}")"
    fi
fi

cd "${repo_root}"

cmd=(
    python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml
    --overrides
    --task_name "${task_name}"
    --task_config "${task_config}"
    --ckpt_setting "${ckpt_setting}"
    --expert_data_num "${expert_data_num}"
    --seed "${seed}"
)

if [[ -n "${ckpt_path}" ]]; then
    cmd+=(--ckpt_path "${ckpt_path}")
fi

PYTHONWARNINGS=ignore::UserWarning "${cmd[@]}"
