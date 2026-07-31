#!/usr/bin/env bash
set -euo pipefail

project_root="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1
    pwd
)"
cd "${project_root}"

output_dir="result/dataset2_cooccur_lift_online_promotion_20260729/double-replay-retry2"
log_path="logs/dataset2_cooccur_lift_online_promotion_replay_retry2.log"

if [[ -e "${output_dir}" ]]; then
    echo "refusing to overwrite existing output: ${output_dir}" >&2
    exit 1
fi
if [[ -e "${log_path}" ]]; then
    echo "refusing to overwrite existing log: ${log_path}" >&2
    exit 1
fi

mkdir -p logs
exec >"${log_path}" 2>&1

export JITTOR_HOME="${project_root}/.deps/jittor-home"
export TMPDIR="${project_root}/.deps/tmp"
export CUDA_HOME="${project_root}/.venv/jittor_nv126_overlay"
export nvcc_path="${CUDA_HOME}/bin/nvcc"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONUNBUFFERED=1

echo "[promotion-replay] started_at=$(date --iso-8601=seconds)"
echo "[promotion-replay] output_dir=${output_dir}"

exec .venv/bin/python -u scripts/replay_dataset2_cooccur_lift_checkpoint.py \
    --checkpoint checkpoints/d1_time_ramp_g050_d2_cooccur_lift_aux_w050_20260729.pkl \
    --checkpoint-report result/dataset2_cooccur_lift_online_promotion_20260729/checkpoint/checkpoint-integration-report.json \
    --candidate-zip result/d1_time_ramp_g050_d2_cooccur_lift_aux_w050_20260729/result.zip \
    --output-dir "${output_dir}" \
    --data-dir data \
    --batch-size 4096
