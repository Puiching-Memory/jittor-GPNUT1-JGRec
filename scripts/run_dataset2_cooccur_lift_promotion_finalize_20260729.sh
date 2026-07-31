#!/usr/bin/env bash
set -euo pipefail

project_root="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1
    pwd
)"
cd "${project_root}"

status_dir="result/dataset2_cooccur_lift_online_promotion_20260729/auto-finalize-v1"
replay_dir="result/dataset2_cooccur_lift_online_promotion_20260729/double-replay-final-v1"
status_file="${status_dir}/status.json"
orchestrator_log="logs/dataset2_cooccur_lift_online_promotion_finalize_20260729.log"

if [[ -e "${status_dir}" || -e "${replay_dir}" || -e "${orchestrator_log}" ]]; then
    echo "refusing to overwrite an existing finalization artifact" >&2
    exit 1
fi

mkdir -p "${status_dir}" logs
exec >"${orchestrator_log}" 2>&1

export JITTOR_HOME="${project_root}/.deps/jittor-home"
export TMPDIR="${project_root}/.deps/tmp"
export CUDA_HOME="${project_root}/.venv/jittor_nv126_overlay"
export nvcc_path="${CUDA_HOME}/bin/nvcc"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONUNBUFFERED=1

phase="initializing"
write_status() {
    local status="$1"
    local exit_code="$2"
    printf '{"status":"%s","phase":"%s","exit_code":%s,"updated_at":"%s"}\n' \
        "${status}" \
        "${phase}" \
        "${exit_code}" \
        "$(date --iso-8601=seconds)" \
        >"${status_file}.tmp"
    mv "${status_file}.tmp" "${status_file}"
}
on_exit() {
    local exit_code="$?"
    if [[ "${phase}" != "passed" ]]; then
        write_status "failed" "${exit_code}"
    fi
}
trap on_exit EXIT

write_status "running" 0
echo "[finalize] started_at=$(date --iso-8601=seconds)"

phase="waiting_for_previous_diagnostic"
write_status "running" 0
while pgrep -f '[p]ython.*_diagnose_cooccur_aux_sequential_parity.py' >/dev/null; do
    sleep 10
done

phase="focused_green"
write_status "running" 0
.venv/bin/python -m pytest \
    tests/test_cooccur_lift_checkpoint.py::test_auxiliary_prediction_is_bracketed_by_jittor_cache_cleanup \
    -q \
    >"${status_dir}/focused-green.log" 2>&1

phase="bounded_sequential_parity"
write_status "running" 0
.venv/bin/python -u scripts/_diagnose_cooccur_aux_sequential_parity.py \
    >"${status_dir}/bounded-sequential-parity.log" 2>&1

phase="full_double_replay"
write_status "running" 0
.venv/bin/python -u scripts/replay_dataset2_cooccur_lift_checkpoint.py \
    --checkpoint checkpoints/d1_time_ramp_g050_d2_cooccur_lift_aux_w050_20260729.pkl \
    --checkpoint-report result/dataset2_cooccur_lift_online_promotion_20260729/checkpoint/checkpoint-integration-report.json \
    --candidate-zip result/d1_time_ramp_g050_d2_cooccur_lift_aux_w050_20260729/result.zip \
    --output-dir "${replay_dir}" \
    --data-dir data \
    --batch-size 4096 \
    >"${status_dir}/full-double-replay.log" 2>&1

test -s "${replay_dir}/replay-report.json"
phase="passed"
write_status "passed" 0
echo "[finalize] passed_at=$(date --iso-8601=seconds)"
