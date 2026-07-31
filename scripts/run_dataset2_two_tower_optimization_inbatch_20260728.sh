#!/usr/bin/env bash
set -uo pipefail

cd /home/edu/workspace/jittor-GPNUT1-JGRec
source .workspace-env.sh

run_prefix="dataset2_two_tower_opt_inbatch_seed60_20260728"
control_dir="result/dataset2_two_tower_listwise_200k_seed60_20260724"
full100_prefix="cache/supervised_features/dataset2_full100_matched_train_seed60_20260723"
cache_report="result/dataset2_matched_full100_seed60_20260723/cache-build-report.json"
mkdir -p logs result

resource_gate() {
  local gpu_pids
  local available_kib
  gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
  available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  if [[ -n "${gpu_pids}" ]]; then
    printf 'resource gate: GPU already has compute PIDs: %s\n' "${gpu_pids}"
    return 75
  fi
  if [[ "${available_kib}" -lt 12582912 ]]; then
    printf 'resource gate: only %s KiB memory available\n' "${available_kib}"
    return 75
  fi
}

run_arm() {
  local arm="$1"
  local output_dir="result/${run_prefix}_${arm}"
  local log_file="logs/${run_prefix}_${arm}.log"
  local exit_file="logs/${run_prefix}_${arm}.exit"

  if [[ -e "${output_dir}" || -e "${exit_file}" ]]; then
    printf 'refusing to overwrite arm=%s\n' "${arm}"
    return 73
  fi
  resource_gate || return $?
  nice -n 10 uv run --no-sync python \
    scripts/evaluate_dataset2_two_tower_optimization_inbatch.py \
    --train-csv data/dataset2/train.csv \
    --test-csv data/dataset2/test.csv \
    --full100-prefix "${full100_prefix}" \
    --cache-report "${cache_report}" \
    --control-model "${control_dir}/candidate-model.npz" \
    --control-report "${control_dir}/evaluation-report.json" \
    --output-dir "${output_dir}" \
    --arm "${arm}" \
    --seed 60 \
    --val-ratio 0.15 \
    --context-ratio 0.75 \
    --validation-queries 20000 \
    --epochs 50 \
    --patience 3 \
    --batch-size 512 \
    --sampling-workers 16 \
    --max-samples 200000 \
    --negatives 99 \
    >"${log_file}" 2>&1
  local status=$?
  printf '%s\n' "${status}" >"${exit_file}"
  return "${status}"
}

status=0
for arm in optimizer_only inbatch_only combined; do
  run_arm "${arm}" || {
    status=$?
    break
  }
done

printf '%s\n' "${status}" >"logs/${run_prefix}.exit"
exit "${status}"
