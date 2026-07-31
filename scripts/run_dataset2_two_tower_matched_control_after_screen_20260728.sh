#!/usr/bin/env bash
set -uo pipefail

cd /home/edu/workspace/jittor-GPNUT1-JGRec
source .workspace-env.sh

run_prefix="dataset2_two_tower_opt_inbatch_seed60_20260728"
master_exit="logs/${run_prefix}.exit"
control_dir="result/${run_prefix}_matched_control"
matched_dir="result/${run_prefix}_matched_screen"
control_log="logs/${run_prefix}_matched_control.log"
control_exit="logs/${run_prefix}_matched_control.exit"
waiter_exit="logs/${run_prefix}_matched_finalize.exit"

while [[ ! -f "${master_exit}" ]]; do
  sleep 30
done
if [[ "$(cat "${master_exit}")" != "0" ]]; then
  printf 'candidate screen failed before matched control\n'
  printf '74\n' >"${waiter_exit}"
  exit 74
fi

while true; do
  gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
  available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  if [[ -z "${gpu_pids}" && "${available_kib}" -ge 12582912 ]]; then
    break
  fi
  sleep 30
done

if [[ -e "${control_dir}" || -e "${matched_dir}" ]]; then
  printf 'refusing to overwrite matched control or report\n'
  printf '73\n' >"${waiter_exit}"
  exit 73
fi

nice -n 10 uv run --no-sync python \
  scripts/evaluate_dataset2_two_tower_optimization_inbatch.py \
  --train-csv data/dataset2/train.csv \
  --test-csv data/dataset2/test.csv \
  --full100-prefix \
  cache/supervised_features/dataset2_full100_matched_train_seed60_20260723 \
  --cache-report \
  result/dataset2_matched_full100_seed60_20260723/cache-build-report.json \
  --control-model \
  result/dataset2_two_tower_listwise_200k_seed60_20260724/candidate-model.npz \
  --control-report \
  result/dataset2_two_tower_listwise_200k_seed60_20260724/evaluation-report.json \
  --output-dir "${control_dir}" \
  --arm control \
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
  >"${control_log}" 2>&1
status=$?
printf '%s\n' "${status}" >"${control_exit}"
if [[ "${status}" != "0" ]]; then
  printf '%s\n' "${status}" >"${waiter_exit}"
  exit "${status}"
fi

uv run --no-sync python \
  scripts/finalize_dataset2_two_tower_optimization_inbatch.py \
  --control-dir "${control_dir}" \
  --candidate-dir "result/${run_prefix}_optimizer_only" \
  --candidate-dir "result/${run_prefix}_inbatch_only" \
  --candidate-dir "result/${run_prefix}_combined" \
  --output-dir "${matched_dir}" \
  >"logs/${run_prefix}_matched_finalize.log" 2>&1
status=$?
printf '%s\n' "${status}" >"${waiter_exit}"
exit "${status}"
