#!/usr/bin/env bash
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="$root/result/dataset1_k256_static_setwise_dual_horizon_20260730"
cache_dir="$run_dir/cache"
mkdir -p "$run_dir"
echo "$$" > "$run_dir/pipeline.pid"

while [[ ! -f "$cache_dir/build.exit" ]]; do
  sleep 30
done
cache_code="$(cat "$cache_dir/build.exit")"
if [[ "$cache_code" != "0" ]]; then
  echo "cache_failed:$cache_code" > "$run_dir/pipeline.status"
  echo "$cache_code" > "$run_dir/pipeline.exit"
  exit "$cache_code"
fi

export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24
export NUMEXPR_NUM_THREADS=24
export CUDA_VISIBLE_DEVICES=0

set +e
.venv/bin/python scripts/train_select_dataset1_k256_static_setwise_dual_horizon.py \
  --plan docs/experiments/dataset1-k256-static-setwise-dual-horizon.preregistered.json \
  --plan-sha256 docs/experiments/dataset1-k256-static-setwise-dual-horizon.preregistered.json.sha256 \
  --source-checkpoint checkpoints/d1_time_ramp_g050_d2_cooccur_lift_aux_w050_20260729.pkl \
  --cache-prefix cache/supervised_features/dataset1_k256_static_setwise_recent200k_full100_20260730 \
  --cache-report "$cache_dir/train-cache-report.json" \
  --reference-cache-prefix cache/supervised_features/dataset1_joint_recent200k_full100_seed60_20260726 \
  --output-dir "$run_dir/internal" \
  > "$run_dir/internal.log" 2>&1
internal_code=$?
set -e
echo "$internal_code" > "$run_dir/internal.exit"
if [[ "$internal_code" != "0" ]]; then
  echo "internal_rejected_or_failed:$internal_code" > "$run_dir/pipeline.status"
  echo "$internal_code" > "$run_dir/pipeline.exit"
  exit "$internal_code"
fi

echo "internal_selected_external_starting" > "$run_dir/pipeline.status"
set +e
bash scripts/run_dataset1_k256_static_setwise_external_20260730.sh \
  > "$run_dir/external.log" 2>&1
external_code=$?
set -e
echo "$external_code" > "$run_dir/external.exit"
if [[ "$external_code" == "0" ]]; then
  echo "external_accepted" > "$run_dir/pipeline.status"
else
  echo "external_rejected_or_failed:$external_code" > "$run_dir/pipeline.status"
fi
echo "$external_code" > "$run_dir/pipeline.exit"
exit "$external_code"
