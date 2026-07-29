#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24
export NUMEXPR_NUM_THREADS=24
export CUDA_VISIBLE_DEVICES=0

exec .venv/bin/python scripts/evaluate_dataset1_k256_static_setwise_external.py \
  --plan docs/experiments/dataset1-k256-static-setwise-dual-horizon.preregistered.json \
  --plan-sha256 docs/experiments/dataset1-k256-static-setwise-dual-horizon.preregistered.json.sha256 \
  --selection-lock result/dataset1_k256_static_setwise_dual_horizon_20260730/internal/selection-lock.json \
  --selection-lock-sha256 result/dataset1_k256_static_setwise_dual_horizon_20260730/internal/selection-lock.sha256 \
  --selection-report result/dataset1_k256_static_setwise_dual_horizon_20260730/internal/selection-report.json \
  --source-checkpoint checkpoints/d1_time_ramp_g050_d2_cooccur_lift_aux_w050_20260729.pkl \
  --train-cache-prefix cache/supervised_features/dataset1_k256_static_setwise_recent200k_full100_20260730 \
  --train-cache-report result/dataset1_k256_static_setwise_dual_horizon_20260730/cache/train-cache-report.json \
  --external-cache-prefix cache/supervised_features/dataset1_k256_static_setwise_recent200k_full100_val_20260730 \
  --external-cache-report result/dataset1_k256_static_setwise_dual_horizon_20260730/cache/validation-cache-report.json \
  --reference-external-report result/dataset1_joint_recent200k_full100_seed60_20260726/validation-cache-report.json \
  --output-dir result/dataset1_k256_static_setwise_dual_horizon_20260730/external
