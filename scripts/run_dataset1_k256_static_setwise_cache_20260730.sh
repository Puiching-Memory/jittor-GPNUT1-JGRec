#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3
export CUDA_VISIBLE_DEVICES=0

exec .venv/bin/python scripts/build_dataset1_full100_train_cache.py \
  --checkpoint checkpoints/d1_time_ramp_g050_d2_cooccur_lift_aux_w050_20260729.pkl \
  --train-csv data/dataset1/train.csv \
  --output-prefix cache/supervised_features/dataset1_k256_static_setwise_recent200k_full100_20260730 \
  --report result/dataset1_k256_static_setwise_dual_horizon_20260730/cache/train-cache-report.json \
  --validation-output-prefix cache/supervised_features/dataset1_k256_static_setwise_recent200k_full100_val_20260730 \
  --validation-report result/dataset1_k256_static_setwise_dual_horizon_20260730/cache/validation-cache-report.json \
  --train-rows 200000 \
  --validation-rows 20000 \
  --train-selection recent \
  --batch-rows 4096 \
  --structure-workers 8 \
  --comparison-structure-workers 4 \
  --minimum-parallel-speedup 1.05 \
  --minimum-comparison-speedup 1.05 \
  --minimum-memory-reserve-gib 8 \
  --prediction-history-limit 256
