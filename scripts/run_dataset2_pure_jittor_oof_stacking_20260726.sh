#!/usr/bin/env bash
set -euo pipefail

cd /home/edu/workspace/jittor-GPNUT1-JGRec

exec .venv/bin/python scripts/train_dataset2_pure_jittor_oof_stacking.py \
  --phase all \
  --train-cache-prefix cache/supervised_features/dataset2_joint_recent200k_full100_seed60_20260725 \
  --validation-cache-prefix cache/supervised_features/dataset2_joint_recent200k_full100_val_seed60_20260725 \
  --train-cache-report result/dataset2_joint_recent200k_full100_seed60_20260725/train-cache-report.json \
  --validation-cache-report result/dataset2_joint_recent200k_full100_seed60_20260725/validation-cache-report.json \
  --champion-validation-scores result/dataset2_conservative_window_blend_20260726/artifacts/validation-conservative-blend.npy \
  --output-dir result/dataset2_pure_jittor_oof_stacking_20260726 \
  --warmup-rows 40000 \
  --fold-rows 40000 \
  --fold-count 4 \
  --meta-train-fold-count 3 \
  --expert-validation-rows 8000 \
  --expert-epochs 6 \
  --expert-batch-size 256 \
  --meta-epochs 12 \
  --meta-batch-size 512 \
  --learning-rate 0.001 \
  --seed 60 \
  --minimum-full-delta 0.0002 \
  --device cuda
