#!/usr/bin/env bash
set -euo pipefail

cd /home/edu/workspace/jittor-GPNUT1-JGRec

.venv/bin/python scripts/train_dataset2_oof_utility_router_v3.py \
  --oof-dir \
    result/dataset2_bounded_source_multi_horizon_oof_20260727 \
  --train-cache-prefix \
    cache/supervised_features/dataset2_joint_recent200k_full100_seed60_20260725 \
  --train-cache-report \
    result/dataset2_joint_recent200k_full100_seed60_20260725/train-cache-report.json \
  --output-dir \
    result/dataset2_oof_utility_router_v3_20260727 \
  --hidden-dim 128 \
  --warmup-epochs 8 \
  --epochs 24 \
  --device cuda
