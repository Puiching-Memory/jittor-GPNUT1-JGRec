#!/usr/bin/env bash
set -euo pipefail

cd /home/edu/workspace/jittor-GPNUT1-JGRec

.venv/bin/python scripts/train_dataset2_high_confidence_topk_router.py \
  --oof-dir \
    result/dataset2_bounded_source_multi_horizon_oof_20260727 \
  --train-cache-prefix \
    cache/supervised_features/dataset2_joint_recent200k_full100_seed60_20260725 \
  --train-cache-report \
    result/dataset2_joint_recent200k_full100_seed60_20260725/train-cache-report.json \
  --output-dir \
    result/dataset2_high_confidence_topk_residual_router_v2_20260727 \
  --hidden-dim 128 \
  --epochs 30 \
  --nonzero-weight 64 \
  --device cuda
