#!/usr/bin/env bash
set -euo pipefail

cd /home/edu/workspace/jittor-GPNUT1-JGRec

exec .venv/bin/python \
  scripts/retrain_dataset2_pure_jittor_oof_stable_meta.py \
  --base-experiment-dir \
    result/dataset2_pure_jittor_oof_stacking_20260726 \
  --output-dir \
    result/dataset2_pure_jittor_oof_stacking_stable_v2_tieneutral_20260727 \
  --champion-validation-scores \
    result/dataset2_conservative_window_blend_20260726/artifacts/validation-conservative-blend.npy \
  --meta-epochs 12 \
  --meta-batch-size 512 \
  --learning-rate 0.001 \
  --seed 60 \
  --minimum-full-delta 0.0002
