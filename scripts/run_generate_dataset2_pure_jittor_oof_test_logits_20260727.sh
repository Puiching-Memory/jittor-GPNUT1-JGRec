#!/usr/bin/env bash
set -euo pipefail

cd /home/edu/workspace/jittor-GPNUT1-JGRec

exec .venv/bin/python \
  scripts/generate_dataset2_pure_jittor_oof_test_logits.py \
  --source-checkpoint \
    checkpoints/d1_time_ramp_g050_d2_window_conservative_a030_seed60_20260726.pkl \
  --base-experiment-dir \
    result/dataset2_pure_jittor_oof_stacking_20260726 \
  --stable-experiment-dir \
    result/dataset2_pure_jittor_oof_stacking_stable_v2_tieneutral_20260727 \
  --output-dir \
    result/dataset2_pure_jittor_oof_stacking_test_logits_20260727 \
  --data-dir data \
  --batch-size 512
