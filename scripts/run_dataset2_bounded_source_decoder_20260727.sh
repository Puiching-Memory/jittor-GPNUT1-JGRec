#!/usr/bin/env bash
set -euo pipefail

cd /home/edu/workspace/jittor-GPNUT1-JGRec

.venv/bin/python scripts/train_dataset2_bounded_source_decoder.py \
  --phase all \
  --train-cache-prefix \
    cache/supervised_features/dataset2_joint_recent200k_full100_seed60_20260725 \
  --sequence-cache-dir \
    cache/source_conditioned/dataset2_abcd_recent200k_full100_20260727 \
  --base-result-dir \
    result/dataset2_source_conditioned_cst_abcd_20260727 \
  --base-cache-dir \
    cache/bounded_id_residual/dataset2_frozen_a_20260727 \
  --validation-cache-prefix \
    cache/supervised_features/dataset2_joint_recent200k_full100_val_seed60_20260725 \
  --full-base-validation-expert-logits \
    result/dataset2_pure_jittor_oof_stacking_20260726/full-validation-expert-logits.npy \
  --champion-validation-scores \
    result/dataset2_conservative_window_blend_20260726/artifacts/validation-conservative-blend.npy \
  --output-dir \
    result/dataset2_bounded_source_decoder_20260727 \
  --device cuda
