#!/usr/bin/env bash
set -euo pipefail

cd /home/edu/workspace/jittor-GPNUT1-JGRec

.venv/bin/python \
  scripts/generate_dataset2_bounded_source_multi_horizon_oof.py \
  --train-csv data/dataset2/train.csv \
  --train-cache-prefix \
    cache/supervised_features/dataset2_joint_recent200k_full100_seed60_20260725 \
  --sequence-cache-dir \
    cache/source_conditioned/dataset2_abcd_recent200k_full100_20260727 \
  --base-result-dir \
    result/dataset2_source_conditioned_cst_abcd_20260727 \
  --decoder-result-dir \
    result/dataset2_bounded_source_decoder_20260727 \
  --output-dir \
    result/dataset2_bounded_source_multi_horizon_oof_20260727 \
  --device cuda
