#!/usr/bin/env bash
set -uo pipefail

cd /home/edu/workspace/jittor-GPNUT1-JGRec
source .workspace-env.sh

output_dir="result/dataset2_two_hop_decay_full100_seed60_20260723"
log_file="logs/dataset2_two_hop_decay_full100_seed60_20260723.log"
exit_file="logs/dataset2_two_hop_decay_full100_seed60_20260723.exit"

uv run --no-sync python scripts/evaluate_dataset2_two_hop_decay_full100.py \
  --checkpoint checkpoints/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722.pkl \
  --cache-prefix cache/supervised_features/4baa722bf26e5d50356da26ac5f479cb54324ddb \
  --train-csv data/dataset2/train.csv \
  --test-csv data/dataset2/test.csv \
  --output-dir "$output_dir" \
  --boost-rounds 308 \
  --mlp-weight 0.07 \
  --min-full-delta 0.002 \
  --decay-ratio 0.05 \
  --source-history-limit 64 \
  --seed 60 \
  --num-threads 16 \
  >"$log_file" 2>&1
status=$?
printf '%s\n' "$status" >"$exit_file"
exit "$status"
