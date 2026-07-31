#!/usr/bin/env bash
set -uo pipefail

cd /home/edu/workspace/jittor-GPNUT1-JGRec
source .workspace-env.sh

output_dir="result/dataset2_matched_full100_seed60_20260723"
full_prefix="cache/supervised_features/dataset2_full100_matched_train_seed60_20260723"
source_prefix="cache/supervised_features/4baa722bf26e5d50356da26ac5f479cb54324ddb"
checkpoint="checkpoints/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722.pkl"
replay_report="result/dataset2_full100_train_seed60_20260723/replay-report.json"
log_file="logs/dataset2_matched_full100_seed60_20260723.log"
exit_file="logs/dataset2_matched_full100_seed60_20260723.exit"

mkdir -p "$output_dir" logs cache/supervised_features
uv run --no-sync python scripts/build_dataset2_full100_train_cache.py \
  --checkpoint "$checkpoint" \
  --source-cache-prefix "$source_prefix" \
  --train-csv data/dataset2/train.csv \
  --replay-report "$replay_report" \
  --output-prefix "$full_prefix" \
  --report "$output_dir/cache-build-report.json" \
  --candidate-count 100 \
  --batch-rows 4096 \
  >"$log_file" 2>&1
status=$?

if [ "$status" -eq 0 ]; then
  uv run --no-sync python scripts/evaluate_dataset2_matched_train_caches.py \
    --checkpoint "$checkpoint" \
    --source-cache-prefix "$source_prefix" \
    --full100-prefix "$full_prefix" \
    --cache-report "$output_dir/cache-build-report.json" \
    --output-dir "$output_dir" \
    --boost-rounds 308 \
    --mlp-weight 0.07 \
    --min-full-delta 0.002 \
    --seed 60 \
    --num-threads 16 \
    >>"$log_file" 2>&1
  status=$?
fi

printf '%s\n' "$status" >"$exit_file"
exit "$status"
