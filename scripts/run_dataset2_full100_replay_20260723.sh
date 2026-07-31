#!/usr/bin/env bash
set -uo pipefail

cd /home/edu/workspace/jittor-GPNUT1-JGRec
source .workspace-env.sh

output_dir="result/dataset2_full100_train_seed60_20260723"
log_file="logs/dataset2_full100_replay_seed60_20260723.log"
exit_file="logs/dataset2_full100_replay_seed60_20260723.exit"

mkdir -p "$output_dir" logs
uv run --no-sync python scripts/preflight_dataset2_full100_replay.py \
  --checkpoint checkpoints/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722.pkl \
  --cache-prefix cache/supervised_features/4baa722bf26e5d50356da26ac5f479cb54324ddb \
  --train-csv data/dataset2/train.csv \
  --output "$output_dir/replay-report.json" \
  --replay-rows 4096 \
  >"$log_file" 2>&1
status=$?
printf '%s\n' "$status" >"$exit_file"
exit "$status"
