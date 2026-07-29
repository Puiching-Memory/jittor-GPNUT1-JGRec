#!/usr/bin/env bash
set -uo pipefail

cd /home/edu/workspace/jittor-GPNUT1-JGRec
source .workspace-env.sh

run_name="dataset2_two_tower_listwise_200k_seed60_20260724"
output_dir="result/${run_name}"
log_file="logs/${run_name}.log"
exit_file="logs/${run_name}.exit"

mkdir -p result logs
uv run --no-sync python scripts/evaluate_dataset2_two_tower_listwise_200k.py \
  --train-csv data/dataset2/train.csv \
  --test-csv data/dataset2/test.csv \
  --full100-prefix cache/supervised_features/dataset2_full100_matched_train_seed60_20260723 \
  --cache-report result/dataset2_matched_full100_seed60_20260723/cache-build-report.json \
  --output-dir "$output_dir" \
  --seed 60 \
  --val-ratio 0.15 \
  --context-ratio 0.75 \
  --validation-queries 20000 \
  --epochs 50 \
  --patience 3 \
  --batch-size 512 \
  --sampling-workers 16 \
  --candidate-max-samples 200000 \
  --candidate-negatives 99 \
  --min-full-delta 0.002 \
  >"$log_file" 2>&1
status=$?

printf '%s\n' "$status" >"$exit_file"
exit "$status"
