#!/usr/bin/env bash
set -uo pipefail

cd /home/edu/workspace/jittor-GPNUT1-JGRec
source .workspace-env.sh

run_name="d2_twotower200k_listwise_fullreranker_seed60_20260724"
checkpoint="checkpoints/d2_twotower200k_listwise_fullreranker_seed60_20260724.pkl"
cache_dir="cache/supervised_features/twotower200k_listwise_seed60_20260724"
log_file="logs/${run_name}.log"
exit_file="logs/${run_name}.exit"

mkdir -p logs checkpoints "$cache_dir"
uv run --no-sync jgrec-build \
  --dataset dataset2 \
  --run-name "$run_name" \
  --seed 60 \
  --selection-metric mrr \
  --max-train-events 50000 \
  --max-val-events 20000 \
  --num-negatives 31 \
  --train-num-negatives 31 \
  --val-num-negatives 99 \
  --test-candidate-negative-ratio 1.0 \
  --structure-predict-neighbor-limit 512 \
  --source-profile-predict-history-limit 512 \
  --fusion-mode ensemble \
  --two-tower-embedding-dim 64 \
  --two-tower-hidden-dim 64 \
  --two-tower-epochs 50 \
  --two-tower-max-samples 200000 \
  --two-tower-num-negatives 99 \
  --two-tower-test-candidate-negative-ratio 1.0 \
  --two-tower-objective listwise \
  --two-tower-early-stop-metric mrr \
  --supervised-feature-memmap \
  --supervised-feature-cache-dir "$cache_dir" \
  --negative-sampling-workers 16 \
  --save-checkpoint "$checkpoint" \
  >"$log_file" 2>&1
status=$?

printf '%s\n' "$status" >"$exit_file"
exit "$status"
