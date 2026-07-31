#!/usr/bin/env bash
set -uo pipefail

cd /home/edu/workspace/jittor-GPNUT1-JGRec
source .workspace-env.sh

run_name="d2_twotower200k_lgbm_tune_seed60_20260724"
log_file="logs/${run_name}.log"
exit_file="logs/${run_name}.exit"
output_dir="result/${run_name}"

mkdir -p logs
uv run --no-sync python scripts/tune_dataset2_lgbm_cached.py \
  --checkpoint checkpoints/d2_twotower200k_listwise_fullreranker_seed60_20260724.pkl.tmp \
  --cache-prefix cache/supervised_features/twotower200k_listwise_seed60_20260724/453b7d812e6e3e23c373957ddeefb5d1f9802333 \
  --output-dir "$output_dir" \
  --tune-rows 10000 \
  --seed 60 \
  --num-threads 16 \
  --num-boost-round 800 \
  --early-stopping-rounds 60 \
  >"$log_file" 2>&1
status=$?

printf '%s\n' "$status" >"$exit_file"
exit "$status"
