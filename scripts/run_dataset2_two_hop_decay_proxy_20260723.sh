#!/usr/bin/env bash
set -uo pipefail

cd /home/edu/workspace/jittor-GPNUT1-JGRec
source .workspace-env.sh

output_dir="result/dataset2_two_hop_decay_proxy_seed60_20260723"
exit_file="logs/dataset2_two_hop_decay_proxy_seed60_20260723.exit"

uv run --no-sync python scripts/evaluate_dataset2_two_hop_decay_proxy.py \
  --checkpoint checkpoints/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722.pkl \
  --train-csv data/dataset2/train.csv \
  --test-csv data/dataset2/test.csv \
  --output-dir "$output_dir"
status=$?
printf '%s\n' "$status" > "$exit_file"
exit "$status"
