#!/usr/bin/env bash
set -uo pipefail

cd /home/edu/workspace/jittor-GPNUT1-JGRec
source .workspace-env.sh

validation_dir="result/dataset2_two_hop_decay_full100_seed60_20260723"
output_dir="result/d1_champion_d2_twohop_decay_seed60_20260723"
output_checkpoint="checkpoints/d1_champion_d2_twohop_decay_seed60_20260723.pkl"
log_file="logs/build_dataset2_two_hop_decay_candidate_20260723.log"
exit_file="logs/build_dataset2_two_hop_decay_candidate_20260723.exit"

uv run --no-sync python scripts/build_dataset2_two_hop_decay_candidate.py \
  --source-checkpoint checkpoints/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722.pkl \
  --evaluation-report "$validation_dir/full100-report.json" \
  --lgbm-model "$validation_dir/dataset2-two-hop-decay-lgbm.txt" \
  --champion-dataset1 result/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722/csv/dataset1.csv \
  --dataset2-train data/dataset2/train.csv \
  --output-checkpoint "$output_checkpoint" \
  --output-dir "$output_dir" \
  --data-dir data \
  --batch-size 2048 \
  >"$log_file" 2>&1
status=$?
printf '%s\n' "$status" >"$exit_file"
exit "$status"
