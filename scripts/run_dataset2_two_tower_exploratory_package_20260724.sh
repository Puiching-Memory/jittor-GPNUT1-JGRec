#!/usr/bin/env bash
set -uo pipefail

cd /home/edu/workspace/jittor-GPNUT1-JGRec
source .workspace-env.sh

run_name="d1_champion_d2_twotower200k_exploratory_seed60_20260724"
log_file="logs/${run_name}.log"
exit_file="logs/${run_name}.exit"

mkdir -p logs
uv run --no-sync python scripts/build_dataset2_lgbm_tuned_candidate.py \
  --source-checkpoint checkpoints/d1_champion_d2_twotower200k_base_seed60_20260724.pkl \
  --tuning-report result/d2_twotower200k_lgbm_tune_seed60_20260724/tuning-report.json \
  --lgbm-model result/d2_twotower200k_lgbm_tune_seed60_20260724/dataset2-lgbm.txt \
  --champion-dataset1 result/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722/csv/dataset1.csv \
  --output-checkpoint checkpoints/${run_name}.pkl \
  --output-dir result/${run_name} \
  --data-dir data \
  --batch-size 2048 \
  --allow-rejected-tuning \
  >"$log_file" 2>&1
status=$?

printf '%s\n' "$status" >"$exit_file"
exit "$status"
