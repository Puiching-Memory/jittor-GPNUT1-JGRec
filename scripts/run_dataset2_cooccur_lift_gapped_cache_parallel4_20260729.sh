#!/usr/bin/env bash
set -euo pipefail

root="/home/edu/workspace/jittor-GPNUT1-JGRec"
run_root="$root/result/dataset2_cooccur_lift_successor_v2_duel_20260729"
output_dir="$run_root/gapped-cache-v2-parallel4"
pid_path="$run_root/gapped-cache-v2-parallel4.pid"

cd "$root"
printf '%s\n' "$$" > "$pid_path"
exec nice -n 10 ionice -c 2 -n 7 "$root/.venv/bin/python" \
  scripts/build_dataset2_cooccur_lift_gapped_cache.py \
  --checkpoint checkpoints/d1_time_ramp_g050_d2_cooccur_lift_aux_w050_20260729.pkl \
  --train-csv data/dataset2/train.csv \
  --validation-plan docs/experiments/cooccur-lift-successor-v2-duel.validation-plan.json \
  --output-dir "$output_dir" \
  --batch-rows 4096 \
  --structure-workers 4 \
  --minimum-parallel-speedup 1.5
