#!/usr/bin/env bash
set -euo pipefail

root="/home/edu/workspace/jittor-GPNUT1-JGRec"
training_pid="553416"
training_tag="dataset2_matched_lgbm_setwise_seed60_20260724"
candidate_tag="d1_champion_d2_matched_reranker_seed60_20260724"
supervisor_dir="$root/result/dataset2_matched_conditional_package_seed60_20260724"
output_dir="$root/result/$candidate_tag"
checkpoint="$root/checkpoints/$candidate_tag.pkl"
log_path="$root/logs/$candidate_tag.log"
pid_path="$supervisor_dir/package-supervisor.pid"

for protected in \
  "$supervisor_dir" \
  "$output_dir" \
  "$checkpoint" \
  "$checkpoint.tmp" \
  "$log_path"
do
  if [[ -e "$protected" ]]; then
    echo "refusing to overwrite existing artifact: $protected" >&2
    exit 1
  fi
done

mkdir -p "$supervisor_dir" "$root/logs"
setsid -f bash -lc "
  printf '%s\n' \$\$ > '$pid_path'
  while kill -0 '$training_pid' 2>/dev/null; do
    sleep 30
  done
  cd '$root'
  source .workspace-env.sh
  exec uv run --no-sync python scripts/build_dataset2_matched_reranker_candidate.py \
    --source-checkpoint checkpoints/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722.pkl \
    --evaluation-report result/$training_tag/artifacts/evaluation-report.json \
    --lgbm-model result/$training_tag/artifacts/dataset2-matched-lgbm.txt \
    --setwise-model result/$training_tag/artifacts/dataset2-setwise.npz \
    --champion-dataset1 result/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722/csv/dataset1.csv \
    --output-checkpoint checkpoints/$candidate_tag.pkl \
    --output-dir result/$candidate_tag \
    --data-dir data \
    --batch-size 2048
" >"$log_path" 2>&1 </dev/null &

for _ in {1..20}; do
  if [[ -s "$pid_path" ]]; then
    echo "started supervisor pid=$(cat "$pid_path") log=$log_path"
    exit 0
  fi
  sleep 0.1
done
echo "detached supervisor did not publish its pid" >&2
exit 1
