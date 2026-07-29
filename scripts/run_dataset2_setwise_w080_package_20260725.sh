#!/usr/bin/env bash
set -euo pipefail

root="/home/edu/workspace/jittor-GPNUT1-JGRec"
supervisor_tag="dataset2_setwise_w080_package_seed60_20260725"
candidate_tag="d1_champion_d2_setwise_w080_seed60_20260725"
supervisor_dir="$root/result/$supervisor_tag"
candidate_dir="$root/result/$candidate_tag"
checkpoint="$root/checkpoints/$candidate_tag.pkl"
log_path="$root/logs/$candidate_tag.log"
pid_path="$supervisor_dir/pipeline.pid"
exit_path="$supervisor_dir/pipeline.exit"

for protected in \
  "$supervisor_dir" \
  "$candidate_dir" \
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
  cd '$root'
  source .workspace-env.sh
  set +e
  uv run --no-sync python scripts/build_dataset2_matched_reranker_candidate.py \
    --source-checkpoint checkpoints/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722.pkl \
    --evaluation-report result/dataset2_setwise_high_weight_scan_seed60_20260725/artifacts/setwise-high-weight-report.json \
    --lgbm-model result/dataset2_joint_lgbm_setwise_seed60_20260725/artifacts/dataset2-matched-lgbm.txt \
    --setwise-model result/dataset2_joint_lgbm_setwise_seed60_20260725/artifacts/dataset2-setwise.npz \
    --champion-dataset1 result/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722/csv/dataset1.csv \
    --output-checkpoint checkpoints/$candidate_tag.pkl \
    --output-dir result/$candidate_tag \
    --data-dir data \
    --batch-size 2048
  status=\$?
  printf '%s\n' \"\$status\" > '$exit_path'
  exit \"\$status\"
" >"$log_path" 2>&1 </dev/null &

for _ in {1..20}; do
  if [[ -s "$pid_path" ]]; then
    echo "started pid=$(cat "$pid_path") log=$log_path"
    exit 0
  fi
  sleep 0.1
done
echo "detached worker did not publish its pid" >&2
exit 1
