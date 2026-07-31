#!/usr/bin/env bash
set -euo pipefail

root="/home/edu/workspace/jittor-GPNUT1-JGRec"
tag="dataset1_time_ramped_setwise_blend_20260726"
run_dir="$root/result/$tag"
log_path="$root/logs/$tag.log"
pid_path="$run_dir/pipeline.pid"
exit_path="$run_dir/pipeline.exit"

for protected in "$run_dir" "$log_path"; do
  if [[ -e "$protected" ]]; then
    echo "refusing to overwrite existing artifact: $protected" >&2
    exit 1
  fi
done

mkdir -p "$run_dir" "$root/logs"
setsid -f bash -lc "
  printf '%s\n' \$\$ > '$pid_path'
  cd '$root'
  source .workspace-env.sh
  set +e
  uv run --no-sync python \
    scripts/select_dataset1_time_ramped_setwise_blend.py \
    --checkpoint \
      checkpoints/d1_champion_d2_setwise_w080_seed60_20260725.pkl \
    --validation-features \
      cache/supervised_features/dataset1_joint_recent200k_full100_val_seed60_20260726.val.npy \
    --validation-times \
      cache/supervised_features/dataset1_joint_recent200k_full100_val_seed60_20260726.val-time.npy \
    --validation-cache-report \
      result/dataset1_joint_recent200k_full100_seed60_20260726/validation-cache-report.json \
    --setwise-prediction \
      result/dataset1_full100_setwise_seed60_20260726/validation-setwise-recent_100k.npy \
    --source-evaluation-report \
      result/dataset1_full100_setwise_seed60_20260726/evaluation-report.json \
    --output-dir '$run_dir/artifacts' \
    --batch-size 512 \
    --minimum-prefix-delta 0.0002
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
