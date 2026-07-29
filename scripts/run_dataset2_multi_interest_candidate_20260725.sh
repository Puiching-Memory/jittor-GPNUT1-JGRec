#!/usr/bin/env bash
set -euo pipefail

root="/home/edu/workspace/jittor-GPNUT1-JGRec"
tag="d1_champion_d2_multi_interest_proxy_v3_seed60_20260725"
run_dir="$root/result/$tag"
checkpoint="$root/checkpoints/$tag.pkl"
log_path="$root/logs/$tag.log"
pid_path="$root/result/dataset2_multi_interest_proxy_seed60_20260725/package_v3.pid"
exit_path="$root/result/dataset2_multi_interest_proxy_seed60_20260725/package_v3.exit"

for protected in "$run_dir" "$checkpoint" "$checkpoint.tmp" "$log_path"; do
  if [[ -e "$protected" ]]; then
    echo "refusing to overwrite existing artifact: $protected" >&2
    exit 1
  fi
done

setsid -f bash -lc "
  printf '%s\n' \$\$ > '$pid_path'
  cd '$root'
  source .workspace-env.sh
  set +e
  uv run --no-sync python scripts/build_dataset2_multi_interest_candidate.py \
    --source-checkpoint checkpoints/d1_champion_d2_setwise_w080_seed60_20260725.pkl \
    --proxy-report result/dataset2_multi_interest_proxy_seed60_20260725/artifacts/multi-interest-report.json \
    --proxy-model result/dataset2_multi_interest_proxy_seed60_20260725/artifacts/dataset2-multi-interest-setwise.npz \
    --validation-features cache/supervised_features/dataset2_joint_recent200k_full100_val_seed60_20260725.val.npy \
    --validation-proxy result/dataset2_multi_interest_proxy_seed60_20260725/artifacts/multi-interest.val.npy \
    --validation-cache-prefix cache/supervised_features/dataset2_joint_recent200k_full100_val_seed60_20260725 \
    --train-csv data/dataset2/train.csv \
    --champion-dataset1 result/d1_champion_d2_setwise_w080_seed60_20260725/csv/dataset1.csv \
    --output-checkpoint checkpoints/$tag.pkl \
    --output-dir result/$tag \
    --data-dir data \
    --seed 60 \
    --history-limit 64 \
    --validation-prefix-rows 1922091 \
    --validation-proxy-seed 2160 \
    --final-proxy-seed 3160 \
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
