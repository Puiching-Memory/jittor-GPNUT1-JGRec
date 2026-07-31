#!/usr/bin/env bash
set -euo pipefail

root="/home/edu/workspace/jittor-GPNUT1-JGRec"
tag="dataset2_activity_adaptive_multi_interest_seed60_20260726"
run_dir="$root/result/$tag"
artifact_dir="$run_dir/artifacts"
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
    scripts/evaluate_dataset2_activity_adaptive_multi_interest.py \
    --checkpoint \
      checkpoints/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722.pkl \
    --train-csv data/dataset2/train.csv \
    --train-cache-prefix \
      cache/supervised_features/dataset2_joint_recent200k_full100_seed60_20260725 \
    --train-cache-report \
      result/dataset2_joint_recent200k_full100_seed60_20260725/train-cache-report.json \
    --validation-cache-prefix \
      cache/supervised_features/dataset2_joint_recent200k_full100_val_seed60_20260725 \
    --validation-cache-report \
      result/dataset2_joint_recent200k_full100_seed60_20260725/validation-cache-report.json \
    --old-proxy-train \
      result/dataset2_multi_interest_proxy_seed60_20260725/artifacts/multi-interest.train.npy \
    --old-proxy-validation \
      result/dataset2_multi_interest_proxy_seed60_20260725/artifacts/multi-interest.val.npy \
    --old-proxy-report \
      result/dataset2_multi_interest_proxy_seed60_20260725/artifacts/multi-interest-report.json \
    --champion-setwise-model \
      result/dataset2_joint_lgbm_setwise_seed60_20260725/artifacts/dataset2-setwise.npz \
    --old-multi-interest-setwise-model \
      result/dataset2_multi_interest_proxy_seed60_20260725/artifacts/dataset2-multi-interest-setwise.npz \
    --output-dir '$artifact_dir' \
    --seed 60 \
    --feature-batch-rows 512 \
    --fusion-batch-size 256 \
    --fusion-epochs 10 \
    --fusion-patience 2 \
    --setwise-weight 0.80 \
    --minimum-slice1-delta 0.001 \
    --minimum-q12-delta -0.001
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
