#!/usr/bin/env bash
set -euo pipefail

root="/home/edu/workspace/jittor-GPNUT1-JGRec"
tag="dataset2_setwise_window_diversity_20260726"
output="$root/result/$tag"
log_path="$root/logs/$tag.log"
pid_path="$root/logs/$tag.pid"
exit_path="$root/logs/$tag.exit"

for protected in "$output" "$log_path" "$pid_path" "$exit_path"
do
  if [[ -e "$protected" ]]; then
    echo "refusing to overwrite existing artifact: $protected" >&2
    exit 1
  fi
done

mkdir -p "$root/logs"
setsid -f bash -lc "
  printf '%s\n' \$\$ > '$pid_path'
  cd '$root'
  source .workspace-env.sh
  set +e
  uv run --no-sync python scripts/run_dataset2_setwise_window_diversity.py \
    train-select \
    --checkpoint checkpoints/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722.pkl \
    --train-cache-prefix cache/supervised_features/dataset2_joint_recent200k_full100_seed60_20260725 \
    --train-cache-report result/dataset2_joint_recent200k_full100_seed60_20260725/train-cache-report.json \
    --validation-cache-prefix cache/supervised_features/dataset2_joint_recent200k_full100_val_seed60_20260725 \
    --validation-cache-report result/dataset2_joint_recent200k_full100_seed60_20260725/validation-cache-report.json \
    --champion-evaluation-report result/d1_champion_d2_setwise_w080_seed60_20260725/evaluation-report.json \
    --recent200k-model result/dataset2_joint_lgbm_setwise_seed60_20260725/artifacts/dataset2-setwise.npz \
    --frozen-dataset1-csv result/d1_champion_d2_setwise_w080_seed60_20260725/csv/dataset1.csv \
    --output-dir result/$tag \
    --setwise-epochs 10 \
    --setwise-patience 2 \
    --setwise-batch-size 256 \
    --setwise-hidden-dim 32 \
    --setwise-learning-rate 0.001
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
