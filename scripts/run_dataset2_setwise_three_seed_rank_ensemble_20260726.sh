#!/usr/bin/env bash
set -euo pipefail

root="/home/edu/workspace/jittor-GPNUT1-JGRec"
tag="dataset2_setwise_three_seed_rank_ensemble_20260726"
output="$root/result/$tag"
log_path="$root/logs/$tag.log"
pid_path="$output/pipeline.pid"
exit_path="$output/pipeline.exit"

for protected in "$output" "$log_path"
do
  if [[ -e "$protected" ]]; then
    echo "refusing to overwrite existing artifact: $protected" >&2
    exit 1
  fi
done

mkdir -p "$output" "$root/logs"
setsid -f bash -lc "
  printf '%s\n' \$\$ > '$pid_path'
  cd '$root'
  source .workspace-env.sh
  set +e
  uv run --no-sync python scripts/train_evaluate_dataset2_setwise_three_seed.py \
    --checkpoint checkpoints/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722.pkl \
    --train-cache-prefix cache/supervised_features/dataset2_joint_recent200k_full100_seed60_20260725 \
    --train-cache-report result/dataset2_joint_recent200k_full100_seed60_20260725/train-cache-report.json \
    --validation-cache-prefix cache/supervised_features/dataset2_joint_recent200k_full100_val_seed60_20260725 \
    --validation-cache-report result/dataset2_joint_recent200k_full100_seed60_20260725/validation-cache-report.json \
    --source-evaluation-report result/dataset2_joint_lgbm_setwise_seed60_20260725/artifacts/evaluation-report.json \
    --seed60-model result/dataset2_joint_lgbm_setwise_seed60_20260725/artifacts/dataset2-setwise.npz \
    --output-dir result/$tag/artifacts \
    --setwise-weight 0.80 \
    --min-full-delta 0.001 \
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
