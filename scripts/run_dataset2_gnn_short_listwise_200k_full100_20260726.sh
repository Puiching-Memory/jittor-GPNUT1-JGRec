#!/usr/bin/env bash
set -euo pipefail

root="/home/edu/workspace/jittor-GPNUT1-JGRec"
tag="dataset2_gnn_short_listwise_200k_full100_seed60_20260726_v2"
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
  uv run --no-sync python scripts/train_dataset2_gnn_short_listwise_200k_full100.py \
    --checkpoint checkpoints/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722.pkl \
    --train-csv data/dataset2/train.csv \
    --train-prefix cache/supervised_features/dataset2_joint_recent200k_full100_seed60_20260725 \
    --validation-prefix cache/supervised_features/dataset2_joint_recent200k_full100_val_seed60_20260725 \
    --output-dir '$run_dir/artifacts' \
    --seed 60 \
    --epochs 50 \
    --patience 3 \
    --batch-size 512 \
    --validation-batch-size 256 \
    --progress-every 25
  status=\$?
  printf '%s\n' \"\$status\" > '$exit_path'
  exit \"\$status\"
" >"$log_path" 2>&1 </dev/null &

for _ in {1..30}; do
  if [[ -s "$pid_path" ]]; then
    echo "started pid=$(cat "$pid_path") log=$log_path"
    exit 0
  fi
  sleep 0.1
done
echo "detached worker did not publish its pid" >&2
exit 1
