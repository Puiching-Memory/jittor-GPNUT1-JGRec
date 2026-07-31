#!/usr/bin/env bash
set -euo pipefail

root="/home/edu/workspace/jittor-GPNUT1-JGRec"
tag="dataset2_partial_listwise_expert_blend_20260728"
run_dir="$root/result/$tag"
log_path="$root/logs/$tag-score.log"
pid_path="$run_dir/score.pid"
exit_path="$run_dir/score.exit"

for protected in \
  "$log_path" \
  "$pid_path" \
  "$exit_path" \
  "$run_dir/champion-probabilities.npy" \
  "$run_dir/listwise-mlp-probabilities.npy" \
  "$run_dir/listwise-two-tower-probabilities.npy" \
  "$run_dir/score-report.json"
do
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
  uv run --no-sync python scripts/score_dataset2_partial_listwise_experts.py \
    --checkpoint checkpoints/d1_time_ramp_g050_d2_gnn_short_none_e50_edges40000_setwise_w080_seed60_20260727.pkl \
    --validation-cache-prefix cache/supervised_features/dataset2_joint_recent200k_full100_val_seed60_20260725 \
    --validation-cache-report result/dataset2_joint_recent200k_full100_seed60_20260725/validation-cache-report.json \
    --short-none-scores result/dataset2_targeted_gnn_edges_seed60_20260725/artifacts/short_none.val-scores.npy \
    --listwise-mlp-model result/dataset2_listwise_mlp_seed60_20260723/dataset2-listwise-mlp.npz \
    --listwise-two-tower-model result/dataset2_two_tower_listwise_200k_seed60_20260724/candidate-model.npz \
    --two-tower-report result/dataset2_two_tower_listwise_200k_seed60_20260724/evaluation-report.json \
    --train-csv data/dataset2/train.csv \
    --frozen-config '$run_dir/frozen-config.json' \
    --output-dir '$run_dir' \
    --batch-size 256
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
