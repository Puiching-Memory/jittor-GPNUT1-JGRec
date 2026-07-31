#!/usr/bin/env bash
set -euo pipefail

root="/home/edu/workspace/jittor-GPNUT1-JGRec"
tag="dataset2_listwise_mlp_exact_rolling_20260728"
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
  uv run --no-sync python scripts/train_dataset2_listwise_mlp_exact_rolling.py \
    --checkpoint checkpoints/d1_time_ramp_g050_d2_gnn_short_none_e50_edges40000_setwise_w080_seed60_20260727.pkl \
    --train-cache-prefix cache/supervised_features/dataset2_joint_recent200k_full100_seed60_20260725 \
    --train-cache-report result/dataset2_joint_recent200k_full100_seed60_20260725/train-cache-report.json \
    --short-none-scores result/dataset2_targeted_gnn_edges_seed60_20260725/artifacts/short_none.train-scores.npy \
    --source-weight-config result/dataset2_partial_listwise_expert_blend_20260728/frozen-config.json \
    --output-dir '$artifact_dir'
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
