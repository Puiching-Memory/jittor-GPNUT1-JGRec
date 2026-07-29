#!/usr/bin/env bash
set -euo pipefail

root="/home/edu/workspace/jittor-GPNUT1-JGRec"
tag="dataset2_setwise_prob_seed_bag_v1_20260728"
run_dir="$root/result/$tag"
artifact_dir="$run_dir/artifacts"
selection_dir="$run_dir/rolling-selection"
log_path="$root/logs/$tag.log"
pid_path="$run_dir/pipeline.pid"
exit_path="$run_dir/pipeline.exit"
stage_path="$run_dir/pipeline.stage"

for protected in "$run_dir" "$log_path"; do
  if [[ -e "$protected" ]]; then
    echo "refusing to overwrite existing artifact: $protected" >&2
    exit 1
  fi
done

mkdir -p "$run_dir" "$root/logs"
setsid -f bash -lc "
  printf '%s\n' \$\$ > '$pid_path'
  printf '%s\n' training > '$stage_path'
  cd '$root'
  source .workspace-env.sh
  set +e
  uv run --no-sync python \
    scripts/train_dataset2_setwise_prob_seed_bag_rolling.py \
    --frozen-config \
      docs/experiments/setwise-prob-seed-bag-v1.frozen.json \
    --checkpoint \
      checkpoints/d1_time_ramp_g050_d2_gnn_short_none_e50_edges40000_setwise_w080_seed60_20260727.pkl \
    --train-cache-prefix \
      cache/supervised_features/dataset2_joint_recent200k_full100_seed60_20260725 \
    --train-cache-report \
      result/dataset2_joint_recent200k_full100_seed60_20260725/train-cache-report.json \
    --short-none-scores \
      result/dataset2_targeted_gnn_edges_seed60_20260725/artifacts/short_none.train-scores.npy \
    --source-rolling-manifest \
      result/dataset2_listwise_mlp_exact_rolling_20260728/artifacts/rolling-manifest.json \
    --output-dir '$artifact_dir'
  producer_status=\$?
  if [[ \$producer_status -ne 0 ]]; then
    printf '%s\n' producer_failed > '$stage_path'
    printf '%s\n' \"\$producer_status\" > '$exit_path'
    exit \"\$producer_status\"
  fi

  printf '%s\n' rolling_selection > '$stage_path'
  uv run --no-sync python scripts/select_robust_integrated_weight.py \
    --manifest '$artifact_dir/rolling-manifest.json' \
    --output-dir '$selection_dir'
  selection_status=\$?
  if [[ \$selection_status -eq 0 ]]; then
    printf '%s\n' rolling_selected_external_not_opened > '$stage_path'
  elif [[ \$selection_status -eq 2 ]]; then
    printf '%s\n' rolling_rejected_closed_no_external > '$stage_path'
  else
    printf '%s\n' rolling_selector_failed > '$stage_path'
  fi
  printf '%s\n' \"\$selection_status\" > '$exit_path'
  exit \"\$selection_status\"
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
