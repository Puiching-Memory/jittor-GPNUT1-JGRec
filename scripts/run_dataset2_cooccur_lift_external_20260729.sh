#!/usr/bin/env bash
set -euo pipefail

root="/home/edu/workspace/jittor-GPNUT1-JGRec"
tag="dataset2_cooccur_lift_aux_expert_v1_20260728_compact_retry2"
run_dir="$root/result/$tag"
output_dir="$run_dir/external-materialization"
log_path="$root/logs/$tag-external-materialization.log"
pid_path="$run_dir/external-materialization.pid"
exit_path="$run_dir/external-materialization.exit"

for protected in "$output_dir" "$log_path" "$pid_path" "$exit_path"; do
  if [[ -e "$protected" ]]; then
    echo "refusing to overwrite existing artifact: $protected" >&2
    exit 1
  fi
done

setsid -f bash -lc "
  printf '%s\n' \$\$ > '$pid_path'
  cd '$root'
  source .workspace-env.sh
  export CUDA_VISIBLE_DEVICES=0
  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4
  export OPENBLAS_NUM_THREADS=4
  export NUMEXPR_NUM_THREADS=4
  set +e
  uv run --no-sync python \
    scripts/materialize_dataset2_cooccur_lift_external.py \
    --frozen-config \
      docs/experiments/cooccur-lift-aux-expert-v1.frozen.json \
    --selection-lock \
      '$run_dir/rolling-selection/selection-lock.json' \
    --checkpoint \
      checkpoints/d1_time_ramp_g050_d2_gnn_short_none_e50_edges40000_setwise_w080_seed60_20260727.pkl \
    --train-cache-prefix \
      cache/supervised_features/dataset2_joint_recent200k_full100_seed60_20260725 \
    --train-cache-report \
      result/dataset2_joint_recent200k_full100_seed60_20260725/train-cache-report.json \
    --train-lift-features \
      '$run_dir/artifacts/lift-features.npy' \
    --train-run-contract \
      '$run_dir/artifacts/run-contract.json' \
    --train-short-none \
      result/dataset2_targeted_gnn_edges_seed60_20260725/artifacts/short_none.train-scores.npy \
    --validation-cache-prefix \
      cache/supervised_features/dataset2_joint_recent200k_full100_val_seed60_20260725 \
    --validation-cache-report \
      result/dataset2_joint_recent200k_full100_seed60_20260725/validation-cache-report.json \
    --validation-short-none \
      result/dataset2_targeted_gnn_edges_seed60_20260725/artifacts/short_none.val-scores.npy \
    --train-csv data/dataset2/train.csv \
    --baseline-probabilities \
      result/dataset2_partial_listwise_expert_blend_20260728/champion-probabilities.npy \
    --output-dir '$output_dir'
  status=\$?
  printf '%s\n' \"\$status\" > '$exit_path'
  exit \"\$status\"
" >"$log_path" 2>&1 </dev/null

for _ in {1..30}; do
  if [[ -s "$pid_path" ]]; then
    echo "started pid=$(cat "$pid_path") log=$log_path"
    exit 0
  fi
  sleep 0.1
done
echo "detached external materializer did not publish its pid" >&2
exit 1
