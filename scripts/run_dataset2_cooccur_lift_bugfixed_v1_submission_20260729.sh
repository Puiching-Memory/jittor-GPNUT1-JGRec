#!/usr/bin/env bash
set -euo pipefail

root="/home/edu/workspace/jittor-GPNUT1-JGRec"
tag="dataset2_cooccur_lift_bugfixed_v1_refit_cpu_retry1_20260729"
run_dir="$root/result/$tag"
log_path="$root/logs/$tag.log"
pid_path="$run_dir/pipeline.pid"
exit_path="$run_dir/pipeline.exit"
status_path="$run_dir/status.json"

if [[ -e "$run_dir" || -e "$log_path" ]]; then
  echo "refusing to overwrite an existing bugfixed v1 run" >&2
  exit 1
fi

mkdir -p "$run_dir"
setsid -f bash -lc "
  printf '%s\n' \$\$ > '$pid_path'
  cd '$root'
  source .workspace-env.sh
  export CUDA_VISIBLE_DEVICES=0
  export OMP_NUM_THREADS=24
  export MKL_NUM_THREADS=24
  export OPENBLAS_NUM_THREADS=24
  export NUMEXPR_NUM_THREADS=24

  printf '%s\n' '{\"stage\":\"training\",\"status\":\"running\"}' \
    > '$status_path'
  set +e
  uv run --no-sync python \
    scripts/train_dataset2_cooccur_lift_bugfixed_v1.py \
    --candidate-contract \
      docs/experiments/cooccur-lift-aux-expert-v1-bugfixed-refit.preregistered.json \
    --frozen-config \
      docs/experiments/cooccur-lift-aux-expert-v1.frozen.json \
    --selection-lock \
      result/dataset2_cooccur_lift_aux_expert_v1_20260728_compact_retry2/rolling-selection/selection-lock.json \
    --source-checkpoint \
      checkpoints/d1_time_ramp_g050_d2_gnn_short_none_e50_edges40000_setwise_w080_seed60_20260727.pkl \
    --train-cache-prefix \
      cache/supervised_features/dataset2_joint_recent200k_full100_seed60_20260725 \
    --train-cache-report \
      result/dataset2_joint_recent200k_full100_seed60_20260725/train-cache-report.json \
    --validation-cache-report \
      result/dataset2_joint_recent200k_full100_seed60_20260725/validation-cache-report.json \
    --train-lift-features \
      result/dataset2_cooccur_lift_aux_expert_v1_20260728_compact_retry2/artifacts/lift-features.npy \
    --train-short-none \
      result/dataset2_targeted_gnn_edges_seed60_20260725/artifacts/short_none.train-scores.npy \
    --output-dir '$run_dir/training'
  status=\$?
  if [[ \$status -ne 0 ]]; then
    printf '%s\n' '{\"stage\":\"training\",\"status\":\"failed\"}' \
      > '$status_path'
    printf '%s\n' \"\$status\" > '$exit_path'
    exit \"\$status\"
  fi

  printf '%s\n' '{\"stage\":\"online_materialization\",\"status\":\"running\"}' \
    > '$status_path'
  uv run --no-sync python \
    scripts/materialize_dataset2_cooccur_lift_test.py \
    --frozen-config \
      docs/experiments/cooccur-lift-aux-expert-v1.frozen.json \
    --selection-lock \
      result/dataset2_cooccur_lift_aux_expert_v1_20260728_compact_retry2/rolling-selection/selection-lock.json \
    --candidate-contract \
      docs/experiments/cooccur-lift-aux-expert-v1-bugfixed-refit.preregistered.json \
    --training-report '$run_dir/training/training-report.json' \
    --source-checkpoint \
      checkpoints/d1_time_ramp_g050_d2_gnn_short_none_e50_edges40000_setwise_w080_seed60_20260727.pkl \
    --auxiliary-model \
      '$run_dir/training/cooccur-lift-bugfixed-v1-seed33100.npz' \
    --train-csv data/dataset2/train.csv \
    --test-csv data/dataset2/test.csv \
    --output-dir '$run_dir/online-materialization' \
    --batch-size 4096 \
    --query-order source_grouped
  status=\$?
  if [[ \$status -ne 0 ]]; then
    printf '%s\n' \
      '{\"stage\":\"online_materialization\",\"status\":\"failed\"}' \
      > '$status_path'
    printf '%s\n' \"\$status\" > '$exit_path'
    exit \"\$status\"
  fi

  printf '%s\n' '{\"stage\":\"packaging\",\"status\":\"running\"}' \
    > '$status_path'
  uv run --no-sync python \
    scripts/package_dataset2_cooccur_lift_candidate.py \
    --frozen-config \
      docs/experiments/cooccur-lift-aux-expert-v1.frozen.json \
    --selection-lock \
      result/dataset2_cooccur_lift_aux_expert_v1_20260728_compact_retry2/rolling-selection/selection-lock.json \
    --candidate-contract \
      docs/experiments/cooccur-lift-aux-expert-v1-bugfixed-refit.preregistered.json \
    --training-report '$run_dir/training/training-report.json' \
    --test-materialization-report \
      '$run_dir/online-materialization/test-materialization-report.json' \
    --champion-zip \
      result/d1_time_ramp_g050_d2_gnn_short_none_e50_edges40000_setwise_w080_seed60_20260727/result.zip \
    --output-dir '$run_dir/submission' \
    --expected-champion-zip-sha256 \
      104f68dc82aed862600be3328f779d80e04746283c0ec75193a3582266438193 \
    --expected-dataset1-sha256 \
      81285eddfba612b9c05e96075be29a15718d8a94839d143e3083bf5353644369 \
    --expected-dataset2-sha256 \
      b5544d15fac4bd6d5737c5e7e30d5d413553d1e11109d5ee23acb7b18513cc3a
  status=\$?
  if [[ \$status -ne 0 ]]; then
    printf '%s\n' '{\"stage\":\"packaging\",\"status\":\"failed\"}' \
      > '$status_path'
    printf '%s\n' \"\$status\" > '$exit_path'
    exit \"\$status\"
  fi

  sha256sum '$run_dir/submission/result.zip' \
    > '$run_dir/submission/result.zip.sha256'
  printf '%s\n' '{\"stage\":\"complete\",\"status\":\"complete\"}' \
    > '$status_path'
  printf '%s\n' '0' > '$exit_path'
  exit 0
" >"$log_path" 2>&1 </dev/null

for _ in {1..50}; do
  if [[ -s "$pid_path" ]]; then
    echo "started pid=$(cat "$pid_path") log=$log_path"
    exit 0
  fi
  sleep 0.1
done
echo "detached bugfixed v1 pipeline did not publish its pid" >&2
exit 1
