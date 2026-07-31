#!/usr/bin/env bash
set -euo pipefail

root="/home/edu/workspace/jittor-GPNUT1-JGRec"
cache_tag="dataset2_joint_recent200k_full100_seed60_20260725"
validation_tag="dataset2_joint_recent200k_full100_val_seed60_20260725"
training_tag="dataset2_joint_lgbm_setwise_seed60_20260725"
candidate_tag="d1_champion_d2_joint_reranker_seed60_20260725"
train_prefix="$root/cache/supervised_features/$cache_tag"
validation_prefix="$root/cache/supervised_features/$validation_tag"
cache_result="$root/result/$cache_tag"
training_result="$root/result/$training_tag"
candidate_result="$root/result/$candidate_tag"
checkpoint="$root/checkpoints/$candidate_tag.pkl"
log_path="$root/logs/$candidate_tag.log"
pid_path="$cache_result/pipeline.pid"
exit_path="$cache_result/pipeline.exit"

for protected in \
  "$train_prefix.train.npy" \
  "$train_prefix.train-candidates.npy" \
  "$train_prefix.train-src.npy" \
  "$train_prefix.train-dst.npy" \
  "$train_prefix.train-time.npy" \
  "$train_prefix.train-row-indices.npy" \
  "$train_prefix.progress.json" \
  "$validation_prefix.val.npy" \
  "$validation_prefix.val-candidates.npy" \
  "$validation_prefix.val-src.npy" \
  "$validation_prefix.val-dst.npy" \
  "$validation_prefix.val-time.npy" \
  "$validation_prefix.val-row-indices.npy" \
  "$validation_prefix.progress.json" \
  "$cache_result" \
  "$training_result" \
  "$candidate_result" \
  "$checkpoint" \
  "$checkpoint.tmp" \
  "$log_path"
do
  if [[ -e "$protected" ]]; then
    echo "refusing to overwrite existing artifact: $protected" >&2
    exit 1
  fi
done

mkdir -p "$cache_result" "$root/logs"
setsid -f bash -lc "
  printf '%s\n' \$\$ > '$pid_path'
  cd '$root'
  source .workspace-env.sh
  set +e
  (
    set -e
    uv run --no-sync python scripts/build_dataset2_full100_train_cache.py \
      --checkpoint checkpoints/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722.pkl \
      --source-cache-prefix cache/supervised_features/4baa722bf26e5d50356da26ac5f479cb54324ddb \
      --train-csv data/dataset2/train.csv \
      --replay-report result/dataset2_full100_train_seed60_20260723/replay-report.json \
      --output-prefix cache/supervised_features/$cache_tag \
      --report result/$cache_tag/train-cache-report.json \
      --validation-output-prefix cache/supervised_features/$validation_tag \
      --validation-report result/$cache_tag/validation-cache-report.json \
      --candidate-count 100 \
      --train-rows 200000 \
      --train-selection recent \
      --validation-rows 20000 \
      --batch-rows 4096

    uv run --no-sync python scripts/train_dataset2_matched_lgbm_setwise.py \
      --checkpoint checkpoints/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722.pkl \
      --train-cache-prefix cache/supervised_features/$cache_tag \
      --train-cache-report result/$cache_tag/train-cache-report.json \
      --validation-cache-prefix cache/supervised_features/$validation_tag \
      --validation-cache-report result/$cache_tag/validation-cache-report.json \
      --output-dir result/$training_tag/artifacts \
      --seed 60 \
      --num-threads 16 \
      --lgbm-max-rounds 800 \
      --lgbm-patience 60 \
      --setwise-epochs 10 \
      --setwise-patience 2 \
      --setwise-batch-size 256 \
      --setwise-hidden-dim 32 \
      --setwise-learning-rate 0.001 \
      --mlp-weight 0.07 \
      --min-full-delta 0.002

    uv run --no-sync python scripts/build_dataset2_matched_reranker_candidate.py \
      --source-checkpoint checkpoints/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722.pkl \
      --evaluation-report result/$training_tag/artifacts/evaluation-report.json \
      --lgbm-model result/$training_tag/artifacts/dataset2-matched-lgbm.txt \
      --setwise-model result/$training_tag/artifacts/dataset2-setwise.npz \
      --champion-dataset1 result/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722/csv/dataset1.csv \
      --output-checkpoint checkpoints/$candidate_tag.pkl \
      --output-dir result/$candidate_tag \
      --data-dir data \
      --batch-size 2048
  )
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
