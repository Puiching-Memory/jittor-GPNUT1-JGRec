#!/usr/bin/env bash
set -euo pipefail

root="/home/edu/workspace/jittor-GPNUT1-JGRec"
tag="dataset2_recent200k_full100_seed60_20260724"
prefix="$root/cache/supervised_features/$tag"
result_dir="$root/result/$tag"
log_path="$root/logs/$tag.log"
pid_path="$result_dir/cache-build.pid"

for protected in \
  "$prefix.train.npy" \
  "$prefix.train-candidates.npy" \
  "$prefix.train-src.npy" \
  "$prefix.train-dst.npy" \
  "$prefix.train-time.npy" \
  "$prefix.train-row-indices.npy" \
  "$prefix.progress.json" \
  "$root/cache/supervised_features/.$tag.train.npy.part" \
  "$root/cache/supervised_features/.$tag.train-candidates.npy.part" \
  "$result_dir/cache-build-report.json" \
  "$log_path" \
  "$pid_path"
do
  if [[ -e "$protected" ]]; then
    echo "refusing to overwrite existing artifact: $protected" >&2
    exit 1
  fi
done

mkdir -p "$result_dir" "$root/logs"
setsid -f bash -lc "
  printf '%s\n' \$\$ > '$pid_path'
  cd '$root'
  source .workspace-env.sh
  exec uv run --no-sync python scripts/build_dataset2_full100_train_cache.py \
    --checkpoint checkpoints/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722.pkl \
    --source-cache-prefix cache/supervised_features/4baa722bf26e5d50356da26ac5f479cb54324ddb \
    --train-csv data/dataset2/train.csv \
    --replay-report result/dataset2_full100_train_seed60_20260723/replay-report.json \
    --output-prefix cache/supervised_features/$tag \
    --report result/$tag/cache-build-report.json \
    --candidate-count 100 \
    --train-rows 200000 \
    --train-selection recent \
    --batch-rows 4096
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
