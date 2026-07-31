#!/usr/bin/env bash
set -euo pipefail

root="/home/edu/workspace/jittor-GPNUT1-JGRec"
validation_pid="551772"
validation_tag="dataset2_recent200k_full100_matched_val_seed60_20260724"
training_tag="dataset2_matched_lgbm_setwise_seed60_20260724"
output_dir="$root/result/$training_tag"
log_path="$root/logs/$training_tag.log"
pid_path="$output_dir/training-supervisor.pid"

for protected in "$output_dir" "$log_path"; do
  if [[ -e "$protected" ]]; then
    echo "refusing to overwrite existing artifact: $protected" >&2
    exit 1
  fi
done

mkdir -p "$output_dir" "$root/logs"
setsid -f bash -lc "
  printf '%s\n' \$\$ > '$pid_path'
  while kill -0 '$validation_pid' 2>/dev/null; do
    sleep 30
  done
  cd '$root'
  source .workspace-env.sh
  uv run --no-sync python - <<'PY'
import json
from pathlib import Path

report = json.loads(
    Path(
        'result/dataset2_recent200k_full100_matched_val_seed60_20260724/'
        'validation-cache-report.json'
    ).read_text(encoding='utf-8')
)
if report.get('status') != 'complete' or not report.get('train_replay', {}).get('matched'):
    raise SystemExit('matched validation did not pass; refusing to train rerankers')
PY
  exec uv run --no-sync python scripts/train_dataset2_matched_lgbm_setwise.py \
    --checkpoint checkpoints/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722.pkl \
    --train-cache-prefix cache/supervised_features/dataset2_recent200k_full100_seed60_20260724 \
    --train-cache-report result/dataset2_recent200k_full100_seed60_20260724/cache-build-report.json \
    --validation-cache-prefix cache/supervised_features/$validation_tag \
    --validation-cache-report result/$validation_tag/validation-cache-report.json \
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
" >"$log_path" 2>&1 </dev/null &

for _ in {1..20}; do
  if [[ -s "$pid_path" ]]; then
    echo "started supervisor pid=$(cat "$pid_path") log=$log_path"
    exit 0
  fi
  sleep 0.1
done
echo "detached supervisor did not publish its pid" >&2
exit 1
