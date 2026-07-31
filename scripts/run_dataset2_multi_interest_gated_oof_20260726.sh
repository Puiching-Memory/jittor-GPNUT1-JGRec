#!/usr/bin/env bash
set -euo pipefail

root="/home/edu/workspace/jittor-GPNUT1-JGRec"
tag="dataset2_multi_interest_gated_oof_v2_seed60_20260726"
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
  uv run --no-sync python scripts/evaluate_dataset2_multi_interest_gated_oof.py \
    --checkpoint checkpoints/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722.pkl \
    --validation-features cache/supervised_features/dataset2_joint_recent200k_full100_val_seed60_20260725.val.npy \
    --validation-proxy result/dataset2_multi_interest_proxy_seed60_20260725/artifacts/multi-interest.val.npy \
    --validation-cache-report result/dataset2_joint_recent200k_full100_seed60_20260725/validation-cache-report.json \
    --proxy-report result/dataset2_multi_interest_proxy_seed60_20260725/artifacts/multi-interest-report.json \
    --champion-setwise-model result/dataset2_joint_lgbm_setwise_seed60_20260725/artifacts/dataset2-setwise.npz \
    --candidate-setwise-model result/dataset2_multi_interest_proxy_seed60_20260725/artifacts/dataset2-multi-interest-setwise.npz \
    --output-dir '$artifact_dir' \
    --batch-size 256 \
    --setwise-weight 0.80 \
    --minimum-full-delta 0.002 \
    --minimum-confidence-threshold 0.005 \
    --maximum-gate-coverage 0.35
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
