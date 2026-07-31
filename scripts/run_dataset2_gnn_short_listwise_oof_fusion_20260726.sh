#!/usr/bin/env bash
set -euo pipefail

root="/home/edu/workspace/jittor-GPNUT1-JGRec"
tag="dataset2_gnn_short_listwise_oof_fusion_seed60_20260726"
run_dir="$root/result/$tag"
oof_dir="$run_dir/oof-artifacts"
evaluation_dir="$run_dir/evaluation-artifacts"
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
  uv run --no-sync python scripts/build_dataset2_gnn_short_listwise_oof.py \
    --checkpoint checkpoints/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722.pkl \
    --train-csv data/dataset2/train.csv \
    --train-cache-prefix cache/supervised_features/dataset2_joint_recent200k_full100_seed60_20260725 \
    --train-cache-report result/dataset2_joint_recent200k_full100_seed60_20260725/train-cache-report.json \
    --output-dir '$oof_dir' \
    --seed 60 \
    --burn-in 25000 \
    --fold-size 25000 \
    --epochs 50 \
    --patience 3 \
    --batch-size 512 \
    --internal-val-ratio 0.1 \
    --progress-every 100
  status=\$?
  if [[ \$status -eq 0 ]]; then
    uv run --no-sync python scripts/evaluate_dataset2_gnn_short_listwise_fusion.py \
      --checkpoint checkpoints/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722.pkl \
      --gnn-model result/dataset2_gnn_short_listwise_200k_full100_seed60_20260726_v2/artifacts/best-gnn-short-listwise.npz \
      --gnn-training-report result/dataset2_gnn_short_listwise_200k_full100_seed60_20260726_v2/artifacts/training-report.json \
      --train-csv data/dataset2/train.csv \
      --train-cache-prefix cache/supervised_features/dataset2_joint_recent200k_full100_seed60_20260725 \
      --train-cache-report result/dataset2_joint_recent200k_full100_seed60_20260725/train-cache-report.json \
      --oof-training-features '$oof_dir/train-oof-gnn-short-listwise.npy' \
      --oof-training-report '$oof_dir/oof-build-report.json' \
      --validation-cache-prefix cache/supervised_features/dataset2_joint_recent200k_full100_val_seed60_20260725 \
      --validation-cache-report result/dataset2_joint_recent200k_full100_seed60_20260725/validation-cache-report.json \
      --champion-report result/dataset2_setwise_high_weight_scan_seed60_20260725/artifacts/setwise-high-weight-report.json \
      --output-dir '$evaluation_dir' \
      --seed 60 \
      --score-batch-size 256 \
      --fusion-batch-size 256 \
      --fusion-epochs 10 \
      --fusion-patience 2 \
      --setwise-weight 0.80 \
      --min-full-delta 0.002
    status=\$?
  fi
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
