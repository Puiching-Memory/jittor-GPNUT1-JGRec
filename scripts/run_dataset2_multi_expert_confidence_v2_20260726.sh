#!/usr/bin/env bash
set -euo pipefail

root="/home/edu/workspace/jittor-GPNUT1-JGRec"
tag="dataset2_multi_expert_confidence_v2_r3_20260726"
output_dir="$root/result/$tag"
log_path="$root/logs/$tag.log"
pid_path="$root/logs/$tag.pid"
exit_path="$root/logs/$tag.exit"

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
  set +e
  uv run --no-sync python scripts/run_dataset2_multi_expert_confidence_v2.py train-select \
    --validation-features cache/supervised_features/dataset2_joint_recent200k_full100_val_seed60_20260725.val.npy \
    --validation-cache-report result/dataset2_joint_recent200k_full100_seed60_20260725/validation-cache-report.json \
    --validation-proxy result/dataset2_multi_interest_proxy_seed60_20260725/artifacts/multi-interest.val.npy \
    --proxy-report result/dataset2_multi_interest_proxy_seed60_20260725/artifacts/multi-interest-report.json \
    --multi-interest-model result/dataset2_multi_interest_proxy_seed60_20260725/artifacts/dataset2-multi-interest-setwise.npz \
    --current-gate-model result/d1_champion_d2_multi_interest_confidence_gate_seed60_20260726/confidence-gate.pkl \
    --current-gate-report result/d1_champion_d2_multi_interest_confidence_gate_seed60_20260726/candidate-report.json \
    --window-selection-report result/dataset2_setwise_window_diversity_20260726/selection-report.json \
    --window-evaluation-report result/dataset2_setwise_window_diversity_20260726/evaluation-report.json \
    --window-artifacts-dir result/dataset2_setwise_window_diversity_20260726/artifacts \
    --frozen-dataset1-csv result/d1_champion_d2_multi_interest_confidence_gate_seed60_20260726/csv/dataset1.csv \
    --output-dir 'result/$tag'
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
