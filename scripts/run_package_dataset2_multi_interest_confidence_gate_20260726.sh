#!/usr/bin/env bash
set -euo pipefail

root="/home/edu/workspace/jittor-GPNUT1-JGRec"
tag="d1_champion_d2_multi_interest_confidence_gate_seed60_20260726"
run_dir="$root/result/$tag"
log_path="$root/logs/$tag.log"
pid_path="$root/result/dataset2_multi_interest_gated_oof_v2_seed60_20260726/package.pid"
exit_path="$root/result/dataset2_multi_interest_gated_oof_v2_seed60_20260726/package.exit"

for protected in "$run_dir" "$log_path" "$exit_path"; do
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
  uv run --no-sync python scripts/package_dataset2_multi_interest_confidence_gate.py \
    --oof-report result/dataset2_multi_interest_gated_oof_v2_seed60_20260726/artifacts/gated-oof-report.json \
    --paired-diagnostics result/dataset2_multi_interest_gated_oof_v2_seed60_20260726/artifacts/paired-query-diagnostics.npz \
    --champion-dataset1 result/d1_champion_d2_setwise_w080_seed60_20260725/csv/dataset1.csv \
    --champion-dataset2 result/d1_champion_d2_setwise_w080_seed60_20260725/csv/dataset2.csv \
    --candidate-dataset2 result/d1_champion_d2_multi_interest_proxy_v3_package_seed60_20260726/csv/dataset2.csv \
    --output-dir '$run_dir' \
    --data-dir data
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
