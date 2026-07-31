#!/usr/bin/env bash

set -u
set -o pipefail

repo="$HOME/workspace/jittor-GPNUT1-JGRec"
run_name="d1_segment_policy_d2_champion_seed60_20260723"
output_dir="$repo/result/$run_name"
output_checkpoint="$repo/checkpoints/${run_name}.pkl"
run_log="$repo/logs/${run_name}.log"
status_file="$repo/logs/${run_name}.status"

cd "$repo" || exit 90
source .workspace-env.sh || exit 91
mkdir -p "$repo/logs"

for target in "$output_dir" "$output_checkpoint" "${output_checkpoint}.tmp" "$run_log"; do
  if [[ -e "$target" ]]; then
    printf 'status=refused\nreason=target_exists\ntarget=%s\n' "$target" > "$status_file"
    exit 92
  fi
done

command=(
  uv run python scripts/build_segment_fusion_candidate.py
  --source-checkpoint "$repo/checkpoints/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722.pkl"
  --tuning-report "$repo/result/segment_fusion_tuning_d1d2_seed60_20260722_v3/segment-fusion-report.json"
  --dataset1-gate "$repo/result/segment_fusion_tuning_d1d2_seed60_20260722_v3/dataset1-gate.pkl"
  --champion-dataset2 "$repo/result/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722/csv/dataset2.csv"
  --output-checkpoint "$output_checkpoint"
  --output-dir "$output_dir"
  --batch-size 2048
)

started_epoch=$(date +%s)
started_at=$(date -Is)
printf -v command_string '%q ' "${command[@]}"
{
  printf 'status=running\n'
  printf 'run_name=%s\n' "$run_name"
  printf 'started_at=%s\n' "$started_at"
  printf 'runner_pid=%s\n' "$$"
  printf 'command=%s\n' "$command_string"
  printf 'output_dir=%s\n' "$output_dir"
  printf 'output_checkpoint=%s\n' "$output_checkpoint"
  printf 'run_log=%s\n' "$run_log"
} > "$status_file"

PYTHONUNBUFFERED=1 "${command[@]}" > "$run_log" 2>&1
command_exit=$?
finished_epoch=$(date +%s)
finished_at=$(date -Is)
duration_s=$((finished_epoch - started_epoch))
case "$command_exit" in
  0) final_status=passed ;;
  *) final_status=failed ;;
esac
{
  printf 'status=%s\n' "$final_status"
  printf 'run_name=%s\n' "$run_name"
  printf 'started_at=%s\n' "$started_at"
  printf 'finished_at=%s\n' "$finished_at"
  printf 'duration_s=%s\n' "$duration_s"
  printf 'exit_code=%s\n' "$command_exit"
  printf 'output_dir=%s\n' "$output_dir"
  printf 'output_checkpoint=%s\n' "$output_checkpoint"
  printf 'run_log=%s\n' "$run_log"
} > "${status_file}.tmp"
mv "${status_file}.tmp" "$status_file"

exit "$command_exit"
