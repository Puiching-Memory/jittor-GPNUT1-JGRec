#!/usr/bin/env bash

set -u
set -o pipefail

repo="$HOME/workspace/jittor-GPNUT1-JGRec"
run_name="d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722"
source_checkpoint="$repo/checkpoints/lgbm_fullmrr_v99_t31_towers50_k512_d1d2_seed60_20260722.pkl"
tuning_dir="$repo/result/dataset2_lgbm_tune_pseudob_seed60_20260722_v2"
champion_dataset1="$repo/result/lgbm_fullmrr_v99_t31_towers50_k512_d1d2_seed60_20260722/csv/dataset1.csv"
output_checkpoint="$repo/checkpoints/${run_name}.pkl"
output_dir="$repo/result/$run_name"
run_log="$repo/logs/${run_name}.log"
status_file="$repo/logs/${run_name}.status"

cd "$repo" || exit 90
source .workspace-env.sh || exit 91
mkdir -p "$repo/logs" "$repo/checkpoints"

for required in "$source_checkpoint" "$tuning_dir/tuning-report.json" "$tuning_dir/dataset2-lgbm.txt" "$champion_dataset1"; do
  if [[ ! -f "$required" ]]; then
    printf 'status=refused\nreason=missing_input\ntarget=%s\n' "$required" > "$status_file"
    exit 92
  fi
done
for target in "$output_dir" "$output_checkpoint" "${output_checkpoint}.tmp" "$run_log"; do
  if [[ -e "$target" ]]; then
    printf 'status=refused\nreason=target_exists\ntarget=%s\n' "$target" > "$status_file"
    exit 93
  fi
done

command=(
  uv run python scripts/build_dataset2_lgbm_tuned_candidate.py
  --source-checkpoint "$source_checkpoint"
  --tuning-report "$tuning_dir/tuning-report.json"
  --lgbm-model "$tuning_dir/dataset2-lgbm.txt"
  --champion-dataset1 "$champion_dataset1"
  --output-checkpoint "$output_checkpoint"
  --output-dir "$output_dir"
  --data-dir "$repo/data"
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
  printf 'output_checkpoint=%s\n' "$output_checkpoint"
  printf 'output_dir=%s\n' "$output_dir"
  printf 'run_log=%s\n' "$run_log"
} > "$status_file"

PYTHONUNBUFFERED=1 "${command[@]}" > "$run_log" 2>&1
command_exit=$?
finished_epoch=$(date +%s)
finished_at=$(date -Is)
duration_s=$((finished_epoch - started_epoch))
final_status=failed
if ((command_exit == 0)); then
  final_status=finished
fi
{
  printf 'status=%s\n' "$final_status"
  printf 'run_name=%s\n' "$run_name"
  printf 'started_at=%s\n' "$started_at"
  printf 'finished_at=%s\n' "$finished_at"
  printf 'duration_s=%s\n' "$duration_s"
  printf 'exit_code=%s\n' "$command_exit"
  printf 'output_checkpoint=%s\n' "$output_checkpoint"
  printf 'output_checkpoint_bytes=%s\n' "$(stat -c %s "$output_checkpoint" 2>/dev/null || printf '0')"
  printf 'result_zip=%s\n' "$output_dir/result.zip"
  printf 'result_zip_bytes=%s\n' "$(stat -c %s "$output_dir/result.zip" 2>/dev/null || printf '0')"
  printf 'run_log=%s\n' "$run_log"
} > "${status_file}.tmp"
mv "${status_file}.tmp" "$status_file"

exit "$command_exit"
