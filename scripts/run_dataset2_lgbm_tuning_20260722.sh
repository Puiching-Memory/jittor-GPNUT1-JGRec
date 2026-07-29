#!/usr/bin/env bash

set -u
set -o pipefail

repo="$HOME/workspace/jittor-GPNUT1-JGRec"
run_name="dataset2_lgbm_tune_pseudob_seed60_20260722_v2"
checkpoint="$repo/checkpoints/lgbm_fullmrr_v99_t31_towers50_k512_d1d2_seed60_20260722.pkl"
cache_prefix="$repo/cache/supervised_features/4baa722bf26e5d50356da26ac5f479cb54324ddb"
output_dir="$repo/result/$run_name"
run_log="$repo/logs/${run_name}.log"
status_file="$repo/logs/${run_name}.status"

cd "$repo" || exit 90
source .workspace-env.sh || exit 91
mkdir -p "$repo/logs"

for target in "$output_dir" "$run_log"; do
  if [[ -e "$target" ]]; then
    printf 'status=refused\nreason=target_exists\ntarget=%s\n' "$target" > "$status_file"
    exit 92
  fi
done

command=(
  uv run python scripts/tune_dataset2_lgbm_cached.py
  --checkpoint "$checkpoint"
  --cache-prefix "$cache_prefix"
  --output-dir "$output_dir"
  --tune-rows 10000
  --seed 60
  --num-threads 16
  --num-boost-round 800
  --early-stopping-rounds 60
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
  printf 'checkpoint=%s\n' "$checkpoint"
  printf 'cache_prefix=%s\n' "$cache_prefix"
  printf 'output_dir=%s\n' "$output_dir"
  printf 'run_log=%s\n' "$run_log"
} > "$status_file"

PYTHONUNBUFFERED=1 "${command[@]}" > "$run_log" 2>&1
command_exit=$?
finished_epoch=$(date +%s)
finished_at=$(date -Is)
duration_s=$((finished_epoch - started_epoch))
case "$command_exit" in
  0) final_status=passed ;;
  2) final_status=rejected ;;
  *) final_status=failed ;;
esac
{
  printf 'status=%s\n' "$final_status"
  printf 'run_name=%s\n' "$run_name"
  printf 'started_at=%s\n' "$started_at"
  printf 'finished_at=%s\n' "$finished_at"
  printf 'duration_s=%s\n' "$duration_s"
  printf 'exit_code=%s\n' "$command_exit"
  printf 'checkpoint=%s\n' "$checkpoint"
  printf 'cache_prefix=%s\n' "$cache_prefix"
  printf 'output_dir=%s\n' "$output_dir"
  printf 'run_log=%s\n' "$run_log"
} > "${status_file}.tmp"
mv "${status_file}.tmp" "$status_file"

exit "$command_exit"
