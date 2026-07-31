#!/usr/bin/env bash

set -u
set -o pipefail

repo="$HOME/workspace/jittor-GPNUT1-JGRec"
run_name="lgbm_fullmrr_v99_t31_towers50_k512_d1d2_seed60_20260722"
checkpoint="$repo/checkpoints/${run_name}.pkl"
cache_dir="$repo/cache/supervised_features"
log_dir="$repo/logs"
run_log="$log_dir/${run_name}.log"
status_file="$log_dir/${run_name}.status"
verify_log="$log_dir/${run_name}.verify.log"
result_dir="$repo/result/$run_name"
result_zip="$result_dir/result.zip"

cd "$repo" || exit 90
source .workspace-env.sh || exit 91
mkdir -p "$log_dir" "$repo/checkpoints" "$cache_dir"

for target in "$result_dir" "$checkpoint" "${checkpoint}.tmp"; do
  if [[ -e "$target" ]]; then
    printf 'status=refused\nreason=target_exists\ntarget=%s\n' "$target" > "$status_file"
    exit 92
  fi
done

command=(
  uv run jgrec-build
  --model hybrid
  --run-name "$run_name"
  --seed 60
  --selection-metric mrr
  --num-negatives 31
  --train-num-negatives 31
  --val-num-negatives 99
  --max-train-events 50000
  --max-val-events 20000
  --test-candidate-negative-ratio 1.0
  --structure-predict-neighbor-limit 512
  --source-profile-predict-history-limit 512
  --fusion-mode ensemble
  --gnn-epochs 50
  --seq-epochs 50
  --two-tower-epochs 50
  --source-profile-epochs 50
  --supervised-feature-cache-dir "$cache_dir"
  --save-checkpoint "$checkpoint"
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
  printf 'cache_dir=%s\n' "$cache_dir"
  printf 'checkpoint=%s\n' "$checkpoint"
  printf 'result_zip=%s\n' "$result_zip"
} > "$status_file"

{
  printf 'run_name=%s\n' "$run_name"
  printf 'started_at=%s\n' "$started_at"
  printf 'command=%s\n' "$command_string"
  printf '%s\n' 'validation_candidates=100 (1 positive + 99 negatives)'
  printf '%s\n' 'lightgbm_early_stop_metric=full_candidate_mrr'
} > "$run_log"

PYTHONUNBUFFERED=1 "${command[@]}" >> "$run_log" 2>&1
command_exit=$?
verify_exit=1

if ((command_exit == 0)); then
  .venv/bin/python - "$checkpoint" "$result_zip" > "$verify_log" 2>&1 <<'PY'
import hashlib
import sys
import zipfile
from pathlib import Path

from jgrec.contest_checkpoint import load_checkpoint_dataset, load_checkpoint_metadata

checkpoint = Path(sys.argv[1])
result_zip = Path(sys.argv[2])
metadata = load_checkpoint_metadata(checkpoint)
expected = {"dataset1", "dataset2"}
if set(metadata.get("datasets", ())) != expected:
    raise RuntimeError(f"checkpoint datasets mismatch: {metadata.get('datasets')!r}")
for dataset in sorted(expected):
    state = load_checkpoint_dataset(checkpoint, dataset)
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"checkpoint state is empty or invalid: {dataset}")
if not result_zip.is_file():
    raise FileNotFoundError(result_zip)
with zipfile.ZipFile(result_zip) as archive:
    bad_member = archive.testzip()
    if bad_member is not None:
        raise RuntimeError(f"corrupt zip member: {bad_member}")
    csv_names = sorted(name for name in archive.namelist() if name.endswith(".csv"))
    if len(csv_names) != 2:
        raise RuntimeError(f"expected two CSV files, got: {csv_names!r}")
digest = hashlib.sha256(result_zip.read_bytes()).hexdigest()
print(f"checkpoint_bytes={checkpoint.stat().st_size}")
print(f"result_zip_bytes={result_zip.stat().st_size}")
print(f"result_zip_sha256={digest}")
print(f"csv_names={csv_names}")
PY
  verify_exit=$?
fi

finished_epoch=$(date +%s)
finished_at=$(date -Is)
duration_s=$((finished_epoch - started_epoch))
final_status=failed
final_exit=$command_exit
if ((command_exit == 0 && verify_exit == 0)); then
  final_status=finished
else
  ((final_exit == 0)) && final_exit=93
fi

{
  printf 'status=%s\n' "$final_status"
  printf 'run_name=%s\n' "$run_name"
  printf 'started_at=%s\n' "$started_at"
  printf 'finished_at=%s\n' "$finished_at"
  printf 'duration_s=%s\n' "$duration_s"
  printf 'command_exit_code=%s\n' "$command_exit"
  printf 'verify_exit_code=%s\n' "$verify_exit"
  printf 'exit_code=%s\n' "$final_exit"
  printf 'cache_dir=%s\n' "$cache_dir"
  printf 'checkpoint=%s\n' "$checkpoint"
  printf 'checkpoint_bytes=%s\n' "$(stat -c %s "$checkpoint" 2>/dev/null || printf '0')"
  printf 'result_zip=%s\n' "$result_zip"
  printf 'result_zip_bytes=%s\n' "$(stat -c %s "$result_zip" 2>/dev/null || printf '0')"
  printf 'run_log=%s\n' "$run_log"
  printf 'verify_log=%s\n' "$verify_log"
} > "${status_file}.tmp"
mv "${status_file}.tmp" "$status_file"

exit "$final_exit"
