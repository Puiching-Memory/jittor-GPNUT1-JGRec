#!/usr/bin/env bash
set -u
set -o pipefail

root="/home/edu/workspace/jittor-GPNUT1-JGRec"
run_root="$root/result/dataset2_cooccur_lift_successor_v2_duel_20260729"
cache_pid="${1:?cache PID is required}"
cache_dir="${2:?cache directory is required}"
pipeline_id="${3:?pipeline id is required}"
status_path="$run_root/pipeline-status-$pipeline_id.json"
pid_path="$run_root/pipeline-watcher-$pipeline_id.pid"
duel_dir="$run_root/duel-$pipeline_id"
selection_dir="$run_root/selection-$pipeline_id"
python="$root/.venv/bin/python"

checkpoint="$root/checkpoints/d1_time_ramp_g050_d2_cooccur_lift_aux_w050_20260729.pkl"
validation_plan="$root/docs/experiments/cooccur-lift-successor-v2-duel.validation-plan.json"
plan_lock="$run_root/plan-v2/validation-plan-lock.json"
execution_contract="$root/docs/experiments/cooccur-lift-successor-v2-duel.execution.preregistered.json"
bugfixed_v1_contract="$root/docs/experiments/cooccur-lift-aux-expert-v1-bugfixed-refit.preregistered.json"
full_only_config="$root/docs/experiments/cooccur-lift-full-only-v2.preregistered.json"
gap_aware_config="$root/docs/experiments/cooccur-lift-gap-aware-v2.preregistered.json"
near_cache_prefix="$root/cache/supervised_features/dataset2_joint_recent200k_full100_seed60_20260725"
near_short_none="$root/result/dataset2_targeted_gnn_edges_seed60_20260725/artifacts/short_none.train-scores.npy"
near_lift="$root/result/dataset2_cooccur_lift_aux_expert_v1_20260728_compact_retry2/artifacts/lift-features.npy"
near_v1_manifest="$root/result/dataset2_cooccur_lift_aux_expert_v1_20260728_compact_retry2/artifacts/rolling-manifest.json"

cd "$root"
printf '%s\n' "$$" > "$pid_path"

write_status() {
  local status="$1"
  local phase="$2"
  local detail="$3"
  "$python" - "$status_path" "$status" "$phase" "$detail" <<'PY'
import json
import os
import sys
from datetime import datetime
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "protocol": "cooccur_lift_successor_v2_pipeline_v1",
    "status": sys.argv[2],
    "phase": sys.argv[3],
    "detail": sys.argv[4],
    "external_authorized": False,
    "external_opened": False,
    "updated_at": datetime.now().astimezone().isoformat(),
}
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
    encoding="utf-8",
)
os.replace(temporary, path)
PY
}

write_status "running" "waiting_for_gapped_cache" "waiting for PID $cache_pid"
echo "[pipeline] waiting for gapped cache PID $cache_pid"
while [[ -r "/proc/$cache_pid/cmdline" ]] \
  && grep -aq "build_dataset2_cooccur_lift_gapped_cache.py" "/proc/$cache_pid/cmdline"; do
  sleep 60
done

cache_report="$cache_dir/cache-report.json"
if [[ ! -f "$cache_report" ]]; then
  write_status "failed" "gapped_cache" "cache process ended without cache-report.json"
  echo "[pipeline] cache process ended without a report" >&2
  exit 20
fi

write_status "running" "validating_gapped_cache" "verifying report and artifact hashes"
"$python" - "$cache_report" "$validation_plan" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


report_path = Path(sys.argv[1])
plan_path = Path(sys.argv[2])
report = json.loads(report_path.read_text(encoding="utf-8"))
plan = json.loads(plan_path.read_text(encoding="utf-8"))
if report.get("status") != "complete":
    raise SystemExit("gapped cache report is not complete")
if report.get("external_scores_read") is not False:
    raise SystemExit("gapped cache report consumed external scores")
if report.get("validation_plan_sha256") != sha256(plan_path):
    raise SystemExit("gapped cache validation-plan hash differs")
if report.get("checkpoint_sha256") != plan["baseline"]["checkpoint_sha256"]:
    raise SystemExit("gapped cache checkpoint hash differs")
artifacts = report.get("artifacts")
if not isinstance(artifacts, dict) or not artifacts:
    raise SystemExit("gapped cache report has no artifact descriptors")
for name, descriptor in artifacts.items():
    path = Path(descriptor["path"])
    if not path.is_file():
        raise SystemExit(f"missing gapped cache artifact: {name}")
    if path.stat().st_size != int(descriptor["bytes"]):
        raise SystemExit(f"gapped cache artifact size differs: {name}")
    if sha256(path) != descriptor["sha256"]:
        raise SystemExit(f"gapped cache artifact hash differs: {name}")
print(f"[pipeline] verified {len(artifacts)} gapped cache artifacts")
PY
verify_code=$?
if [[ "$verify_code" -ne 0 ]]; then
  write_status "failed" "validating_gapped_cache" "cache report or artifact verification failed"
  exit "$verify_code"
fi

if [[ -e "$duel_dir" || -e "$selection_dir" ]]; then
  write_status "failed" "pre_duel" "refusing to overwrite an existing duel or selection directory"
  echo "[pipeline] refusing to overwrite an existing result directory" >&2
  exit 21
fi

write_status "running" "training_duel" "near plus gapped folds; external disabled"
echo "[pipeline] starting frozen successor duel"
nice -n 10 ionice -c 2 -n 7 "$python" \
  scripts/train_dataset2_cooccur_lift_successor_v2_duel.py \
  --v1-checkpoint "$checkpoint" \
  --validation-plan "$validation_plan" \
  --plan-lock "$plan_lock" \
  --execution-contract "$execution_contract" \
  --bugfixed-v1-contract "$bugfixed_v1_contract" \
  --full-only-config "$full_only_config" \
  --gap-aware-config "$gap_aware_config" \
  --near-cache-prefix "$near_cache_prefix" \
  --near-short-none "$near_short_none" \
  --near-lift "$near_lift" \
  --near-v1-manifest "$near_v1_manifest" \
  --gapped-cache-dir "$cache_dir" \
  --output-dir "$duel_dir"
duel_code=$?
if [[ "$duel_code" -ne 0 ]]; then
  write_status "failed" "training_duel" "successor duel exited nonzero"
  exit "$duel_code"
fi

write_status "running" "standard_selection" "running frozen selector; external disabled"
echo "[pipeline] starting standard selector"
nice -n 10 ionice -c 2 -n 7 "$python" \
  scripts/select_standard_rolling_candidate.py \
  --manifest "$duel_dir/rolling-manifest.json" \
  --plan-lock "$plan_lock" \
  --output-dir "$selection_dir"
selection_code=$?
case "$selection_code" in
  0)
    write_status "complete" "selected_external_blocked" "candidate selected; external remains unauthorized"
    ;;
  2)
    write_status "complete" "no_candidate_selected" "all candidates rejected; external remains unopened"
    ;;
  *)
    write_status "failed" "standard_selection" "selector exited unexpectedly"
    exit "$selection_code"
    ;;
esac

echo "[pipeline] terminal selector state reached; external was not opened"
