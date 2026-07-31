#!/usr/bin/env bash
set -euo pipefail

root="/home/edu/workspace/jittor-GPNUT1-JGRec"
run_dir="$root/result/dataset2_cooccur_lift_successor_v2_external_20260729"
status_path="$run_dir/status.json"
finish_pid="$run_dir/online-package-finish.pid"
finish_exit="$run_dir/online-package-finish.exit"
contract="$root/docs/experiments/cooccur-lift-gap-aware-v2.online-package.preregistered.json"
config="$root/docs/experiments/cooccur-lift-gap-aware-v2.preregistered.json"
selection_lock="$root/result/dataset2_cooccur_lift_successor_v2_duel_20260729/selection-v5-cpu-replay-wiringfix/selection-lock.json"
materialization_dir="$run_dir/external-materialization"
external_state="$run_dir/external-evaluation"
online_dir="$run_dir/online-materialization"
submission_dir="$run_dir/submission"
v1_zip="$root/result/dataset2_cooccur_lift_bugfixed_v1_refit_cpu_retry1_20260729/submission/result.zip"

if [[ ! -d "$external_state" || -e "$online_dir" || -e "$submission_dir" || -e "$finish_pid" || -e "$finish_exit" ]]; then
  echo "finish precondition failed: accepted external must exist and online/package outputs must be absent" >&2
  exit 1
fi
printf '%s\n' "$$" > "$finish_pid"

write_status() {
  local status="$1"
  local phase="$2"
  local detail="$3"
  "$root/.venv/bin/python" - "$status_path" "$status" "$phase" "$detail" <<'PY'
import json
import os
import sys
from datetime import datetime
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "protocol": "cooccur_lift_successor_v2_external_pipeline_v1",
    "status": sys.argv[2],
    "phase": sys.argv[3],
    "detail": sys.argv[4],
    "external_decision_role": "safety_gate_only",
    "external_effect_size_estimation_authorized": False,
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

trap 'code=$?; printf "%s\n" "$code" > "$finish_exit"; if [[ "$code" -ne 0 ]]; then write_status "failed" "online_package_finish" "finish command exited nonzero"; fi' EXIT

cd "$root"
source .workspace-env.sh
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24
export NUMEXPR_NUM_THREADS=24

write_status "running" "accepted_external_preflight" "validating accepted seven-gate summary and frozen online semantics"
uv run --no-sync python - "$contract" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path.cwd()
run = root / "result/dataset2_cooccur_lift_successor_v2_external_20260729"
contract = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

external_paths = {
    "receipt_sha256": run / "external-evaluation/external-open-receipt.json",
    "report_sha256": run / "external-evaluation/external-evaluation-report.json",
    "gate_summary_sha256": run / "external-gate-summary.json",
}
for field, path in external_paths.items():
    if sha256(path) != contract["external_decision"][field]:
        raise SystemExit(f"accepted external artifact differs: {field}")

summary = json.loads(
    external_paths["gate_summary_sha256"].read_text(encoding="utf-8")
)
if (
    summary.get("status") != "accepted"
    or summary.get("package_authorized") is not True
    or len(summary.get("gates", {})) != 7
    or not all(summary["gates"].values())
    or summary.get("decision_role") != "safety_gate_only"
    or summary.get("effect_size_estimation_authorized") is not False
):
    raise SystemExit("external seven-gate summary is not accepted")

paths = {
    "test_materializer_script": root / "scripts/materialize_dataset2_cooccur_lift_successor_v2_test.py",
    "package_script": root / "scripts/package_dataset2_cooccur_lift_successor_v2.py",
}
for name, path in paths.items():
    if sha256(path) != contract["implementation_sha256"][name]:
        raise SystemExit(f"online implementation differs: {name}")

inputs = {
    "source_checkpoint": root / "checkpoints/d1_time_ramp_g050_d2_gnn_short_none_e50_edges40000_setwise_w080_seed60_20260727.pkl",
    "train_csv": root / "data/dataset2/train.csv",
    "test_csv": root / "data/dataset2/test.csv",
    "bugfixed_v1_champion_zip": root / "result/dataset2_cooccur_lift_bugfixed_v1_refit_cpu_retry1_20260729/submission/result.zip",
}
for name, path in inputs.items():
    if sha256(path) != contract["input_sha256"][name]:
        raise SystemExit(f"online frozen input differs: {name}")
print("accepted external and frozen online/package inputs verified", flush=True)
PY

write_status "running" "online_materialization" "scoring temporal support indicator with 61109 collapsed rows"
uv run --no-sync python \
  scripts/materialize_dataset2_cooccur_lift_successor_v2_test.py \
  --candidate-config "$config" \
  --selection-lock "$selection_lock" \
  --external-report "$external_state/external-evaluation-report.json" \
  --source-checkpoint \
    "$root/checkpoints/d1_time_ramp_g050_d2_gnn_short_none_e50_edges40000_setwise_w080_seed60_20260727.pkl" \
  --auxiliary-model \
    "$materialization_dir/cooccur_lift_gap_aware_v2-seed33100.npz" \
  --train-csv "$root/data/dataset2/train.csv" \
  --test-csv "$root/data/dataset2/test.csv" \
  --output-dir "$online_dir" \
  --batch-size 4096

write_status "running" "packaging" "building v2 package on the byte-frozen bugfixed V1 champion"
uv run --no-sync python \
  scripts/package_dataset2_cooccur_lift_successor_v2.py \
  --candidate-config "$config" \
  --selection-lock "$selection_lock" \
  --external-report "$external_state/external-evaluation-report.json" \
  --test-materialization-report \
    "$online_dir/test-materialization-report.json" \
  --champion-zip "$v1_zip" \
  --output-dir "$submission_dir" \
  --expected-champion-zip-sha256 \
    b90960c3427f70e2745bcb381289fca4625c208ebfaefb43ecdbc7a7387ff2f0 \
  --expected-dataset1-sha256 \
    81285eddfba612b9c05e96075be29a15718d8a94839d143e3083bf5353644369 \
  --expected-dataset2-sha256 \
    702f46d6a14b36e5330cac315ceefb130e54d4a68f9a173ce9d65c5a1d06192f

sha256sum "$submission_dir/result.zip" \
  > "$submission_dir/result.zip.sha256"
write_status "complete" "complete" "seven safety gates passed and submission package is complete"
