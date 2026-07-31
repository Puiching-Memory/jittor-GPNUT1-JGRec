#!/usr/bin/env bash
set -euo pipefail

root="/home/edu/workspace/jittor-GPNUT1-JGRec"
run_dir="$root/result/dataset2_cooccur_lift_successor_v2_external_20260729"
status_path="$run_dir/status.json"
resume_pid="$run_dir/external-open-resume.pid"
resume_exit="$run_dir/external-open-resume.exit"
open_contract="$root/docs/experiments/cooccur-lift-gap-aware-v2.external-open.preregistered.json"
config="$root/docs/experiments/cooccur-lift-gap-aware-v2.preregistered.json"
selection_lock="$root/result/dataset2_cooccur_lift_successor_v2_duel_20260729/selection-v5-cpu-replay-wiringfix/selection-lock.json"
materialization_dir="$run_dir/external-materialization"
external_state="$run_dir/external-evaluation"
online_dir="$run_dir/online-materialization"
submission_dir="$run_dir/submission"
v1_zip="$root/result/dataset2_cooccur_lift_bugfixed_v1_refit_cpu_retry1_20260729/submission/result.zip"

if [[ ! -d "$run_dir" || -e "$external_state" || -e "$resume_pid" || -e "$resume_exit" ]]; then
  echo "resume precondition failed: run must exist and external state/resume markers must be absent" >&2
  exit 1
fi
printf '%s\n' "$$" > "$resume_pid"

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

trap 'code=$?; printf "%s\n" "$code" > "$resume_exit"; if [[ "$code" -ne 0 ]] && ! grep -q "\"status\": \"complete\"" "$status_path" 2>/dev/null; then write_status "failed" "external_open_resume" "resume command exited nonzero"; fi' EXIT

cd "$root"
source .workspace-env.sh
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24
export NUMEXPR_NUM_THREADS=24

write_status "running" "external_open_preflight" "validating frozen scores, runner, and zero prior opens"
uv run --no-sync python - "$open_contract" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path.cwd()
run = root / "result/dataset2_cooccur_lift_successor_v2_external_20260729"
contract = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

artifacts = {
    "materialization_report": run / "external-materialization/external-materialization-report.json",
    "external_manifest": run / "external-materialization/external-manifest.json",
    "external_preflight_report": run / "external-preflight-report.json",
    "external_baseline_scores": run / "external-materialization/external-baseline-v1.npy",
    "external_candidate_scores": run / "external-materialization/external-candidate-v2.npy",
    "gap_aware_model": run / "external-materialization/cooccur_lift_gap_aware_v2-seed33100.npz",
}
implementations = {
    "external_evaluator_script": root / "scripts/evaluate_standard_external_gate.py",
    "standard_validation_protocol": root / "src/jgrec/standard_validation_protocol.py",
    "preflight_script": root / "scripts/preflight_dataset2_cooccur_lift_successor_v2_external.py",
    "test_materializer_script": root / "scripts/materialize_dataset2_cooccur_lift_successor_v2_test.py",
    "package_script": root / "scripts/package_dataset2_cooccur_lift_successor_v2.py",
}
post_gate = {
    "bugfixed_v1_champion_zip": root / "result/dataset2_cooccur_lift_bugfixed_v1_refit_cpu_retry1_20260729/submission/result.zip",
    "test_csv": root / "data/dataset2/test.csv",
}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

for group, paths in (
    ("artifact_sha256", artifacts),
    ("implementation_sha256", implementations),
    ("post_gate_sha256", post_gate),
):
    for name, path in paths.items():
        if sha256(path) != contract[group][name]:
            raise SystemExit(f"external-open frozen hash differs: {name}")
if (run / "external-evaluation").exists():
    raise SystemExit("external state appeared before the one-shot call")
print("external-open immutable inputs verified; prior open count is zero", flush=True)
PY

uv run --no-sync python \
  scripts/preflight_dataset2_cooccur_lift_successor_v2_external.py \
  --candidate-config "$config" \
  --selection-lock "$selection_lock" \
  --materialization-report \
    "$materialization_dir/external-materialization-report.json" \
  --manifest "$materialization_dir/external-manifest.json" \
  --state-dir "$external_state" \
  --output "$run_dir/external-preflight-resume-report.json" \
  >/dev/null
if [[ "$(sha256sum "$run_dir/external-preflight-resume-report.json" | cut -d' ' -f1)" != "09d8b9ac15b448be2d8d64f48f689fd0b86ea465b7d23ac152f7c92bf29d7aca" ]]; then
  write_status "failed" "external_open_preflight" "repeated preflight differs from the frozen preflight"
  exit 31
fi

write_status "running" "external_safety_gate" "opening the preregistered external exactly once"
set +e
uv run --no-sync python \
  scripts/evaluate_standard_external_gate.py \
  --manifest "$materialization_dir/external-manifest.json" \
  --selection-lock "$selection_lock" \
  --state-dir "$external_state" \
  >/dev/null
external_code=$?
set -e

if [[ ! -f "$external_state/external-open-receipt.json" || ! -f "$external_state/external-evaluation-report.json" ]]; then
  write_status "failed" "external_safety_gate" "evaluator exited without a receipt and report; no gate decision exists"
  exit 30
fi

uv run --no-sync python - \
  "$external_state/external-evaluation-report.json" \
  "$run_dir/external-gate-summary.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
report = json.loads(report_path.read_text(encoding="utf-8"))
gate_names = {
    "mrr_meets_minimum",
    "hit_at_1_meets_minimum",
    "hit_at_3_meets_minimum",
    "hit_at_10_meets_minimum",
    "ndcg_at_10_meets_minimum",
    "mean_rank_meets_maximum",
    "improved_minus_worsened_meets_minimum",
}
gates = report.get("gates")
if not isinstance(gates, dict) or set(gates) != gate_names:
    raise SystemExit("external report does not contain the exact seven gates")
summary = {
    "schema_version": 1,
    "protocol": "cooccur_lift_successor_v2_external_gate_summary_v1",
    "status": report["status"],
    "decision_role": "safety_gate_only",
    "effect_size_estimation_authorized": False,
    "selection_lock_sha256": report["selection_lock_sha256"],
    "external_manifest_sha256": report["external_manifest_sha256"],
    "external_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    "gates": gates,
    "failed_gates": report["failed_gates"],
    "package_authorized": report["package_authorized"],
}
with output_path.open("x", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

if [[ "$external_code" -eq 2 ]]; then
  write_status "complete" "external_rejected" "one or more of the seven safety gates failed; package forbidden"
  exit 2
fi
if [[ "$external_code" -ne 0 ]]; then
  write_status "failed" "external_safety_gate" "evaluator report exists but exit code is inconsistent"
  exit "$external_code"
fi

write_status "running" "online_materialization" "seven safety gates passed; scoring deployed support states"
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
