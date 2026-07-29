#!/usr/bin/env bash
set -euo pipefail

root="/home/edu/workspace/jittor-GPNUT1-JGRec"
run_dir="$root/result/dataset2_cooccur_lift_successor_v2_external_20260729"
status_path="$run_dir/status.json"
pid_path="$run_dir/pipeline.pid"
exit_path="$run_dir/pipeline.exit"

contract="$root/docs/experiments/cooccur-lift-gap-aware-v2.external-execution.preregistered.json"
config="$root/docs/experiments/cooccur-lift-gap-aware-v2.preregistered.json"
selection_lock="$root/result/dataset2_cooccur_lift_successor_v2_duel_20260729/selection-v5-cpu-replay-wiringfix/selection-lock.json"
v1_contract="$root/docs/experiments/cooccur-lift-aux-expert-v1-bugfixed-refit.preregistered.json"
v1_training="$root/result/dataset2_cooccur_lift_bugfixed_v1_refit_cpu_retry1_20260729/training/training-report.json"
v1_model="$root/result/dataset2_cooccur_lift_bugfixed_v1_refit_cpu_retry1_20260729/training/cooccur-lift-bugfixed-v1-seed33100.npz"
source_checkpoint="$root/checkpoints/d1_time_ramp_g050_d2_gnn_short_none_e50_edges40000_setwise_w080_seed60_20260727.pkl"
train_cache_prefix="$root/cache/supervised_features/dataset2_joint_recent200k_full100_seed60_20260725"
train_cache_report="$root/result/dataset2_joint_recent200k_full100_seed60_20260725/train-cache-report.json"
validation_cache_prefix="$root/cache/supervised_features/dataset2_joint_recent200k_full100_val_seed60_20260725"
validation_cache_report="$root/result/dataset2_joint_recent200k_full100_seed60_20260725/validation-cache-report.json"
train_lift="$root/result/dataset2_cooccur_lift_aux_expert_v1_20260728_compact_retry2/artifacts/lift-features.npy"
train_short_none="$root/result/dataset2_targeted_gnn_edges_seed60_20260725/artifacts/short_none.train-scores.npy"
validation_short_none="$root/result/dataset2_targeted_gnn_edges_seed60_20260725/artifacts/short_none.val-scores.npy"
prior_external="$root/result/dataset2_partial_listwise_expert_blend_20260728/champion-probabilities.npy"
v1_zip="$root/result/dataset2_cooccur_lift_bugfixed_v1_refit_cpu_retry1_20260729/submission/result.zip"
train_csv="$root/data/dataset2/train.csv"
test_csv="$root/data/dataset2/test.csv"

materialization_dir="$run_dir/external-materialization"
preflight_report="$run_dir/external-preflight-report.json"
external_state="$run_dir/external-evaluation"
online_dir="$run_dir/online-materialization"
submission_dir="$run_dir/submission"

if [[ -e "$run_dir" ]]; then
  echo "refusing to overwrite existing successor external run: $run_dir" >&2
  exit 1
fi
mkdir -p "$run_dir"
printf '%s\n' "$$" > "$pid_path"

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

trap 'code=$?; printf "%s\n" "$code" > "$exit_path"; if [[ "$code" -ne 0 ]] && ! grep -q "\"status\": \"complete\"" "$status_path" 2>/dev/null; then write_status "failed" "pipeline" "pipeline command exited nonzero"; fi' EXIT

cd "$root"
source .workspace-env.sh
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24
export NUMEXPR_NUM_THREADS=24

write_status "running" "frozen_preflight" "validating immutable scripts and post-gate assets"
uv run --no-sync python - "$contract" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path.cwd()
contract = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
paths = {
    "preflight_script": root / "scripts/preflight_dataset2_cooccur_lift_successor_v2_external.py",
    "test_materializer_script": root / "scripts/materialize_dataset2_cooccur_lift_successor_v2_test.py",
    "package_script": root / "scripts/package_dataset2_cooccur_lift_successor_v2.py",
    "bugfixed_v1_champion_zip": root / "result/dataset2_cooccur_lift_bugfixed_v1_refit_cpu_retry1_20260729/submission/result.zip",
    "test_csv": root / "data/dataset2/test.csv",
}
for name, path in paths.items():
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != contract["post_gate_sha256"][name]:
        raise SystemExit(f"frozen post-gate asset differs: {name}")
print("frozen execution and post-gate assets verified", flush=True)
PY

write_status "running" "external_materialization" "training two deterministic CPU replays; external metrics unread"
uv run --no-sync python \
  scripts/materialize_dataset2_cooccur_lift_successor_v2_external.py \
  --execution-contract "$contract" \
  --candidate-config "$config" \
  --selection-lock "$selection_lock" \
  --bugfixed-v1-contract "$v1_contract" \
  --bugfixed-v1-training-report "$v1_training" \
  --bugfixed-v1-model "$v1_model" \
  --source-checkpoint "$source_checkpoint" \
  --train-cache-prefix "$train_cache_prefix" \
  --train-cache-report "$train_cache_report" \
  --validation-cache-prefix "$validation_cache_prefix" \
  --validation-cache-report "$validation_cache_report" \
  --train-lift-features "$train_lift" \
  --train-short-none "$train_short_none" \
  --validation-short-none "$validation_short_none" \
  --train-csv "$train_csv" \
  --prior-external-probabilities "$prior_external" \
  --output-dir "$materialization_dir"

write_status "running" "external_preflight" "validating manifest and score artifacts without opening external"
uv run --no-sync python \
  scripts/preflight_dataset2_cooccur_lift_successor_v2_external.py \
  --candidate-config "$config" \
  --selection-lock "$selection_lock" \
  --materialization-report \
    "$materialization_dir/external-materialization-report.json" \
  --manifest "$materialization_dir/external-manifest.json" \
  --state-dir "$external_state" \
  --output "$preflight_report"

write_status "running" "external_safety_gate" "opening the preregistered external exactly once"
set +e
uv run --no-sync python \
  scripts/evaluate_standard_external_gate.py \
  --manifest "$materialization_dir/external-manifest.json" \
  --selection-lock "$selection_lock" \
  --state-dir "$external_state"
external_code=$?
set -e
if [[ "$external_code" -eq 2 ]]; then
  write_status "complete" "external_rejected" "one or more of the seven safety gates failed; package forbidden"
  exit 2
fi
if [[ "$external_code" -ne 0 ]]; then
  write_status "failed" "external_safety_gate" "one-shot evaluator failed after opening"
  exit "$external_code"
fi

write_status "running" "online_materialization" "seven safety gates passed; scoring deployed support states"
uv run --no-sync python \
  scripts/materialize_dataset2_cooccur_lift_successor_v2_test.py \
  --candidate-config "$config" \
  --selection-lock "$selection_lock" \
  --external-report "$external_state/external-evaluation-report.json" \
  --source-checkpoint "$source_checkpoint" \
  --auxiliary-model \
    "$materialization_dir/cooccur_lift_gap_aware_v2-seed33100.npz" \
  --train-csv "$train_csv" \
  --test-csv "$test_csv" \
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
