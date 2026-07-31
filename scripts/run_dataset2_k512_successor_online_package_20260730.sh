#!/usr/bin/env bash
set -euo pipefail

root="/home/edu/workspace/jittor-GPNUT1-JGRec"
run_dir="$root/result/dataset2_k512_cooccur_lift_successor_v2_rerun_20260729"
contract="$root/docs/experiments/cooccur-lift-k512-gap-aware-v2.online-package.preregistered.json"
preflight="$run_dir/package-preflight.json"
status_path="$run_dir/package-pipeline-status.json"
pid_path="$run_dir/package-pipeline.pid"
exit_path="$run_dir/package-pipeline.exit"
v1_online="$run_dir/v1-online-materialization"
v1_submission="$run_dir/v1-submission"
v1_lock="$run_dir/contracts/v1-online-baseline-lock.json"
v2_online="$run_dir/online-materialization"
v2_submission="$run_dir/submission"

if [[ ! -f "$preflight" || -e "$pid_path" || -e "$exit_path" ]]; then
  echo "package pipeline precondition failed" >&2
  exit 1
fi
if [[ -e "$v1_online" || -e "$v1_submission" || -e "$v1_lock" \
      || -e "$v2_online" || -e "$v2_submission" ]]; then
  echo "refusing to overwrite package pipeline output" >&2
  exit 1
fi

printf '%s\n' "$$" > "$pid_path"
cd "$root"
source .workspace-env.sh
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24
export NUMEXPR_NUM_THREADS=24

write_status() {
  local status="$1"
  local stage="$2"
  local detail="$3"
  .venv/bin/python - "$status_path" "$status" "$stage" "$detail" <<'PY'
import json
import os
import sys
from datetime import datetime
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "protocol": "cooccur_lift_k512_successor_v2_online_package_pipeline_v1",
    "status": sys.argv[2],
    "stage": sys.argv[3],
    "detail": sys.argv[4],
    "updated_at": datetime.now().astimezone().isoformat(),
    "selected_weight": 0.5,
    "external_decision_role": "safety_gate_only",
    "external_effect_size_used": False,
}
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, path)
PY
}

trap 'code=$?; printf "%s\n" "$code" > "$exit_path"; if [[ "$code" -ne 0 ]]; then write_status "failed" "pipeline" "package pipeline exited nonzero"; fi' EXIT

write_status "running" "preflight_recheck" "verifying frozen package contract and current-run lineage"
.venv/bin/python - "$contract" "$preflight" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

contract_path = Path(sys.argv[1])
preflight_path = Path(sys.argv[2])
preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
contract_sha256 = hashlib.sha256(contract_path.read_bytes()).hexdigest()
if (
    preflight.get("status") != "passed"
    or preflight.get("contract_sha256") != contract_sha256
    or preflight.get("candidate_id") != "cooccur_lift_gap_aware_v2"
    or float(preflight.get("selected_weight", -1.0)) != 0.5
    or preflight.get("all_seven_gates_passed") is not True
    or preflight.get("outputs_absent") is not True
    or preflight.get("external_effect_size_used") is not False
):
    raise ValueError("package preflight receipt differs")
PY

write_status "running" "v1_online_materialization" "materializing current K512 bugfixed V1 test probabilities"
uv run --no-sync python scripts/materialize_dataset2_cooccur_lift_test.py \
  --frozen-config \
    docs/experiments/cooccur-lift-aux-expert-v1.frozen.json \
  --selection-lock \
    result/dataset2_cooccur_lift_aux_expert_v1_20260728_compact_retry2/rolling-selection/selection-lock.json \
  --candidate-contract \
    "$run_dir/contracts/bugfixed-v1.json" \
  --training-report \
    "$run_dir/v1-full-origin/training-report.json" \
  --source-checkpoint \
    checkpoints/d1_time_ramp_g050_d2_gnn_short_none_e50_edges40000_setwise_w080_seed60_20260727.pkl \
  --auxiliary-model \
    "$run_dir/v1-full-origin/cooccur-lift-bugfixed-v1-seed33100.npz" \
  --train-csv data/dataset2/train.csv \
  --test-csv data/dataset2/test.csv \
  --output-dir "$v1_online" \
  --batch-size 4096 \
  --query-order source_grouped

write_status "running" "v1_packaging" "building current K512 bugfixed V1 online baseline package"
uv run --no-sync python scripts/package_dataset2_cooccur_lift_candidate.py \
  --frozen-config \
    docs/experiments/cooccur-lift-aux-expert-v1.frozen.json \
  --selection-lock \
    result/dataset2_cooccur_lift_aux_expert_v1_20260728_compact_retry2/rolling-selection/selection-lock.json \
  --candidate-contract \
    "$run_dir/contracts/bugfixed-v1.json" \
  --training-report \
    "$run_dir/v1-full-origin/training-report.json" \
  --test-materialization-report \
    "$v1_online/test-materialization-report.json" \
  --champion-zip \
    result/d1_time_ramp_g050_d2_gnn_short_none_e50_edges40000_setwise_w080_seed60_20260727/result.zip \
  --output-dir "$v1_submission" \
  --expected-champion-zip-sha256 \
    104f68dc82aed862600be3328f779d80e04746283c0ec75193a3582266438193 \
  --expected-dataset1-sha256 \
    81285eddfba612b9c05e96075be29a15718d8a94839d143e3083bf5353644369 \
  --expected-dataset2-sha256 \
    b5544d15fac4bd6d5737c5e7e30d5d413553d1e11109d5ee23acb7b18513cc3a

write_status "running" "v1_baseline_lock" "freezing generated current K512 V1 package before v2 scoring"
.venv/bin/python - "$contract" "$v1_online" "$v1_submission" "$v1_lock" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

contract_path = Path(sys.argv[1])
online = Path(sys.argv[2])
submission = Path(sys.argv[3])
lock_path = Path(sys.argv[4])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

materialization_path = online / "test-materialization-report.json"
package_path = submission / "candidate-report.json"
zip_path = submission / "result.zip"
materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
package = json.loads(package_path.read_text(encoding="utf-8"))
if (
    materialization.get("status") != "complete_online_candidate_materialization"
    or materialization.get("candidate_id")
    != "cooccur_lift_aux_expert_v1_k512_weighted_normalizer_refit_20260729"
    or materialization.get("auxiliary_model_sha256")
    != "3a592856c828fbaca2da08d8ee155e2eca09232e3ba4737bbca110e841e2f8eb"
    or materialization.get("candidate_contract_sha256")
    != "0e61c8ccb6883edc4b0ee361f29045641e5c275b2b6d36c2db738ad2aa49869f"
    or materialization.get("test_csv_sha256")
    != "389e330d6a21317cc1a0a013c878850c2324e916b2392901b3c039384e201372"
    or materialization.get("shape") != [153420, 100]
    or materialization.get("external_report_reused") is not False
    or materialization.get("production_checkpoint_modified") is not False
):
    raise ValueError("current K512 V1 online materialization differs")
if (
    package.get("status") != "online_candidate"
    or float(package["dataset2"].get("auxiliary_weight", -1.0)) != 0.5
    or package["dataset1"].get("sha256")
    != "81285eddfba612b9c05e96075be29a15718d8a94839d143e3083bf5353644369"
    or package["dataset2"].get("champion_member_sha256")
    != "b5544d15fac4bd6d5737c5e7e30d5d413553d1e11109d5ee23acb7b18513cc3a"
    or package["expert"].get("model_sha256")
    != "3a592856c828fbaca2da08d8ee155e2eca09232e3ba4737bbca110e841e2f8eb"
    or package.get("result_zip_sha256") != sha256(zip_path)
):
    raise ValueError("current K512 V1 package differs")
payload = {
    "schema_version": 1,
    "protocol": "cooccur_lift_k512_v1_online_baseline_lock_v1",
    "status": "frozen_before_successor_test_scoring",
    "online_package_contract_sha256": sha256(contract_path),
    "selected_weight": 0.5,
    "v1_model_sha256": materialization["auxiliary_model_sha256"],
    "v1_materialization_report_sha256": sha256(materialization_path),
    "v1_package_report_sha256": sha256(package_path),
    "v1_zip": str(zip_path.resolve()),
    "v1_zip_sha256": sha256(zip_path),
    "dataset1_member_sha256": package["dataset1"]["sha256"],
    "dataset2_member_sha256": package["dataset2"]["sha256"],
    "external_effect_size_used": False,
    "weight_rescan_used": False,
}
with lock_path.open("x", encoding="utf-8") as handle:
    json.dump(
        payload,
        handle,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    handle.write("\n")
print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
PY

write_status "running" "v2_online_materialization" "materializing selected gap-aware v2 test probabilities"
uv run --no-sync python \
  scripts/materialize_dataset2_cooccur_lift_successor_v2_test.py \
  --candidate-config \
    docs/experiments/cooccur-lift-gap-aware-v2.preregistered.json \
  --selection-lock "$run_dir/selection/selection-lock.json" \
  --external-report \
    "$run_dir/external-state/external-evaluation-report.json" \
  --source-checkpoint \
    checkpoints/d1_time_ramp_g050_d2_gnn_short_none_e50_edges40000_setwise_w080_seed60_20260727.pkl \
  --auxiliary-model \
    "$run_dir/external-materialization/cooccur_lift_gap_aware_v2-seed33100.npz" \
  --train-csv data/dataset2/train.csv \
  --test-csv data/dataset2/test.csv \
  --output-dir "$v2_online" \
  --batch-size 4096

v1_zip_sha256="$(jq -r '.v1_zip_sha256' "$v1_lock")"
dataset1_sha256="$(jq -r '.dataset1_member_sha256' "$v1_lock")"
dataset2_sha256="$(jq -r '.dataset2_member_sha256' "$v1_lock")"

write_status "running" "v2_packaging" "building accepted gap-aware v2 package on current K512 V1"
uv run --no-sync python \
  scripts/package_dataset2_cooccur_lift_successor_v2.py \
  --candidate-config \
    docs/experiments/cooccur-lift-gap-aware-v2.preregistered.json \
  --selection-lock "$run_dir/selection/selection-lock.json" \
  --external-report \
    "$run_dir/external-state/external-evaluation-report.json" \
  --test-materialization-report \
    "$v2_online/test-materialization-report.json" \
  --champion-zip "$v1_submission/result.zip" \
  --output-dir "$v2_submission" \
  --expected-champion-zip-sha256 "$v1_zip_sha256" \
  --expected-dataset1-sha256 "$dataset1_sha256" \
  --expected-dataset2-sha256 "$dataset2_sha256"

write_status "running" "final_validation" "verifying final zip, members, lineage, support, and authorization"
.venv/bin/python - "$contract" "$v1_lock" "$v2_online" "$v2_submission" <<'PY'
import hashlib
import json
import sys
import zipfile
from pathlib import Path

contract_path = Path(sys.argv[1])
v1_lock_path = Path(sys.argv[2])
online = Path(sys.argv[3])
submission = Path(sys.argv[4])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

v1 = json.loads(v1_lock_path.read_text(encoding="utf-8"))
materialization_path = online / "test-materialization-report.json"
package_path = submission / "successor-package-report.json"
zip_path = submission / "result.zip"
materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
package = json.loads(package_path.read_text(encoding="utf-8"))
support = materialization.get("short_window_support", {})
submission_report = package.get("submission", {})
if (
    materialization.get("status")
    != "complete_online_candidate_materialization"
    or materialization.get("candidate_id") != "cooccur_lift_gap_aware_v2"
    or materialization.get("auxiliary_model_sha256")
    != "5567a4f26f8a06bcf74ed8c3f1ac83fcb31c31e9a18968b7503706d2d873a158"
    or materialization.get("selection_lock_sha256")
    != "1343aceaf36d4718d0f52e2f927a2a1daf0b256ec34168ca2751001b0188e2b7"
    or support.get("collapsed_rows") != 61109
    or support.get("total_rows") != 153420
    or materialization.get("external_effect_size_used") is not False
    or materialization.get("production_checkpoint_modified") is not False
):
    raise ValueError("successor test materialization differs")
if (
    package.get("status") != "complete"
    or package.get("candidate_id") != "cooccur_lift_gap_aware_v2"
    or package.get("selection_lock_sha256")
    != "1343aceaf36d4718d0f52e2f927a2a1daf0b256ec34168ca2751001b0188e2b7"
    or package.get("bugfixed_v1_champion_zip_sha256")
    != v1["v1_zip_sha256"]
    or package.get("external_effect_size_used") is not False
    or submission_report.get("result_zip_sha256") != sha256(zip_path)
    or submission_report["dataset1"].get("sha256")
    != v1["dataset1_member_sha256"]
    or submission_report["dataset2"].get("champion_member_sha256")
    != v1["dataset2_member_sha256"]
    or float(submission_report["dataset2"].get("auxiliary_weight", -1.0))
    != 0.5
    or submission_report["expert"].get("model_sha256")
    != "5567a4f26f8a06bcf74ed8c3f1ac83fcb31c31e9a18968b7503706d2d873a158"
):
    raise ValueError("successor package lineage differs")
with zipfile.ZipFile(zip_path) as archive:
    if sorted(archive.namelist()) != ["dataset1.csv", "dataset2.csv"]:
        raise ValueError("final zip members differ")
    member_sha256 = {
        name: hashlib.sha256(archive.read(name)).hexdigest()
        for name in archive.namelist()
    }
if (
    member_sha256["dataset1.csv"]
    != submission_report["dataset1"]["sha256"]
    or member_sha256["dataset2.csv"]
    != submission_report["dataset2"]["sha256"]
):
    raise ValueError("final zip member hash differs")
validation = {
    "schema_version": 1,
    "protocol": "cooccur_lift_k512_successor_v2_final_package_validation_v1",
    "status": "passed",
    "online_package_contract_sha256": sha256(contract_path),
    "v1_baseline_lock_sha256": sha256(v1_lock_path),
    "test_materialization_report_sha256": sha256(materialization_path),
    "successor_package_report_sha256": sha256(package_path),
    "result_zip": str(zip_path.resolve()),
    "result_zip_bytes": zip_path.stat().st_size,
    "result_zip_sha256": sha256(zip_path),
    "member_sha256": member_sha256,
    "selected_weight": 0.5,
    "collapsed_rows": 61109,
    "all_seven_external_gates_passed": True,
    "external_effect_size_used": False,
}
validation_path = submission / "final-package-validation.json"
with validation_path.open("x", encoding="utf-8") as handle:
    json.dump(
        validation,
        handle,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    handle.write("\n")
(submission / "result.zip.sha256").write_text(
    f"{validation['result_zip_sha256']}  {zip_path}\n",
    encoding="utf-8",
)
print(json.dumps(validation, ensure_ascii=False, sort_keys=True), flush=True)
PY

write_status "complete" "complete" "final package validated and ready for local transfer"
printf '%s\n' "0" > "$exit_path"
trap - EXIT
