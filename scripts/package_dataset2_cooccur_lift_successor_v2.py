from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from jgrec.cooccur_lift_successor_external import (
    authorize_successor_package,
    validate_successor_external_setup,
)
from jgrec.partial_listwise_submission import (
    build_partial_listwise_submission,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Package accepted gap-aware v2 probabilities on top of the "
            "byte-frozen bugfixed V1 champion package."
        )
    )
    parser.add_argument("--candidate-config", required=True, type=Path)
    parser.add_argument("--selection-lock", required=True, type=Path)
    parser.add_argument("--external-report", required=True, type=Path)
    parser.add_argument(
        "--test-materialization-report",
        required=True,
        type=Path,
    )
    parser.add_argument("--champion-zip", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-champion-zip-sha256", required=True)
    parser.add_argument("--expected-dataset1-sha256", required=True)
    parser.add_argument("--expected-dataset2-sha256", required=True)
    args = parser.parse_args()

    setup = validate_successor_external_setup(
        candidate_config_path=args.candidate_config,
        selection_lock_path=args.selection_lock,
    )
    external = _read_json(args.external_report)
    authorization = authorize_successor_package(
        external_report=external,
        external_report_sha256=_sha256(args.external_report),
        expected_selection_lock_sha256=setup.selection_lock_sha256,
        expected_candidate_id=setup.candidate_id,
        expected_config_sha256=setup.config_sha256,
    )
    materialization = _read_json(args.test_materialization_report)
    if (
        materialization.get("status")
        != "complete_online_candidate_materialization"
        or materialization.get("candidate_id") != setup.candidate_id
        or materialization.get("candidate_config_sha256")
        != setup.config_sha256
        or materialization.get("selection_lock_sha256")
        != setup.selection_lock_sha256
        or materialization.get("external_report_sha256")
        != authorization["external_report_sha256"]
        or materialization.get("production_checkpoint_modified") is not False
        or materialization.get("external_effect_size_used") is not False
        or float(materialization.get("selected_weight", -1.0))
        != setup.selected_weight
    ):
        raise ValueError("successor test materialization binding differs")
    support = materialization.get("short_window_support")
    if (
        not isinstance(support, dict)
        or support.get("collapsed_rows") != 61_109
        or support.get("total_rows") != 153_420
    ):
        raise ValueError("successor deployed support evidence differs")

    probabilities = Path(str(materialization["probabilities"]))
    if _sha256(probabilities) != materialization["probabilities_sha256"]:
        raise ValueError("successor probability hash differs")
    model = Path(str(materialization["auxiliary_model"]))
    if _sha256(model) != materialization["auxiliary_model_sha256"]:
        raise ValueError("successor model hash differs")

    report = build_partial_listwise_submission(
        champion_zip=args.champion_zip,
        expert_scores_path=probabilities,
        output_dir=args.output_dir,
        auxiliary_weight=setup.selected_weight,
        expected_rows={"dataset1": 61_051, "dataset2": 153_420},
        expected_columns=100,
        expected_champion_zip_sha256=(
            args.expected_champion_zip_sha256
        ),
        expected_dataset1_sha256=args.expected_dataset1_sha256,
        expected_dataset2_sha256=args.expected_dataset2_sha256,
        expert_name=setup.candidate_id,
        expert_model_sha256=materialization["auxiliary_model_sha256"],
        candidate_manifest_sha256=materialization[
            "test_candidate_fingerprint"
        ],
        selection_lock_sha256=setup.selection_lock_sha256,
        expert_score_transform="setwise_probability",
        expert_source_sha256=materialization["probabilities_sha256"],
        dataset2_mode="cooccur_lift_gap_aware_v2_blend_on_bugfixed_v1",
    )
    wrapper = {
        "schema_version": 1,
        "protocol": "cooccur_lift_successor_v2_package_v1",
        "status": "complete",
        "candidate_id": setup.candidate_id,
        "candidate_config_sha256": setup.config_sha256,
        "selection_lock_sha256": setup.selection_lock_sha256,
        "external_authorization": authorization,
        "external_effect_size_used": False,
        "bugfixed_v1_champion_zip_sha256": _sha256(args.champion_zip),
        "test_materialization_report_sha256": _sha256(
            args.test_materialization_report
        ),
        "submission": report,
    }
    _write_json(args.output_dir / "successor-package-report.json", wrapper)
    print(json.dumps(wrapper, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
