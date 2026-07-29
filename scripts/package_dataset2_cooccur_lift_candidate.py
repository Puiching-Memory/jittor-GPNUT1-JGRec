from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from jgrec.cooccur_lift_bugfixed_v1 import (
    validate_bugfixed_v1_materialization_inputs,
)
from jgrec.partial_listwise_submission import (
    build_partial_listwise_submission,
)
from jgrec.rankers.hybrid.cooccur_lift import (
    load_frozen_cooccur_lift_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Package the accepted cooccur-lift auxiliary probabilities "
            "with the byte-identical Dataset1 champion member."
        )
    )
    parser.add_argument("--frozen-config", required=True, type=Path)
    parser.add_argument("--selection-lock", required=True, type=Path)
    parser.add_argument("--external-report", type=Path)
    parser.add_argument("--candidate-contract", type=Path)
    parser.add_argument("--training-report", type=Path)
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

    config = load_frozen_cooccur_lift_config(args.frozen_config)
    lock = _read_json(args.selection_lock)
    materialization = _read_json(args.test_materialization_report)
    lock_sha256 = _sha256(args.selection_lock)
    selected_weight = _validate_package_evidence(
        args=args,
        config=config,
        lock=lock,
        lock_sha256=lock_sha256,
        materialization=materialization,
    )

    probabilities = Path(str(materialization["probabilities"]))
    if _sha256(probabilities) != materialization.get(
        "probabilities_sha256"
    ):
        raise ValueError("test auxiliary probability hash differs")
    auxiliary_model = Path(str(materialization["auxiliary_model"]))
    if _sha256(auxiliary_model) != materialization.get(
        "auxiliary_model_sha256"
    ):
        raise ValueError("auxiliary model hash differs")

    report = build_partial_listwise_submission(
        champion_zip=args.champion_zip,
        expert_scores_path=probabilities,
        output_dir=args.output_dir,
        auxiliary_weight=selected_weight,
        expected_rows={"dataset1": 61_051, "dataset2": 153_420},
        expected_columns=100,
        expected_champion_zip_sha256=(
            args.expected_champion_zip_sha256
        ),
        expected_dataset1_sha256=args.expected_dataset1_sha256,
        expected_dataset2_sha256=args.expected_dataset2_sha256,
        expert_name=config.integration_id,
        expert_model_sha256=materialization[
            "auxiliary_model_sha256"
        ],
        candidate_manifest_sha256=materialization[
            "test_candidate_fingerprint"
        ],
        selection_lock_sha256=lock_sha256,
        expert_score_transform="setwise_probability",
        expert_source_sha256=materialization[
            "probabilities_sha256"
        ],
        dataset2_mode="cooccur_lift_aux_expert_blend",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _validate_package_evidence(
    *,
    args: argparse.Namespace,
    config: Any,
    lock: dict[str, Any],
    lock_sha256: str,
    materialization: dict[str, Any],
) -> float:
    if (
        materialization.get("status")
        != "complete_online_candidate_materialization"
        or materialization.get("production_checkpoint_modified") is not False
        or materialization.get("selection_lock_sha256") != lock_sha256
        or materialization.get("integration_id") != config.integration_id
    ):
        raise ValueError("test materialization evidence differs")

    bugfixed_paths = (
        args.candidate_contract,
        args.training_report,
    )
    if any(path is not None for path in bugfixed_paths):
        if not all(path is not None for path in bugfixed_paths):
            raise ValueError(
                "bugfixed v1 package requires candidate and training reports"
            )
        if args.external_report is not None:
            raise ValueError(
                "bugfixed v1 package must not reuse historical external evidence"
            )
        contract = _read_json(args.candidate_contract)
        training = _read_json(args.training_report)
        auxiliary_model = Path(str(materialization["auxiliary_model"]))
        source_checkpoint = Path(
            str(materialization["source_checkpoint"])
        )
        evidence = validate_bugfixed_v1_materialization_inputs(
            contract=contract,
            contract_sha256=_sha256(args.candidate_contract),
            training_report=training,
            actual_model_sha256=_sha256(auxiliary_model),
            actual_source_checkpoint_sha256=_sha256(source_checkpoint),
        )
        if (
            materialization.get("candidate_id")
            != evidence["candidate_id"]
            or materialization.get("evidence_mode")
            != "bugfixed_v1_training_evidence"
            or materialization.get("scoring_device") != "cuda"
            or materialization.get("candidate_contract_sha256")
            != _sha256(args.candidate_contract)
            or materialization.get("training_report_sha256")
            != _sha256(args.training_report)
            or materialization.get("external_report_reused") is not False
        ):
            raise ValueError("bugfixed v1 materialization binding differs")
        selected_weight = float(evidence["selected_weight"])
    else:
        if args.external_report is None:
            raise ValueError(
                "historical package requires an accepted external report"
            )
        external = _read_json(args.external_report)
        external_sha256 = _sha256(args.external_report)
        if (
            external.get("status") != "accepted"
            or external.get("integration_id") != config.integration_id
            or external.get("selection_lock_sha256") != lock_sha256
            or materialization.get("external_report_sha256")
            != external_sha256
        ):
            raise ValueError("accepted historical external evidence differs")
        selected_weight = float(external.get("selected_weight"))

    if (
        selected_weight != 0.5
        or float(lock.get("selected_weight", -1.0)) != selected_weight
        or float(materialization.get("selected_weight", -2.0))
        != selected_weight
        or lock.get("integration_id") != config.integration_id
    ):
        raise ValueError("evidence selected weight differs")
    return selected_weight


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
