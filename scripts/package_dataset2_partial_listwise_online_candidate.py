from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from jgrec.partial_listwise_submission import (
    build_partial_listwise_submission,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Package the byte-identical Dataset1 champion with the frozen "
            "Dataset2 Two-Tower 0.20 partial blend."
        )
    )
    parser.add_argument("--champion-zip", required=True, type=Path)
    parser.add_argument("--expert-scores", required=True, type=Path)
    parser.add_argument(
        "--expert-materialization-report",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--auxiliary-candidate-report",
        required=True,
        type=Path,
    )
    parser.add_argument("--delivery-lock", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-champion-zip-sha256", required=True)
    parser.add_argument("--expected-dataset1-sha256", required=True)
    parser.add_argument("--expected-dataset2-sha256", required=True)
    parser.add_argument("--expected-auxiliary-zip-sha256", required=True)
    parser.add_argument("--expected-auxiliary-member-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-test-csv-sha256", required=True)
    parser.add_argument("--expected-delivery-lock-sha256", required=True)
    args = parser.parse_args()

    materialization = json.loads(
        args.expert_materialization_report.read_text(encoding="utf-8")
    )
    if materialization.get("status") != "passed":
        raise ValueError("expert materialization report is not passed")
    if (
        materialization["source_zip_sha256"]
        != args.expected_auxiliary_zip_sha256
    ):
        raise ValueError("expert materialization ZIP hash differs")
    if (
        materialization["source_member_sha256"]
        != args.expected_auxiliary_member_sha256
    ):
        raise ValueError("expert materialization member hash differs")
    auxiliary_report = json.loads(
        args.auxiliary_candidate_report.read_text(encoding="utf-8")
    )
    if (
        auxiliary_report["result_zip_sha256"]
        != args.expected_auxiliary_zip_sha256
        or auxiliary_report["dataset2_sha256"]
        != args.expected_auxiliary_member_sha256
        or auxiliary_report["output_checkpoint_sha256"]
        != args.expected_checkpoint_sha256
    ):
        raise ValueError("auxiliary candidate provenance differs")
    delivery_lock = json.loads(
        args.delivery_lock.read_text(encoding="utf-8")
    )
    actual_delivery_lock_sha256 = _sha256(args.delivery_lock)
    if (
        delivery_lock.get("status")
        != "frozen_before_direct_candidate_blend"
        or actual_delivery_lock_sha256
        != args.expected_delivery_lock_sha256
    ):
        raise ValueError("delivery lock differs")
    locked_test_csv_sha256 = delivery_lock.get(
        "dataset2_test_csv_sha256"
    )
    if locked_test_csv_sha256 != args.expected_test_csv_sha256:
        raise ValueError("Dataset2 test CSV hash differs")

    report = build_partial_listwise_submission(
        champion_zip=args.champion_zip,
        expert_scores_path=args.expert_scores,
        output_dir=args.output_dir,
        auxiliary_weight=0.20,
        expected_rows={"dataset1": 61_051, "dataset2": 153_420},
        expected_columns=100,
        expected_champion_zip_sha256=(
            args.expected_champion_zip_sha256
        ),
        expected_dataset1_sha256=args.expected_dataset1_sha256,
        expected_dataset2_sha256=args.expected_dataset2_sha256,
        expert_name="listwise_two_tower_full_reranker",
        expert_model_sha256=args.expected_checkpoint_sha256,
        candidate_manifest_sha256=locked_test_csv_sha256,
        selection_lock_sha256=actual_delivery_lock_sha256,
        expert_score_transform="persisted_full_reranker_probability",
        expert_source_sha256=args.expected_auxiliary_zip_sha256,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
