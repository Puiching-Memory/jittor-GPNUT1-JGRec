from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.cooccur_lift_promotion import TieSafeServiceComparison
from jgrec.rankers.hybrid.cooccur_lift import INTEGRATION_ID

PROMOTED_CHECKPOINT_SHA256 = (
    "796d8d21a0c706ad11f244385b314d471d522c3b807748a54fe4ac78722f5880"
)
ACCEPTED_CANDIDATE_SHA256 = (
    "7ebfeb7ea29d8dcd03a43a7433a43ddc8de0e24245d93115ecbf8ebd17ef50eb"
)
PREWIRING_RECEIPT_SHA256 = (
    "ff45301fcc3ec797477c17c72ea9687a5507dc32e1dd2299fac178245b60a7ee"
)
EXPECTED_REPLAY_SHA256 = (
    "2b4012edb3a9d18675b00417553b0366438db44a9d5680a7566e693cd20b21e0"
)
EXPECTED_ROWS = 153_420
EXPECTED_CANDIDATES = 100
NUMERIC_TOLERANCE = 5e-7


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize the user-authorized raw/tie-safe equivalence split "
            "for the Dataset2 cooccur-lift checkpoint promotion."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-report", required=True, type=Path)
    parser.add_argument("--candidate-zip", required=True, type=Path)
    parser.add_argument("--replay-a", required=True, type=Path)
    parser.add_argument("--replay-b", required=True, type=Path)
    parser.add_argument("--bounded-raw-log", required=True, type=Path)
    parser.add_argument("--prewiring-receipt", required=True, type=Path)
    parser.add_argument("--superseded-status", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--output-status", required=True, type=Path)
    args = parser.parse_args()

    for path in (
        args.output_report,
        args.output_manifest,
        args.output_status,
    ):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite: {path}")

    checkpoint_report = _read_json(args.checkpoint_report)
    prewiring_receipt = _read_json(args.prewiring_receipt)
    superseded_status = _read_json(args.superseded_status)
    _validate_checkpoint_report(checkpoint_report)
    _validate_prewiring_receipt(
        prewiring_receipt,
        checkpoint_report=checkpoint_report,
    )
    _validate_superseded_status(superseded_status)

    _require_hash(
        args.checkpoint,
        PROMOTED_CHECKPOINT_SHA256,
        "promoted checkpoint",
    )
    _require_hash(
        args.candidate_zip,
        ACCEPTED_CANDIDATE_SHA256,
        "accepted candidate ZIP",
    )
    _require_hash(
        args.prewiring_receipt,
        PREWIRING_RECEIPT_SHA256,
        "immutable pre-wiring receipt",
    )

    replay_a_sha256 = _sha256(args.replay_a)
    replay_b_sha256 = _sha256(args.replay_b)
    if (
        replay_a_sha256 != EXPECTED_REPLAY_SHA256
        or replay_b_sha256 != EXPECTED_REPLAY_SHA256
        or not _files_equal(args.replay_a, args.replay_b)
    ):
        raise RuntimeError(
            "full standard-load replays are not the frozen byte-identical "
            "artifacts"
        )

    raw_evidence = _read_last_json_object(args.bounded_raw_log)
    _validate_raw_evidence(raw_evidence)
    service_comparison = _compare_service_boundary(
        replay_path=args.replay_a,
        candidate_zip=args.candidate_zip,
    )
    if (
        service_comparison["status"] != "passed"
        or service_comparison["rows"] != EXPECTED_ROWS
        or service_comparison["candidate_count"] != EXPECTED_CANDIDATES
    ):
        raise RuntimeError("tie-safe service equivalence did not pass")

    checkpoint_report_sha256 = _sha256(args.checkpoint_report)
    bounded_raw_log_sha256 = _sha256(args.bounded_raw_log)
    superseded_status_sha256 = _sha256(args.superseded_status)
    report = {
        "schema_version": 1,
        "status": "accepted",
        "promotion_status": "promoted",
        "integration_id": INTEGRATION_ID,
        "selected_weight": 0.5,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": PROMOTED_CHECKPOINT_SHA256,
        "checkpoint_bytes": int(args.checkpoint.stat().st_size),
        "checkpoint_report": str(args.checkpoint_report.resolve()),
        "checkpoint_report_sha256": checkpoint_report_sha256,
        "accepted_candidate_zip": str(args.candidate_zip.resolve()),
        "accepted_candidate_zip_sha256": ACCEPTED_CANDIDATE_SHA256,
        "replay_a": str(args.replay_a.resolve()),
        "replay_a_sha256": replay_a_sha256,
        "replay_b": str(args.replay_b.resolve()),
        "replay_b_sha256": replay_b_sha256,
        "byte_identical_full_replays": True,
        "standard_load_replays": 2,
        "dataset2_rows": EXPECTED_ROWS,
        "raw_model_equivalence": {
            "status": "passed",
            "contract": (
                "immutable_artifact_identity_plus_bounded_sequential_"
                "worst_known_drift_batch_numeric_replay"
            ),
            "checkpoint_state_identity": {
                "source_checkpoint_sha256": checkpoint_report[
                    "source_checkpoint_sha256"
                ],
                "auxiliary_model_sha256": checkpoint_report[
                    "auxiliary_model_sha256"
                ],
                "lift_features_sha256": checkpoint_report[
                    "auxiliary_state_audit"
                ]["lift_features_sha256"],
                "query_fingerprints_sha256": checkpoint_report[
                    "auxiliary_state_audit"
                ]["query_fingerprints_sha256"],
                "dataset1_pickle_sha256": checkpoint_report[
                    "dataset1_pickle_sha256"
                ],
                "dataset1_reload_pickle_sha256": checkpoint_report[
                    "dataset1_reload_pickle_sha256"
                ],
                "protected_dataset2_top_level_sha256": checkpoint_report[
                    "protected_dataset2_top_level_sha256"
                ],
            },
            "bounded_raw_log": str(args.bounded_raw_log.resolve()),
            "bounded_raw_log_sha256": bounded_raw_log_sha256,
            "bounded_sequential_evidence": raw_evidence,
            "maximum_allowed_absolute_error": NUMERIC_TOLERANCE,
            "full_153420_row_raw_numeric_replay_claimed": False,
            "coverage_note": (
                "Artifact hashes cover the complete model and feature "
                "contract; numeric replay covers twelve sequential batches "
                "through the worst-known drift row."
            ),
        },
        "tie_safe_service_equivalence": {
            **service_comparison,
            "replay_byte_identity_required": True,
            "replay_byte_identity_passed": True,
            "full_order_equivalence_claimed": False,
            "boundary_explanation": (
                "The accepted ZIP blends an already tie-safe champion CSV "
                "with stored auxiliary probabilities and writes eight "
                "decimals. Standard serving blends checkpoint heads first, "
                "then applies final tie handling and writes seventeen "
                "significant digits."
            ),
        },
        "protocol_amendment": {
            "status": "user_authorized",
            "authorized_in_conversation": True,
            "authorized_on": "2026-07-29",
            "change": (
                "split raw-model numeric equivalence from tie-safe service "
                "equivalence"
            ),
            "formula_changed": False,
            "weight_changed": False,
            "model_retrained": False,
            "metrics_reselected": False,
        },
        "prewiring_receipt": str(args.prewiring_receipt.resolve()),
        "prewiring_receipt_sha256": PREWIRING_RECEIPT_SHA256,
        "superseded_failed_status": {
            "path": str(args.superseded_status.resolve()),
            "sha256": superseded_status_sha256,
            "payload": superseded_status,
            "preserved_unchanged": True,
            "superseded_reason": (
                "obsolete combined final-CSV numeric gate crossed a "
                "serialization and tie-postprocessing boundary"
            ),
        },
        "package_authorized": True,
        "promotion_authorized": True,
    }
    _write_json_atomic(args.output_report, report)
    report_sha256 = _sha256(args.output_report)

    manifest = {
        "schema_version": 1,
        "status": "promoted",
        "decision_status": "accepted",
        "promotion_status": "promoted",
        "integration_id": INTEGRATION_ID,
        "champion_checkpoint": str(args.checkpoint.resolve()),
        "champion_checkpoint_sha256": PROMOTED_CHECKPOINT_SHA256,
        "champion_checkpoint_bytes": int(args.checkpoint.stat().st_size),
        "selected_weight": 0.5,
        "online_score": float(prewiring_receipt["online_score"]),
        "accepted_candidate_zip_sha256": ACCEPTED_CANDIDATE_SHA256,
        "prewiring_receipt_sha256": PREWIRING_RECEIPT_SHA256,
        "checkpoint_report_sha256": checkpoint_report_sha256,
        "replay_report": str(args.output_report.resolve()),
        "replay_report_sha256": report_sha256,
        "replay_sha256": EXPECTED_REPLAY_SHA256,
        "raw_model_equivalence": "passed",
        "tie_safe_service_equivalence": "passed",
        "user_authorized_protocol_amendment": True,
        "package_authorized": True,
        "promotion_authorized": True,
        "prohibited_actions_observed": False,
    }
    _write_json_atomic(args.output_manifest, manifest)
    manifest_sha256 = _sha256(args.output_manifest)

    status = {
        "schema_version": 1,
        "status": "accepted",
        "promotion_status": "promoted",
        "integration_id": INTEGRATION_ID,
        "checkpoint_sha256": PROMOTED_CHECKPOINT_SHA256,
        "replay_report_sha256": report_sha256,
        "promoted_manifest_sha256": manifest_sha256,
        "supersedes": str(args.superseded_status.resolve()),
        "superseded_status_sha256": superseded_status_sha256,
    }
    _write_json_atomic(args.output_status, status)
    print(
        json.dumps(
            {
                "status": status,
                "manifest": manifest,
                "service_comparison": service_comparison,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _validate_checkpoint_report(report: dict[str, Any]) -> None:
    if (
        report.get("status") != "complete"
        or report.get("integration_id") != INTEGRATION_ID
        or report.get("standard_hydrate_passed") is not True
        or report.get("double_replay_required") is not True
        or float(report.get("selected_weight", -1.0)) != 0.5
        or report.get("output_checkpoint_sha256")
        != PROMOTED_CHECKPOINT_SHA256
        or report.get("candidate_zip_sha256")
        != ACCEPTED_CANDIDATE_SHA256
        or report.get("promotion_receipt_sha256")
        != PREWIRING_RECEIPT_SHA256
        or report.get("dataset1_pickle_sha256")
        != report.get("dataset1_reload_pickle_sha256")
    ):
        raise ValueError("checkpoint integration report differs")


def _validate_prewiring_receipt(
    receipt: dict[str, Any],
    *,
    checkpoint_report: dict[str, Any],
) -> None:
    if (
        receipt.get("status")
        != "online_score_passed_before_checkpoint_wiring"
        or receipt.get("integration_id") != INTEGRATION_ID
        or receipt.get("candidate_zip_sha256")
        != ACCEPTED_CANDIDATE_SHA256
        or float(receipt.get("selected_weight", -1.0)) != 0.5
        or receipt.get("checkpoint_wiring_authorized") is not True
        or receipt.get("double_replay_required") is not True
        or receipt.get("weight_rescan_authorized") is not False
        or receipt.get("formula_change_authorized") is not False
        or receipt.get("model_retraining_authorized") is not False
        or receipt.get("source_checkpoint_sha256")
        != checkpoint_report.get("source_checkpoint_sha256")
        or receipt.get("auxiliary_model_sha256")
        != checkpoint_report.get("auxiliary_model_sha256")
    ):
        raise ValueError("immutable pre-wiring receipt differs")


def _validate_superseded_status(status: dict[str, Any]) -> None:
    if (
        status.get("status") != "failed"
        or status.get("phase") != "full_double_replay"
    ):
        raise ValueError("superseded status is not the preserved old failure")


def _validate_raw_evidence(evidence: dict[str, Any]) -> None:
    batch = evidence.get("batch_summary")
    if not isinstance(batch, dict):
        raise ValueError("bounded raw evidence has no batch summary")
    if (
        evidence.get("status") != "passed"
        or int(evidence.get("sequential_batches", 0)) < 12
        or float(batch.get("maximum_absolute_error", float("inf")))
        > NUMERIC_TOLERANCE
        or int(batch.get("rows_above_5e_7", -1)) != 0
        or int(batch.get("values_above_5e_7", -1)) != 0
        or int(batch.get("top1_disagreements", -1)) != 0
        or float(
            evidence.get(
                "target_maximum_absolute_error",
                float("inf"),
            )
        )
        > NUMERIC_TOLERANCE
    ):
        raise ValueError("bounded raw-model equivalence did not pass")


def _compare_service_boundary(
    *,
    replay_path: Path,
    candidate_zip: Path,
) -> dict[str, object]:
    comparison = TieSafeServiceComparison(
        numeric_tolerance=NUMERIC_TOLERANCE,
        diagnostic_top_ks=(1, 3, 10),
    )
    accepted_rows: list[np.ndarray] = []
    served_rows: list[np.ndarray] = []
    with (
        zipfile.ZipFile(candidate_zip) as archive,
        archive.open("dataset2.csv", "r") as accepted,
        replay_path.open("rb") as served,
    ):
        while True:
            accepted_line = accepted.readline()
            served_line = served.readline()
            if not accepted_line and not served_line:
                break
            if not accepted_line or not served_line:
                raise ValueError("accepted and served row counts differ")
            accepted_rows.append(
                _parse_csv_line(
                    accepted_line,
                    label=f"accepted:{comparison.rows + len(accepted_rows) + 1}",
                )
            )
            served_rows.append(
                _parse_csv_line(
                    served_line,
                    label=f"served:{comparison.rows + len(served_rows) + 1}",
                )
            )
            if len(accepted_rows) == 1024:
                comparison.update(
                    np.stack(accepted_rows),
                    np.stack(served_rows),
                )
                accepted_rows.clear()
                served_rows.clear()
    if accepted_rows:
        comparison.update(
            np.stack(accepted_rows),
            np.stack(served_rows),
        )
    return comparison.finalize()


def _parse_csv_line(line: bytes, *, label: str) -> np.ndarray:
    values = np.fromstring(
        line.decode("ascii"),
        sep=",",
        dtype=np.float64,
    )
    if values.shape != (EXPECTED_CANDIDATES,) or not np.all(
        np.isfinite(values)
    ):
        raise ValueError(f"{label} is not a finite 100-column row")
    return values


def _read_last_json_object(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    position = 0
    last: dict[str, Any] | None = None
    while position < len(text):
        start = text.find("{", position)
        if start < 0:
            break
        try:
            payload, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            position = start + 1
            continue
        if isinstance(payload, dict):
            last = payload
        position = end
    if last is None:
        raise ValueError(f"{path} contains no JSON object")
    return last


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} hash differs: expected {expected}, got {actual}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files_equal(first: Path, second: Path) -> bool:
    if first.stat().st_size != second.stat().st_size:
        return False
    with first.open("rb") as left, second.open("rb") as right:
        while True:
            left_block = left.read(4 * 1024 * 1024)
            right_block = right.read(4 * 1024 * 1024)
            if left_block != right_block:
                return False
            if not left_block:
                return True


if __name__ == "__main__":
    raise SystemExit(main())
