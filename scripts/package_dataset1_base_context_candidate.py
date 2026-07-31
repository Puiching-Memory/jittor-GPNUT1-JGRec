from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.core.io import discover_datasets
from jgrec.core.types import DatasetResult, TrainingReport
from jgrec.rankers.hybrid.base_context_gate import (
    BASE_CONTEXT_INTEGRATION_ID,
)
from jgrec.submission import (
    expected_test_rows,
    validate_submission_file,
    write_zip,
)

CHAMPION_TIE_SAFE_ZIP_SHA256 = (
    "085da277f6f20429a2f9e4872438de2f7dca672eea41ba5a8e7fe1d99fb50730"
)
CHAMPION_TIE_SAFE_DATASET2_SHA256 = (
    "fc917052556abcb234cf8c55cd1417aa2550ea3fd013d0d8eae7d84041e107bb"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Package the twice-replayed Dataset1 base-context candidate "
            "with the byte-identical current tie-safe Dataset2 champion."
        )
    )
    parser.add_argument("--dataset1-csv", required=True, type=Path)
    parser.add_argument("--replay-report", required=True, type=Path)
    parser.add_argument("--checkpoint-report", required=True, type=Path)
    parser.add_argument("--champion-result-zip", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    started = time.time()
    replay = _read_json(args.replay_report)
    checkpoint = _read_json(args.checkpoint_report)
    _validate_reports(replay, checkpoint)
    _require_hash(
        args.dataset1_csv,
        replay["replay_a_sha256"],
        "Dataset1 replay",
    )
    _require_hash(
        args.champion_result_zip,
        CHAMPION_TIE_SAFE_ZIP_SHA256,
        "tie-safe champion package",
    )

    datasets = {
        dataset.name: dataset
        for dataset in discover_datasets(args.data_dir)
    }
    dataset1_rows = expected_test_rows(datasets["dataset1"])
    dataset2_rows = expected_test_rows(datasets["dataset2"])
    args.output_dir.mkdir(parents=True)
    csv_dir = args.output_dir / "csv"
    csv_dir.mkdir()
    dataset1_output = csv_dir / "dataset1.csv"
    dataset2_output = csv_dir / "dataset2.csv"
    shutil.copyfile(args.dataset1_csv, dataset1_output)
    with zipfile.ZipFile(args.champion_result_zip, "r") as archive:
        if archive.testzip() is not None:
            raise ValueError("tie-safe champion ZIP failed CRC validation")
        if set(archive.namelist()) != {"dataset1.csv", "dataset2.csv"}:
            raise ValueError("tie-safe champion ZIP has unexpected members")
        with (
            archive.open("dataset2.csv", "r") as source,
            dataset2_output.open("wb") as destination,
        ):
            shutil.copyfileobj(source, destination)

    validate_submission_file(
        dataset1_output,
        expected_rows=dataset1_rows,
    )
    validate_submission_file(
        dataset2_output,
        expected_rows=dataset2_rows,
    )
    _require_hash(
        dataset2_output,
        CHAMPION_TIE_SAFE_DATASET2_SHA256,
        "extracted tie-safe Dataset2",
    )
    dataset1_ties = _diagnose_csv_ties(dataset1_output)
    dataset2_ties = _diagnose_csv_ties(dataset2_output)
    if (
        dataset1_ties["rows_with_exact_ties"] != 0
        or dataset2_ties["rows_with_exact_ties"] != 0
    ):
        raise RuntimeError("candidate package contains exact score ties")

    results = [
        DatasetResult(
            name="dataset1",
            rows=dataset1_rows,
            output_path=dataset1_output,
            training_report=TrainingReport(model_name="hybrid"),
        ),
        DatasetResult(
            name="dataset2",
            rows=dataset2_rows,
            output_path=dataset2_output,
            training_report=TrainingReport(model_name="hybrid"),
        ),
    ]
    zip_path = args.output_dir / "result.zip"
    write_zip(results, zip_path)
    with zipfile.ZipFile(zip_path, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"candidate ZIP CRC failed: {bad_member}")
        members = archive.namelist()
    report = {
        "status": "complete",
        "package_gate_passed": True,
        "submission_authorized": False,
        "integration_id": BASE_CONTEXT_INTEGRATION_ID,
        "checkpoint": checkpoint["output_checkpoint"],
        "checkpoint_sha256": checkpoint["output_checkpoint_sha256"],
        "checkpoint_report": str(args.checkpoint_report.resolve()),
        "checkpoint_report_sha256": _sha256(args.checkpoint_report),
        "replay_report": str(args.replay_report.resolve()),
        "replay_report_sha256": _sha256(args.replay_report),
        "dataset1_mode": "locked_base_context_v1_time_ramp_g050",
        "dataset1_rows": dataset1_rows,
        "dataset1_sha256": _sha256(dataset1_output),
        "dataset1_ties": dataset1_ties,
        "dataset2_mode": "byte_copy_from_online_tie_safe_champion",
        "dataset2_source_zip_sha256": CHAMPION_TIE_SAFE_ZIP_SHA256,
        "dataset2_expected_sha256": (
            CHAMPION_TIE_SAFE_DATASET2_SHA256
        ),
        "dataset2_rows": dataset2_rows,
        "dataset2_sha256": _sha256(dataset2_output),
        "dataset2_ties": dataset2_ties,
        "result_zip": str(zip_path.resolve()),
        "result_zip_bytes": zip_path.stat().st_size,
        "result_zip_sha256": _sha256(zip_path),
        "zip_members": members,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.output_dir / "candidate-report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


def _validate_reports(
    replay: dict[str, Any],
    checkpoint: dict[str, Any],
) -> None:
    if (
        checkpoint.get("status") != "complete"
        or checkpoint.get("package_authorized") is not True
        or checkpoint.get("integration_id")
        != BASE_CONTEXT_INTEGRATION_ID
        or checkpoint.get("standard_hydrate_passed") is not True
    ):
        raise ValueError("checkpoint report does not authorize packaging")
    if (
        replay.get("status") != "passed"
        or replay.get("integration_id") != BASE_CONTEXT_INTEGRATION_ID
        or replay.get("byte_identical") is not True
        or int(replay.get("standard_load_replays", 0)) != 2
        or replay.get("checkpoint_sha256")
        != checkpoint.get("output_checkpoint_sha256")
        or replay.get("tie_report", {}).get("rows_with_exact_ties") != 0
    ):
        raise ValueError("replay report does not authorize packaging")


def _diagnose_csv_ties(path: Path) -> dict[str, int]:
    rows = 0
    rows_with_ties = 0
    duplicate_adjacencies = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            values = np.fromstring(line, sep=",", dtype=np.float64)
            if values.size == 0:
                raise ValueError(f"{path}:{line_number} is empty")
            ordered = np.sort(values)
            duplicates = int(
                np.count_nonzero(ordered[1:] == ordered[:-1])
            )
            rows += 1
            rows_with_ties += int(duplicates > 0)
            duplicate_adjacencies += duplicates
    return {
        "rows": rows,
        "rows_with_exact_ties": rows_with_ties,
        "duplicate_adjacencies": duplicate_adjacencies,
    }


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch: actual={actual} expected={expected}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
