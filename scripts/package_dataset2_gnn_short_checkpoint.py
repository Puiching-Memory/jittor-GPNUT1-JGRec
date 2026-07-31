from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from jgrec.core.io import discover_datasets
from jgrec.core.types import DatasetResult, TrainingReport
from jgrec.submission import (
    expected_test_rows,
    validate_submission_file,
    write_zip,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Package the current Dataset1 champion with the verified "
            "short_none 50/40k Dataset2 checkpoint replay."
        )
    )
    parser.add_argument("--dataset1-csv", required=True, type=Path)
    parser.add_argument("--dataset2-csv", required=True, type=Path)
    parser.add_argument("--replay-report", required=True, type=Path)
    parser.add_argument("--checkpoint-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--dataset1-sha256", required=True)
    parser.add_argument("--dataset2-sha256", required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    replay = _read_json(args.replay_report)
    checkpoint = _read_json(args.checkpoint_report)
    if (
        replay.get("status") != "passed"
        or not replay.get("byte_identical")
        or replay.get("standard_load_replays") != 2
    ):
        raise RuntimeError("two-load replay report did not authorize packaging")
    if (
        checkpoint.get("status") != "complete"
        or not checkpoint.get("standard_hydrate_passed")
        or checkpoint.get("encoder_retrained")
    ):
        raise RuntimeError("checkpoint integration report did not authorize packaging")
    _require_hash(args.dataset1_csv, args.dataset1_sha256, "Dataset1 CSV")
    _require_hash(args.dataset2_csv, args.dataset2_sha256, "Dataset2 CSV")
    if replay["replay_a_sha256"] != args.dataset2_sha256:
        raise RuntimeError("Dataset2 CSV differs from replay report")

    datasets = {
        dataset.name: dataset
        for dataset in discover_datasets(args.data_dir)
    }
    dataset1_rows = expected_test_rows(datasets["dataset1"])
    dataset2_rows = expected_test_rows(datasets["dataset2"])
    validate_submission_file(args.dataset1_csv, expected_rows=dataset1_rows)
    validate_submission_file(args.dataset2_csv, expected_rows=dataset2_rows)

    csv_dir = args.output_dir / "csv"
    csv_dir.mkdir(parents=True)
    dataset1_output = csv_dir / "dataset1.csv"
    dataset2_output = csv_dir / "dataset2.csv"
    shutil.copyfile(args.dataset1_csv, dataset1_output)
    shutil.copyfile(args.dataset2_csv, dataset2_output)
    _require_hash(dataset1_output, args.dataset1_sha256, "copied Dataset1")
    _require_hash(dataset2_output, args.dataset2_sha256, "copied Dataset2")

    dataset1_result = DatasetResult(
        name="dataset1",
        rows=dataset1_rows,
        output_path=dataset1_output,
        training_report=TrainingReport(model_name="hybrid"),
    )
    dataset2_result = DatasetResult(
        name="dataset2",
        rows=dataset2_rows,
        output_path=dataset2_output,
        training_report=TrainingReport(model_name="hybrid"),
    )
    zip_path = args.output_dir / "result.zip"
    write_zip([dataset1_result, dataset2_result], zip_path)
    report = {
        "status": "complete",
        "submission_authorized": False,
        "dataset1_mode": "current_gamma050_champion_byte_copy",
        "dataset1_rows": dataset1_rows,
        "dataset1_sha256": _sha256(dataset1_output),
        "dataset2_mode": "gnn_short_none_e50_edges40000_setwise_w080",
        "dataset2_rows": dataset2_rows,
        "dataset2_sha256": _sha256(dataset2_output),
        "checkpoint": checkpoint["output_checkpoint"],
        "checkpoint_sha256": checkpoint["output_checkpoint_sha256"],
        "checkpoint_report": str(args.checkpoint_report.resolve()),
        "checkpoint_report_sha256": _sha256(args.checkpoint_report),
        "replay_report": str(args.replay_report.resolve()),
        "replay_report_sha256": _sha256(args.replay_report),
        "result_zip": str(zip_path.resolve()),
        "result_zip_bytes": zip_path.stat().st_size,
        "result_zip_sha256": _sha256(zip_path),
    }
    _write_json(args.output_dir / "candidate-report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch: actual={actual} expected={expected}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
