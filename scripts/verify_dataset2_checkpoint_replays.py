from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from jgrec.contest_checkpoint import load_checkpoint_metadata
from jgrec.core.io import discover_datasets
from jgrec.submission import expected_test_rows, validate_submission_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify two independent standard-load Dataset2 checkpoint replays "
            "produce byte-identical submission CSVs."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--integration-report", required=True, type=Path)
    parser.add_argument("--replay-a", required=True, type=Path)
    parser.add_argument("--replay-b", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    integration = _read_json(args.integration_report)
    if integration.get("status") != "complete":
        raise RuntimeError("checkpoint integration report is incomplete")
    _require_hash(
        args.checkpoint,
        integration["output_checkpoint_sha256"],
        "checkpoint",
    )
    metadata = load_checkpoint_metadata(args.checkpoint)
    if metadata.get("dataset2_integration") != (
        "gnn_short_none_e50_edges40000_setwise"
    ):
        raise RuntimeError("checkpoint metadata has the wrong integration")

    datasets = {
        dataset.name: dataset
        for dataset in discover_datasets(args.data_dir)
    }
    dataset2 = datasets["dataset2"]
    rows = expected_test_rows(dataset2)
    validate_submission_file(args.replay_a, expected_rows=rows)
    validate_submission_file(args.replay_b, expected_rows=rows)
    replay_a_hash = _sha256(args.replay_a)
    replay_b_hash = _sha256(args.replay_b)
    if replay_a_hash != replay_b_hash:
        raise RuntimeError(
            "independent checkpoint replays produced different Dataset2 CSVs"
        )
    if not _files_equal(args.replay_a, args.replay_b):
        raise RuntimeError("replay hashes matched but bytes differ")

    report = {
        "status": "passed",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "integration_report": str(args.integration_report.resolve()),
        "integration_report_sha256": _sha256(args.integration_report),
        "replay_a": str(args.replay_a.resolve()),
        "replay_a_sha256": replay_a_hash,
        "replay_b": str(args.replay_b.resolve()),
        "replay_b_sha256": replay_b_hash,
        "dataset2_rows": rows,
        "byte_identical": True,
        "standard_load_replays": 2,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_report, report)
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


def _files_equal(first: Path, second: Path) -> bool:
    if first.stat().st_size != second.stat().st_size:
        return False
    with first.open("rb") as left, second.open("rb") as right:
        while True:
            left_block = left.read(1024 * 1024)
            right_block = right.read(1024 * 1024)
            if left_block != right_block:
                return False
            if not left_block:
                return True


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
