from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.core.io import discover_datasets
from jgrec.core.memory import release_memory
from jgrec.core.runner import build_dataset_submission
from jgrec.rankers.hybrid.base_context_gate import (
    BASE_CONTEXT_INTEGRATION_ID,
)
from jgrec.rankers.registry import create_ranker
from jgrec.submission import (
    expected_test_rows,
    validate_submission_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the accepted Dataset1 base-context checkpoint twice "
            "through the standard hydration and prediction path."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.time()
    checkpoint_report = _read_json(args.checkpoint_report)
    if (
        checkpoint_report.get("status") != "complete"
        or checkpoint_report.get("package_authorized") is not True
        or checkpoint_report.get("integration_id")
        != BASE_CONTEXT_INTEGRATION_ID
        or checkpoint_report.get("standard_hydrate_passed") is not True
    ):
        raise ValueError("checkpoint report does not authorize replay")
    _require_hash(
        args.checkpoint,
        checkpoint_report["output_checkpoint_sha256"],
        "candidate checkpoint",
    )
    datasets = {
        dataset.name: dataset
        for dataset in discover_datasets(args.data_dir)
    }
    dataset1 = datasets["dataset1"]
    expected_rows = expected_test_rows(dataset1)

    replay_paths: list[Path] = []
    for replay_name in ("replay-a", "replay-b"):
        output_dir = args.output_dir / replay_name
        state = load_checkpoint_dataset(args.checkpoint, "dataset1")
        ranker = create_ranker("hybrid", None)
        ranker.hydrate(state)
        result = build_dataset_submission(
            dataset=dataset1,
            ranker=ranker,
            output_dir=output_dir,
            batch_size=args.batch_size,
            verbose=True,
            fit_ranker=False,
        )
        path = Path(result.output_path)
        validate_submission_file(path, expected_rows=expected_rows)
        replay_paths.append(path)
        del result, ranker, state
        gc.collect()
        release_memory()

    replay_a, replay_b = replay_paths
    replay_a_hash = _sha256(replay_a)
    replay_b_hash = _sha256(replay_b)
    if replay_a_hash != replay_b_hash or not _files_equal(
        replay_a,
        replay_b,
    ):
        raise RuntimeError(
            "independent Dataset1 checkpoint replays are not byte-identical"
        )
    tie_report = _diagnose_csv_ties(replay_a)
    if tie_report["rows_with_exact_ties"] != 0:
        raise RuntimeError("Dataset1 candidate replay contains exact ties")
    report = {
        "status": "passed",
        "integration_id": BASE_CONTEXT_INTEGRATION_ID,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_report": str(args.checkpoint_report.resolve()),
        "checkpoint_report_sha256": _sha256(args.checkpoint_report),
        "replay_a": str(replay_a.resolve()),
        "replay_a_sha256": replay_a_hash,
        "replay_b": str(replay_b.resolve()),
        "replay_b_sha256": replay_b_hash,
        "dataset1_rows": expected_rows,
        "byte_identical": True,
        "standard_load_replays": 2,
        "tie_report": tie_report,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.output_dir / "replay-report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


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
