from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.cooccur_lift_promotion import TieSafeServiceComparison
from jgrec.core.io import discover_datasets
from jgrec.core.memory import release_memory
from jgrec.core.runner import build_dataset_submission
from jgrec.rankers.hybrid.cooccur_lift import INTEGRATION_ID
from jgrec.rankers.hybrid.cooccur_lift_checkpoint import (
    CooccurLiftAuxiliaryState,
)
from jgrec.rankers.registry import create_ranker
from jgrec.submission import (
    expected_test_rows,
    validate_submission_file,
)

MAXIMUM_ACCEPTED_ROUNDING_ERROR = 5e-7


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the promoted Dataset2 cooccur-lift checkpoint twice and "
            "compare it with the online-accepted ZIP."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-report", required=True, type=Path)
    parser.add_argument("--candidate-zip", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    report = _read_json(args.checkpoint_report)
    if (
        report.get("status") != "complete"
        or report.get("integration_id") != INTEGRATION_ID
        or report.get("standard_hydrate_passed") is not True
        or report.get("double_replay_required") is not True
        or float(report.get("selected_weight", -1.0)) != 0.5
    ):
        raise ValueError("checkpoint report does not authorize double replay")
    _require_hash(
        args.checkpoint,
        report["output_checkpoint_sha256"],
        "promoted checkpoint",
    )
    _require_hash(
        args.candidate_zip,
        report["candidate_zip_sha256"],
        "online candidate ZIP",
    )
    args.output_dir.mkdir(parents=True)
    started = time.time()

    datasets = {
        dataset.name: dataset
        for dataset in discover_datasets(args.data_dir)
    }
    dataset2 = datasets["dataset2"]
    expected_rows = expected_test_rows(dataset2)
    replay_paths: list[Path] = []
    replay_durations: dict[str, float] = {}

    for replay_name in ("replay-a", "replay-b"):
        replay_started = time.time()
        state = load_checkpoint_dataset(args.checkpoint, "dataset2")
        ranker = create_ranker("hybrid", None)
        ranker.hydrate(state)
        if not isinstance(
            ranker.impl.cooccur_lift_auxiliary_state,
            CooccurLiftAuxiliaryState,
        ):
            raise RuntimeError("standard load omitted cooccur-lift state")
        if ranker.impl.cooccur_lift_auxiliary_model is None:
            raise RuntimeError("standard load omitted cooccur-lift model")
        result = build_dataset_submission(
            dataset=dataset2,
            ranker=ranker,
            output_dir=args.output_dir / replay_name,
            batch_size=args.batch_size,
            verbose=True,
            fit_ranker=False,
        )
        path = Path(result.output_path)
        validate_submission_file(path, expected_rows=expected_rows)
        replay_paths.append(path)
        replay_durations[replay_name] = time.time() - replay_started
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
            "independent Dataset2 checkpoint replays are not byte-identical"
        )
    tie_report = _diagnose_csv_ties(replay_a)
    if tie_report["rows_with_exact_ties"] != 0:
        raise RuntimeError("Dataset2 checkpoint replay contains exact ties")
    accepted_comparison = _compare_with_zip_member(
        replay_a,
        args.candidate_zip,
        member="dataset2.csv",
    )
    if (
        accepted_comparison["status"] != "passed"
        or accepted_comparison["rows"] != expected_rows
    ):
        raise RuntimeError(
            "checkpoint replay is not tie-safe service equivalent to the "
            "online-accepted candidate"
        )

    final_report = {
        "status": "passed",
        "integration_id": INTEGRATION_ID,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": report["output_checkpoint_sha256"],
        "checkpoint_report": str(args.checkpoint_report.resolve()),
        "checkpoint_report_sha256": _sha256(args.checkpoint_report),
        "candidate_zip": str(args.candidate_zip.resolve()),
        "candidate_zip_sha256": report["candidate_zip_sha256"],
        "replay_a": str(replay_a.resolve()),
        "replay_a_sha256": replay_a_hash,
        "replay_b": str(replay_b.resolve()),
        "replay_b_sha256": replay_b_hash,
        "dataset2_rows": expected_rows,
        "byte_identical": True,
        "standard_load_replays": 2,
        "batch_size": args.batch_size,
        "replay_durations_seconds": replay_durations,
        "tie_report": tie_report,
        "tie_safe_service_equivalence": accepted_comparison,
        "protocol_amendment": (
            "raw numeric equivalence is verified before serving "
            "postprocessing; final CSV numeric deltas are diagnostic"
        ),
        "package_authorized": True,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.output_dir / "replay-report.json", final_report)
    print(json.dumps(final_report, indent=2, sort_keys=True), flush=True)
    return 0


def _compare_with_zip_member(
    replay_path: Path,
    candidate_zip: Path,
    *,
    member: str,
) -> dict[str, Any]:
    comparison = TieSafeServiceComparison(
        numeric_tolerance=MAXIMUM_ACCEPTED_ROUNDING_ERROR,
        diagnostic_top_ks=(1, 3, 10),
    )
    rows = 0
    with (
        zipfile.ZipFile(candidate_zip) as archive,
        archive.open(member, "r") as accepted,
        replay_path.open("rb") as replay,
    ):
        while True:
            accepted_line = accepted.readline()
            replay_line = replay.readline()
            if not accepted_line and not replay_line:
                break
            if not accepted_line or not replay_line:
                raise ValueError("accepted ZIP and replay row counts differ")
            accepted_values = _parse_csv_line(
                accepted_line,
                label=f"{member}:{rows + 1}",
            )
            replay_values = _parse_csv_line(
                replay_line,
                label=f"{replay_path}:{rows + 1}",
            )
            if accepted_values.shape != replay_values.shape:
                raise ValueError(
                    "accepted ZIP and replay column counts differ"
                )
            comparison.update(
                accepted_values[np.newaxis, :],
                replay_values[np.newaxis, :],
            )
            rows += 1
    report = comparison.finalize()
    report["member"] = member
    return report


def _parse_csv_line(line: bytes, *, label: str) -> np.ndarray:
    values = np.fromstring(
        line.decode("ascii"),
        sep=",",
        dtype=np.float64,
    )
    if values.shape != (100,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{label} is not a finite 100-column row")
    return values


def _diagnose_csv_ties(path: Path) -> dict[str, int]:
    rows = 0
    rows_with_ties = 0
    duplicate_adjacencies = 0
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            values = _parse_csv_line(
                line,
                label=f"{path}:{line_number}",
            )
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
            left_block = left.read(4 * 1024 * 1024)
            right_block = right.read(4 * 1024 * 1024)
            if left_block != right_block:
                return False
            if not left_block:
                return True


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
