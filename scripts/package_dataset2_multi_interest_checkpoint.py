from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.core.io import discover_datasets
from jgrec.core.runner import build_dataset_submission
from jgrec.core.types import DatasetResult, TrainingReport
from jgrec.rankers.registry import create_ranker
from jgrec.submission import (
    expected_test_rows,
    validate_submission_file,
    write_zip,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--proxy-report", required=True, type=Path)
    parser.add_argument("--champion-dataset1", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.time()

    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    ranker = create_ranker("hybrid", None)
    ranker.hydrate(state)
    if ranker.impl.multi_interest_proxy_state is None:
        raise RuntimeError("checkpoint has no multi-interest proxy state")
    datasets = {
        dataset.name: dataset
        for dataset in discover_datasets(args.data_dir)
    }
    csv_dir = args.output_dir / "csv"
    csv_dir.mkdir()
    dataset1_output = csv_dir / "dataset1.csv"
    dataset2_output = csv_dir / "dataset2.csv"
    shutil.copyfile(args.champion_dataset1, dataset1_output)
    validate_submission_file(
        dataset1_output,
        expected_rows=expected_test_rows(datasets["dataset1"]),
    )
    if _sha256(dataset1_output) != _sha256(args.champion_dataset1):
        raise RuntimeError("Dataset1 byte copy differs")
    dataset2_result = build_dataset_submission(
        dataset=datasets["dataset2"],
        ranker=ranker,
        output_dir=csv_dir,
        batch_size=args.batch_size,
        verbose=True,
        fit_ranker=False,
    )
    validate_submission_file(
        dataset2_output,
        expected_rows=expected_test_rows(datasets["dataset2"]),
    )
    dataset1_result = DatasetResult(
        name="dataset1",
        rows=expected_test_rows(datasets["dataset1"]),
        output_path=dataset1_output,
        training_report=TrainingReport(model_name="hybrid"),
    )
    zip_path = args.output_dir / "result.zip"
    write_zip([dataset1_result, dataset2_result], zip_path)
    report = {
        "status": "complete",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "checkpoint_sha256": _sha256(args.checkpoint),
        "proxy_report_sha256": _sha256(args.proxy_report),
        "dataset1_sha256": _sha256(dataset1_output),
        "dataset2_sha256": _sha256(dataset2_output),
        "result_zip": str(zip_path.resolve()),
        "result_zip_bytes": zip_path.stat().st_size,
        "result_zip_sha256": _sha256(zip_path),
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "candidate-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
