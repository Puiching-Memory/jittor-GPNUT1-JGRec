from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.core.io import discover_datasets
from jgrec.core.runner import build_dataset_submission
from jgrec.core.types import DatasetResult, TrainingReport
from jgrec.rankers.registry import create_ranker
from jgrec.submission import expected_test_rows, validate_submission_file, write_zip


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Dataset1 pure-MLP control submission.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--champion-dataset2", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    datasets = {dataset.name: dataset for dataset in discover_datasets(args.data_dir)}
    dataset1 = datasets["dataset1"]
    dataset2 = datasets["dataset2"]
    csv_dir = args.output_dir / "csv"
    dataset1_output = csv_dir / "dataset1.csv"
    dataset2_output = csv_dir / "dataset2.csv"
    zip_path = args.output_dir / "result.zip"
    for target in (dataset1_output, dataset2_output, zip_path):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite existing control artifact: {target}")

    state = load_checkpoint_dataset(args.checkpoint, "dataset1")
    ranker = create_ranker("hybrid", None)
    ranker.hydrate(state)
    impl = getattr(ranker, "impl", None)
    if impl is None or impl.lgbm_result is None:
        raise ValueError("checkpoint does not contain an ensemble-capable hybrid ranker")
    impl.lgbm_result = None

    dataset1_result = build_dataset_submission(
        dataset=dataset1,
        ranker=ranker,
        output_dir=csv_dir,
        batch_size=args.batch_size,
        verbose=True,
        fit_ranker=False,
    )
    validate_submission_file(dataset1_output, expected_rows=expected_test_rows(dataset1))

    csv_dir.mkdir(parents=True, exist_ok=True)
    champion_dataset2_hash = _sha256(args.champion_dataset2)
    shutil.copyfile(args.champion_dataset2, dataset2_output)
    validate_submission_file(dataset2_output, expected_rows=expected_test_rows(dataset2))
    copied_dataset2_hash = _sha256(dataset2_output)
    if copied_dataset2_hash != champion_dataset2_hash:
        raise RuntimeError("copied Dataset2 CSV hash differs from the champion source")

    dataset2_result = DatasetResult(
        name="dataset2",
        rows=expected_test_rows(dataset2),
        output_path=dataset2_output,
        training_report=TrainingReport(model_name="hybrid"),
    )
    write_zip([dataset1_result, dataset2_result], zip_path)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset1_mode": "pure_mlp",
        "dataset1_rows": dataset1_result.rows,
        "dataset1_sha256": _sha256(dataset1_output),
        "dataset2_mode": "byte_copy_from_new_champion",
        "dataset2_rows": dataset2_result.rows,
        "dataset2_sha256": copied_dataset2_hash,
        "result_zip": str(zip_path.resolve()),
        "result_zip_bytes": zip_path.stat().st_size,
        "result_zip_sha256": _sha256(zip_path),
    }
    report_path = args.output_dir / "control-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
