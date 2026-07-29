from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
from pathlib import Path

from jgrec.contest_checkpoint import (
    ContestCheckpointWriter,
    load_checkpoint_dataset,
    load_checkpoint_metadata,
)
from jgrec.core.io import discover_datasets
from jgrec.core.runner import build_dataset_submission
from jgrec.core.types import DatasetResult, TrainingReport
from jgrec.rankers.hybrid.lgbm_tuning import apply_tuned_lgbm_result
from jgrec.rankers.registry import create_ranker
from jgrec.submission import expected_test_rows, validate_submission_file, write_zip


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Dataset2-only tuned LightGBM candidate.")
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--tuning-report", required=True, type=Path)
    parser.add_argument("--lgbm-model", required=True, type=Path)
    parser.add_argument("--champion-dataset1", required=True, type=Path)
    parser.add_argument("--output-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--allow-rejected-tuning",
        action="store_true",
        help="Explicitly allow an exploratory package from a tuning report that failed its offline gate.",
    )
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite candidate directory: {args.output_dir}")
    if args.output_checkpoint.exists() or args.output_checkpoint.with_suffix(
        f"{args.output_checkpoint.suffix}.tmp"
    ).exists():
        raise FileExistsError(f"refusing to overwrite candidate checkpoint: {args.output_checkpoint}")

    tuning_report = json.loads(args.tuning_report.read_text(encoding="utf-8"))
    model_text = args.lgbm_model.read_text(encoding="utf-8")
    source_metadata = load_checkpoint_metadata(args.source_checkpoint)
    extra_metadata = {
        key: value
        for key, value in source_metadata.items()
        if key not in {"format", "version", "model_name", "datasets"}
    }
    extra_metadata.update(
        {
            "derived_from": str(args.source_checkpoint.resolve()),
            "dataset2_lgbm_tuning_report": str(args.tuning_report.resolve()),
        }
    )
    writer = ContestCheckpointWriter(
        args.output_checkpoint,
        model_name=str(source_metadata["model_name"]),
        expected_datasets=tuple(source_metadata["datasets"]),
        metadata=extra_metadata,
    )
    dataset2_state = None
    try:
        dataset1_state = load_checkpoint_dataset(args.source_checkpoint, "dataset1")
        writer.add_dataset("dataset1", dataset1_state)
        del dataset1_state
        gc.collect()

        dataset2_state = load_checkpoint_dataset(args.source_checkpoint, "dataset2")
        current_lgbm = dataset2_state.get("lgbm_result")
        if current_lgbm is None:
            raise ValueError("source Dataset2 checkpoint has no LightGBM result")
        dataset2_state["lgbm_result"] = apply_tuned_lgbm_result(
            current_lgbm,
            model_text=model_text,
            report=tuning_report,
            allow_rejected_report=args.allow_rejected_tuning,
        )
        writer.add_dataset("dataset2", dataset2_state)
        writer.finalize()
    except Exception:
        writer.abort()
        raise

    if dataset2_state is None:
        raise RuntimeError("Dataset2 state was not created")
    datasets = {dataset.name: dataset for dataset in discover_datasets(args.data_dir)}
    dataset1 = datasets["dataset1"]
    dataset2 = datasets["dataset2"]
    csv_dir = args.output_dir / "csv"
    dataset1_output = csv_dir / "dataset1.csv"
    dataset2_output = csv_dir / "dataset2.csv"
    zip_path = args.output_dir / "result.zip"
    csv_dir.mkdir(parents=True, exist_ok=False)

    source_dataset1_hash = _sha256(args.champion_dataset1)
    shutil.copyfile(args.champion_dataset1, dataset1_output)
    validate_submission_file(dataset1_output, expected_rows=expected_test_rows(dataset1))
    copied_dataset1_hash = _sha256(dataset1_output)
    if copied_dataset1_hash != source_dataset1_hash:
        raise RuntimeError("copied Dataset1 CSV hash differs from the champion source")

    ranker = create_ranker("hybrid", None)
    ranker.hydrate(dataset2_state)
    del dataset2_state
    gc.collect()
    dataset2_result = build_dataset_submission(
        dataset=dataset2,
        ranker=ranker,
        output_dir=csv_dir,
        batch_size=args.batch_size,
        verbose=True,
        fit_ranker=False,
    )
    validate_submission_file(dataset2_output, expected_rows=expected_test_rows(dataset2))
    dataset1_result = DatasetResult(
        name="dataset1",
        rows=expected_test_rows(dataset1),
        output_path=dataset1_output,
        training_report=TrainingReport(model_name="hybrid"),
    )
    write_zip([dataset1_result, dataset2_result], zip_path)

    report = {
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "output_checkpoint": str(args.output_checkpoint.resolve()),
        "output_checkpoint_bytes": args.output_checkpoint.stat().st_size,
        "output_checkpoint_sha256": _sha256(args.output_checkpoint),
        "tuning_report": str(args.tuning_report.resolve()),
        "tuning_gate_passed": bool(tuning_report["gate_passed"]),
        "exploratory_override": bool(
            args.allow_rejected_tuning and not tuning_report["gate_passed"]
        ),
        "dataset2_winner": tuning_report["winner"],
        "dataset1_mode": "byte_copy_from_online_champion",
        "dataset1_rows": dataset1_result.rows,
        "dataset1_sha256": copied_dataset1_hash,
        "dataset2_mode": "tuned_lgbm_overlay",
        "dataset2_rows": dataset2_result.rows,
        "dataset2_sha256": _sha256(dataset2_output),
        "result_zip": str(zip_path.resolve()),
        "result_zip_bytes": zip_path.stat().st_size,
        "result_zip_sha256": _sha256(zip_path),
    }
    report_path = args.output_dir / "candidate-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    shutil.copyfile(args.tuning_report, args.output_dir / "tuning-report.json")
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
