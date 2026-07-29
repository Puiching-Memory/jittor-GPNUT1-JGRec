from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
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
from jgrec.rankers.hybrid.segment_fusion import SegmentGateResult
from jgrec.rankers.registry import create_ranker
from jgrec.submission import expected_test_rows, validate_submission_file, write_zip


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a guarded segment-fusion candidate.")
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--tuning-report", required=True, type=Path)
    parser.add_argument("--dataset1-gate", required=True, type=Path)
    parser.add_argument("--champion-dataset2", required=True, type=Path)
    parser.add_argument("--output-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite candidate directory: {args.output_dir}")
    checkpoint_temp = args.output_checkpoint.with_suffix(f"{args.output_checkpoint.suffix}.tmp")
    if args.output_checkpoint.exists() or checkpoint_temp.exists():
        raise FileExistsError(f"refusing to overwrite candidate checkpoint: {args.output_checkpoint}")

    tuning_report = json.loads(args.tuning_report.read_text(encoding="utf-8"))
    dataset_reports = tuning_report["datasets"]
    if not dataset_reports["dataset1"]["accepted"]:
        raise ValueError("Dataset1 segment gate did not pass the frozen validation protocol")
    if dataset_reports["dataset2"]["accepted"]:
        raise ValueError("this guarded build expects Dataset2 to retain the champion fallback")
    with args.dataset1_gate.open("rb") as handle:
        dataset1_gate = pickle.load(handle)
    if not isinstance(dataset1_gate, SegmentGateResult):
        raise TypeError("Dataset1 gate artifact has an incompatible type")

    source_metadata = load_checkpoint_metadata(args.source_checkpoint)
    extra_metadata = {
        key: value
        for key, value in source_metadata.items()
        if key not in {"format", "version", "model_name", "datasets"}
    }
    extra_metadata.update(
        {
            "derived_from": str(args.source_checkpoint.resolve()),
            "segment_fusion_report": str(args.tuning_report.resolve()),
            "segment_fusion_datasets": ("dataset1",),
        }
    )
    writer = ContestCheckpointWriter(
        args.output_checkpoint,
        model_name=str(source_metadata["model_name"]),
        expected_datasets=tuple(source_metadata["datasets"]),
        metadata=extra_metadata,
    )
    dataset1_state = None
    try:
        dataset1_state = load_checkpoint_dataset(args.source_checkpoint, "dataset1")
        dataset1_state["segment_gate_result"] = dataset1_gate
        writer.add_dataset("dataset1", dataset1_state)

        dataset2_state = load_checkpoint_dataset(args.source_checkpoint, "dataset2")
        dataset2_state["segment_gate_result"] = None
        writer.add_dataset("dataset2", dataset2_state)
        del dataset2_state
        writer.finalize()
    except Exception:
        writer.abort()
        raise

    if dataset1_state is None:
        raise RuntimeError("Dataset1 state was not created")
    roundtrip_dataset1 = load_checkpoint_dataset(args.output_checkpoint, "dataset1")
    roundtrip_dataset2 = load_checkpoint_dataset(args.output_checkpoint, "dataset2")
    if not isinstance(roundtrip_dataset1.get("segment_gate_result"), SegmentGateResult):
        raise RuntimeError("Dataset1 gate was lost during checkpoint roundtrip")
    if roundtrip_dataset2.get("segment_gate_result") is not None:
        raise RuntimeError("Dataset2 checkpoint did not preserve the scalar fallback")
    del roundtrip_dataset1, roundtrip_dataset2
    gc.collect()

    datasets = {dataset.name: dataset for dataset in discover_datasets(args.data_dir)}
    dataset1 = datasets["dataset1"]
    dataset2 = datasets["dataset2"]
    csv_dir = args.output_dir / "csv"
    dataset1_output = csv_dir / "dataset1.csv"
    dataset2_output = csv_dir / "dataset2.csv"
    zip_path = args.output_dir / "result.zip"
    csv_dir.mkdir(parents=True, exist_ok=False)

    ranker = create_ranker("hybrid", None)
    ranker.hydrate(dataset1_state)
    del dataset1_state
    gc.collect()
    dataset1_result = build_dataset_submission(
        dataset=dataset1,
        ranker=ranker,
        output_dir=csv_dir,
        batch_size=args.batch_size,
        verbose=True,
        fit_ranker=False,
    )
    validate_submission_file(dataset1_output, expected_rows=expected_test_rows(dataset1))
    del ranker
    gc.collect()

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
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "output_checkpoint": str(args.output_checkpoint.resolve()),
        "output_checkpoint_bytes": args.output_checkpoint.stat().st_size,
        "output_checkpoint_sha256": _sha256(args.output_checkpoint),
        "tuning_report": str(args.tuning_report.resolve()),
        "dataset1_gate": str(args.dataset1_gate.resolve()),
        "dataset1_validation": dataset_reports["dataset1"]["winner"],
        "dataset1_mode": "segment_policy_gate",
        "dataset1_rows": dataset1_result.rows,
        "dataset1_sha256": _sha256(dataset1_output),
        "dataset2_mode": "byte_copy_from_online_champion",
        "dataset2_rows": dataset2_result.rows,
        "dataset2_sha256": copied_dataset2_hash,
        "result_zip": str(zip_path.resolve()),
        "result_zip_bytes": zip_path.stat().st_size,
        "result_zip_sha256": _sha256(zip_path),
    }
    report_path = args.output_dir / "candidate-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    shutil.copyfile(args.tuning_report, args.output_dir / "segment-fusion-report.json")
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
