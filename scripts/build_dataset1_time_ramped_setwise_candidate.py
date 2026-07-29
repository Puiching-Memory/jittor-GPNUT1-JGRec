from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.contest_checkpoint import (
    ContestCheckpointWriter,
    load_checkpoint_dataset,
    load_checkpoint_metadata,
)
from jgrec.core.cuda import require_jittor_cuda
from jgrec.core.io import discover_datasets, read_test_queries
from jgrec.core.runner import build_dataset_submission
from jgrec.core.types import DatasetResult, TrainingReport
from jgrec.rankers.hybrid.fusion import FusionResult
from jgrec.rankers.registry import create_ranker
from jgrec.submission import (
    expected_test_rows,
    validate_submission_file,
    write_zip,
)

FROZEN_DATASET2_SHA256 = (
    "d7e0d574789aa6b507e592fffb5da054839aba663f63ed58134ea2d951b1ae1e"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Dataset1 time-ramped Setwise package authorized by "
            "the independent chronological gate."
        )
    )
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--selection-report", required=True, type=Path)
    parser.add_argument("--evaluation-report", required=True, type=Path)
    parser.add_argument("--source-setwise-report", required=True, type=Path)
    parser.add_argument("--setwise-model", required=True, type=Path)
    parser.add_argument("--champion-dataset2", required=True, type=Path)
    parser.add_argument("--output-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    import jittor as jt  # noqa: PLC0415

    require_jittor_cuda(jt)
    _require_new_output(args.output_dir, args.output_checkpoint)
    started = time.time()

    selection = _read_json(args.selection_report)
    evaluation = _read_json(args.evaluation_report)
    source_setwise_report = _read_json(args.source_setwise_report)
    power = _authorized_power(
        selection,
        evaluation,
        selection_report_sha256=_sha256(args.selection_report),
    )
    expected_checkpoint_hash = selection["frozen_config"][
        "checkpoint_sha256"
    ]
    _require_hash(
        args.source_checkpoint,
        expected_checkpoint_hash,
        "source checkpoint",
    )
    model_report = source_setwise_report["models"]["recent_100k"]
    _require_hash(
        args.setwise_model,
        model_report["model_sha256"],
        "recent-100k Setwise model",
    )
    _require_hash(
        args.champion_dataset2,
        FROZEN_DATASET2_SHA256,
        "frozen Dataset2 CSV",
    )

    datasets = {
        dataset.name: dataset
        for dataset in discover_datasets(args.data_dir)
    }
    dataset1 = datasets["dataset1"]
    dataset2 = datasets["dataset2"]
    test_queries = read_test_queries(dataset1.test_path)
    if not test_queries:
        raise ValueError("Dataset1 test queries are empty")
    minimum_time = int(test_queries.time.min())
    maximum_time = int(test_queries.time.max())
    if maximum_time <= minimum_time:
        raise ValueError("Dataset1 test query horizon has no time span")
    del test_queries

    source_metadata = load_checkpoint_metadata(args.source_checkpoint)
    dataset1_state = load_checkpoint_dataset(
        args.source_checkpoint,
        "dataset1",
    )
    setwise_state, setwise_result, hidden_dim = _load_setwise_expert(
        args.setwise_model,
        model_report,
        expected_source_features=len(dataset1_state["feature_names"]),
    )
    dataset1_state["time_ramp_setwise_fusion_state"] = setwise_state
    dataset1_state["time_ramp_setwise_result"] = setwise_result
    dataset1_state["time_ramp_setwise_hidden_dim"] = hidden_dim
    dataset1_state["time_ramp_config"] = {
        "power": power,
        "minimum_time": float(minimum_time),
        "maximum_time": float(maximum_time),
    }

    extra_metadata = {
        key: value
        for key, value in source_metadata.items()
        if key not in {"format", "version", "model_name", "datasets"}
    }
    extra_metadata.update(
        {
            "derived_from": str(args.source_checkpoint.resolve()),
            "dataset1_time_ramp_selection": str(
                args.selection_report.resolve()
            ),
            "dataset1_time_ramp_evaluation": str(
                args.evaluation_report.resolve()
            ),
            "dataset1_time_ramp_power": power,
            "dataset1_time_ramp_minimum_time": minimum_time,
            "dataset1_time_ramp_maximum_time": maximum_time,
            "dataset1_time_ramp_expert": "recent_100k_setwise",
        }
    )
    writer = ContestCheckpointWriter(
        args.output_checkpoint,
        model_name=str(source_metadata["model_name"]),
        expected_datasets=tuple(source_metadata["datasets"]),
        metadata=extra_metadata,
    )
    source_dataset2_state_hash = ""
    try:
        writer.add_dataset("dataset1", dataset1_state)
        dataset2_state = load_checkpoint_dataset(
            args.source_checkpoint,
            "dataset2",
        )
        source_dataset2_state_hash = _pickle_sha256(dataset2_state)
        writer.add_dataset("dataset2", dataset2_state)
        writer.finalize()
    except BaseException:
        writer.abort()
        raise
    finally:
        del dataset1_state
        if "dataset2_state" in locals():
            del dataset2_state
        gc.collect()

    reloaded_dataset1 = load_checkpoint_dataset(
        args.output_checkpoint,
        "dataset1",
    )
    reloaded_ranker = create_ranker("hybrid", None)
    reloaded_ranker.hydrate(reloaded_dataset1)
    actual_config = reloaded_ranker.impl.time_ramp_config
    if (
        reloaded_ranker.impl.time_ramp_setwise_fusion is None
        or reloaded_ranker.impl.time_ramp_setwise_result is None
        or actual_config is None
        or actual_config
        != {
            "power": power,
            "minimum_time": float(minimum_time),
            "maximum_time": float(maximum_time),
        }
    ):
        raise RuntimeError("reloaded Dataset1 time-ramp checkpoint differs")
    del reloaded_dataset1
    gc.collect()

    reloaded_dataset2 = load_checkpoint_dataset(
        args.output_checkpoint,
        "dataset2",
    )
    output_dataset2_state_hash = _pickle_sha256(reloaded_dataset2)
    del reloaded_dataset2
    gc.collect()
    if output_dataset2_state_hash != source_dataset2_state_hash:
        raise RuntimeError("Dataset2 checkpoint state changed during packaging")

    csv_dir = args.output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=False)
    dataset1_output = csv_dir / "dataset1.csv"
    dataset2_output = csv_dir / "dataset2.csv"
    zip_path = args.output_dir / "result.zip"

    dataset1_result = build_dataset_submission(
        dataset=dataset1,
        ranker=reloaded_ranker,
        output_dir=csv_dir,
        batch_size=args.batch_size,
        verbose=True,
        fit_ranker=False,
    )
    validate_submission_file(
        dataset1_output,
        expected_rows=expected_test_rows(dataset1),
    )
    shutil.copyfile(args.champion_dataset2, dataset2_output)
    validate_submission_file(
        dataset2_output,
        expected_rows=expected_test_rows(dataset2),
    )
    copied_dataset2_hash = _sha256(dataset2_output)
    if copied_dataset2_hash != FROZEN_DATASET2_SHA256:
        raise RuntimeError("copied Dataset2 CSV differs from champion")
    dataset2_result = DatasetResult(
        name="dataset2",
        rows=expected_test_rows(dataset2),
        output_path=dataset2_output,
        training_report=TrainingReport(model_name="hybrid"),
    )
    write_zip([dataset1_result, dataset2_result], zip_path)

    report = {
        "status": "complete",
        "winner": "dataset1_time_ramped_recent100k_setwise",
        "selected_power": power,
        "test_time_bounds": [minimum_time, maximum_time],
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": _sha256(args.source_checkpoint),
        "selection_report": str(args.selection_report.resolve()),
        "selection_report_sha256": _sha256(args.selection_report),
        "evaluation_report": str(args.evaluation_report.resolve()),
        "evaluation_report_sha256": _sha256(args.evaluation_report),
        "offline_gate": evaluation["gate"],
        "setwise_model_sha256": _sha256(args.setwise_model),
        "output_checkpoint": str(args.output_checkpoint.resolve()),
        "output_checkpoint_bytes": args.output_checkpoint.stat().st_size,
        "output_checkpoint_sha256": _sha256(args.output_checkpoint),
        "dataset1_mode": "global_time_ramp_champion_to_recent100k_setwise",
        "dataset1_rows": dataset1_result.rows,
        "dataset1_sha256": _sha256(dataset1_output),
        "dataset2_mode": "byte_copy_from_online_setwise_champion",
        "dataset2_rows": dataset2_result.rows,
        "dataset2_sha256": copied_dataset2_hash,
        "source_dataset2_state_pickle_sha256": source_dataset2_state_hash,
        "output_dataset2_state_pickle_sha256": output_dataset2_state_hash,
        "result_zip": str(zip_path.resolve()),
        "result_zip_bytes": zip_path.stat().st_size,
        "result_zip_sha256": _sha256(zip_path),
        "elapsed_seconds": time.time() - started,
    }
    _write_json_atomic(args.output_dir / "candidate-report.json", report)
    shutil.copyfile(
        args.selection_report,
        args.output_dir / "selection-report.json",
    )
    shutil.copyfile(
        args.evaluation_report,
        args.output_dir / "evaluation-report.json",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _authorized_power(
    selection: dict[str, Any],
    evaluation: dict[str, Any],
    *,
    selection_report_sha256: str,
) -> float:
    if not selection.get("gate_passed"):
        raise RuntimeError("prefix selection did not pass")
    if (
        not evaluation.get("gate_passed")
        or not evaluation.get("package_authorized")
    ):
        raise RuntimeError("independent chronological gate did not pass")
    if (
        evaluation.get("selection_report_sha256")
        != selection_report_sha256
    ):
        raise ValueError("evaluation did not authorize this selection report")
    selected_power = selection["selection"]["selected_power"]
    if selected_power is None:
        raise RuntimeError("selection report has no selected power")
    power = float(selected_power)
    if abs(power - float(evaluation["selected_power"])) > 1e-12:
        raise ValueError("selection and gate powers differ")
    return power


def _load_setwise_expert(
    path: Path,
    model_report: dict[str, Any],
    *,
    expected_source_features: int,
) -> tuple[dict[str, np.ndarray], FusionResult, int]:
    with np.load(path, allow_pickle=False) as payload:
        hidden_dim = int(payload["hidden_dim"][0])
        source_feature_count = int(payload["source_feature_count"][0])
        training_rows = int(payload["training_rows"][0])
        if source_feature_count != expected_source_features:
            raise ValueError("Setwise model source feature count differs")
        if training_rows != int(model_report["training_rows"]):
            raise ValueError("Setwise model training scale differs")
        if int(payload["context_transform_version"][0]) != 1:
            raise ValueError("unsupported Setwise context transform")
        state = {
            key.removeprefix("state__"): np.asarray(
                payload[key],
                dtype=np.float32,
            )
            for key in payload.files
            if key.startswith("state__")
        }
        result = FusionResult(
            best_val_ap=float(model_report["best_selection_ap"]),
            best_val_mrr=float(model_report["best_selection_mrr"]),
            state=state,
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
            feature_indices=tuple(
                int(value) for value in payload["feature_indices"]
            ),
            candidate_name="dataset1_time_ramp_recent100k_setwise",
        )
    return state, result, hidden_dim


def _require_new_output(output_dir: Path, checkpoint: Path) -> None:
    temporary_checkpoint = checkpoint.with_suffix(
        f"{checkpoint.suffix}.tmp"
    )
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    if checkpoint.exists() or temporary_checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite: {checkpoint}")


def _pickle_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    class HashWriter:
        def write(self, data: bytes | pickle.PickleBuffer) -> int:
            view = memoryview(data)
            digest.update(view)
            return view.nbytes

    pickle.dump(value, HashWriter(), protocol=pickle.HIGHEST_PROTOCOL)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch: actual={actual} expected={expected}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
