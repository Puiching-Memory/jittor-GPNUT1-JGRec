from __future__ import annotations

import argparse
import builtins
import gc
import hashlib
import json
import os
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
from jgrec.core.io import discover_datasets
from jgrec.core.runner import build_dataset_submission
from jgrec.core.types import DatasetResult, TestQueryArray, TrainingReport
from jgrec.rankers.hybrid.candidate_set_transformer import (
    load_candidate_set_ensemble_checkpoint,
    predict_candidate_set_ensemble_probabilities,
    snapshot_candidate_set_ensemble,
)
from jgrec.rankers.registry import create_ranker
from jgrec.submission import (
    expected_test_rows,
    validate_submission_file,
    write_zip,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace Dataset2 Setwise/LightGBM with the selected pure-Jittor "
            "Candidate-Set Transformer ensemble and build a submission."
        )
    )
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--source-candidate-report",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--ensemble-checkpoint",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--ensemble-evaluation-report",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--validation-cache-prefix",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--expected-validation-probabilities",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--frozen-dataset1-csv",
        required=True,
        type=Path,
    )
    parser.add_argument("--output-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    import jittor as jt  # noqa: PLC0415

    require_jittor_cuda(jt)
    _require_new_output(args.output_dir, args.output_checkpoint)
    started = time.time()
    source_report = _read_json(args.source_candidate_report)
    ensemble_report = _read_json(args.ensemble_evaluation_report)
    if (
        ensemble_report.get("status") != "passed"
        or not ensemble_report.get("gate", {}).get("passed")
    ):
        raise RuntimeError("pure-Jittor ensemble did not pass its gate")
    _require_hash(
        args.source_checkpoint,
        source_report["output_checkpoint_sha256"],
        "source checkpoint",
    )
    _require_hash(
        args.ensemble_checkpoint,
        ensemble_report["checkpoint"]["sha256"],
        "Candidate-Set Transformer ensemble",
    )
    _require_hash(
        args.expected_validation_probabilities,
        ensemble_report["validation"]["probabilities_sha256"],
        "ensemble validation probabilities",
    )
    ensemble = load_candidate_set_ensemble_checkpoint(
        args.ensemble_checkpoint
    )

    source_metadata = load_checkpoint_metadata(args.source_checkpoint)
    dataset1_state = load_checkpoint_dataset(
        args.source_checkpoint,
        "dataset1",
    )
    dataset2_state = load_checkpoint_dataset(
        args.source_checkpoint,
        "dataset2",
    )
    feature_names = tuple(dataset2_state["feature_names"])
    for result in ensemble.results:
        if result.feature_names != feature_names:
            raise ValueError(
                "ensemble features do not match source Dataset2 encoder"
            )
    dataset2_state.update(
        {
            "fusion_state": None,
            "fusion_result": None,
            "lgbm_result": None,
            "segment_gate_result": None,
            "setwise_fusion_state": None,
            "setwise_fusion_result": None,
            "time_ramp_setwise_fusion_state": None,
            "time_ramp_setwise_result": None,
            "time_ramp_config": None,
            "conservative_window_fusion_states": {},
            "conservative_window_results": {},
            "conservative_window_hidden_dims": {},
            "conservative_window_config": None,
            "multi_interest_proxy_state": None,
            "candidate_set_ensemble_state": (
                snapshot_candidate_set_ensemble(ensemble)
            ),
        }
    )

    extra_metadata = {
        key: value
        for key, value in source_metadata.items()
        if key not in {"format", "version", "model_name", "datasets"}
    }
    extra_metadata.update(
        {
            "derived_from": str(args.source_checkpoint.resolve()),
            "dataset2_reranker": (
                "pure_jittor_candidate_set_transformer_ensemble"
            ),
            "dataset2_trainable_frameworks": ("jittor",),
            "dataset2_non_jittor_trainable_models": (),
            "dataset2_external_ml_runtime_dependencies": (),
            "dataset2_ensemble_checkpoint_sha256": _sha256(
                args.ensemble_checkpoint
            ),
            "dataset2_ensemble_weights": ensemble.weights,
        }
    )
    writer = ContestCheckpointWriter(
        args.output_checkpoint,
        model_name=str(source_metadata["model_name"]),
        expected_datasets=tuple(source_metadata["datasets"]),
        metadata=extra_metadata,
    )
    try:
        writer.add_dataset("dataset1", dataset1_state)
        writer.add_dataset("dataset2", dataset2_state)
        writer.finalize()
    except BaseException:
        writer.abort()
        raise
    finally:
        del dataset1_state, dataset2_state, ensemble
        gc.collect()

    reloaded_dataset2 = load_checkpoint_dataset(
        args.output_checkpoint,
        "dataset2",
    )
    ranker = create_ranker("hybrid", None)
    original_import = builtins.__import__

    def block_external_ml(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] in {"lightgbm", "sklearn"}:
            raise RuntimeError(
                f"forbidden Dataset2 inference import: {name}"
            )
        if name in {"fusion", "fusion_lgbm"}:
            raise RuntimeError(
                f"legacy Dataset2 fusion import during hydrate: {name}"
            )
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = block_external_ml
    try:
        ranker.hydrate(reloaded_dataset2)
        validation_smoke = _validate_checkpoint_predictions(
            ranker,
            args.validation_cache_prefix,
            args.expected_validation_probabilities,
        )
    finally:
        builtins.__import__ = original_import
    del reloaded_dataset2
    gc.collect()

    datasets = {
        dataset.name: dataset
        for dataset in discover_datasets(args.data_dir)
    }
    dataset1 = datasets["dataset1"]
    dataset2 = datasets["dataset2"]
    csv_dir = args.output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=False)
    dataset1_output = csv_dir / "dataset1.csv"
    dataset2_output = csv_dir / "dataset2.csv"
    zip_path = args.output_dir / "result.zip"

    shutil.copyfile(args.frozen_dataset1_csv, dataset1_output)
    validate_submission_file(
        dataset1_output,
        expected_rows=expected_test_rows(dataset1),
    )
    dataset1_result = DatasetResult(
        name="dataset1",
        rows=expected_test_rows(dataset1),
        output_path=dataset1_output,
        training_report=TrainingReport(model_name="hybrid"),
    )
    dataset2_result = build_dataset_submission(
        dataset=dataset2,
        ranker=ranker,
        output_dir=csv_dir,
        batch_size=args.batch_size,
        verbose=True,
        fit_ranker=False,
    )
    validate_submission_file(
        dataset2_output,
        expected_rows=expected_test_rows(dataset2),
    )
    write_zip([dataset1_result, dataset2_result], zip_path)

    report = {
        "status": "complete",
        "winner": "dataset2_pure_jittor_candidate_set_transformer",
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": source_report[
            "output_checkpoint_sha256"
        ],
        "ensemble_checkpoint": str(
            args.ensemble_checkpoint.resolve()
        ),
        "ensemble_checkpoint_sha256": _sha256(
            args.ensemble_checkpoint
        ),
        "ensemble_weights": list(
            ranker.impl.candidate_set_ensemble.weights
        ),
        "offline_comparison": ensemble_report["comparison"],
        "validation_smoke": validation_smoke,
        "output_checkpoint": str(args.output_checkpoint.resolve()),
        "output_checkpoint_bytes": args.output_checkpoint.stat().st_size,
        "output_checkpoint_sha256": _sha256(args.output_checkpoint),
        "checkpoint_scope": {
            "dataset2_trainable_frameworks": ["jittor"],
            "dataset2_non_jittor_trainable_models": [],
            "dataset2_external_ml_runtime_dependencies": [],
            "dataset1": "byte_preserved_from_existing_champion",
        },
        "dataset1_mode": "byte_copy_from_existing_champion_submission",
        "dataset1_rows": dataset1_result.rows,
        "dataset1_sha256": _sha256(dataset1_output),
        "dataset2_mode": "pure_jittor_candidate_set_transformer",
        "dataset2_rows": dataset2_result.rows,
        "dataset2_sha256": _sha256(dataset2_output),
        "result_zip": str(zip_path.resolve()),
        "result_zip_bytes": zip_path.stat().st_size,
        "result_zip_sha256": _sha256(zip_path),
        "elapsed_seconds": time.time() - started,
    }
    _write_json_atomic(args.output_dir / "candidate-report.json", report)
    shutil.copyfile(
        args.ensemble_evaluation_report,
        args.output_dir / "evaluation-report.json",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _validate_checkpoint_predictions(
    ranker: Any,
    cache_prefix: Path,
    expected_path: Path,
) -> dict[str, Any]:
    indices = np.asarray(
        [0, 6_666, 6_667, 13_333, 13_334, 19_999],
        dtype=np.int64,
    )
    feature_path = Path(f"{cache_prefix}.val.npy")
    expected = np.load(expected_path, mmap_mode="r", allow_pickle=False)
    validation_features = np.load(
        feature_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    direct = predict_candidate_set_ensemble_probabilities(
        ranker.impl.candidate_set_ensemble,
        np.asarray(validation_features[indices], dtype=np.float32),
        batch_size=len(indices),
    )
    expected_rows = np.asarray(expected[indices], dtype=np.float32)
    direct_error = float(np.max(np.abs(direct - expected_rows)))
    if not np.allclose(direct, expected_rows, rtol=0.0, atol=5e-7):
        raise RuntimeError(
            "reloaded candidate-set head differs from gate artifact: "
            f"max_abs_error={direct_error}"
        )

    queries = TestQueryArray(
        src=np.asarray(
            np.load(
                Path(f"{cache_prefix}.val-src.npy"),
                mmap_mode="r",
                allow_pickle=False,
            )[indices],
            dtype=np.int32,
        ),
        time=np.asarray(
            np.load(
                Path(f"{cache_prefix}.val-time.npy"),
                mmap_mode="r",
                allow_pickle=False,
            )[indices],
            dtype=np.int64,
        ),
        candidates=np.asarray(
            np.load(
                Path(f"{cache_prefix}.val-candidates.npy"),
                mmap_mode="r",
                allow_pickle=False,
            )[indices],
            dtype=np.int32,
        ),
    )
    integrated = ranker.predict_batch(queries).astype(
        np.float32,
        copy=False,
    )
    if integrated.shape != expected_rows.shape:
        raise RuntimeError(
            "integrated candidate-set prediction shape is invalid"
        )
    if not np.isfinite(integrated).all():
        raise RuntimeError(
            "integrated candidate-set predictions are not finite"
        )
    row_sum_error = float(
        np.max(np.abs(integrated.sum(axis=1) - 1.0))
    )
    if row_sum_error > 5e-6:
        raise RuntimeError(
            "integrated candidate-set probabilities are not normalized: "
            f"max_row_sum_error={row_sum_error}"
        )
    integrated_cache_difference = float(
        np.max(np.abs(integrated - expected_rows))
    )
    direct_ranking_equal = bool(
        np.array_equal(
            np.argsort(-direct, axis=1, kind="stable"),
            np.argsort(-expected_rows, axis=1, kind="stable"),
        )
    )
    if not direct_ranking_equal:
        raise RuntimeError(
            "reloaded candidate-set head rankings differ"
        )
    return {
        "rows": indices.tolist(),
        "feature_source": "locked causal validation cache",
        "direct_head_max_abs_error": direct_error,
        "direct_head_ranking_order_equal": direct_ranking_equal,
        "production_chain_finite": True,
        "production_chain_max_probability_sum_error": row_sum_error,
        "production_vs_causal_cache_comparison": (
            "not_applicable_full_trained_encoder_differs_from_train_end_cache"
        ),
        "production_vs_causal_cache_max_abs_difference_diagnostic": (
            integrated_cache_difference
        ),
        "blocked_imports": ["lightgbm", "sklearn", "legacy fusion"],
    }


def _require_new_output(output_dir: Path, checkpoint: Path) -> None:
    temporary_checkpoint = checkpoint.with_suffix(
        f"{checkpoint.suffix}.tmp"
    )
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    if checkpoint.exists() or temporary_checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite: {checkpoint}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: actual={actual} expected={expected}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
