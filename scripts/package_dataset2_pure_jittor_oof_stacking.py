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
    CandidateSetEnsembleCheckpoint,
    load_candidate_set_checkpoint,
)
from jgrec.rankers.hybrid.oof_models import (
    PureJittorOOFStackingCheckpoint,
    load_candidate_set_mlp_checkpoint,
    predict_pure_jittor_oof_stacking_scores,
    snapshot_pure_jittor_oof_stacking,
)
from jgrec.rankers.hybrid.oof_stacking import (
    STABLE_EXPERT_LOGIT_FEATURE_VERSION,
)
from jgrec.rankers.registry import create_ranker
from jgrec.submission import (
    expected_test_rows,
    validate_submission_file,
    write_zip,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Package the selected pure-Jittor Dataset2 OOF stack."
    )
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--source-candidate-report",
        required=True,
        type=Path,
    )
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument(
        "--validation-cache-prefix",
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
    args = parser.parse_args()

    import jittor as jt  # noqa: PLC0415

    require_jittor_cuda(jt)
    _require_new_output(args.output_dir, args.output_checkpoint)
    started = time.time()
    source_report = _read_json(args.source_candidate_report)
    evaluation = _read_json(
        args.experiment_dir / "evaluation-report.json"
    )
    meta_report = _read_json(args.experiment_dir / "meta-report.json")
    if (
        evaluation.get("status") != "passed"
        or not evaluation.get("gate", {}).get("passed")
    ):
        raise RuntimeError("OOF stacking did not pass the external gate")
    if (
        evaluation.get("metric_protocol")
        != "tie_neutral_average_rank"
        or evaluation.get("stable_feature_version")
        != STABLE_EXPERT_LOGIT_FEATURE_VERSION
    ):
        raise RuntimeError(
            "OOF stacking gate predates the current tie-neutral stable "
            "feature contract"
        )
    _require_hash(
        args.source_checkpoint,
        source_report["output_checkpoint_sha256"],
        "source checkpoint",
    )

    full_by_name = {
        row["expert"]: row for row in evaluation["full_experts"]
    }
    cst_pairs = []
    for name in ("cst_main", "cst_residual"):
        path = args.experiment_dir / "full-experts" / f"{name}.npz"
        _require_hash(
            path,
            full_by_name[name]["checkpoint_sha256"],
            name,
        )
        cst_pairs.append(load_candidate_set_checkpoint(path))
    setwise_path = (
        args.experiment_dir / "full-experts" / "setwise_mlp.npz"
    )
    _require_hash(
        setwise_path,
        full_by_name["setwise_mlp"]["checkpoint_sha256"],
        "setwise_mlp",
    )
    setwise_pair = load_candidate_set_mlp_checkpoint(setwise_path)
    meta_path = args.experiment_dir / "meta-stacking-mlp.npz"
    _require_hash(
        meta_path,
        meta_report["meta_checkpoint_sha256"],
        "meta stacking MLP",
    )
    meta_pair = load_candidate_set_mlp_checkpoint(meta_path)
    stacking = PureJittorOOFStackingCheckpoint(
        expert_names=("cst_main", "cst_residual", "setwise_mlp"),
        cst_experts=CandidateSetEnsembleCheckpoint(
            models=tuple(pair[0] for pair in cst_pairs),
            results=tuple(pair[1] for pair in cst_pairs),
            weights=(0.5, 0.5),
        ),
        setwise_mlp=setwise_pair,
        meta_mlp=meta_pair,
        meta_weight=float(evaluation["meta_weight"]),
    )
    stacking_state = snapshot_pure_jittor_oof_stacking(stacking)

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
    for result in stacking.cst_experts.results:
        if result.feature_names != feature_names:
            raise ValueError("OOF CST features differ from source encoder")
    if stacking.setwise_mlp[1].feature_names != feature_names:
        raise ValueError("OOF Setwise features differ from source encoder")
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
            "candidate_set_ensemble_state": None,
            "oof_stacking_state": stacking_state,
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
            "dataset2_reranker": "pure_jittor_oof_stacking",
            "dataset2_trainable_frameworks": ("jittor",),
            "dataset2_non_jittor_trainable_models": (),
            "dataset2_external_ml_runtime_dependencies": (),
            "dataset2_oof_expert_names": stacking.expert_names,
            "dataset2_oof_meta_weight": stacking.meta_weight,
            "dataset2_oof_evaluation_report_sha256": _sha256(
                args.experiment_dir / "evaluation-report.json"
            ),
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
        del dataset1_state, dataset2_state
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
        validation_replay = _validate_checkpoint_predictions(
            ranker,
            args.validation_cache_prefix,
            args.experiment_dir
            / "full-validation-selected-scores.npy",
            args.batch_size,
        )
    finally:
        builtins.__import__ = original_import
    del reloaded_dataset2, stacking
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
    test_scores_path = args.output_dir / "dataset2-test-stacked-scores.npy"
    test_scores = np.loadtxt(
        dataset2_output,
        delimiter=",",
        dtype=np.float32,
    )
    _save_array_atomic(test_scores_path, test_scores)
    del test_scores
    write_zip([dataset1_result, dataset2_result], zip_path)

    report = {
        "status": "complete",
        "winner": "dataset2_pure_jittor_oof_stacking",
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": source_report[
            "output_checkpoint_sha256"
        ],
        "offline_comparison": evaluation["comparison"],
        "meta_weight": evaluation["meta_weight"],
        "validation_replay": validation_replay,
        "output_checkpoint": str(args.output_checkpoint.resolve()),
        "output_checkpoint_bytes": args.output_checkpoint.stat().st_size,
        "output_checkpoint_sha256": _sha256(args.output_checkpoint),
        "checkpoint_scope": {
            "dataset2_trainable_frameworks": ["jittor"],
            "dataset2_non_jittor_trainable_models": [],
            "dataset2_external_ml_runtime_dependencies": [],
            "dataset1": "byte_preserved_from_existing_champion",
        },
        "dataset1_rows": dataset1_result.rows,
        "dataset1_sha256": _sha256(dataset1_output),
        "dataset2_rows": dataset2_result.rows,
        "dataset2_sha256": _sha256(dataset2_output),
        "dataset2_test_scores": str(test_scores_path.resolve()),
        "dataset2_test_scores_sha256": _sha256(test_scores_path),
        "result_zip": str(zip_path.resolve()),
        "result_zip_bytes": zip_path.stat().st_size,
        "result_zip_sha256": _sha256(zip_path),
        "elapsed_seconds": time.time() - started,
    }
    _write_json_atomic(args.output_dir / "candidate-report.json", report)
    shutil.copyfile(
        args.experiment_dir / "evaluation-report.json",
        args.output_dir / "evaluation-report.json",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _validate_checkpoint_predictions(
    ranker: Any,
    cache_prefix: Path,
    expected_path: Path,
    batch_size: int,
) -> dict[str, Any]:
    validation_features = np.load(
        Path(f"{cache_prefix}.val.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    expected = np.load(
        expected_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    direct = predict_pure_jittor_oof_stacking_scores(
        ranker.impl.oof_stacking,
        validation_features,
        batch_size=batch_size,
    )
    max_error = float(np.max(np.abs(direct - expected)))
    if not np.allclose(direct, expected, rtol=0.0, atol=5e-7):
        raise RuntimeError(
            "reloaded OOF stack differs from gate artifact: "
            f"max_abs_error={max_error}"
        )
    indices = np.asarray(
        [0, 6_666, 6_667, 13_333, 13_334, 19_999],
        dtype=np.int64,
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
    integrated = ranker.predict_batch(queries)
    if (
        integrated.shape != (len(indices), validation_features.shape[1])
        or not np.isfinite(integrated).all()
        or np.any(integrated < 0.0)
        or np.any(integrated > 1.0)
    ):
        raise RuntimeError("integrated OOF stacking scores are invalid")
    return {
        "rows": int(expected.shape[0]),
        "direct_head_max_abs_error": max_error,
        "direct_head_ranking_order_equal": bool(
            np.array_equal(
                np.argsort(-direct, axis=1, kind="stable"),
                np.argsort(-expected, axis=1, kind="stable"),
            )
        ),
        "production_chain_smoke_rows": indices.tolist(),
        "production_chain_finite_and_bounded": True,
        "blocked_imports": ["lightgbm", "sklearn", "legacy fusion"],
    }


def _require_new_output(output_dir: Path, checkpoint: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    if checkpoint.exists() or checkpoint.with_suffix(
        f"{checkpoint.suffix}.tmp"
    ).exists():
        raise FileExistsError(f"refusing to overwrite: {checkpoint}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def _save_array_atomic(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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
