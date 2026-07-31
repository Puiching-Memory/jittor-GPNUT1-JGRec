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
from jgrec.core.io import discover_datasets
from jgrec.core.runner import build_dataset_submission
from jgrec.core.types import DatasetResult, TrainingReport
from jgrec.rankers.hybrid.conservative_window_blend import (
    conservative_window_scores,
)
from jgrec.rankers.hybrid.fusion import FusionResult, predict_logits
from jgrec.rankers.hybrid.fusion_lgbm import predict_logits_lgbm
from jgrec.rankers.hybrid.setwise import setwise_context_features
from jgrec.rankers.hybrid.window_diversity import blend_expert_subset
from jgrec.rankers.registry import create_ranker
from jgrec.submission import (
    expected_test_rows,
    validate_submission_file,
    write_zip,
)

EXPECTED_ALPHA = 0.30
EXPECTED_EXTRAS = (
    "recent100k",
    "recent200k_decay100k",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Dataset2 conservative window package authorized by "
            "the independent chronological gate."
        )
    )
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--source-candidate-report", required=True, type=Path)
    parser.add_argument("--selection-report", required=True, type=Path)
    parser.add_argument("--evaluation-report", required=True, type=Path)
    parser.add_argument("--output-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    import jittor as jt  # noqa: PLC0415

    require_jittor_cuda(jt)
    _require_new_output(args.output_dir, args.output_checkpoint)
    started = time.time()

    source_candidate = _read_json(args.source_candidate_report)
    selection = _read_json(args.selection_report)
    evaluation = _read_json(args.evaluation_report)
    alpha = _authorized_alpha(
        selection,
        evaluation,
        selection_report_sha256=_sha256(args.selection_report),
    )
    _require_hash(
        args.source_checkpoint,
        source_candidate["output_checkpoint_sha256"],
        "source checkpoint",
    )
    frozen_path = Path(selection["frozen_config"])
    _require_hash(
        frozen_path,
        selection["frozen_config_sha256"],
        "conservative frozen config",
    )
    frozen = _read_json(frozen_path)
    source_window_selection_path = Path(
        frozen["source"]["selection_report"]
    )
    _require_hash(
        source_window_selection_path,
        frozen["source"]["selection_report_sha256"],
        "source window selection",
    )
    source_window_selection = _read_json(source_window_selection_path)
    source_window_frozen_path = Path(
        source_window_selection["frozen_config"]
    )
    _require_hash(
        source_window_frozen_path,
        source_window_selection["frozen_config_sha256"],
        "source window frozen config",
    )
    source_window_frozen = _read_json(source_window_frozen_path)
    frozen_dataset1_csv = Path(evaluation["frozen_dataset1_csv"])
    _require_hash(
        frozen_dataset1_csv,
        evaluation["frozen_dataset1_csv_sha256"],
        "frozen Dataset1 CSV",
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
    source_dataset1_state_hash = _pickle_sha256(dataset1_state)
    source_dataset2_state_hash = _pickle_sha256(dataset2_state)
    if dataset2_state.get("conservative_window_config") is not None:
        raise ValueError("source Dataset2 already has conservative window state")

    extra_states: dict[str, dict[str, np.ndarray]] = {}
    extra_results: dict[str, FusionResult] = {}
    extra_hidden_dims: dict[str, int] = {}
    extra_model_hashes: dict[str, str] = {}
    source_feature_count = len(dataset2_state["feature_names"])
    for name in EXPECTED_EXTRAS:
        expert_report = source_window_selection["experts"][name]
        model_path = Path(expert_report["model"])
        _require_hash(
            model_path,
            expert_report["model_sha256"],
            f"{name} model",
        )
        state, result, hidden_dim = _load_setwise_expert(
            model_path,
            expert_name=name,
            expected_source_features=source_feature_count,
            expected_train_rows=int(expert_report["train_rows"]),
            best_prefix_mrr=expert_report["best_prefix_mrr"],
        )
        extra_states[name] = state
        extra_results[name] = result
        extra_hidden_dims[name] = hidden_dim
        extra_model_hashes[name] = _sha256(model_path)
    dataset2_state["conservative_window_fusion_states"] = extra_states
    dataset2_state["conservative_window_results"] = extra_results
    dataset2_state["conservative_window_hidden_dims"] = extra_hidden_dims
    dataset2_state["conservative_window_config"] = {"alpha": alpha}

    extra_metadata = {
        key: value
        for key, value in source_metadata.items()
        if key not in {"format", "version", "model_name", "datasets"}
    }
    extra_metadata.update(
        {
            "derived_from": str(args.source_checkpoint.resolve()),
            "dataset2_conservative_window_selection": str(
                args.selection_report.resolve()
            ),
            "dataset2_conservative_window_evaluation": str(
                args.evaluation_report.resolve()
            ),
            "dataset2_conservative_window_alpha": alpha,
            "dataset2_conservative_window_extras": list(EXPECTED_EXTRAS),
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

    reloaded_dataset1 = load_checkpoint_dataset(
        args.output_checkpoint,
        "dataset1",
    )
    output_dataset1_state_hash = _pickle_sha256(reloaded_dataset1)
    del reloaded_dataset1
    gc.collect()
    if output_dataset1_state_hash != source_dataset1_state_hash:
        raise RuntimeError("Dataset1 checkpoint state changed during packaging")

    reloaded_dataset2 = load_checkpoint_dataset(
        args.output_checkpoint,
        "dataset2",
    )
    output_dataset2_state_hash = _pickle_sha256(reloaded_dataset2)
    reloaded_ranker = create_ranker("hybrid", None)
    reloaded_ranker.hydrate(reloaded_dataset2)
    impl = reloaded_ranker.impl
    if (
        impl.conservative_window_config != {"alpha": alpha}
        or tuple(impl.conservative_window_fusions) != EXPECTED_EXTRAS
    ):
        raise RuntimeError("reloaded conservative window checkpoint differs")
    del reloaded_dataset2
    gc.collect()

    validation_smoke = _validate_checkpoint_predictions(
        reloaded_ranker,
        source_window_frozen,
        Path(evaluation["selected_prediction"]),
        evaluation["selected_prediction_sha256"],
    )

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

    shutil.copyfile(frozen_dataset1_csv, dataset1_output)
    validate_submission_file(
        dataset1_output,
        expected_rows=expected_test_rows(dataset1),
    )
    copied_dataset1_hash = _sha256(dataset1_output)
    if copied_dataset1_hash != evaluation["frozen_dataset1_csv_sha256"]:
        raise RuntimeError("copied Dataset1 CSV differs from frozen champion")
    dataset1_result = DatasetResult(
        name="dataset1",
        rows=expected_test_rows(dataset1),
        output_path=dataset1_output,
        training_report=TrainingReport(model_name="hybrid"),
    )
    dataset2_result = build_dataset_submission(
        dataset=dataset2,
        ranker=reloaded_ranker,
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
        "winner": "dataset2_conservative_window_alpha030",
        "selected_alpha": alpha,
        "extra_experts": list(EXPECTED_EXTRAS),
        "extra_model_sha256": extra_model_hashes,
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": _sha256(args.source_checkpoint),
        "selection_report": str(args.selection_report.resolve()),
        "selection_report_sha256": _sha256(args.selection_report),
        "evaluation_report": str(args.evaluation_report.resolve()),
        "evaluation_report_sha256": _sha256(args.evaluation_report),
        "offline_gate": evaluation["gate"],
        "validation_smoke": validation_smoke,
        "output_checkpoint": str(args.output_checkpoint.resolve()),
        "output_checkpoint_bytes": args.output_checkpoint.stat().st_size,
        "output_checkpoint_sha256": _sha256(args.output_checkpoint),
        "source_dataset1_state_pickle_sha256": source_dataset1_state_hash,
        "output_dataset1_state_pickle_sha256": output_dataset1_state_hash,
        "source_dataset2_state_pickle_sha256": source_dataset2_state_hash,
        "output_dataset2_state_pickle_sha256": output_dataset2_state_hash,
        "dataset1_mode": "byte_copy_from_online_time_ramp_champion",
        "dataset1_rows": dataset1_result.rows,
        "dataset1_sha256": copied_dataset1_hash,
        "dataset2_mode": "conservative_window_residual_alpha030",
        "dataset2_rows": dataset2_result.rows,
        "dataset2_sha256": _sha256(dataset2_output),
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


def _authorized_alpha(
    selection: dict[str, Any],
    evaluation: dict[str, Any],
    *,
    selection_report_sha256: str,
) -> float:
    if (
        selection.get("status") != "locked_before_forward_gate"
        or not selection.get("gate_unlocked")
        or selection.get("forward_metrics_read") is not False
    ):
        raise RuntimeError("conservative selection did not unlock the gate")
    if (
        not evaluation.get("gate_passed")
        or not evaluation.get("production_followup_authorized")
    ):
        raise RuntimeError("independent conservative gate did not pass")
    if (
        evaluation.get("selection_report_sha256")
        != selection_report_sha256
    ):
        raise ValueError("evaluation did not authorize this selection report")
    alpha = float(selection["selected_alpha"])
    if (
        abs(alpha - float(evaluation["selected_alpha"])) > 1e-12
        or abs(alpha - EXPECTED_ALPHA) > 1e-12
    ):
        raise ValueError("selection and gate alpha differ")
    return alpha


def _load_setwise_expert(
    path: Path,
    *,
    expert_name: str,
    expected_source_features: int,
    expected_train_rows: int,
    best_prefix_mrr: float | None,
) -> tuple[dict[str, np.ndarray], FusionResult, int]:
    with np.load(path, allow_pickle=False) as payload:
        hidden_dim = int(payload["hidden_dim"][0])
        source_feature_count = int(payload["source_feature_count"][0])
        train_rows = int(payload["train_rows"][0])
        if source_feature_count != expected_source_features:
            raise ValueError(f"{expert_name} source feature count differs")
        if train_rows != expected_train_rows:
            raise ValueError(f"{expert_name} training scale differs")
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
            best_val_ap=0.0,
            best_val_mrr=(
                0.0 if best_prefix_mrr is None else float(best_prefix_mrr)
            ),
            state=state,
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
            feature_indices=tuple(
                int(value) for value in payload["feature_indices"]
            ),
            candidate_name=f"dataset2_conservative_{expert_name}",
        )
    return state, result, hidden_dim


def _validate_checkpoint_predictions(
    ranker: Any,
    source_window_frozen: dict[str, Any],
    expected_prediction_path: Path,
    expected_prediction_sha256: str,
) -> dict[str, Any]:
    _require_hash(
        expected_prediction_path,
        expected_prediction_sha256,
        "gate prediction",
    )
    validation_path = Path(source_window_frozen["validation_features"])
    validation_features = np.load(
        validation_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    expected = np.load(
        expected_prediction_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    indices = np.asarray([0, 6_666, 6_667, 13_333, 13_334, 19_999])
    raw_features = np.asarray(
        validation_features[indices],
        dtype=np.float32,
    )
    impl = ranker.impl
    context = setwise_context_features(raw_features)
    probabilities: dict[str, np.ndarray] = {}
    main_result = impl.setwise_fusion_result
    if main_result is None or impl.setwise_fusion is None:
        raise RuntimeError("checkpoint has no Setwise champion")
    main_features = context[..., main_result.feature_indices]
    probabilities["recent200k"] = _softmax(
        predict_logits(
            impl.setwise_fusion,
            main_features,
            main_result.mean,
            main_result.std,
        )
    )
    for name, model in impl.conservative_window_fusions.items():
        result = impl.conservative_window_results[name]
        selected = context[..., result.feature_indices]
        probabilities[name] = _softmax(
            predict_logits(
                model,
                selected,
                result.mean,
                result.std,
            )
        )
    if impl.lgbm_result is None:
        raise RuntimeError("checkpoint has no LightGBM champion expert")
    lgbm_features = raw_features[..., impl.lgbm_result.feature_indices]
    lgbm_probabilities = _softmax(
        predict_logits_lgbm(
            impl.lgbm_result.model_text,
            lgbm_features,
        )
    )
    champion = blend_expert_subset(
        probabilities,
        lgbm_probabilities,
        selected_experts=("recent200k",),
        expert_weight=impl.lgbm_result.mlp_weight,
    )
    window = blend_expert_subset(
        probabilities,
        lgbm_probabilities,
        selected_experts=(
            "recent100k",
            "recent200k",
            "recent200k_decay100k",
        ),
        expert_weight=impl.lgbm_result.mlp_weight,
    )
    actual = conservative_window_scores(
        champion,
        window,
        alpha=impl.conservative_window_config["alpha"],
    )
    expected_rows = np.asarray(expected[indices], dtype=np.float64)
    max_abs_error = float(np.max(np.abs(actual - expected_rows)))
    if not np.allclose(actual, expected_rows, rtol=0.0, atol=5e-6):
        raise RuntimeError(
            "checkpoint conservative predictions differ from gate artifact: "
            f"max_abs_error={max_abs_error}"
        )
    if not np.array_equal(
        np.argsort(-actual, axis=1, kind="stable"),
        np.argsort(-expected_rows, axis=1, kind="stable"),
    ):
        raise RuntimeError("checkpoint conservative rankings differ")
    return {
        "rows": indices.tolist(),
        "feature_source": "locked causal validation cache",
        "max_abs_error": max_abs_error,
        "ranking_order_equal": True,
    }


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


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
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
