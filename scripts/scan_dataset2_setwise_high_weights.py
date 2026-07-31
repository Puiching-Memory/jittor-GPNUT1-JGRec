from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset, set_model_state
from jgrec.rankers.hybrid.full100_training import passes_full100_gate
from jgrec.rankers.hybrid.fusion import FusionMLP, predict_logits
from jgrec.rankers.hybrid.fusion_analysis import (
    inclusive_weight_grid,
    ranking_mrr_three_slices,
    scan_high_weight_blend_on_prefix,
)
from jgrec.rankers.hybrid.fusion_lgbm import predict_logits_lgbm
from jgrec.rankers.hybrid.setwise import SetwiseFeatureView

EXPECTED_SHAPE = (20_000, 100, 63)
SELECTION_STOP = 13_334


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scan Dataset2 Setwise weights 0.80..1.00, selecting only on "
            "the first two chronological validation slices."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--validation-features", required=True, type=Path)
    parser.add_argument("--validation-cache-report", required=True, type=Path)
    parser.add_argument("--source-evaluation-report", required=True, type=Path)
    parser.add_argument("--setwise-model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-full-delta", type=float, default=0.002)
    parser.add_argument("--weight-start", type=float, default=0.80)
    parser.add_argument("--weight-stop", type=float, default=1.00)
    parser.add_argument("--weight-step", type=float, default=0.01)
    parser.add_argument("--reference-weight", type=float)
    parser.add_argument(
        "--report-name",
        default="setwise-high-weight-report.json",
    )
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite Setwise scan: {args.output_dir}"
        )
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    args.output_dir.mkdir(parents=True)
    frozen_path = args.output_dir / "frozen-config.json"
    report_path = args.output_dir / args.report_name
    weights = inclusive_weight_grid(
        args.weight_start,
        args.weight_stop,
        args.weight_step,
    )

    started = time.time()
    validation_report = json.loads(
        args.validation_cache_report.read_text(encoding="utf-8")
    )
    source_evaluation = json.loads(
        args.source_evaluation_report.read_text(encoding="utf-8")
    )
    if validation_report.get("status") != "complete":
        raise RuntimeError("joint validation cache is incomplete")
    expected_validation_sha = validation_report["artifacts"]["features"]["sha256"]
    _require_hash(
        args.validation_features,
        expected_validation_sha,
        "validation features",
    )
    _require_hash(
        args.checkpoint,
        source_evaluation["frozen_config"]["checkpoint_sha256"],
        "champion checkpoint",
    )
    _require_hash(
        args.setwise_model,
        source_evaluation["setwise"]["model_sha256"],
        "Setwise model",
    )

    val_features = np.load(
        args.validation_features,
        mmap_mode="r",
        allow_pickle=False,
    )
    if val_features.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"joint validation tensor shape differs: {val_features.shape}"
        )
    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    feature_names = tuple(str(name) for name in state["feature_names"])
    if tuple(validation_report["feature_names"]) != feature_names:
        raise ValueError("validation and checkpoint feature schemas differ")
    lgbm_result = state.get("lgbm_result")
    if lgbm_result is None:
        raise ValueError("champion checkpoint has no Dataset2 LightGBM")
    feature_indices = tuple(
        int(index) for index in lgbm_result.feature_indices
    )
    if feature_indices != tuple(range(EXPECTED_SHAPE[-1])):
        raise ValueError("frozen scan requires all 63 champion features")

    frozen = {
        "status": "frozen_before_prediction",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "validation_features": str(args.validation_features.resolve()),
        "validation_features_sha256": expected_validation_sha,
        "validation_shape": list(val_features.shape),
        "validation_cache_report": str(
            args.validation_cache_report.resolve()
        ),
        "validation_cache_report_sha256": _sha256(
            args.validation_cache_report
        ),
        "source_evaluation_report": str(
            args.source_evaluation_report.resolve()
        ),
        "source_evaluation_report_sha256": _sha256(
            args.source_evaluation_report
        ),
        "setwise_model": str(args.setwise_model.resolve()),
        "setwise_model_sha256": _sha256(args.setwise_model),
        "weight_grid": {
            "setwise_start": weights[0],
            "setwise_stop": weights[-1],
            "step": args.weight_step,
            "weights_tested": len(weights),
        },
        "reference_weight": args.reference_weight,
        "selection_rows": [0, SELECTION_STOP],
        "forward_rows": [SELECTION_STOP, EXPECTED_SHAPE[0]],
        "selection_uses_forward_rows": False,
        "tie_break": "higher_setwise_weight",
        "gate": {
            "minimum_full_mrr_delta": args.min_full_delta,
            "all_three_slices_non_decreasing": True,
        },
    }
    _write_json_atomic(frozen_path, frozen)
    print(json.dumps(frozen, ensure_ascii=False, indent=2), flush=True)

    champion_lgbm = _softmax(
        predict_logits_lgbm(
            lgbm_result.model_text,
            val_features,
        )
    )
    champion_lgbm_metrics = ranking_mrr_three_slices(champion_lgbm)
    expected_lgbm = source_evaluation["baseline"]["lightgbm"]
    _require_metrics_close(
        champion_lgbm_metrics,
        expected_lgbm,
        "champion LightGBM",
    )

    payload = np.load(args.setwise_model, allow_pickle=False)
    hidden_dim = int(payload["hidden_dim"][0])
    source_feature_count = int(payload["source_feature_count"][0])
    if source_feature_count != EXPECTED_SHAPE[-1]:
        raise ValueError("Setwise source feature count differs")
    if int(payload["context_transform_version"][0]) != 1:
        raise ValueError("unsupported Setwise context transform")
    setwise_state = {
        key.removeprefix("state__"): np.asarray(
            payload[key],
            dtype=np.float32,
        )
        for key in payload.files
        if key.startswith("state__")
    }
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    setwise_indices = tuple(
        int(value) for value in payload["feature_indices"]
    )
    setwise_model = FusionMLP(
        input_dim=len(setwise_indices),
        hidden_dim=hidden_dim,
    )
    set_model_state(setwise_model, setwise_state)
    setwise_view = SetwiseFeatureView(val_features)
    setwise_logits = _predict_streaming(
        setwise_model,
        setwise_view,
        mean,
        std,
        feature_indices=setwise_indices,
        batch_size=args.batch_size,
    )
    setwise = _softmax(setwise_logits)
    del setwise_logits, setwise_model, setwise_view, payload, state
    gc.collect()
    setwise_metrics = ranking_mrr_three_slices(setwise)
    _require_metrics_close(
        setwise_metrics,
        source_evaluation["setwise"]["expert"],
        "Setwise expert",
    )

    scan = scan_high_weight_blend_on_prefix(
        setwise,
        champion_lgbm,
        selection_stop=SELECTION_STOP,
        primary_weights=weights,
    )
    trials: list[dict[str, Any]] = []
    for setwise_weight in weights:
        blended = (
            setwise_weight * setwise
            + (1.0 - setwise_weight) * champion_lgbm
        )
        metrics = ranking_mrr_three_slices(blended)
        trials.append(
            {
                "setwise_weight": setwise_weight,
                "selection_mrr": _selection_mrr(blended),
                "metrics": metrics,
            }
        )
    selected = next(
        trial
        for trial in trials
        if trial["setwise_weight"] == scan.primary_weight
    )
    if args.reference_weight is None:
        baseline = {
            key: float(value)
            for key, value in source_evaluation["baseline"]["fixed_blend"].items()
        }
    else:
        reference_blend = (
            args.reference_weight * setwise
            + (1.0 - args.reference_weight) * champion_lgbm
        )
        baseline = ranking_mrr_three_slices(reference_blend)
        source_setwise = source_evaluation.get("setwise", {})
        if abs(
            float(source_setwise.get("selected_weight", np.nan))
            - args.reference_weight
        ) <= 1e-12:
            _require_metrics_close(
                baseline,
                source_setwise["fixed_blend"],
                "reference blend",
            )
    selected_metrics = {
        key: float(value) for key, value in selected["metrics"].items()
    }
    slice_keys = ("slice_0", "slice_1", "slice_2")
    slice_deltas = [
        selected_metrics[key] - baseline[key] for key in slice_keys
    ]
    full_delta = selected_metrics["full"] - baseline["full"]
    passed = passes_full100_gate(
        baseline_full_mrr=baseline["full"],
        candidate_full_mrr=selected_metrics["full"],
        baseline_slice_mrrs=tuple(baseline[key] for key in slice_keys),
        candidate_slice_mrrs=tuple(
            selected_metrics[key] for key in slice_keys
        ),
        min_full_delta=args.min_full_delta,
    )
    if (
        scan.weights_tested != len(trials)
        or trials[0]["setwise_weight"] != weights[0]
        or trials[-1]["setwise_weight"] != weights[-1]
    ):
        raise RuntimeError("Setwise high-weight grid is incomplete")
    report = {
        "status": "passed" if passed else "rejected",
        "gate_passed": passed,
        "package_authorized": passed,
        "package_generated": False,
        "winner": "setwise" if passed else None,
        "frozen_config": frozen,
        "baseline": {
            "setwise_weight": args.reference_weight,
            "fixed_blend": baseline,
            "lightgbm": champion_lgbm_metrics,
        },
        "setwise": {
            "gate_passed": passed,
            "selected_weight": scan.primary_weight,
            "selection_mrr": scan.selection_mrr,
            "fixed_blend": selected_metrics,
            "expert": setwise_metrics,
            "full_delta": full_delta,
            "slice_deltas": slice_deltas,
            "best_val_ap": source_evaluation["setwise"]["best_val_ap"],
            "best_val_mrr": source_evaluation["setwise"]["best_val_mrr"],
            "model_path": str(args.setwise_model.resolve()),
            "model_sha256": _sha256(args.setwise_model),
        },
        "gate": {
            "minimum_full_mrr_delta": args.min_full_delta,
            "full_delta_passed": bool(
                full_delta + 1e-12 >= args.min_full_delta
            ),
            "all_three_slices_non_decreasing": bool(
                all(delta >= 0.0 for delta in slice_deltas)
            ),
            "forward_slice_delta": slice_deltas[2],
        },
        "trials": trials,
        "elapsed_seconds": time.time() - started,
    }
    _write_json_atomic(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if passed else 2


def _predict_streaming(
    model: FusionMLP,
    features: Any,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    feature_indices: tuple[int, ...],
    batch_size: int,
) -> np.ndarray:
    scores = np.empty(features.shape[:2], dtype=np.float32)
    for start in range(0, features.shape[0], batch_size):
        end = min(start + batch_size, features.shape[0])
        batch = np.asarray(features[start:end], dtype=np.float32)
        if feature_indices != tuple(range(batch.shape[-1])):
            batch = batch[..., feature_indices]
        scores[start:end] = predict_logits(model, batch, mean, std)
    return scores


def _selection_mrr(scores: np.ndarray) -> float:
    values = np.asarray(scores[:SELECTION_STOP])
    positive = values[:, 0:1]
    greater = np.sum(values[:, 1:] > positive, axis=1)
    equal = np.sum(values[:, 1:] == positive, axis=1)
    return float(np.mean(1.0 / (1.0 + greater + 0.5 * equal)))


def _require_metrics_close(
    actual: dict[str, float],
    expected: dict[str, float],
    label: str,
) -> None:
    for key, expected_value in expected.items():
        if abs(float(actual[key]) - float(expected_value)) > 1e-10:
            raise RuntimeError(
                f"{label} reproduction failed for {key}: "
                f"actual={actual[key]:.16f} expected={expected_value:.16f}"
            )


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch: actual={actual} expected={expected}"
        )


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
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
