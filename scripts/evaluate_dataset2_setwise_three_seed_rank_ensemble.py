from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset, set_model_state
from jgrec.rankers.hybrid.full100_training import passes_full100_gate
from jgrec.rankers.hybrid.fusion import FusionMLP, predict_logits
from jgrec.rankers.hybrid.fusion_analysis import (
    ranking_mrr_three_slices,
    uniform_rank_average,
)
from jgrec.rankers.hybrid.fusion_lgbm import predict_logits_lgbm
from jgrec.rankers.hybrid.setwise import SetwiseFeatureView

EXPECTED_SHAPE = (20_000, 100, 63)
SEEDS = (17, 41, 60)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the fixed Dataset2 three-seed Setwise rank ensemble."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--validation-features", required=True, type=Path)
    parser.add_argument("--validation-cache-report", required=True, type=Path)
    parser.add_argument(
        "--seed60-source-evaluation-report",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--champion-evaluation-report",
        required=True,
        type=Path,
    )
    parser.add_argument("--seed17-model", required=True, type=Path)
    parser.add_argument("--seed41-model", required=True, type=Path)
    parser.add_argument("--seed60-model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--setwise-weight", type=float, default=0.80)
    parser.add_argument("--min-full-delta", type=float, default=0.001)
    args = parser.parse_args()

    if abs(args.setwise_weight - 0.80) > 1e-12:
        raise ValueError("three-seed experiment fixes the Setwise weight at 0.80")
    report_path = args.output_dir / "evaluation-report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {report_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    validation_report = _read_json(args.validation_cache_report)
    seed60_source = _read_json(args.seed60_source_evaluation_report)
    champion_evaluation = _read_json(args.champion_evaluation_report)
    expected_validation_sha = validation_report["artifacts"]["features"]["sha256"]
    _require_hash(
        args.validation_features,
        expected_validation_sha,
        "validation features",
    )
    _require_hash(
        args.checkpoint,
        champion_evaluation["frozen_config"]["checkpoint_sha256"],
        "champion checkpoint",
    )
    _require_hash(
        args.seed60_model,
        seed60_source["setwise"]["model_sha256"],
        "seed-60 Setwise model",
    )
    if abs(
        float(champion_evaluation["setwise"]["selected_weight"])
        - args.setwise_weight
    ) > 1e-12:
        raise ValueError("champion evaluation does not use the fixed 0.80 weight")

    validation_features = np.load(
        args.validation_features,
        mmap_mode="r",
        allow_pickle=False,
    )
    if validation_features.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"validation feature shape mismatch: {validation_features.shape}"
        )
    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    feature_names = tuple(str(name) for name in state["feature_names"])
    if feature_names != tuple(validation_report["feature_names"]):
        raise ValueError("checkpoint and validation feature schemas differ")
    lgbm_result = state.get("lgbm_result")
    if lgbm_result is None:
        raise ValueError("champion checkpoint has no Dataset2 LightGBM expert")
    if tuple(int(index) for index in lgbm_result.feature_indices) != tuple(
        range(EXPECTED_SHAPE[-1])
    ):
        raise ValueError("three-seed evaluation requires all 63 features")

    champion = {
        key: float(value)
        for key, value in champion_evaluation["setwise"]["fixed_blend"].items()
    }
    lgbm = _softmax(
        predict_logits_lgbm(lgbm_result.model_text, validation_features)
    )
    validation_view = SetwiseFeatureView(validation_features)
    model_paths = {
        17: args.seed17_model,
        41: args.seed41_model,
        60: args.seed60_model,
    }
    final_scores: list[np.ndarray] = []
    seed_reports: dict[str, Any] = {}
    for seed in SEEDS:
        model_path = model_paths[seed]
        model, payload = _load_setwise_model(model_path)
        logits = _predict_streaming(
            model,
            validation_view,
            payload["mean"],
            payload["std"],
            feature_indices=payload["feature_indices"],
            batch_size=args.batch_size,
        )
        probabilities = _softmax(logits)
        blended = (
            args.setwise_weight * probabilities
            + (1.0 - args.setwise_weight) * lgbm
        )
        expert_metrics = ranking_mrr_three_slices(probabilities)
        blend_metrics = ranking_mrr_three_slices(blended)
        if seed == 60:
            _require_metrics_close(
                expert_metrics,
                seed60_source["setwise"]["expert"],
                "seed-60 Setwise expert",
            )
            _require_metrics_close(
                blend_metrics,
                champion,
                "seed-60 champion blend",
            )
        prediction_path = args.output_dir / f"validation-blend-seed{seed}.npy"
        prediction = np.asarray(blended, dtype=np.float32)
        if prediction_path.exists():
            existing = np.load(prediction_path, allow_pickle=False)
            if not np.array_equal(existing, prediction):
                raise ValueError(
                    f"seed-{seed} saved validation prediction differs"
                )
        else:
            np.save(prediction_path, prediction)
        final_scores.append(blended)
        seed_reports[str(seed)] = {
            "model": str(model_path.resolve()),
            "model_sha256": _sha256(model_path),
            "setwise_expert": expert_metrics,
            "fixed_blend": blend_metrics,
            "validation_prediction": str(prediction_path.resolve()),
            "validation_prediction_sha256": _sha256(prediction_path),
        }
        print(
            f"[three-seed-eval] seed={seed} "
            f"blend_mrr={blend_metrics['full']:.8f}",
            flush=True,
        )
        del model, logits, probabilities, blended, prediction
        gc.collect()

    ensemble_scores = uniform_rank_average(tuple(final_scores))
    ensemble_path = args.output_dir / "validation-uniform-rank-average.npy"
    np.save(ensemble_path, np.asarray(ensemble_scores, dtype=np.float32))
    ensemble = ranking_mrr_three_slices(ensemble_scores)
    metric_keys = ("full", "slice_0", "slice_1", "slice_2")
    deltas = {key: ensemble[key] - champion[key] for key in metric_keys}
    passed = passes_full100_gate(
        baseline_full_mrr=champion["full"],
        candidate_full_mrr=ensemble["full"],
        baseline_slice_mrrs=tuple(
            champion[f"slice_{index}"] for index in range(3)
        ),
        candidate_slice_mrrs=tuple(
            ensemble[f"slice_{index}"] for index in range(3)
        ),
        min_full_delta=args.min_full_delta,
    )
    frozen = {
        "status": "frozen_before_evaluation",
        "seeds": list(SEEDS),
        "aggregation": (
            "uniform query-local rank-percentile average of each seed's "
            "0.80 Setwise + 0.20 champion LightGBM score"
        ),
        "weight_search": False,
        "setwise_weight": args.setwise_weight,
        "minimum_full_mrr_delta": args.min_full_delta,
        "validation_features": str(args.validation_features.resolve()),
        "validation_features_sha256": expected_validation_sha,
        "champion_evaluation_report": str(
            args.champion_evaluation_report.resolve()
        ),
        "champion_evaluation_report_sha256": _sha256(
            args.champion_evaluation_report
        ),
        "seed60_source_evaluation_report": str(
            args.seed60_source_evaluation_report.resolve()
        ),
        "seed60_source_evaluation_report_sha256": _sha256(
            args.seed60_source_evaluation_report
        ),
    }
    report = {
        "status": "passed" if passed else "rejected",
        "gate_passed": passed,
        "package_authorized": passed,
        "package_generated": False,
        "frozen_config": frozen,
        "champion": champion,
        "seeds": seed_reports,
        "ensemble": ensemble,
        "delta_vs_champion": deltas,
        "ensemble_prediction": str(ensemble_path.resolve()),
        "ensemble_prediction_sha256": _sha256(ensemble_path),
        "gate": {
            "minimum_full_mrr_delta": args.min_full_delta,
            "full_delta_passed": bool(
                deltas["full"] + 1e-12 >= args.min_full_delta
            ),
            "all_three_slices_non_decreasing": bool(
                all(deltas[f"slice_{index}"] >= 0.0 for index in range(3))
            ),
        },
        "elapsed_seconds": time.time() - started,
    }
    _write_json_atomic(report_path, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if passed else 2


def _load_setwise_model(path: Path) -> tuple[FusionMLP, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    if int(payload["source_feature_count"][0]) != EXPECTED_SHAPE[-1]:
        raise ValueError("Setwise source feature count differs")
    if int(payload["context_transform_version"][0]) != 1:
        raise ValueError("unsupported Setwise context transform")
    feature_indices = tuple(
        int(value) for value in payload["feature_indices"]
    )
    state = {
        key.removeprefix("state__"): np.asarray(value, dtype=np.float32)
        for key, value in payload.items()
        if key.startswith("state__")
    }
    model = FusionMLP(
        input_dim=len(feature_indices),
        hidden_dim=int(payload["hidden_dim"][0]),
    )
    set_model_state(model, state)
    return model, {
        "mean": np.asarray(payload["mean"], dtype=np.float32),
        "std": np.asarray(payload["std"], dtype=np.float32),
        "feature_indices": feature_indices,
    }


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


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _require_metrics_close(
    actual: dict[str, float],
    expected: dict[str, float],
    label: str,
) -> None:
    for key, expected_value in expected.items():
        if abs(float(actual[key]) - float(expected_value)) > 1e-10:
            raise ValueError(
                f"{label} metric mismatch for {key}: "
                f"actual={actual[key]} expected={expected_value}"
            )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
