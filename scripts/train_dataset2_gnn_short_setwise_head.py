from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.rankers.hybrid.config import graph_window_edge_parameters
from jgrec.rankers.hybrid.fusion import (
    FusionConfig,
    fit_fusion_mlp_listwise_streaming,
)
from jgrec.rankers.hybrid.fusion_analysis import ranking_mrr_three_slices
from jgrec.rankers.hybrid.fusion_lgbm import predict_logits_lgbm
from jgrec.rankers.hybrid.setwise import SetwiseFeatureView

VARIANT = "short_none"
WINDOW_NAME = "gnn_short"
GRAPH_EPOCHS = 50
GRAPH_MAX_TRAIN_EDGES = 40_000


class _ColumnOverlay:
    def __init__(
        self,
        source: Any,
        replacement: np.ndarray,
        column: int,
    ) -> None:
        if replacement.shape != source.shape[:2]:
            raise ValueError("replacement graph scores must match cache rows")
        self._source = source
        self._replacement = replacement
        self._column = int(column)
        self.shape = tuple(int(value) for value in source.shape)
        self.ndim = 3
        self.size = int(np.prod(self.shape, dtype=np.int64))

    def __getitem__(self, key: Any) -> np.ndarray:
        values = np.array(self._source[key], dtype=np.float32, copy=True)
        values[..., self._column] = self._replacement[key]
        return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce and save the Dataset2 Setwise head matched to the "
            "winning short_none GNN at 50 epochs and 40k train edges."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument("--train-cache-report", required=True, type=Path)
    parser.add_argument("--validation-cache-prefix", required=True, type=Path)
    parser.add_argument("--validation-cache-report", required=True, type=Path)
    parser.add_argument("--edge-report", required=True, type=Path)
    parser.add_argument("--score-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--fusion-epochs", type=int, default=10)
    parser.add_argument("--fusion-batch-size", type=int, default=256)
    parser.add_argument("--fusion-patience", type=int, default=2)
    parser.add_argument("--setwise-weight", type=float, default=0.80)
    parser.add_argument("--min-full-delta", type=float, default=0.001)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.time()

    edge_report = _read_json(args.edge_report)
    trial = edge_report.get("trials", {}).get(VARIANT)
    if edge_report.get("status") != "complete" or trial is None:
        raise ValueError("edge report has no completed short_none trial")
    frozen_graph = edge_report["frozen_config"]["graph_config"]
    if (
        int(frozen_graph["epochs"]) != GRAPH_EPOCHS
        or int(frozen_graph["max_train_edges"]) != GRAPH_MAX_TRAIN_EDGES
        or trial["window"] != WINDOW_NAME
        or trial["weighting"] != "none"
    ):
        raise ValueError("edge report is not the winning short_none 50/40k run")
    expected_checkpoint_hash = edge_report["frozen_config"][
        "checkpoint_sha256"
    ]
    _require_hash(args.checkpoint, expected_checkpoint_hash, "checkpoint")

    train_report = _read_json(args.train_cache_report)
    val_report = _read_json(args.validation_cache_report)
    train_path = Path(f"{args.train_cache_prefix}.train.npy")
    val_path = Path(f"{args.validation_cache_prefix}.val.npy")
    train_score_path = args.score_dir / f"{VARIANT}.train-scores.npy"
    val_score_path = args.score_dir / f"{VARIANT}.val-scores.npy"
    _require_hash(
        train_path,
        edge_report["frozen_config"]["train_cache_sha256"],
        "training feature cache",
    )
    _require_hash(
        val_path,
        edge_report["frozen_config"]["validation_cache_sha256"],
        "validation feature cache",
    )
    _require_hash(
        train_score_path,
        trial["train_score_sha256"],
        "short_none training scores",
    )
    _require_hash(
        val_score_path,
        trial["val_score_sha256"],
        "short_none validation scores",
    )

    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    config = state["config"]
    graph_config = config.graph_config()
    short_weighting, _ = graph_window_edge_parameters(
        graph_config,
        WINDOW_NAME,
    )
    if (
        graph_config.epochs != GRAPH_EPOCHS
        or graph_config.max_train_edges != GRAPH_MAX_TRAIN_EDGES
        or short_weighting != "none"
    ):
        raise ValueError("checkpoint service encoder is not short_none 50/40k")
    feature_names = tuple(str(name) for name in state["feature_names"])
    if feature_names != tuple(train_report["feature_names"]):
        raise ValueError("checkpoint and training feature schemas differ")
    if feature_names != tuple(val_report["feature_names"]):
        raise ValueError("checkpoint and validation feature schemas differ")
    gnn_column = feature_names.index(WINDOW_NAME)

    train_features = np.load(train_path, mmap_mode="r", allow_pickle=False)
    val_features = np.load(val_path, mmap_mode="r", allow_pickle=False)
    train_scores = np.load(
        train_score_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    val_scores = np.load(val_score_path, mmap_mode="r", allow_pickle=False)
    expected_train_shape = (200_000, 100, len(feature_names))
    expected_val_shape = (20_000, 100, len(feature_names))
    if train_features.shape != expected_train_shape:
        raise ValueError(f"unexpected training shape: {train_features.shape}")
    if val_features.shape != expected_val_shape:
        raise ValueError(f"unexpected validation shape: {val_features.shape}")
    if train_scores.shape != expected_train_shape[:2]:
        raise ValueError(f"unexpected training score shape: {train_scores.shape}")
    if val_scores.shape != expected_val_shape[:2]:
        raise ValueError(f"unexpected validation score shape: {val_scores.shape}")

    lgbm_result = state.get("lgbm_result")
    if lgbm_result is None:
        raise ValueError("checkpoint has no Dataset2 LightGBM expert")
    lgbm_probabilities = _softmax(
        predict_logits_lgbm(lgbm_result.model_text, val_features)
    )
    _require_close(
        ranking_mrr_three_slices(lgbm_probabilities),
        edge_report["lightgbm"],
        "LightGBM",
    )

    jt.flags.use_cuda = 1
    fusion_config = FusionConfig(
        epochs=args.fusion_epochs,
        batch_size=args.fusion_batch_size,
        lr=0.001,
        weight_decay=0.0,
        hidden_dim=32,
        selection_metric="mrr",
        early_stop_patience=args.fusion_patience,
    )
    train_view = SetwiseFeatureView(
        _ColumnOverlay(train_features, train_scores, gnn_column)
    )
    val_view = SetwiseFeatureView(
        _ColumnOverlay(val_features, val_scores, gnn_column)
    )
    model, result, history = fit_fusion_mlp_listwise_streaming(
        train_view,
        val_view,
        fusion_config,
        np.random.default_rng(args.seed),
        verbose=True,
        candidate_name="dataset2_gnn_short_none_e50_edges40000",
    )
    setwise_probabilities = _softmax(
        _predict_streaming(
            model,
            val_view,
            result.mean,
            result.std,
            result.feature_indices,
            args.fusion_batch_size,
        )
    )
    blend = (
        args.setwise_weight * setwise_probabilities
        + (1.0 - args.setwise_weight) * lgbm_probabilities
    )
    actual_setwise = ranking_mrr_three_slices(setwise_probabilities)
    actual_blend = ranking_mrr_three_slices(blend)
    source_trial_delta = _metric_delta(
        actual_blend,
        trial["fixed_blend"],
    )
    source_trial_max_abs_delta = max(
        abs(value)
        for value in source_trial_delta.values()
    )

    model_path = args.output_dir / "dataset2-gnn-short-none-e50-edges40000-setwise.npz"
    _save_setwise_model(
        model_path,
        result=result,
        hidden_dim=fusion_config.hidden_dim,
        source_feature_count=len(feature_names),
    )
    champion = {
        key: float(value)
        for key, value in edge_report["champion"].items()
    }
    delta = _metric_delta(actual_blend, champion)
    gate_passed = bool(
        delta["full"] >= args.min_full_delta
        and all(delta[f"slice_{index}"] >= 0.0 for index in range(3))
    )
    report = {
        "status": "passed" if gate_passed else "rejected",
        "gate_passed": gate_passed,
        "package_authorized": gate_passed,
        "variant": VARIANT,
        "window": WINDOW_NAME,
        "weighting": "none",
        "graph_epochs": GRAPH_EPOCHS,
        "graph_max_train_edges": GRAPH_MAX_TRAIN_EDGES,
        "setwise_weight": args.setwise_weight,
        "feature_count": len(feature_names),
        "context_feature_count": int(train_view.shape[-1]),
        "gnn_feature_column": gnn_column,
        "champion": champion,
        "candidate": actual_blend,
        "delta_vs_champion": delta,
        "source_trial_candidate": trial["fixed_blend"],
        "delta_vs_source_trial": source_trial_delta,
        "source_trial_max_abs_delta": source_trial_max_abs_delta,
        "source_trial_reproduced_exactly": (
            source_trial_max_abs_delta <= 1e-10
            and abs(
                float(result.best_val_mrr)
                - float(trial["best_val_mrr"])
            )
            <= 1e-10
        ),
        "setwise": actual_setwise,
        "lightgbm": ranking_mrr_three_slices(lgbm_probabilities),
        "setwise_best_val_ap": float(result.best_val_ap),
        "setwise_best_val_mrr": float(result.best_val_mrr),
        "setwise_history": list(history),
        "setwise_model": str(model_path.resolve()),
        "setwise_model_sha256": _sha256(model_path),
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": _sha256(args.checkpoint),
        "source_edge_report": str(args.edge_report.resolve()),
        "source_edge_report_sha256": _sha256(args.edge_report),
        "train_score_sha256": _sha256(train_score_path),
        "val_score_sha256": _sha256(val_score_path),
        "minimum_full_delta": args.min_full_delta,
        "all_slices_non_decreasing": all(
            delta[f"slice_{index}"] >= 0.0 for index in range(3)
        ),
        "encoder_retrained": False,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.output_dir / "evaluation-report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if gate_passed else 2


def _predict_streaming(
    model: Any,
    features: Any,
    mean: np.ndarray,
    std: np.ndarray,
    feature_indices: tuple[int, ...],
    batch_size: int,
) -> np.ndarray:
    scores = np.empty(features.shape[:2], dtype=np.float32)
    with jt.no_grad():
        for start in range(0, features.shape[0], batch_size):
            end = min(start + batch_size, features.shape[0])
            selected = np.asarray(features[start:end], dtype=np.float32)
            selected = selected[..., feature_indices]
            normalized = ((selected - mean) / std).astype(
                np.float32,
                copy=False,
            )
            scores[start:end] = np.asarray(
                model(jt.array(normalized, dtype=jt.float32)).numpy(),
                dtype=np.float32,
            )
    return scores


def _save_setwise_model(
    path: Path,
    *,
    result: Any,
    hidden_dim: int,
    source_feature_count: int,
) -> None:
    payload = {
        "mean": np.asarray(result.mean, dtype=np.float32),
        "std": np.asarray(result.std, dtype=np.float32),
        "feature_indices": np.asarray(result.feature_indices, dtype=np.int32),
        "hidden_dim": np.asarray([hidden_dim], dtype=np.int32),
        "source_feature_count": np.asarray(
            [source_feature_count],
            dtype=np.int32,
        ),
        "context_transform_version": np.asarray([1], dtype=np.int32),
    }
    payload.update(
        {
            f"state__{key}": np.asarray(value, dtype=np.float32)
            for key, value in result.state.items()
        }
    )
    np.savez_compressed(path, **payload)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.asarray(logits, dtype=np.float64)
    shifted -= shifted.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def _require_close(
    actual: dict[str, float],
    expected: dict[str, float],
    label: str,
) -> None:
    for key, value in expected.items():
        if abs(float(actual[key]) - float(value)) > 1e-10:
            raise RuntimeError(
                f"{label} reproduction failed for {key}: "
                f"actual={actual[key]} expected={value}"
            )


def _metric_delta(
    candidate: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, float]:
    return {
        key: float(candidate[key] - baseline[key])
        for key in ("full", "slice_0", "slice_1", "slice_2")
    }


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
