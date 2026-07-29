from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.core.io import read_interactions
from jgrec.core.types import TestQueryArray
from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid.fusion import (
    FusionConfig,
    fit_fusion_mlp_listwise_streaming,
)
from jgrec.rankers.hybrid.fusion_analysis import ranking_mrr_three_slices
from jgrec.rankers.hybrid.fusion_lgbm import predict_logits_lgbm
from jgrec.rankers.hybrid.gnn import (
    GRAPH_WINDOW_FRACTIONS,
    GraphTower,
    _graph_window_data,
    _mapped_edges,
)
from jgrec.rankers.hybrid.gnn_experiment import (
    GNN_CAPACITY_VARIANTS,
    resolve_gnn_capacity_experiment,
)
from jgrec.rankers.hybrid.setwise import SetwiseFeatureView

VARIANTS = GNN_CAPACITY_VARIANTS


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
        description="Evaluate isolated Dataset2 short/recent GNN edge weights."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument("--train-cache-report", required=True, type=Path)
    parser.add_argument("--validation-cache-prefix", required=True, type=Path)
    parser.add_argument("--validation-cache-report", required=True, type=Path)
    parser.add_argument("--source-evaluation-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--score-batch-rows", type=int, default=4096)
    parser.add_argument("--fusion-batch-size", type=int, default=256)
    parser.add_argument("--fusion-epochs", type=int, default=10)
    parser.add_argument("--fusion-patience", type=int, default=2)
    parser.add_argument("--setwise-weight", type=float, default=0.80)
    parser.add_argument("--min-full-delta", type=float, default=0.001)
    parser.add_argument(
        "--variant",
        action="append",
        choices=tuple(VARIANTS),
        help="Run only the named graph variant; repeat for multiple variants.",
    )
    parser.add_argument("--graph-epochs", type=int)
    parser.add_argument("--graph-max-train-edges", type=int)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.time()
    jt.flags.use_cuda = 1

    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    config = state["config"]
    baseline_graph_config = config.graph_config()
    graph_config, variants = resolve_gnn_capacity_experiment(
        baseline_graph_config,
        variant_names=tuple(args.variant or VARIANTS),
        epochs=(
            baseline_graph_config.epochs
            if args.graph_epochs is None
            else args.graph_epochs
        ),
        max_train_edges=(
            baseline_graph_config.max_train_edges
            if args.graph_max_train_edges is None
            else args.graph_max_train_edges
        ),
    )
    feature_names = tuple(str(name) for name in state["feature_names"])
    train_report = _read_json(args.train_cache_report)
    val_report = _read_json(args.validation_cache_report)
    source_report = _read_json(args.source_evaluation_report)
    split = val_report["split"]
    train_end = int(split["train_end"])
    train_cache_path = Path(f"{args.train_cache_prefix}.train.npy")
    val_cache_path = Path(f"{args.validation_cache_prefix}.val.npy")
    train_features = np.load(train_cache_path, mmap_mode="r", allow_pickle=False)
    val_features = np.load(val_cache_path, mmap_mode="r", allow_pickle=False)
    if train_features.shape != (200_000, 100, len(feature_names)):
        raise ValueError(f"unexpected training shape: {train_features.shape}")
    if val_features.shape != (20_000, 100, len(feature_names)):
        raise ValueError(f"unexpected validation shape: {val_features.shape}")

    train_src = _load_sidecar(args.train_cache_prefix, "src")
    train_time = _load_sidecar(args.train_cache_prefix, "time")
    train_candidates = _load_sidecar(args.train_cache_prefix, "candidates")
    val_src = _load_sidecar(args.validation_cache_prefix, "src", split="val")
    val_time = _load_sidecar(args.validation_cache_prefix, "time", split="val")
    val_candidates = _load_sidecar(
        args.validation_cache_prefix,
        "candidates",
        split="val",
    )
    train_queries = TestQueryArray(
        src=train_src,
        time=train_time,
        candidates=train_candidates,
    )
    val_queries = TestQueryArray(
        src=val_src,
        time=val_time,
        candidates=val_candidates,
    )

    interactions = read_interactions(args.train_csv).sort_by_time()
    if config.max_fit_events > 0 and len(interactions) > config.max_fit_events:
        interactions = interactions.tail(config.max_fit_events)
    context_end = int(train_report["split"]["context_end"])
    if train_end > len(interactions) or context_end >= train_end:
        raise ValueError("cache split is incompatible with interaction table")
    id_map = NodeIdMap.from_interactions(interactions)

    frozen = {
        "status": "frozen_before_training",
        "variants": variants,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "train_cache_sha256": _sha256(train_cache_path),
        "validation_cache_sha256": _sha256(val_cache_path),
        "baseline_graph_config": {
            "epochs": baseline_graph_config.epochs,
            "max_train_edges": baseline_graph_config.max_train_edges,
        },
        "graph_config": {
            "model": graph_config.model_name,
            "embedding_dim": graph_config.embedding_dim,
            "layers": graph_config.layers,
            "epochs": graph_config.epochs,
            "max_train_edges": graph_config.max_train_edges,
            "time_decay_ratio": graph_config.time_decay_ratio,
        },
        "split": {
            "context_end": context_end,
            "train_end": train_end,
        },
        "selection_slices": [0, 1],
        "forward_slice": 2,
        "minimum_full_delta": args.min_full_delta,
        "setwise_weight": args.setwise_weight,
    }
    _write_json(args.output_dir / "frozen-config.json", frozen)
    print(json.dumps(frozen, indent=2), flush=True)

    graph_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for variant_index, (variant, (window_name, weighting)) in enumerate(
        variants.items()
    ):
        train_score_path = args.output_dir / f"{variant}.train-scores.npy"
        val_score_path = args.output_dir / f"{variant}.val-scores.npy"
        train_scores = _fit_and_score_window(
            interactions=interactions[:context_end],
            queries=train_queries,
            id_map=id_map,
            graph_config=graph_config,
            window_name=window_name,
            weighting=weighting,
            seed=args.seed + 1000 + (0 if window_name == "gnn_short" else 100),
            batch_rows=args.score_batch_rows,
            output_path=train_score_path,
        )
        val_scores = _fit_and_score_window(
            interactions=interactions[:train_end],
            queries=val_queries,
            id_map=id_map,
            graph_config=graph_config,
            window_name=window_name,
            weighting=weighting,
            seed=args.seed + 2000 + (0 if window_name == "gnn_short" else 100),
            batch_rows=args.score_batch_rows,
            output_path=val_score_path,
        )
        graph_scores[variant] = (train_scores, val_scores)
        print(
            f"[graph-variant] {variant_index + 1}/{len(variants)} "
            f"name={variant} elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    lgbm_result = state.get("lgbm_result")
    if lgbm_result is None:
        raise ValueError("checkpoint has no Dataset2 LightGBM")
    lgbm_probabilities = _softmax(
        predict_logits_lgbm(lgbm_result.model_text, val_features)
    )
    expected_lgbm = source_report["baseline"]["lightgbm"]
    actual_lgbm = ranking_mrr_three_slices(lgbm_probabilities)
    _require_close(actual_lgbm, expected_lgbm, "LightGBM")

    fusion_config = FusionConfig(
        epochs=args.fusion_epochs,
        batch_size=args.fusion_batch_size,
        lr=0.001,
        weight_decay=0.0,
        hidden_dim=32,
        selection_metric="mrr",
        early_stop_patience=args.fusion_patience,
    )
    trials: dict[str, Any] = {}
    for variant, (window_name, _) in variants.items():
        column = feature_names.index(window_name)
        train_scores, val_scores = graph_scores[variant]
        train_view = SetwiseFeatureView(
            _ColumnOverlay(train_features, train_scores, column)
        )
        val_view = SetwiseFeatureView(
            _ColumnOverlay(val_features, val_scores, column)
        )
        model, result, history = fit_fusion_mlp_listwise_streaming(
            train_view,
            val_view,
            fusion_config,
            np.random.default_rng(args.seed),
            verbose=True,
            candidate_name=f"dataset2_gnn_{variant}",
        )
        probabilities = _softmax(
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
            args.setwise_weight * probabilities
            + (1.0 - args.setwise_weight) * lgbm_probabilities
        )
        metrics = ranking_mrr_three_slices(blend)
        trials[variant] = {
            "window": window_name,
            "weighting": variants[variant][1],
            "setwise": ranking_mrr_three_slices(probabilities),
            "fixed_blend": metrics,
            "best_val_mrr": result.best_val_mrr,
            "history": list(history),
            "train_score_sha256": _sha256(
                args.output_dir / f"{variant}.train-scores.npy"
            ),
            "val_score_sha256": _sha256(
                args.output_dir / f"{variant}.val-scores.npy"
            ),
        }
        del model, probabilities, blend, train_view, val_view
        gc.collect()
        print(f"[fusion-variant] {variant} metrics={metrics}", flush=True)

    champion = {
        key: float(value)
        for key, value in source_report["setwise"]["fixed_blend"].items()
    }
    edge_variants = (
        "short_repeat",
        "short_time_decay",
        "recent_time_decay",
    )
    complete_edge_comparison = all(
        name in trials
        for name in (*edge_variants, "short_none", "recent_none")
    )
    comparisons: dict[str, Any] = {}
    if complete_edge_comparison:
        for variant in edge_variants:
            control = (
                "short_none"
                if variant.startswith("short_")
                else "recent_none"
            )
            candidate = trials[variant]["fixed_blend"]
            control_metrics = trials[control]["fixed_blend"]
            comparisons[variant] = {
                "control": control,
                "delta_vs_control": _metric_delta(
                    candidate,
                    control_metrics,
                ),
                "delta_vs_champion": _metric_delta(
                    candidate,
                    champion,
                ),
                "selection_score": float(
                    (
                        candidate["slice_0"]
                        - control_metrics["slice_0"]
                        + candidate["slice_1"]
                        - control_metrics["slice_1"]
                    )
                    / 2.0
                ),
            }
        selected = max(
            comparisons,
            key=lambda name: comparisons[name]["selection_score"],
        )
    else:
        comparisons = {
            variant: {
                "delta_vs_champion": _metric_delta(
                    trial["fixed_blend"],
                    champion,
                ),
                "selection_score": float(
                    (
                        trial["fixed_blend"]["slice_0"]
                        - champion["slice_0"]
                        + trial["fixed_blend"]["slice_1"]
                        - champion["slice_1"]
                    )
                    / 2.0
                ),
            }
            for variant, trial in trials.items()
        }
        selected = max(
            comparisons,
            key=lambda name: comparisons[name]["selection_score"],
        )
    selected_metrics = trials[selected]["fixed_blend"]
    delta = _metric_delta(selected_metrics, champion)
    edge_gate_passed = bool(
        delta["full"] >= args.min_full_delta
        and all(delta[f"slice_{index}"] >= 0.0 for index in range(3))
    )
    report = {
        "status": "complete",
        "frozen_config": frozen,
        "champion": champion,
        "lightgbm": actual_lgbm,
        "trials": trials,
        "comparisons": comparisons,
        "selected_on_slices_0_1": selected,
        "forward_gate": {
            "delta_vs_champion": delta,
            "minimum_full_delta": args.min_full_delta,
            "all_slices_non_decreasing": all(
                delta[f"slice_{index}"] >= 0.0 for index in range(3)
            ),
            "passed": edge_gate_passed,
        },
        "candidate_aligned_graph_objective_authorized": edge_gate_passed,
        "elapsed_seconds": time.time() - started,
    }
    report_name = (
        "edge-weight-report.json"
        if tuple(variants) == tuple(VARIANTS)
        and graph_config == baseline_graph_config
        else "gnn-capacity-report.json"
    )
    _write_json(args.output_dir / report_name, report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


def _fit_and_score_window(
    *,
    interactions: Any,
    queries: TestQueryArray,
    id_map: NodeIdMap,
    graph_config: Any,
    window_name: str,
    weighting: str,
    seed: int,
    batch_rows: int,
    output_path: Path,
) -> np.ndarray:
    window_index = ("gnn_full", "gnn_recent", "gnn_short").index(window_name)
    config = replace(
        graph_config,
        edge_weighting="none",
        full_edge_weighting=None,
        recent_edge_weighting=(
            weighting if window_name == "gnn_recent" else None
        ),
        short_edge_weighting=(
            weighting if window_name == "gnn_short" else None
        ),
    )
    mapped_edges = _mapped_edges(interactions, id_map, config)
    edge_count = max(
        1,
        int(len(mapped_edges) * GRAPH_WINDOW_FRACTIONS[window_index]),
    )
    window_edges = mapped_edges[-edge_count:]
    rng = np.random.default_rng(seed)
    edge_index, edge_weight = _graph_window_data(
        window_edges,
        config,
        rng,
        window_name=window_name,
    )
    jt.set_seed(seed)
    tower = GraphTower(id_map=id_map, config=config)
    tower._fit_one_window(
        window_name,
        edge_index,
        edge_weight,
        rng,
        verbose=True,
    )
    scores = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(queries), queries.candidate_count),
    )
    for start in range(0, len(queries), batch_rows):
        end = min(start + batch_rows, len(queries))
        batch = tower.scores_for_query_array(queries[start:end])
        scores[start:end] = batch[..., window_index]
        if end == len(queries) or (start // batch_rows) % 8 == 7:
            scores.flush()
    scores.flush()
    del tower
    gc.collect()
    return scores


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


def _load_sidecar(
    prefix: Path,
    name: str,
    *,
    split: str = "train",
) -> np.ndarray:
    return np.load(
        Path(f"{prefix}.{split}-{name}.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )


def _metric_delta(
    candidate: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, float]:
    return {
        key: float(candidate[key] - baseline[key])
        for key in ("full", "slice_0", "slice_1", "slice_2")
    }


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
