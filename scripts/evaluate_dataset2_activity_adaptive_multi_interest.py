from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from evaluate_dataset2_multi_interest_proxy import (
    _AugmentedFeatures,
    _load_setwise_probabilities,
    _normalize_rows,
    _predict,
    _read_json,
    _save_model,
    _sha256,
    _sidecar,
    _softmax,
    _write_json,
)
from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.core.io import read_interactions
from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid.fusion import (
    FusionConfig,
    fit_fusion_mlp_listwise_streaming,
)
from jgrec.rankers.hybrid.fusion_lgbm import predict_logits_lgbm
from jgrec.rankers.hybrid.gnn import (
    GRAPH_WINDOW_FRACTIONS,
    GraphTower,
    _graph_window_data,
    _mapped_edges,
)
from jgrec.rankers.hybrid.multi_interest_proxy import (
    ACTIVITY_ADAPTIVE_FEATURE_NAMES,
    MULTI_INTEREST_FEATURE_NAMES,
    activity_adaptive_cluster_interests,
    activity_adaptive_features_for_candidate_batch,
    exponential_interest_center,
    hierarchical_interest_centers,
)
from jgrec.rankers.hybrid.segment_fusion import (
    QUERY_SEGMENT_FEATURE_NAMES,
    query_segment_features,
)
from jgrec.rankers.hybrid.setwise import SetwiseFeatureView

SLICE_0_STOP = 6_667
SLICE_1_STOP = 13_334


class _ConcatenatedFeatures:
    def __init__(self, *sources: Any) -> None:
        if not sources:
            raise ValueError("at least one feature source is required")
        first_shape = tuple(sources[0].shape[:2])
        if any(tuple(source.shape[:2]) != first_shape for source in sources):
            raise ValueError("feature source rows differ")
        self._sources = sources
        self.shape = (
            int(first_shape[0]),
            int(first_shape[1]),
            int(sum(source.shape[2] for source in sources)),
        )
        self.ndim = 3
        self.size = int(np.prod(self.shape, dtype=np.int64))

    def __getitem__(self, key: Any) -> np.ndarray:
        return np.concatenate(
            tuple(
                np.asarray(source[key], dtype=np.float32)
                for source in self._sources
            ),
            axis=-1,
            dtype=np.float32,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument("--train-cache-report", required=True, type=Path)
    parser.add_argument("--validation-cache-prefix", required=True, type=Path)
    parser.add_argument("--validation-cache-report", required=True, type=Path)
    parser.add_argument("--old-proxy-train", required=True, type=Path)
    parser.add_argument("--old-proxy-validation", required=True, type=Path)
    parser.add_argument("--old-proxy-report", required=True, type=Path)
    parser.add_argument("--champion-setwise-model", required=True, type=Path)
    parser.add_argument(
        "--old-multi-interest-setwise-model",
        required=True,
        type=Path,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--feature-batch-rows", type=int, default=512)
    parser.add_argument("--fusion-batch-size", type=int, default=256)
    parser.add_argument("--fusion-epochs", type=int, default=10)
    parser.add_argument("--fusion-patience", type=int, default=2)
    parser.add_argument("--setwise-weight", type=float, default=0.80)
    parser.add_argument("--minimum-slice1-delta", type=float, default=0.001)
    parser.add_argument("--minimum-q12-delta", type=float, default=-0.001)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.time()
    jt.flags.use_cuda = 1

    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    config = state["config"]
    feature_names = tuple(str(name) for name in state["feature_names"])
    train_report = _read_json(args.train_cache_report)
    validation_report = _read_json(args.validation_cache_report)
    old_proxy_report = _read_json(args.old_proxy_report)
    context_end = int(train_report["split"]["context_end"])
    train_end = int(validation_report["split"]["train_end"])

    train_path = Path(f"{args.train_cache_prefix}.train.npy")
    validation_path = Path(f"{args.validation_cache_prefix}.val.npy")
    train_features = np.load(train_path, mmap_mode="r", allow_pickle=False)
    validation_features = np.load(
        validation_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    old_train_proxy = np.load(
        args.old_proxy_train,
        mmap_mode="r",
        allow_pickle=False,
    )
    old_validation_proxy = np.load(
        args.old_proxy_validation,
        mmap_mode="r",
        allow_pickle=False,
    )
    train_candidates = _sidecar(
        args.train_cache_prefix,
        "train",
        "candidates",
    )
    train_src = _sidecar(args.train_cache_prefix, "train", "src")
    validation_candidates = _sidecar(
        args.validation_cache_prefix,
        "val",
        "candidates",
    )
    validation_src = _sidecar(args.validation_cache_prefix, "val", "src")
    if train_features.shape != (200_000, 100, len(feature_names)):
        raise ValueError(f"unexpected training cache: {train_features.shape}")
    if validation_features.shape != (20_000, 100, len(feature_names)):
        raise ValueError(
            f"unexpected validation cache: {validation_features.shape}"
        )
    if old_train_proxy.shape != (200_000, 100, 9):
        raise ValueError(f"unexpected old training proxy: {old_train_proxy.shape}")
    if old_validation_proxy.shape != (20_000, 100, 9):
        raise ValueError(
            f"unexpected old validation proxy: {old_validation_proxy.shape}"
        )

    interactions = read_interactions(args.train_csv).sort_by_time()
    if config.max_fit_events > 0 and len(interactions) > config.max_fit_events:
        interactions = interactions.tail(config.max_fit_events)
    id_map = NodeIdMap.from_interactions(interactions)
    frozen = {
        "status": "frozen_before_training",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "train_cache_sha256": _sha256(train_path),
        "validation_cache_sha256": _sha256(validation_path),
        "old_train_proxy_sha256": _sha256(args.old_proxy_train),
        "old_validation_proxy_sha256": _sha256(args.old_proxy_validation),
        "old_multi_interest_model_sha256": _sha256(
            args.old_multi_interest_setwise_model
        ),
        "champion_setwise_model_sha256": _sha256(
            args.champion_setwise_model
        ),
        "context_end": context_end,
        "train_end": train_end,
        "old_feature_names": list(MULTI_INTEREST_FEATURE_NAMES),
        "adaptive_feature_names": list(ACTIVITY_ADAPTIVE_FEATURE_NAMES),
        "source_feature_count": (
            len(feature_names)
            + len(MULTI_INTEREST_FEATURE_NAMES)
            + len(ACTIVITY_ADAPTIVE_FEATURE_NAMES)
        ),
        "adaptive_cluster_count": 4,
        "adaptive_half_life": {
            "base_events": 64.0,
            "minimum_events": 8.0,
            "activity_formula": "64/sqrt(max(activity/64,1))",
        },
        "selection": {
            "early_stop_rows": [0, SLICE_0_STOP],
            "selection_rows": [SLICE_0_STOP, SLICE_1_STOP],
            "slice2_labels_read": False,
            "minimum_slice1_delta_vs_old_multi_interest": (
                args.minimum_slice1_delta
            ),
            "minimum_q12_delta_vs_old_multi_interest": (
                args.minimum_q12_delta
            ),
            "minimum_q4_delta_vs_v1": 0.0,
        },
    }
    _write_json(args.output_dir / "frozen-config.json", frozen)
    print(json.dumps(frozen, indent=2), flush=True)

    train_adaptive_path = args.output_dir / "activity-adaptive.train.npy"
    validation_adaptive_path = (
        args.output_dir / "activity-adaptive.val.npy"
    )
    _build_adaptive_proxy(
        interactions=interactions[:context_end],
        candidates=train_candidates,
        query_src=train_src,
        id_map=id_map,
        graph_config=config.graph_config(),
        seed=args.seed + 1100,
        batch_rows=args.feature_batch_rows,
        output_path=train_adaptive_path,
    )
    _build_adaptive_proxy(
        interactions=interactions[:train_end],
        candidates=validation_candidates,
        query_src=validation_src,
        id_map=id_map,
        graph_config=config.graph_config(),
        seed=args.seed + 2100,
        batch_rows=args.feature_batch_rows,
        output_path=validation_adaptive_path,
    )
    train_adaptive = np.load(
        train_adaptive_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    validation_adaptive = np.load(
        validation_adaptive_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    train_source = _ConcatenatedFeatures(
        train_features,
        old_train_proxy,
        train_adaptive,
    )
    validation_source = _ConcatenatedFeatures(
        validation_features,
        old_validation_proxy,
        validation_adaptive,
    )
    train_view = SetwiseFeatureView(train_source)
    early_stop_source = _ConcatenatedFeatures(
        validation_features[:SLICE_0_STOP],
        old_validation_proxy[:SLICE_0_STOP],
        validation_adaptive[:SLICE_0_STOP],
    )
    early_stop_view = SetwiseFeatureView(early_stop_source)
    fusion_config = FusionConfig(
        epochs=args.fusion_epochs,
        batch_size=args.fusion_batch_size,
        lr=0.001,
        weight_decay=0.0,
        hidden_dim=32,
        selection_metric="mrr",
        early_stop_patience=args.fusion_patience,
    )
    model, result, history = fit_fusion_mlp_listwise_streaming(
        train_view,
        early_stop_view,
        fusion_config,
        np.random.default_rng(args.seed),
        verbose=True,
        candidate_name="dataset2_activity_adaptive_multi_interest",
    )
    validation_view = SetwiseFeatureView(validation_source)
    adaptive_setwise = _softmax(
        _predict(
            model,
            validation_view,
            result.mean,
            result.std,
            result.feature_indices,
            args.fusion_batch_size,
        )
    )
    lgbm_result = state.get("lgbm_result")
    if lgbm_result is None:
        raise ValueError("checkpoint has no LightGBM result")
    lgbm = _softmax(
        predict_logits_lgbm(lgbm_result.model_text, validation_features)
    )
    champion_setwise = _load_setwise_probabilities(
        args.champion_setwise_model,
        validation_features,
        args.fusion_batch_size,
    )
    old_multi_interest_setwise = _load_setwise_probabilities(
        args.old_multi_interest_setwise_model,
        _AugmentedFeatures(validation_features, old_validation_proxy),
        args.fusion_batch_size,
    )
    champion = _blend(champion_setwise, lgbm, args.setwise_weight)
    old_multi_interest = _blend(
        old_multi_interest_setwise,
        lgbm,
        args.setwise_weight,
    )
    adaptive = _blend(adaptive_setwise, lgbm, args.setwise_weight)
    _require_old_baseline(
        old_multi_interest,
        old_proxy_report["candidate"],
    )

    selection = _selection_metrics(
        validation_features,
        feature_names,
        champion,
        old_multi_interest,
        adaptive,
        minimum_slice1_delta=args.minimum_slice1_delta,
        minimum_q12_delta=args.minimum_q12_delta,
    )
    model_path = args.output_dir / "dataset2-activity-adaptive-setwise.npz"
    _save_model(
        model_path,
        result,
        hidden_dim=32,
        source_feature_count=train_source.shape[2],
    )
    scores_path = args.output_dir / "validation-scores.npz"
    np.savez_compressed(
        scores_path,
        champion=np.asarray(champion, dtype=np.float32),
        old_multi_interest=np.asarray(old_multi_interest, dtype=np.float32),
        adaptive=np.asarray(adaptive, dtype=np.float32),
    )
    report = {
        "status": "selected" if selection["passed"] else "no_eligible_candidate",
        "gate_passed": bool(selection["passed"]),
        "slice2_unlocked": bool(selection["passed"]),
        "slice2_metrics_read": False,
        "frozen_config": frozen,
        "selection": selection,
        "history": list(history),
        "artifacts": {
            "train_adaptive_proxy_sha256": _sha256(train_adaptive_path),
            "validation_adaptive_proxy_sha256": _sha256(
                validation_adaptive_path
            ),
            "model_sha256": _sha256(model_path),
            "validation_scores_sha256": _sha256(scores_path),
        },
        "elapsed_seconds": time.time() - started,
    }
    report_path = args.output_dir / "selection-report.json"
    _write_json(report_path, report)
    report_sha = _sha256(report_path)
    (args.output_dir / "selection-report.sha256").write_text(
        f"{report_sha}  {report_path.name}\n",
        encoding="ascii",
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


def _build_adaptive_proxy(
    *,
    interactions: Any,
    candidates: np.ndarray,
    query_src: np.ndarray,
    id_map: NodeIdMap,
    graph_config: Any,
    seed: int,
    batch_rows: int,
    output_path: Path,
) -> None:
    mapped_edges = _mapped_edges(interactions, id_map, graph_config)
    edge_count = max(
        1,
        int(len(mapped_edges) * GRAPH_WINDOW_FRACTIONS[1]),
    )
    recent_edges = mapped_edges[-edge_count:]
    rng = np.random.default_rng(seed)
    edge_index, edge_weight = _graph_window_data(
        recent_edges,
        graph_config,
        rng,
        window_name="gnn_recent",
    )
    jt.set_seed(seed)
    tower = GraphTower(id_map=id_map, config=graph_config)
    tower._fit_one_window(
        "gnn_recent",
        edge_index,
        edge_weight,
        rng,
        verbose=True,
    )
    item_embeddings = _normalize_rows(tower.item_embeddings["gnn_recent"])
    source_ids = np.fromiter(
        (src_id for src_id, _, _ in recent_edges),
        dtype=np.int32,
        count=len(recent_edges),
    )
    destination_ids = np.fromiter(
        (dst_id for _, dst_id, _ in recent_edges),
        dtype=np.int32,
        count=len(recent_edges),
    )
    event_times = np.fromiter(
        (event_time for _, _, event_time in recent_edges),
        dtype=np.int64,
        count=len(recent_edges),
    )
    order = np.argsort(source_ids, kind="stable")
    source_counts = np.bincount(
        source_ids,
        minlength=id_map.num_src,
    )
    source_offsets = np.concatenate(
        (
            np.zeros(1, dtype=np.int64),
            np.cumsum(source_counts, dtype=np.int64),
        )
    )
    embedding_dim = int(item_embeddings.shape[1])
    centers = {
        "decay1": np.zeros(
            (id_map.num_src, 1, embedding_dim),
            dtype=np.float32,
        ),
        "hierarchical3": np.zeros(
            (id_map.num_src, 3, embedding_dim),
            dtype=np.float32,
        ),
        "adaptive_cluster4": np.zeros(
            (id_map.num_src, 4, embedding_dim),
            dtype=np.float32,
        ),
    }
    metadata = {
        name: np.zeros((id_map.num_src, 4), dtype=np.float32)
        for name in ("support", "age", "last_hit", "weight")
    }
    for src_id in range(id_map.num_src):
        start = int(source_offsets[src_id])
        end = int(source_offsets[src_id + 1])
        if start == end:
            continue
        history_indices = order[start:end]
        history = item_embeddings[destination_ids[history_indices]]
        times = event_times[history_indices]
        clusters = activity_adaptive_cluster_interests(
            history,
            times,
            k=4,
        )
        centers["decay1"][src_id, 0] = exponential_interest_center(
            history,
            times,
            half_life_events=clusters.half_life_events,
        )
        centers["hierarchical3"][src_id] = hierarchical_interest_centers(
            history
        )
        centers["adaptive_cluster4"][src_id] = clusters.centers
        metadata["support"][src_id] = clusters.support
        metadata["age"][src_id] = clusters.age
        metadata["last_hit"][src_id] = clusters.last_hit
        metadata["weight"][src_id] = clusters.weights
        if src_id % 20_000 == 19_999:
            print(
                f"[activity-adaptive] sources={src_id + 1}/{id_map.num_src}",
                flush=True,
            )
    del (
        mapped_edges,
        recent_edges,
        edge_index,
        edge_weight,
        tower,
        source_ids,
        destination_ids,
        event_times,
        order,
        source_counts,
        source_offsets,
    )
    gc.collect()

    proxy = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            candidates.shape[0],
            candidates.shape[1],
            len(ACTIVITY_ADAPTIVE_FEATURE_NAMES),
        ),
    )
    for start in range(0, candidates.shape[0], batch_rows):
        end = min(start + batch_rows, candidates.shape[0])
        src_ids = id_map.src_ids(np.asarray(query_src[start:end]))
        dst_ids = id_map.dst_ids(np.asarray(candidates[start:end]))
        valid_src = src_ids >= 0
        valid_dst = dst_ids >= 0
        candidate_vectors = np.zeros(
            (end - start, candidates.shape[1], embedding_dim),
            dtype=np.float32,
        )
        candidate_vectors[valid_dst] = item_embeddings[dst_ids[valid_dst]]
        batch_centers: dict[str, np.ndarray] = {}
        for name, count in (
            ("decay1", 1),
            ("hierarchical3", 3),
            ("adaptive_cluster4", 4),
        ):
            batch = np.zeros(
                (end - start, count, embedding_dim),
                dtype=np.float32,
            )
            batch[valid_src] = centers[name][src_ids[valid_src]]
            batch_centers[name] = batch
        batch_metadata: list[np.ndarray] = []
        for name in ("support", "age", "last_hit", "weight"):
            batch = np.zeros((end - start, 4), dtype=np.float32)
            batch[valid_src] = metadata[name][src_ids[valid_src]]
            batch_metadata.append(batch)
        features = activity_adaptive_features_for_candidate_batch(
            candidate_vectors,
            batch_centers["decay1"],
            batch_centers["hierarchical3"],
            batch_centers["adaptive_cluster4"],
            *batch_metadata,
        )
        features *= valid_dst[..., None]
        if not np.all(np.isfinite(features)):
            raise ValueError("non-finite activity-adaptive proxy")
        proxy[start:end] = features
        if end == candidates.shape[0] or (start // batch_rows) % 16 == 15:
            proxy.flush()
        if end == candidates.shape[0] or (start // batch_rows) % 32 == 31:
            print(
                f"[activity-adaptive] rows={end}/{candidates.shape[0]}",
                flush=True,
            )
    proxy.flush()


def _selection_metrics(
    validation_features: np.ndarray,
    feature_names: tuple[str, ...],
    champion: np.ndarray,
    old_multi_interest: np.ndarray,
    adaptive: np.ndarray,
    *,
    minimum_slice1_delta: float,
    minimum_q12_delta: float,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name, start, end in (
        ("slice_0", 0, SLICE_0_STOP),
        ("slice_1", SLICE_0_STOP, SLICE_1_STOP),
    ):
        champion_mrr = _mrr(champion[start:end])
        old_mrr = _mrr(old_multi_interest[start:end])
        adaptive_mrr = _mrr(adaptive[start:end])
        metrics[name] = {
            "rows": end - start,
            "champion_mrr": champion_mrr,
            "old_multi_interest_mrr": old_mrr,
            "adaptive_mrr": adaptive_mrr,
            "adaptive_delta_vs_champion": adaptive_mrr - champion_mrr,
            "adaptive_delta_vs_old_multi_interest": adaptive_mrr - old_mrr,
        }

    selection_slice = slice(SLICE_0_STOP, SLICE_1_STOP)
    descriptors = query_segment_features(
        validation_features[selection_slice],
        feature_names,
    )
    activity_index = QUERY_SEGMENT_FEATURE_NAMES.index("source_activity")
    source_activity = descriptors[:, activity_index]
    lower, middle, upper = np.quantile(
        source_activity,
        (0.25, 0.5, 0.75),
    )
    masks = {
        "q1": source_activity <= lower,
        "q2": (source_activity > lower) & (source_activity <= middle),
        "q3": (source_activity > middle) & (source_activity <= upper),
        "q4": source_activity > upper,
    }
    segments = {}
    for name, mask in masks.items():
        segments[name] = {
            "rows": int(mask.sum()),
            "activity_min": float(source_activity[mask].min()),
            "activity_max": float(source_activity[mask].max()),
            "champion_mrr": _mrr(champion[selection_slice][mask]),
            "old_multi_interest_mrr": _mrr(
                old_multi_interest[selection_slice][mask]
            ),
            "adaptive_mrr": _mrr(adaptive[selection_slice][mask]),
        }
        segments[name]["adaptive_delta_vs_champion"] = (
            segments[name]["adaptive_mrr"] - segments[name]["champion_mrr"]
        )
        segments[name]["adaptive_delta_vs_old_multi_interest"] = (
            segments[name]["adaptive_mrr"]
            - segments[name]["old_multi_interest_mrr"]
        )
    passed = bool(
        metrics["slice_1"]["adaptive_delta_vs_old_multi_interest"]
        >= minimum_slice1_delta
        and segments["q4"]["adaptive_delta_vs_champion"] >= 0.0
        and segments["q1"]["adaptive_delta_vs_old_multi_interest"]
        >= minimum_q12_delta
        and segments["q2"]["adaptive_delta_vs_old_multi_interest"]
        >= minimum_q12_delta
    )
    return {
        "passed": passed,
        "slices": metrics,
        "source_activity_quantiles": [float(lower), float(middle), float(upper)],
        "source_activity_segments": segments,
        "slice2_labels_read": False,
    }


def _blend(
    setwise: np.ndarray,
    lgbm: np.ndarray,
    setwise_weight: float,
) -> np.ndarray:
    return setwise_weight * setwise + (1.0 - setwise_weight) * lgbm


def _mrr(probabilities: np.ndarray) -> float:
    if probabilities.shape[0] == 0:
        return 0.0
    ranks = 1 + np.sum(
        probabilities[:, 1:] > probabilities[:, :1],
        axis=1,
    )
    return float(np.mean(1.0 / ranks))


def _require_old_baseline(
    probabilities: np.ndarray,
    expected: dict[str, float],
) -> None:
    actual = {
        "slice_0": _mrr(probabilities[:SLICE_0_STOP]),
        "slice_1": _mrr(probabilities[SLICE_0_STOP:SLICE_1_STOP]),
    }
    for key, value in actual.items():
        if abs(value - float(expected[key])) > 1e-10:
            raise RuntimeError(
                f"old multi-interest reproduction failed for {key}: "
                f"{value} != {expected[key]}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
