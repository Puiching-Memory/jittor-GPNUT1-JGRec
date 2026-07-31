from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset, set_model_state
from jgrec.core.io import read_interactions
from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid.fusion import (
    FusionConfig,
    FusionMLP,
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
from jgrec.rankers.hybrid.multi_interest_proxy import (
    cluster_interest_centers,
    temporal_interest_centers,
)
from jgrec.rankers.hybrid.setwise import SetwiseFeatureView

FAMILIES = (("temporal2", 2), ("cluster2", 2), ("cluster4", 4))
PROXY_FEATURE_NAMES = tuple(
    f"multi_interest_{family}_{stat}"
    for family, _ in FAMILIES
    for stat in ("max", "top2", "coverage")
)


class _AugmentedFeatures:
    def __init__(self, source: Any, proxy: Any) -> None:
        if source.shape[:2] != proxy.shape[:2]:
            raise ValueError("source and proxy rows differ")
        self._source = source
        self._proxy = proxy
        self.shape = (
            int(source.shape[0]),
            int(source.shape[1]),
            int(source.shape[2] + proxy.shape[2]),
        )
        self.ndim = 3
        self.size = int(np.prod(self.shape, dtype=np.int64))

    def __getitem__(self, key: Any) -> np.ndarray:
        return np.concatenate(
            (
                np.asarray(self._source[key], dtype=np.float32),
                np.asarray(self._proxy[key], dtype=np.float32),
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
    parser.add_argument("--source-evaluation-report", required=True, type=Path)
    parser.add_argument("--setwise-model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--history-limit", type=int, default=64)
    parser.add_argument("--feature-batch-rows", type=int, default=512)
    parser.add_argument("--fusion-batch-size", type=int, default=256)
    parser.add_argument("--fusion-epochs", type=int, default=10)
    parser.add_argument("--fusion-patience", type=int, default=2)
    parser.add_argument("--setwise-weight", type=float, default=0.80)
    parser.add_argument("--min-full-delta", type=float, default=0.002)
    parser.add_argument("--min-new-edge-delta", type=float, default=0.003)
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
    val_report = _read_json(args.validation_cache_report)
    source_report = _read_json(args.source_evaluation_report)
    context_end = int(train_report["split"]["context_end"])
    train_end = int(val_report["split"]["train_end"])

    train_path = Path(f"{args.train_cache_prefix}.train.npy")
    val_path = Path(f"{args.validation_cache_prefix}.val.npy")
    train_features = np.load(train_path, mmap_mode="r", allow_pickle=False)
    val_features = np.load(val_path, mmap_mode="r", allow_pickle=False)
    train_candidates = _sidecar(args.train_cache_prefix, "train", "candidates")
    train_src = _sidecar(args.train_cache_prefix, "train", "src")
    val_candidates = _sidecar(args.validation_cache_prefix, "val", "candidates")
    val_src = _sidecar(args.validation_cache_prefix, "val", "src")
    val_dst = _sidecar(args.validation_cache_prefix, "val", "dst")
    if train_features.shape != (200_000, 100, len(feature_names)):
        raise ValueError(f"unexpected training cache: {train_features.shape}")
    if val_features.shape != (20_000, 100, len(feature_names)):
        raise ValueError(f"unexpected validation cache: {val_features.shape}")
    if not np.array_equal(val_candidates[:, 0], val_dst):
        raise ValueError("positive validation target is not candidate position zero")

    interactions = read_interactions(args.train_csv).sort_by_time()
    if config.max_fit_events > 0 and len(interactions) > config.max_fit_events:
        interactions = interactions.tail(config.max_fit_events)
    id_map = NodeIdMap.from_interactions(interactions)
    frozen = {
        "status": "frozen_before_training",
        "checkpoint_sha256": _sha256(args.checkpoint),
        "train_cache_sha256": _sha256(train_path),
        "validation_cache_sha256": _sha256(val_path),
        "setwise_model_sha256": _sha256(args.setwise_model),
        "context_end": context_end,
        "train_end": train_end,
        "history_limit": args.history_limit,
        "families": [list(value) for value in FAMILIES],
        "proxy_feature_names": list(PROXY_FEATURE_NAMES),
        "gate": {
            "minimum_full_delta": args.min_full_delta,
            "minimum_new_edge_delta": args.min_new_edge_delta,
            "all_three_slices_non_decreasing": True,
        },
    }
    _write_json(args.output_dir / "frozen-config.json", frozen)
    print(json.dumps(frozen, indent=2), flush=True)

    train_proxy_path = args.output_dir / "multi-interest.train.npy"
    val_proxy_path = args.output_dir / "multi-interest.val.npy"
    _build_proxy(
        interactions=interactions[:context_end],
        candidates=train_candidates,
        query_src=train_src,
        id_map=id_map,
        graph_config=config.graph_config(),
        seed=args.seed + 1100,
        history_limit=args.history_limit,
        batch_rows=args.feature_batch_rows,
        output_path=train_proxy_path,
    )
    _build_proxy(
        interactions=interactions[:train_end],
        candidates=val_candidates,
        query_src=val_src,
        id_map=id_map,
        graph_config=config.graph_config(),
        seed=args.seed + 2100,
        history_limit=args.history_limit,
        batch_rows=args.feature_batch_rows,
        output_path=val_proxy_path,
    )
    train_proxy = np.load(train_proxy_path, mmap_mode="r", allow_pickle=False)
    val_proxy = np.load(val_proxy_path, mmap_mode="r", allow_pickle=False)

    train_view = SetwiseFeatureView(
        _AugmentedFeatures(train_features, train_proxy)
    )
    val_view = SetwiseFeatureView(_AugmentedFeatures(val_features, val_proxy))
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
        val_view,
        fusion_config,
        np.random.default_rng(args.seed),
        verbose=True,
        candidate_name="dataset2_multi_interest_proxy",
    )
    candidate_setwise = _softmax(
        _predict(
            model,
            val_view,
            result.mean,
            result.std,
            result.feature_indices,
            args.fusion_batch_size,
        )
    )

    lgbm_result = state.get("lgbm_result")
    if lgbm_result is None:
        raise ValueError("checkpoint has no LightGBM result")
    lgbm = _softmax(predict_logits_lgbm(lgbm_result.model_text, val_features))
    candidate = (
        args.setwise_weight * candidate_setwise
        + (1.0 - args.setwise_weight) * lgbm
    )
    baseline_setwise = _load_setwise_probabilities(
        args.setwise_model,
        val_features,
        args.fusion_batch_size,
    )
    baseline = (
        args.setwise_weight * baseline_setwise
        + (1.0 - args.setwise_weight) * lgbm
    )
    baseline_metrics = ranking_mrr_three_slices(baseline)
    expected_baseline = source_report["setwise"]["fixed_blend"]
    _require_close(baseline_metrics, expected_baseline)
    candidate_metrics = ranking_mrr_three_slices(candidate)
    deltas = {
        key: float(candidate_metrics[key] - baseline_metrics[key])
        for key in baseline_metrics
    }

    new_edge_mask = _new_edge_mask(
        interactions[:train_end],
        val_src,
        val_dst,
    )
    baseline_new = _mrr(baseline[new_edge_mask])
    candidate_new = _mrr(candidate[new_edge_mask])
    new_delta = candidate_new - baseline_new
    passed = bool(
        deltas["full"] >= args.min_full_delta
        and all(deltas[f"slice_{index}"] >= 0.0 for index in range(3))
        and new_delta >= args.min_new_edge_delta
    )
    model_path = args.output_dir / "dataset2-multi-interest-setwise.npz"
    _save_model(
        model_path,
        result,
        hidden_dim=32,
        source_feature_count=len(feature_names) + len(PROXY_FEATURE_NAMES),
    )
    report = {
        "status": "complete",
        "frozen_config": frozen,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta": deltas,
        "new_edge": {
            "rows": int(new_edge_mask.sum()),
            "baseline_mrr": baseline_new,
            "candidate_mrr": candidate_new,
            "delta": new_delta,
        },
        "setwise_expert": ranking_mrr_three_slices(candidate_setwise),
        "history": list(history),
        "artifacts": {
            "train_proxy_sha256": _sha256(train_proxy_path),
            "validation_proxy_sha256": _sha256(val_proxy_path),
            "model_sha256": _sha256(model_path),
        },
        "gate_passed": passed,
        "formal_multi_interest_tower_authorized": passed,
        "submission_package_generated": False,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.output_dir / "multi-interest-report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


def _build_proxy(
    *,
    interactions: Any,
    candidates: np.ndarray,
    query_src: np.ndarray,
    id_map: NodeIdMap,
    graph_config: Any,
    seed: int,
    history_limit: int,
    batch_rows: int,
    output_path: Path,
) -> None:
    mapped_edges = _mapped_edges(interactions, id_map, graph_config)
    recent_index = 1
    edge_count = max(
        1,
        int(len(mapped_edges) * GRAPH_WINDOW_FRACTIONS[recent_index]),
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
    histories: list[list[int]] = [[] for _ in range(id_map.num_src)]
    for src_id, dst_id, _ in recent_edges:
        history = histories[src_id]
        history.append(dst_id)
        if len(history) > history_limit:
            del history[0]

    centers = {
        family: np.zeros(
            (id_map.num_src, k, item_embeddings.shape[1]),
            dtype=np.float32,
        )
        for family, k in FAMILIES
    }
    for src_id, history in enumerate(histories):
        if not history:
            continue
        values = item_embeddings[np.asarray(history, dtype=np.int32)]
        centers["temporal2"][src_id] = temporal_interest_centers(values)
        centers["cluster2"][src_id] = cluster_interest_centers(values, k=2)
        centers["cluster4"][src_id] = cluster_interest_centers(values, k=4)
    del histories, tower
    gc.collect()

    proxy = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(candidates.shape[0], candidates.shape[1], len(PROXY_FEATURE_NAMES)),
    )
    for start in range(0, candidates.shape[0], batch_rows):
        end = min(start + batch_rows, candidates.shape[0])
        src_ids = id_map.src_ids(np.asarray(query_src[start:end]))
        dst_ids = id_map.dst_ids(np.asarray(candidates[start:end]))
        valid_src = src_ids >= 0
        valid_dst = dst_ids >= 0
        candidate_vectors = np.zeros(
            (end - start, candidates.shape[1], item_embeddings.shape[1]),
            dtype=np.float32,
        )
        candidate_vectors[valid_dst] = item_embeddings[dst_ids[valid_dst]]
        output_column = 0
        for family, _ in FAMILIES:
            batch_centers = np.zeros(
                (end - start, centers[family].shape[1], item_embeddings.shape[1]),
                dtype=np.float32,
            )
            batch_centers[valid_src] = centers[family][src_ids[valid_src]]
            similarities = np.einsum(
                "bcd,bkd->bck",
                candidate_vectors,
                batch_centers,
                optimize=True,
            )
            ordered = np.sort(similarities, axis=2)
            proxy[start:end, :, output_column] = ordered[:, :, -1]
            proxy[start:end, :, output_column + 1] = ordered[:, :, -2]
            proxy[start:end, :, output_column + 2] = np.maximum(
                similarities,
                0.0,
            ).mean(axis=2)
            output_column += 3
        if not np.all(np.isfinite(proxy[start:end])):
            raise ValueError("non-finite multi-interest proxy")
        if end == candidates.shape[0] or (start // batch_rows) % 16 == 15:
            proxy.flush()
        if end == candidates.shape[0] or (start // batch_rows) % 32 == 31:
            print(
                f"[multi-interest] rows={end}/{candidates.shape[0]}",
                flush=True,
            )
    proxy.flush()


def _load_setwise_probabilities(
    path: Path,
    features: Any,
    batch_size: int,
) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    indices = tuple(int(value) for value in payload["feature_indices"])
    model = FusionMLP(
        input_dim=len(indices),
        hidden_dim=int(payload["hidden_dim"][0]),
    )
    set_model_state(
        model,
        {
            key.removeprefix("state__"): np.asarray(payload[key], dtype=np.float32)
            for key in payload.files
            if key.startswith("state__")
        },
    )
    view = SetwiseFeatureView(features)
    return _softmax(
        _predict(
            model,
            view,
            np.asarray(payload["mean"], dtype=np.float32),
            np.asarray(payload["std"], dtype=np.float32),
            indices,
            batch_size,
        )
    )


def _predict(
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
            values = np.asarray(features[start:end], dtype=np.float32)
            normalized = ((values[..., feature_indices] - mean) / std).astype(
                np.float32,
                copy=False,
            )
            scores[start:end] = np.asarray(
                model(jt.array(normalized, dtype=jt.float32)).numpy(),
                dtype=np.float32,
            )
    return scores


def _new_edge_mask(
    prefix: Any,
    src: np.ndarray,
    dst: np.ndarray,
) -> np.ndarray:
    pair_codes = (
        np.asarray(prefix.src, dtype=np.uint64) << np.uint64(32)
    ) | np.asarray(prefix.dst, dtype=np.uint32)
    query_codes = (
        np.asarray(src, dtype=np.uint64) << np.uint64(32)
    ) | np.asarray(dst, dtype=np.uint32)
    return ~np.isin(query_codes, np.unique(pair_codes), assume_unique=False)


def _mrr(probabilities: np.ndarray) -> float:
    if probabilities.shape[0] == 0:
        return 0.0
    positive = probabilities[:, :1]
    ranks = 1 + np.sum(probabilities[:, 1:] > positive, axis=1)
    return float(np.mean(1.0 / ranks))


def _sidecar(prefix: Path, split: str, name: str) -> np.ndarray:
    return np.load(
        Path(f"{prefix}.{split}-{name}.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(
        matrix,
        np.maximum(norms, 1e-12),
        out=np.zeros_like(matrix),
        where=norms > 0.0,
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(values)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _save_model(
    path: Path,
    result: Any,
    *,
    hidden_dim: int,
    source_feature_count: int,
) -> None:
    payload = {
        "mean": np.asarray(result.mean, dtype=np.float32),
        "std": np.asarray(result.std, dtype=np.float32),
        "feature_indices": np.asarray(result.feature_indices, dtype=np.int32),
        "hidden_dim": np.asarray([hidden_dim], dtype=np.int32),
        "source_feature_count": np.asarray([source_feature_count], dtype=np.int32),
        "context_transform_version": np.asarray([1], dtype=np.int32),
    }
    payload.update(
        {
            f"state__{key}": np.asarray(value, dtype=np.float32)
            for key, value in result.state.items()
        }
    )
    np.savez_compressed(path, **payload)


def _require_close(
    actual: dict[str, float],
    expected: dict[str, float],
) -> None:
    for key, expected_value in expected.items():
        if abs(float(actual[key]) - float(expected_value)) > 1e-10:
            raise RuntimeError(
                f"baseline reproduction failed for {key}: "
                f"{actual[key]} != {expected_value}"
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
