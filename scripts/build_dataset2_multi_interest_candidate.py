from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from jgrec.contest_checkpoint import (
    ContestCheckpointWriter,
    load_checkpoint_dataset,
    load_checkpoint_metadata,
    set_model_state,
)
from jgrec.core.io import discover_datasets, read_interactions
from jgrec.core.runner import build_dataset_submission
from jgrec.core.types import DatasetResult, TestQueryArray, TrainingReport
from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid.fusion import FusionMLP, FusionResult
from jgrec.rankers.hybrid.fusion_analysis import ranking_mrr_three_slices
from jgrec.rankers.hybrid.fusion_lgbm import predict_logits_lgbm
from jgrec.rankers.hybrid.gnn import (
    GRAPH_WINDOW_FRACTIONS,
    GraphTower,
    _graph_window_data,
    _mapped_edges,
)
from jgrec.rankers.hybrid.multi_interest_proxy import (
    MULTI_INTEREST_FAMILIES,
    MULTI_INTEREST_FEATURE_NAMES,
    cluster_interest_centers,
    multi_interest_features_for_query_array,
    temporal_interest_centers,
)
from jgrec.rankers.hybrid.setwise import setwise_context_features
from jgrec.rankers.registry import create_ranker
from jgrec.submission import (
    expected_test_rows,
    validate_submission_file,
    write_zip,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--proxy-report", required=True, type=Path)
    parser.add_argument("--proxy-model", required=True, type=Path)
    parser.add_argument("--validation-features", required=True, type=Path)
    parser.add_argument("--validation-proxy", required=True, type=Path)
    parser.add_argument("--validation-cache-prefix", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--champion-dataset1", required=True, type=Path)
    parser.add_argument("--output-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--history-limit", type=int, default=64)
    parser.add_argument("--validation-prefix-rows", type=int, default=1_922_091)
    parser.add_argument("--validation-proxy-seed", type=int, default=2160)
    parser.add_argument("--final-proxy-seed", type=int, default=3160)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    protected = (
        args.output_checkpoint,
        args.output_checkpoint.with_suffix(f"{args.output_checkpoint.suffix}.tmp"),
        args.output_dir,
    )
    existing = [path for path in protected if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite: {existing}")
    args.output_dir.mkdir(parents=True)
    started = time.time()
    jt.flags.use_cuda = 1

    report = _read_json(args.proxy_report)
    if not report.get("gate_passed"):
        raise RuntimeError("proxy experiment did not pass its frozen gate")
    if _sha256(args.proxy_model) != report["artifacts"]["model_sha256"]:
        raise ValueError("proxy model hash differs from report")
    source_metadata = load_checkpoint_metadata(args.source_checkpoint)
    dataset2_state = load_checkpoint_dataset(args.source_checkpoint, "dataset2")
    config = dataset2_state["config"]
    interactions = read_interactions(args.train_csv).sort_by_time()
    if config.max_fit_events > 0 and len(interactions) > config.max_fit_events:
        interactions = interactions.tail(config.max_fit_events)
    id_map = NodeIdMap.from_interactions(interactions)
    checkpoint_map = dataset2_state["id_map"]
    if tuple(checkpoint_map["src_values"]) != id_map.src_values:
        raise ValueError("source checkpoint src map differs from final data")
    if tuple(checkpoint_map["dst_values"]) != id_map.dst_values:
        raise ValueError("source checkpoint dst map differs from final data")

    validation_state = _build_proxy_state(
        interactions[: args.validation_prefix_rows],
        id_map,
        config.graph_config(),
        seed=args.validation_proxy_seed,
        history_limit=args.history_limit,
    )
    validation_proxy = np.load(
        args.validation_proxy,
        mmap_mode="r",
        allow_pickle=False,
    )
    val_src = _sidecar(args.validation_cache_prefix, "src")
    val_candidates = _sidecar(args.validation_cache_prefix, "candidates")
    val_time = _sidecar(args.validation_cache_prefix, "time")
    replay_max_error = 0.0
    rebuilt_validation_proxy = np.empty(
        validation_proxy.shape,
        dtype=np.float32,
    )
    for start in range(0, len(val_src), args.batch_size):
        end = min(start + args.batch_size, len(val_src))
        queries = TestQueryArray(
            src=val_src[start:end],
            time=val_time[start:end],
            candidates=val_candidates[start:end],
        )
        actual = multi_interest_features_for_query_array(
            queries,
            id_map,
            validation_state,
        )
        expected = np.asarray(validation_proxy[start:end], dtype=np.float32)
        replay_max_error = max(
            replay_max_error,
            float(np.max(np.abs(actual - expected))),
        )
        rebuilt_validation_proxy[start:end] = actual

    val_features = np.load(
        args.validation_features,
        mmap_mode="r",
        allow_pickle=False,
    )
    replay_metrics = _validation_metrics(
        dataset2_state,
        args.proxy_model,
        val_features,
        rebuilt_validation_proxy,
        batch_size=256,
    )
    baseline_metrics = report["baseline"]
    replay_deltas = {
        key: float(replay_metrics[key] - baseline_metrics[key])
        for key in baseline_metrics
    }
    replay_gate_passed = bool(
        replay_deltas["full"]
        >= report["frozen_config"]["gate"]["minimum_full_delta"]
        and all(
            replay_deltas[f"slice_{index}"] >= 0.0
            for index in range(3)
        )
    )
    if not replay_gate_passed:
        raise RuntimeError(
            "independent production proxy replay failed the frozen metric gate: "
            f"metrics={replay_metrics} deltas={replay_deltas}"
        )
    del (
        validation_state,
        validation_proxy,
        rebuilt_validation_proxy,
        val_features,
    )
    gc.collect()

    final_proxy_state = _build_proxy_state(
        interactions,
        id_map,
        config.graph_config(),
        seed=args.final_proxy_seed,
        history_limit=args.history_limit,
    )
    payload = np.load(args.proxy_model, allow_pickle=False)
    expected_source_features = (
        len(dataset2_state["feature_names"])
        + len(MULTI_INTEREST_FEATURE_NAMES)
    )
    if int(payload["source_feature_count"][0]) != expected_source_features:
        raise ValueError("proxy model source feature count differs")
    setwise_state = {
        key.removeprefix("state__"): np.asarray(payload[key], dtype=np.float32)
        for key in payload.files
        if key.startswith("state__")
    }
    dataset2_state["setwise_fusion_state"] = setwise_state
    dataset2_state["setwise_fusion_result"] = FusionResult(
        best_val_ap=float(report["history"][3]["val_ap"]),
        best_val_mrr=float(report["setwise_expert"]["full"]),
        state=setwise_state,
        mean=np.asarray(payload["mean"], dtype=np.float32),
        std=np.asarray(payload["std"], dtype=np.float32),
        feature_indices=tuple(int(value) for value in payload["feature_indices"]),
        candidate_name="dataset2_multi_interest_proxy_frozen",
    )
    dataset2_state["setwise_hidden_dim"] = int(payload["hidden_dim"][0])
    dataset2_state["multi_interest_proxy_state"] = final_proxy_state
    current_lgbm = dataset2_state.get("lgbm_result")
    if current_lgbm is None:
        raise ValueError("source checkpoint has no LightGBM expert")
    dataset2_state["lgbm_result"] = replace(current_lgbm, mlp_weight=0.80)
    dataset2_state["segment_gate_result"] = None

    extra_metadata = {
        key: value
        for key, value in source_metadata.items()
        if key not in {"format", "version", "model_name", "datasets"}
    }
    extra_metadata.update(
        {
            "derived_from": str(args.source_checkpoint.resolve()),
            "dataset2_multi_interest_report": str(args.proxy_report.resolve()),
            "dataset2_multi_interest_history_limit": args.history_limit,
            "dataset2_multi_interest_final_seed": args.final_proxy_seed,
        }
    )
    writer = ContestCheckpointWriter(
        args.output_checkpoint,
        model_name=str(source_metadata["model_name"]),
        expected_datasets=tuple(source_metadata["datasets"]),
        metadata=extra_metadata,
    )
    try:
        dataset1_state = load_checkpoint_dataset(
            args.source_checkpoint,
            "dataset1",
        )
        writer.add_dataset("dataset1", dataset1_state)
        del dataset1_state
        gc.collect()
        writer.add_dataset("dataset2", dataset2_state)
        writer.finalize()
    except BaseException:
        writer.abort()
        raise

    reloaded_state = load_checkpoint_dataset(args.output_checkpoint, "dataset2")
    reloaded_ranker = create_ranker("hybrid", None)
    reloaded_ranker.hydrate(reloaded_state)
    if reloaded_ranker.impl.multi_interest_proxy_state is None:
        raise RuntimeError("reloaded checkpoint lost multi-interest state")
    del reloaded_state, dataset2_state, final_proxy_state
    gc.collect()

    datasets = {
        dataset.name: dataset
        for dataset in discover_datasets(args.data_dir)
    }
    csv_dir = args.output_dir / "csv"
    csv_dir.mkdir()
    dataset1_output = csv_dir / "dataset1.csv"
    dataset2_output = csv_dir / "dataset2.csv"
    shutil.copyfile(args.champion_dataset1, dataset1_output)
    validate_submission_file(
        dataset1_output,
        expected_rows=expected_test_rows(datasets["dataset1"]),
    )
    if _sha256(dataset1_output) != _sha256(args.champion_dataset1):
        raise RuntimeError("Dataset1 byte copy differs")
    dataset2_result = build_dataset_submission(
        dataset=datasets["dataset2"],
        ranker=reloaded_ranker,
        output_dir=csv_dir,
        batch_size=args.batch_size,
        verbose=True,
        fit_ranker=False,
    )
    validate_submission_file(
        dataset2_output,
        expected_rows=expected_test_rows(datasets["dataset2"]),
    )
    dataset1_result = DatasetResult(
        name="dataset1",
        rows=expected_test_rows(datasets["dataset1"]),
        output_path=dataset1_output,
        training_report=TrainingReport(model_name="hybrid"),
    )
    zip_path = args.output_dir / "result.zip"
    write_zip([dataset1_result, dataset2_result], zip_path)

    candidate_report = {
        "status": "complete",
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": _sha256(args.source_checkpoint),
        "proxy_report_sha256": _sha256(args.proxy_report),
        "proxy_model_sha256": _sha256(args.proxy_model),
        "validation_proxy_replay_max_abs_error": replay_max_error,
        "validation_metrics": replay_metrics,
        "validation_metric_deltas": replay_deltas,
        "validation_replay_gate_passed": replay_gate_passed,
        "output_checkpoint": str(args.output_checkpoint.resolve()),
        "output_checkpoint_bytes": args.output_checkpoint.stat().st_size,
        "output_checkpoint_sha256": _sha256(args.output_checkpoint),
        "dataset1_sha256": _sha256(dataset1_output),
        "dataset2_sha256": _sha256(dataset2_output),
        "result_zip": str(zip_path.resolve()),
        "result_zip_bytes": zip_path.stat().st_size,
        "result_zip_sha256": _sha256(zip_path),
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.output_dir / "candidate-report.json", candidate_report)
    print(json.dumps(candidate_report, indent=2), flush=True)
    return 0


def _build_proxy_state(
    interactions: Any,
    id_map: NodeIdMap,
    graph_config: Any,
    *,
    seed: int,
    history_limit: int,
) -> dict[str, np.ndarray]:
    mapped_edges = _mapped_edges(interactions, id_map, graph_config)
    edge_count = max(1, int(len(mapped_edges) * GRAPH_WINDOW_FRACTIONS[1]))
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
    state = {
        family: np.zeros(
            (id_map.num_src, count, item_embeddings.shape[1]),
            dtype=np.float32,
        )
        for family, count in MULTI_INTEREST_FAMILIES
    }
    for src_id, history in enumerate(histories):
        if not history:
            continue
        values = item_embeddings[np.asarray(history, dtype=np.int32)]
        state["temporal2"][src_id] = temporal_interest_centers(values)
        state["cluster2"][src_id] = cluster_interest_centers(values, k=2)
        state["cluster4"][src_id] = cluster_interest_centers(values, k=4)
    state["item_embeddings"] = item_embeddings
    return state


def _validation_metrics(
    dataset2_state: dict[str, Any],
    model_path: Path,
    base_features: Any,
    proxy_features: Any,
    *,
    batch_size: int,
) -> dict[str, float]:
    payload = np.load(model_path, allow_pickle=False)
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
    probabilities = np.empty(base_features.shape[:2], dtype=np.float64)
    with jt.no_grad():
        for start in range(0, base_features.shape[0], batch_size):
            end = min(start + batch_size, base_features.shape[0])
            augmented = np.concatenate(
                (
                    np.asarray(base_features[start:end], dtype=np.float32),
                    np.asarray(proxy_features[start:end], dtype=np.float32),
                ),
                axis=-1,
                dtype=np.float32,
            )
            context = setwise_context_features(augmented)[..., indices]
            normalized = (
                (
                    context
                    - np.asarray(payload["mean"], dtype=np.float32)
                )
                / np.asarray(payload["std"], dtype=np.float32)
            ).astype(np.float32, copy=False)
            logits = np.asarray(
                model(jt.array(normalized, dtype=jt.float32)).numpy(),
                dtype=np.float32,
            )
            probabilities[start:end] = _softmax(logits)
    lgbm_result = dataset2_state["lgbm_result"]
    lgbm = _softmax(
        predict_logits_lgbm(lgbm_result.model_text, base_features)
    )
    return ranking_mrr_three_slices(0.80 * probabilities + 0.20 * lgbm)


def _softmax(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits)
    return result / result.sum(axis=1, keepdims=True)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(
        matrix,
        np.maximum(norms, 1e-12),
        out=np.zeros_like(matrix),
        where=norms > 0.0,
    )


def _sidecar(prefix: Path, name: str) -> np.ndarray:
    return np.load(
        Path(f"{prefix}.val-{name}.npy"),
        mmap_mode="r",
        allow_pickle=False,
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
