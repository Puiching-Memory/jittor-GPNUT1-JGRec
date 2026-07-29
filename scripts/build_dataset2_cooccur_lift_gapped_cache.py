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

import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.core.io import read_interactions
from jgrec.core.types import InteractionTable, TestQueryArray
from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid.cooccur_lift_native import (
    materialize_compact_cooccur_lift,
)
from jgrec.rankers.hybrid.full100_training import validate_candidate_matrix
from jgrec.rankers.hybrid.gnn import (
    GRAPH_WINDOW_FRACTIONS,
    GraphTower,
    _graph_window_data,
    _mapped_edges,
)
from jgrec.rankers.hybrid.parallel_structure import (
    ForkedStructureFeatureTower,
    validate_exact_parallel_features,
)
from jgrec.rankers.hybrid.ranker import (
    SupervisedFeatureBuilder,
    TemporalHybridRanker,
)

FEATURE_COUNT = 63
CANDIDATE_COUNT = 100
SHORT_WINDOW_SECONDS = 17_038_080
GNN_SHORT_SEED = 1_060


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the preregistered historical base/lift cache for the "
            "Dataset2 cooccur-lift successor gapped folds."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--validation-plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-rows", type=int, default=4096)
    parser.add_argument("--structure-workers", type=int, default=1)
    parser.add_argument("--minimum-parallel-speedup", type=float, default=1.5)
    args = parser.parse_args()

    if args.batch_rows <= 0:
        raise ValueError("batch rows must be positive")
    if args.structure_workers <= 0:
        raise ValueError("structure workers must be positive")
    if (
        args.structure_workers > 1
        and args.minimum_parallel_speedup <= 1.0
    ):
        raise ValueError("minimum parallel speedup must be greater than one")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.time()

    plan = _read_json(args.validation_plan)
    cache_contract = plan["far_horizon_validation"]["gapped_cache_contract"]
    fold_specs = plan["far_horizon_validation"]["gapped_fold_specs"]
    expected_checkpoint_sha256 = plan["baseline"]["checkpoint_sha256"]
    actual_checkpoint_sha256 = _sha256(args.checkpoint)
    if actual_checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError("checkpoint differs from the frozen v1 baseline")

    interactions = read_interactions(args.train_csv).sort_by_time()
    schedule = _validate_schedule(
        interactions=interactions,
        cache_contract=cache_contract,
        fold_specs=fold_specs,
    )
    row_indices = schedule["query_interaction_rows"]
    query_events = interactions.take(row_indices)
    query_count = len(query_events)
    paths = _artifact_paths(args.output_dir)
    _write_json(
        paths["progress"],
        {
            "status": "schedule_validated",
            "completed_rows": 0,
            "total_rows": query_count,
            "external_scores_read": False,
            "structure_workers": args.structure_workers,
            "schedule": _jsonable(schedule),
        },
    )

    import jittor as jt  # noqa: PLC0415

    if not jt.has_cuda:
        raise RuntimeError("CUDA is required for the frozen base encoder")
    jt.flags.use_cuda = 1
    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    config = state["config"]
    feature_names = tuple(str(name) for name in state["feature_names"])
    if len(feature_names) != FEATURE_COUNT:
        raise ValueError("frozen v1 checkpoint must expose 63 base features")
    if int(config.seed) != int(cache_contract["candidate_seed"]):
        raise ValueError("checkpoint seed differs from the cache contract")
    context_end = int(cache_contract["encoder_context_rows"][1])

    supervised_config = replace(
        config,
        structure_future_only_transition_cooccur=True,
        supervised_feature_cache_dir=None,
        verbose=True,
    )
    ranker = TemporalHybridRanker(recent_window=int(state["recent_window"]))
    ranker.id_map = NodeIdMap.from_interactions(interactions)
    ranker.dataset_profile = state["dataset_profile"]
    encoder_cache = ranker._encoder_state_cache(
        interactions,
        supervised_config,
        verbose=True,
    )
    context_snapshot = (
        encoder_cache.snapshot_for_prefix(context_end)
        if encoder_cache is not None
        else None
    )
    encoder = ranker._timed_fit_encoder(
        "cooccur_lift_gapped_context_encoder",
        interactions[:context_end],
        supervised_config,
        np.random.default_rng(config.seed),
        verbose=True,
        deterministic_snapshot=context_snapshot,
    )
    if tuple(encoder.feature_names) != feature_names:
        raise RuntimeError("historical encoder feature schema drifted")
    if encoder_cache is not None:
        encoder_cache.release_except()
    del context_snapshot, encoder_cache, state
    gc.collect()

    features = np.lib.format.open_memmap(
        paths["features"],
        mode="w+",
        dtype=np.float32,
        shape=(query_count, CANDIDATE_COUNT, FEATURE_COUNT),
    )
    candidates = np.lib.format.open_memmap(
        paths["candidates"],
        mode="w+",
        dtype=np.int32,
        shape=(query_count, CANDIDATE_COUNT),
    )
    build_config = replace(
        supervised_config,
        num_negatives=CANDIDATE_COUNT - 1,
        supervised_feature_batch_size=args.batch_rows,
    )
    builder = SupervisedFeatureBuilder(
        encoder=encoder,
        dst_pool=np.unique(interactions.dst).astype(np.int64, copy=False),
        config=build_config,
        label="cooccur_lift_gapped_features",
    )
    candidate_rng = np.random.default_rng(int(cache_contract["candidate_seed"]))
    total_batches = (query_count + args.batch_rows - 1) // args.batch_rows
    parallel_structure: ForkedStructureFeatureTower | None = None
    parallel_parity: dict[str, Any] | None = None
    for start in range(0, query_count, args.batch_rows):
        stop = min(start + args.batch_rows, query_count)
        queries = builder.batch_for_events(query_events[start:stop], candidate_rng)
        validate_candidate_matrix(
            query_events.dst[start:stop],
            queries.candidates,
            expected_candidate_count=CANDIDATE_COUNT,
        )
        if args.structure_workers > 1 and parallel_structure is None:
            sequential_started = time.perf_counter()
            sequential_features = encoder.features_for_query_array(queries)
            sequential_seconds = time.perf_counter() - sequential_started
            encoder.clear_batch_caches()
            source_structure = encoder.structure
            source_profile = encoder.source_profile
            parallel_structure = ForkedStructureFeatureTower(
                source_structure,
                worker_count=args.structure_workers,
                source_profile=source_profile,
            )
            encoder.structure = parallel_structure
            encoder.source_profile = parallel_structure.source_profile
            try:
                parallel_started = time.perf_counter()
                batch_features = encoder.features_for_query_array(queries)
                parallel_seconds = time.perf_counter() - parallel_started
                parallel_parity = validate_exact_parallel_features(
                    sequential_features,
                    batch_features,
                    sequential_seconds=sequential_seconds,
                    parallel_seconds=parallel_seconds,
                    minimum_speedup=args.minimum_parallel_speedup,
                    worker_pids=parallel_structure.active_worker_pids,
                )
            except Exception:
                parallel_structure.close(terminate=True)
                encoder.structure = source_structure
                encoder.source_profile = source_profile
                parallel_structure = None
                raise
            finally:
                del sequential_features
                gc.collect()
            print(
                "[gapped-cache] exact parallel parity passed "
                f"workers={parallel_parity['worker_count_observed']} "
                f"speedup={parallel_parity['speedup']:.3f}x",
                flush=True,
            )
        else:
            batch_features = encoder.features_for_query_array(queries)
        if batch_features.shape != (
            stop - start,
            CANDIDATE_COUNT,
            FEATURE_COUNT,
        ):
            raise RuntimeError("historical encoder returned an invalid shape")
        if not np.all(np.isfinite(batch_features)):
            raise ValueError("historical base features contain non-finite values")
        features[start:stop] = batch_features
        candidates[start:stop] = queries.candidates
        if stop == query_count or (start // args.batch_rows) % 2 == 1:
            features.flush()
            candidates.flush()
        encoder.clear_batch_caches()
        del queries, batch_features
        gc.collect()
        _write_json(
            paths["progress"],
            {
                "status": "building_base_features",
                "completed_rows": stop,
                "total_rows": query_count,
                "batch": start // args.batch_rows + 1,
                "total_batches": total_batches,
                "external_scores_read": False,
                "elapsed_seconds": time.time() - started,
                "structure_workers": args.structure_workers,
                "parallel_parity": parallel_parity,
            },
        )
        print(
            f"[gapped-cache] base rows={stop}/{query_count} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    features.flush()
    candidates.flush()
    if parallel_structure is not None:
        encoder.structure = parallel_structure.source
        encoder.source_profile = parallel_structure.source_profile_source
        parallel_structure.close()
    del builder, encoder, ranker
    gc.collect()

    _save_array(paths["src"], query_events.src)
    _save_array(paths["dst"], query_events.dst)
    _save_array(paths["time"], query_events.time)
    _save_array(paths["row_indices"], row_indices)
    query_candidates = np.load(
        paths["candidates"],
        mmap_mode="r",
        allow_pickle=False,
    )
    queries = TestQueryArray(
        src=query_events.src,
        time=query_events.time,
        candidates=query_candidates,
    )
    short_none_contract = _materialize_short_none(
        interactions=interactions[:context_end],
        queries=queries,
        id_map=NodeIdMap.from_interactions(interactions),
        graph_config=config.graph_config(),
        batch_rows=args.batch_rows,
        output_path=paths["short_none"],
    )
    _write_json(
        paths["progress"],
        {
            "status": "materializing_lift",
            "completed_rows": query_count,
            "total_rows": query_count,
            "external_scores_read": False,
            "elapsed_seconds": time.time() - started,
        },
    )

    train_count = int(schedule["union_training_cache_rows"][1])
    native_contracts: dict[str, Any] = {}
    native_contracts["train_near"] = _materialize_lift(
        interactions=interactions,
        query_events=query_events[:train_count],
        candidates=query_candidates[:train_count],
        availability_time=query_events.time[:train_count],
        lift_path=paths["lift_train_near"],
        popularity_path=paths["popularity_train_near"],
        work_dir=args.output_dir / "native-train-near",
    )
    score_availability = np.asarray(
        schedule["score_availability_time"],
        dtype=np.int32,
    )
    native_contracts["score_gapped"] = _materialize_lift(
        interactions=interactions,
        query_events=query_events[train_count:],
        candidates=query_candidates[train_count:],
        availability_time=score_availability,
        lift_path=paths["lift_score_gapped"],
        popularity_path=paths["popularity_score_gapped"],
        work_dir=args.output_dir / "native-score-gapped",
    )
    _save_array(paths["score_availability"], score_availability)
    for fold in fold_specs:
        fold_id = str(fold["fold_id"])
        train_start, train_stop = map(int, fold["cache_train_rows"])
        gap_seconds = int(fold["gap_seconds"])
        fold_events = query_events[train_start:train_stop]
        stale_availability = (
            fold_events.time.astype(np.int64) - gap_seconds
        ).astype(np.int32)
        lift_path = args.output_dir / f"lift-train-stale-{fold_id}.npy"
        popularity_path = (
            args.output_dir / f"positive-popularity-train-stale-{fold_id}.npy"
        )
        native_contracts[f"train_stale_{fold_id}"] = _materialize_lift(
            interactions=interactions,
            query_events=fold_events,
            candidates=query_candidates[train_start:train_stop],
            availability_time=stale_availability,
            lift_path=lift_path,
            popularity_path=popularity_path,
            work_dir=args.output_dir / f"native-train-stale-{fold_id}",
        )

    del query_candidates, queries
    gc.collect()
    artifact_report = {
        name: {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for name, path in paths.items()
        if path.exists() and name != "progress"
    }
    for fold in fold_specs:
        fold_id = str(fold["fold_id"])
        for prefix in (
            "lift-train-stale",
            "positive-popularity-train-stale",
        ):
            path = args.output_dir / f"{prefix}-{fold_id}.npy"
            artifact_report[f"{prefix}-{fold_id}"] = {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    report = {
        "status": "complete",
        "protocol": "cooccur_lift_successor_gapped_cache_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": actual_checkpoint_sha256,
        "validation_plan": str(args.validation_plan.resolve()),
        "validation_plan_sha256": _sha256(args.validation_plan),
        "train_csv": str(args.train_csv.resolve()),
        "train_csv_sha256": _sha256(args.train_csv),
        "feature_names": list(feature_names),
        "short_window_seconds": SHORT_WINDOW_SECONDS,
        "structure_workers": args.structure_workers,
        "parallel_parity": parallel_parity,
        "schedule": _jsonable(schedule),
        "short_none": short_none_contract,
        "native_materializers": native_contracts,
        "artifacts": artifact_report,
        "external_scores_read": False,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(paths["report"], report)
    _write_json(
        paths["progress"],
        {
            "status": "complete",
            "completed_rows": query_count,
            "total_rows": query_count,
            "external_scores_read": False,
            "elapsed_seconds": time.time() - started,
        },
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def _validate_schedule(
    *,
    interactions: InteractionTable,
    cache_contract: dict[str, Any],
    fold_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    context_start, context_stop = map(
        int,
        cache_contract["encoder_context_rows"],
    )
    union_start, union_stop = map(
        int,
        cache_contract["union_training_interaction_rows"],
    )
    if context_start != 0 or not 0 < context_stop <= union_start < union_stop:
        raise ValueError("gapped encoder/training row contract is invalid")
    if union_stop > len(interactions):
        raise ValueError("gapped training rows exceed the interaction table")
    train_rows = np.arange(union_start, union_stop, dtype=np.int64)
    query_parts = [train_rows]
    expected_score_cache_start = len(train_rows)
    score_availability_parts: list[np.ndarray] = []
    previous_score_time_max = -1
    previous_train_time_max = -1
    normalized_folds: list[dict[str, Any]] = []
    for fold in fold_specs:
        train_interaction_start, train_interaction_stop = map(
            int,
            fold["train_interaction_rows"],
        )
        train_cache_start, train_cache_stop = map(
            int,
            fold["cache_train_rows"],
        )
        if (
            train_interaction_start != union_start + train_cache_start
            or train_interaction_stop != union_start + train_cache_stop
        ):
            raise ValueError("gapped train cache rows do not map to interaction rows")
        score_start, score_stop = map(int, fold["score_interaction_rows"])
        cache_score_start, cache_score_stop = map(
            int,
            fold["cache_score_rows"],
        )
        if cache_score_start != expected_score_cache_start:
            raise ValueError("gapped score cache rows must be contiguous")
        if cache_score_stop - cache_score_start != score_stop - score_start:
            raise ValueError("gapped score cache and interaction counts differ")
        if not union_stop <= score_start < score_stop <= len(interactions):
            raise ValueError("gapped score rows are outside the internal table")
        train_time_max = int(interactions.time[train_interaction_stop - 1])
        score_time_min = int(interactions.time[score_start])
        score_time_max = int(interactions.time[score_stop - 1])
        gap_seconds = score_time_min - train_time_max
        expected = {
            "train_time_max": train_time_max,
            "score_time_min": score_time_min,
            "score_time_max": score_time_max,
            "gap_seconds": gap_seconds,
        }
        for key, actual in expected.items():
            if int(fold[key]) != actual:
                raise ValueError(f"{fold['fold_id']} {key} differs from data")
        if gap_seconds < int(fold["minimum_gap_seconds"]):
            raise ValueError("gapped fold is shorter than its frozen minimum")
        if gap_seconds < SHORT_WINDOW_SECONDS:
            raise ValueError("gapped fold does not collapse the short window")
        if (
            train_time_max <= previous_train_time_max
            or score_time_min <= previous_score_time_max
        ):
            raise ValueError("gapped fold origins must strictly increase")
        previous_train_time_max = train_time_max
        previous_score_time_max = score_time_max
        score_rows = np.arange(score_start, score_stop, dtype=np.int64)
        query_parts.append(score_rows)
        score_availability_parts.append(
            np.full(len(score_rows), train_time_max, dtype=np.int32)
        )
        expected_score_cache_start = cache_score_stop
        normalized_folds.append(
            {
                "fold_id": str(fold["fold_id"]),
                "train_interaction_rows": [
                    train_interaction_start,
                    train_interaction_stop,
                ],
                "cache_train_rows": [train_cache_start, train_cache_stop],
                "score_interaction_rows": [score_start, score_stop],
                "cache_score_rows": [cache_score_start, cache_score_stop],
                **expected,
            }
        )
    query_rows = np.concatenate(query_parts)
    if not np.all(query_rows[1:] > query_rows[:-1]):
        raise ValueError("gapped query interaction rows must be disjoint and ordered")
    if int(interactions.time[query_rows[-1]]) > int(
        cache_contract["internal_reference_time_max"]
    ):
        raise ValueError("gapped cache crossed the frozen internal reference end")
    return {
        "encoder_context_rows": [context_start, context_stop],
        "union_training_interaction_rows": [union_start, union_stop],
        "union_training_cache_rows": [0, len(train_rows)],
        "query_interaction_rows": query_rows,
        "score_availability_time": np.concatenate(score_availability_parts),
        "folds": normalized_folds,
    }


def _materialize_short_none(
    *,
    interactions: InteractionTable,
    queries: TestQueryArray,
    id_map: NodeIdMap,
    graph_config: Any,
    batch_rows: int,
    output_path: Path,
) -> dict[str, Any]:
    import jittor as jt  # noqa: PLC0415

    config = replace(
        graph_config,
        edge_weighting="none",
        full_edge_weighting=None,
        recent_edge_weighting=None,
        short_edge_weighting="none",
    )
    mapped_edges = _mapped_edges(interactions, id_map, config)
    edge_count = max(1, int(len(mapped_edges) * GRAPH_WINDOW_FRACTIONS[2]))
    window_edges = mapped_edges[-edge_count:]
    rng = np.random.default_rng(GNN_SHORT_SEED)
    edge_index, edge_weight = _graph_window_data(
        window_edges,
        config,
        rng,
        window_name="gnn_short",
    )
    jt.set_seed(GNN_SHORT_SEED)
    tower = GraphTower(id_map=id_map, config=config)
    tower._fit_one_window(
        "gnn_short",
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
        stop = min(start + batch_rows, len(queries))
        scores[start:stop] = tower.scores_for_query_array(
            queries[start:stop]
        )[..., 2]
        if stop == len(queries) or (start // batch_rows) % 8 == 7:
            scores.flush()
    scores.flush()
    del scores, tower
    gc.collect()
    return {
        "window": "gnn_short",
        "edge_weighting": "none",
        "epochs": int(config.epochs),
        "max_train_edges": int(config.max_train_edges),
        "seed": GNN_SHORT_SEED,
        "context_rows": len(interactions),
        "mapped_edges": len(mapped_edges),
        "window_edges": len(window_edges),
        "output_sha256": _sha256(output_path),
    }


def _materialize_lift(
    *,
    interactions: InteractionTable,
    query_events: InteractionTable,
    candidates: np.ndarray,
    availability_time: np.ndarray,
    lift_path: Path,
    popularity_path: Path,
    work_dir: Path,
) -> dict[str, Any]:
    return materialize_compact_cooccur_lift(
        interactions=interactions,
        sources=query_events.src,
        candidates=candidates,
        destinations=query_events.dst,
        event_time=query_events.time,
        availability_time=availability_time,
        short_window=SHORT_WINDOW_SECONDS,
        lift_path=lift_path,
        positive_popularity_path=popularity_path,
        progress_path=work_dir / "progress.json",
        work_dir=work_dir,
    )


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "features": output_dir / "base-features.npy",
        "candidates": output_dir / "candidates.npy",
        "src": output_dir / "src.npy",
        "dst": output_dir / "dst.npy",
        "time": output_dir / "time.npy",
        "row_indices": output_dir / "interaction-row-indices.npy",
        "short_none": output_dir / "short-none.npy",
        "lift_train_near": output_dir / "lift-train-near.npy",
        "popularity_train_near": (
            output_dir / "positive-popularity-train-near.npy"
        ),
        "lift_score_gapped": output_dir / "lift-score-gapped.npy",
        "popularity_score_gapped": (
            output_dir / "positive-popularity-score-gapped.npy"
        ),
        "score_availability": output_dir / "score-availability-time.npy",
        "report": output_dir / "cache-report.json",
        "progress": output_dir / "progress.json",
    }


def _save_array(path: Path, values: np.ndarray) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("wb") as handle:
        np.save(handle, np.asarray(values), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    temp.parent.mkdir(parents=True, exist_ok=True)
    temp.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temp, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
