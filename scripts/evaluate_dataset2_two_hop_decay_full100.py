from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.core.io import read_interactions, read_test_queries
from jgrec.core.types import InteractionTable, TestQueryArray
from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid.fusion import build_fusion_from_state, predict_logits
from jgrec.rankers.hybrid.fusion_lgbm import _flatten_for_ranking, predict_logits_lgbm
from jgrec.rankers.hybrid.lgbm_tuning import predeclared_dataset2_lgbm_grid
from jgrec.rankers.hybrid.oof_hard_negatives import (
    contiguous_oof_folds,
    passes_temporal_mrr_gate,
)
from jgrec.rankers.hybrid.ranker import HybridFeatureEncoder, _sample_events

DECAY_FEATURE_NAME = "cooccur_time_decay_score"
EXPECTED_BASELINE_MRR = 0.5428303297309955
SIGNATURE_FEATURE_NAMES = (
    "dst_popularity",
    "candidate_train_seen",
    "candidate_test_freq",
    "candidate_unseen_test_freq",
    "target_pop_w001",
    "target_pop_share_w001",
    "target_recency_w001",
    "target_pop_w005",
    "target_pop_share_w005",
    "target_recency_w005",
    "target_pop_w020",
    "target_pop_share_w020",
    "target_recency_w020",
    "target_pop_w100",
    "target_pop_share_w100",
    "target_recency_w100",
    "dst_unique_src",
    "dst_pop_rank",
    "dst_degree_log",
)
WEAK_SIGNATURE_FEATURE_NAMES = (
    "dst_popularity",
    "candidate_train_seen",
    "candidate_test_freq",
    "candidate_unseen_test_freq",
    "target_pop_w001",
    "target_pop_share_w001",
    "target_recency_w001",
    "target_pop_w005",
    "target_pop_share_w005",
    "target_recency_w005",
    "target_pop_w020",
    "target_pop_share_w020",
    "target_recency_w020",
    "target_pop_w100",
    "target_pop_share_w100",
    "target_recency_w100",
)
MATCH_FEATURE_NAMES = (
    "pair_strength",
    "repeat_rate",
    "pair_recency",
    "dst_popularity",
    "dst_recency",
    "recent_hit",
    "src_activity",
    "src_recency",
    "candidate_train_seen",
    "candidate_test_freq",
    "candidate_unseen_test_freq",
    "target_pop_w001",
    "target_pop_share_w001",
    "target_recency_w001",
    "target_pop_w005",
    "target_pop_share_w005",
    "target_recency_w005",
    "target_pop_w020",
    "target_pop_share_w020",
    "target_recency_w020",
    "target_pop_w100",
    "target_pop_share_w100",
    "target_recency_w100",
)
STRUCTURE_MATCH_FEATURE_NAMES = (
    "pair_decay_short",
    "pair_decay_medium",
    "pair_decay_long",
    "dst_unique_src",
    "dst_pop_rank",
    "reverse_log_count",
    "reverse_recency",
    "common_neighbors",
    "jaccard",
    "cooccur_score",
    "transition_score",
    "adamic_adar_log",
    "resource_allocation_log",
    "dst_degree_log",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Append one causal two-hop decay feature and evaluate Dataset2 on all 100 candidates."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cache-prefix", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--boost-rounds", type=int, default=308)
    parser.add_argument("--mlp-weight", type=float, default=0.07)
    parser.add_argument("--min-full-delta", type=float, default=0.002)
    parser.add_argument("--decay-ratio", type=float, default=0.05)
    parser.add_argument("--source-history-limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--resolve-batch-rows", type=int, default=32)
    parser.add_argument("--score-batch-rows", type=int, default=512)
    parser.add_argument("--max-resolution-group", type=int, default=4096)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = args.output_dir / "frozen-config.json"
    report_path = args.output_dir / "full100-report.json"
    model_path = args.output_dir / "dataset2-two-hop-decay-lgbm.txt"
    if report_path.exists() or model_path.exists():
        raise FileExistsError("refusing to overwrite a started full-100 decay experiment")

    started = time.time()
    manifest_path = args.cache_prefix.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_features = np.load(
        args.cache_prefix.with_suffix(".train.npy"), mmap_mode="r", allow_pickle=False
    )
    val_features = np.load(
        args.cache_prefix.with_suffix(".val.npy"), mmap_mode="r", allow_pickle=False
    )
    _validate_cache(manifest, train_features, val_features)

    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    config = state["config"]
    feature_names = tuple(str(name) for name in state["feature_names"])
    if len(feature_names) != 63 or train_features.shape[-1] != len(feature_names):
        raise ValueError("the production experiment requires the frozen 63-feature champion cache")
    fusion_result = state["fusion_result"]
    lgbm_result = state.get("lgbm_result")
    if lgbm_result is None:
        raise ValueError("Dataset2 checkpoint has no LightGBM expert")
    mlp_indices = tuple(int(index) for index in fusion_result.feature_indices)
    if mlp_indices != tuple(range(len(feature_names))):
        raise ValueError("the frozen champion MLP must use all 63 cached features")
    if tuple(int(index) for index in lgbm_result.feature_indices) != mlp_indices:
        raise ValueError("the source champion experts must share the original 63 features")
    if abs(float(lgbm_result.mlp_weight) - args.mlp_weight) > 1e-12:
        raise ValueError("fixed MLP weight does not match the source champion")
    if int(config.seed) != args.seed:
        raise ValueError("experiment seed does not match the source checkpoint")

    interactions = read_interactions(args.train_csv).sort_by_time()
    if config.max_fit_events > 0 and len(interactions) > config.max_fit_events:
        interactions = interactions.tail(config.max_fit_events)
    val_size = max(1, int(len(interactions) * config.val_ratio))
    train_end = max(2, len(interactions) - val_size)
    context_end = max(1, min(train_end - 1, int(train_end * config.context_ratio)))
    rng = np.random.default_rng(config.seed)
    sampled_train = _sample_events(
        interactions[context_end:train_end], config.max_train_events, rng
    )
    sampled_val = _sample_events(interactions[train_end:], config.max_val_events, rng)
    if len(sampled_train) != train_features.shape[0] or len(sampled_val) != val_features.shape[0]:
        raise ValueError("reconstructed temporal positives do not align with the supervised cache")

    test_queries = read_test_queries(args.test_csv)
    candidate_values = np.unique(
        np.concatenate((interactions.dst, test_queries.candidates.reshape(-1)))
    ).astype(np.int32, copy=False)
    params = dict(
        dict(
            predeclared_dataset2_lgbm_grid(
                seed=args.seed,
                num_threads=args.num_threads,
            )
        )["lr003"]
    )
    folds = contiguous_oof_folds(row_count=val_features.shape[0], fold_count=3)
    augmented_names = (*feature_names, DECAY_FEATURE_NAME)
    frozen = {
        "status": "frozen_before_decay_recovery_training_and_validation_scoring",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "cache_key": manifest["key"],
        "cache_manifest_sha256": _sha256(manifest_path),
        "train_shape": list(train_features.shape),
        "validation_shape": list(val_features.shape),
        "validation_candidate_count": int(val_features.shape[1]),
        "feature_names": list(augmented_names),
        "new_feature": {
            "name": DECAY_FEATURE_NAME,
            "decay_ratio": args.decay_ratio,
            "source_history_limit": args.source_history_limit,
            "cooccur_history_limit": int(config.structure_cooccur_history_limit),
            "causal_prefixes": {"train": context_end, "validation": train_end},
        },
        "training_scope": "Dataset2 LightGBM only; MLP and all learned towers frozen",
        "boost_rounds": args.boost_rounds,
        "mlp_weight": args.mlp_weight,
        "min_full_delta": args.min_full_delta,
        "expected_baseline_mrr": EXPECTED_BASELINE_MRR,
        "params": params,
        "selection": "none; one frozen fit, no validation tuning or early stopping",
        "validation_slices": [
            [fold.holdout.start, fold.holdout.stop] for fold in folds
        ],
        "gate": "full delta >= 0.002 and every chronological slice delta >= 0",
    }
    if frozen_path.exists():
        existing_frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if existing_frozen != frozen:
            raise RuntimeError("existing frozen config differs from this retry")
    else:
        _write_json(frozen_path, frozen)
    print(json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True), flush=True)

    train_decay_path = args.output_dir / "train.cooccur-time-decay.npy"
    val_decay_path = args.output_dir / "val.cooccur-time-decay.npy"
    recovery_started = time.time()
    train_decay, train_recovery = _build_decay_column(
        label="train",
        prefix=interactions[:context_end],
        positives=sampled_train,
        cached_features=train_features,
        candidate_values=candidate_values,
        state=state,
        feature_names=feature_names,
        decay_ratio=args.decay_ratio,
        source_history_limit=args.source_history_limit,
        resolve_batch_rows=args.resolve_batch_rows,
        score_batch_rows=args.score_batch_rows,
        max_resolution_group=args.max_resolution_group,
    )
    np.save(train_decay_path, train_decay, allow_pickle=False)
    del train_decay
    gc.collect()
    val_decay, val_recovery = _build_decay_column(
        label="validation",
        prefix=interactions[:train_end],
        positives=sampled_val,
        cached_features=val_features,
        candidate_values=candidate_values,
        state=state,
        feature_names=feature_names,
        decay_ratio=args.decay_ratio,
        source_history_limit=args.source_history_limit,
        resolve_batch_rows=args.resolve_batch_rows,
        score_batch_rows=args.score_batch_rows,
        max_resolution_group=args.max_resolution_group,
    )
    np.save(val_decay_path, val_decay, allow_pickle=False)
    del val_decay
    gc.collect()
    recovery_seconds = time.time() - recovery_started
    recovery_report = {
        "train": train_recovery,
        "validation": val_recovery,
        "train_decay_path": str(train_decay_path.resolve()),
        "train_decay_sha256": _sha256(train_decay_path),
        "validation_decay_path": str(val_decay_path.resolve()),
        "validation_decay_sha256": _sha256(val_decay_path),
        "seconds": recovery_seconds,
    }
    _write_json(args.output_dir / "decay-cache-report.json", recovery_report)
    print(json.dumps(recovery_report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)

    train_decay = np.load(train_decay_path, mmap_mode="r", allow_pickle=False)
    augmented_train = np.concatenate((train_features, train_decay[..., None]), axis=2)
    all_indices = tuple(range(len(augmented_names)))
    train_X, train_y, train_group = _flatten_for_ranking(augmented_train, all_indices)
    del augmented_train, train_decay
    gc.collect()

    import lightgbm as lgb  # noqa: PLC0415

    training_started = time.time()
    train_ds = lgb.Dataset(
        train_X,
        label=train_y,
        group=train_group,
        feature_name=list(augmented_names),
        params={"feature_pre_filter": False},
        free_raw_data=True,
    )
    candidate = lgb.train(
        params,
        train_ds,
        num_boost_round=args.boost_rounds,
        callbacks=[lgb.log_evaluation(0)],
    )
    candidate_text = candidate.model_to_string(num_iteration=args.boost_rounds)
    model_path.write_text(candidate_text, encoding="utf-8")
    training_seconds = time.time() - training_started
    del train_X, train_y, train_group, train_ds
    gc.collect()
    print(f"[two-hop-decay] LightGBM training complete seconds={training_seconds:.1f}", flush=True)

    val_decay = np.load(val_decay_path, mmap_mode="r", allow_pickle=False)
    augmented_val = np.concatenate((val_features, val_decay[..., None]), axis=2)
    mlp_model = build_fusion_from_state(
        input_dim=len(mlp_indices),
        hidden_dim=int(state["fusion_hidden_dim"]),
        state=state["fusion_state"],
    )
    mlp = _softmax(
        predict_logits(
            mlp_model,
            val_features,
            fusion_result.mean,
            fusion_result.std,
        )
    )
    baseline_lgbm = _softmax(predict_logits_lgbm(lgbm_result.model_text, val_features))
    candidate_lgbm = _softmax(predict_logits_lgbm(candidate_text, augmented_val))
    baseline_blend = args.mlp_weight * mlp + (1.0 - args.mlp_weight) * baseline_lgbm
    candidate_blend = args.mlp_weight * mlp + (1.0 - args.mlp_weight) * candidate_lgbm
    baseline_metrics = _temporal_mrr(baseline_blend, folds)
    if abs(float(baseline_metrics["full"]) - EXPECTED_BASELINE_MRR) > 1e-12:
        raise RuntimeError(
            "source champion baseline mismatch: "
            f"actual={baseline_metrics['full']:.16f} expected={EXPECTED_BASELINE_MRR:.16f}"
        )
    candidate_metrics = _temporal_mrr(candidate_blend, folds)
    baseline_slices = tuple(float(item["mrr"]) for item in baseline_metrics["slices"])
    candidate_slices = tuple(float(item["mrr"]) for item in candidate_metrics["slices"])
    passed = passes_temporal_mrr_gate(
        candidate_slices=candidate_slices,
        baseline_slices=baseline_slices,
        candidate_full_mrr=float(candidate_metrics["full"]),
        baseline_full_mrr=float(baseline_metrics["full"]),
        min_full_delta=args.min_full_delta,
    )
    slice_deltas = [
        candidate_value - baseline_value
        for candidate_value, baseline_value in zip(
            candidate_slices,
            baseline_slices,
            strict=True,
        )
    ]
    report = {
        "status": "passed" if passed else "rejected",
        "gate_passed": passed,
        "package_authorized": passed,
        "package_generated": False,
        "frozen_config": frozen,
        "recovery": recovery_report,
        "baseline": {
            "lgbm": _temporal_mrr(baseline_lgbm, folds),
            "fixed_blend": baseline_metrics,
        },
        "candidate": {
            "lgbm": _temporal_mrr(candidate_lgbm, folds),
            "fixed_blend": candidate_metrics,
            "blend_full_delta": float(candidate_metrics["full"] - baseline_metrics["full"]),
            "blend_slice_deltas": slice_deltas,
            "decay_feature_importance": {
                "gain": float(candidate.feature_importance(importance_type="gain")[-1]),
                "split": int(candidate.feature_importance(importance_type="split")[-1]),
            },
            "model_path": str(model_path.resolve()),
            "model_sha256": _sha256(model_path),
        },
        "gate": {
            "min_full_delta": args.min_full_delta,
            "full_delta_passed": bool(
                candidate_metrics["full"] - baseline_metrics["full"] + 1e-12
                >= args.min_full_delta
            ),
            "all_slices_non_decreasing": bool(all(delta >= 0.0 for delta in slice_deltas)),
        },
        "training_seconds": training_seconds,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 2


def _build_decay_column(
    *,
    label: str,
    prefix: InteractionTable,
    positives: InteractionTable,
    cached_features: np.ndarray,
    candidate_values: np.ndarray,
    state: dict[str, Any],
    feature_names: tuple[str, ...],
    decay_ratio: float,
    source_history_limit: int,
    resolve_batch_rows: int,
    score_batch_rows: int,
    max_resolution_group: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.time()
    config = state["config"]
    deterministic_config = replace(
        config,
        auto_strategy_enabled=False,
        candidate_prior_enabled=True,
        candidate_prior_include_test_frequency=True,
        target_window_enabled=True,
        structure_enabled=True,
        structure_cooccur_enabled=True,
        structure_transition_enabled=True,
        structure_future_only_transition_cooccur=True,
        structure_cooccur_time_decay_enabled=True,
        structure_cooccur_time_decay_ratio=decay_ratio,
        structure_cooccur_time_decay_source_history_limit=source_history_limit,
        source_profile_enabled=False,
        two_tower_enabled=False,
        gnn_enabled=False,
        seq_enabled=False,
        verbose=False,
    )
    encoder = HybridFeatureEncoder(
        id_map=NodeIdMap.from_interactions(prefix),
        recent_window=int(state["recent_window"]),
        candidate_prior_config=deterministic_config.candidate_prior_config(),
        target_window_config=deterministic_config.target_window_config(),
        structure_config=deterministic_config.structure_config(),
        source_profile_config=deterministic_config.source_profile_config(),
        two_tower_config=deterministic_config.two_tower_config(),
        graph_config=deterministic_config.graph_config(),
        sequence_config=deterministic_config.sequence_config(),
        dataset_profile=state["dataset_profile"],
    )
    encoder.fit(prefix, rng=np.random.default_rng(0), verbose=False)
    if tuple(encoder.feature_names[:-1]) != feature_names or encoder.feature_names[-1] != DECAY_FEATURE_NAME:
        raise RuntimeError("resolver encoder did not preserve the original 63-column order")
    signature_indices = tuple(feature_names.index(name) for name in SIGNATURE_FEATURE_NAMES)
    weak_signature_indices = tuple(
        feature_names.index(name) for name in WEAK_SIGNATURE_FEATURE_NAMES
    )
    match_indices = tuple(feature_names.index(name) for name in MATCH_FEATURE_NAMES)
    structure_match_indices = tuple(
        feature_names.index(name) for name in STRUCTURE_MATCH_FEATURE_NAMES
    )
    structure_start = feature_names.index("pair_decay_short")
    structure_match_local_indices = tuple(
        index - structure_start for index in structure_match_indices
    )
    seen_index = feature_names.index("candidate_train_seen")

    reference_query = TestQueryArray(
        src=np.asarray([-1], dtype=np.int32),
        time=np.asarray([int(prefix.time[-1]) + 1], dtype=np.int32),
        candidates=candidate_values.reshape(1, -1),
    )
    reference_features = encoder.features_for_query_array(reference_query)[0, :, :-1]
    exact_lookup = _signature_lookup(
        candidate_values,
        reference_features,
        signature_indices,
        decimals=None,
    )
    rounded_lookup = _signature_lookup(
        candidate_values,
        reference_features,
        signature_indices,
        decimals=6,
    )
    weak_lookup = _signature_lookup(
        candidate_values,
        reference_features,
        weak_signature_indices,
        decimals=2,
    )
    del reference_features
    gc.collect()

    rows, columns = cached_features.shape[:2]
    resolved_ids = np.full((rows, columns), -1, dtype=np.int32)
    resolved_ids[:, 0] = positives.dst.astype(np.int32, copy=False)
    pending_by_row: dict[int, list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
    override_values: dict[tuple[int, int], float] = {}
    counters = {
        "total_positions": rows * columns,
        "positive_forced": rows,
        "exact_unique": 0,
        "cold_zero": 0,
        "pending_ambiguous_seen": 0,
        "pending_rounded_seen": 0,
        "pending_weak_seen": 0,
        "pending_full_pool_seen": 0,
        "resolved_by_query_features": 0,
        "resolved_equal_decay_tie": 0,
        "ambiguous_conservative_max": 0,
    }
    for start in range(0, rows, 256):
        stop = min(start + 256, rows)
        exact_keys = _signature_keys(
            cached_features[start:stop],
            signature_indices,
            decimals=None,
        )
        rounded_keys = _signature_keys(
            cached_features[start:stop],
            signature_indices,
            decimals=6,
        )
        weak_keys = _signature_keys(
            cached_features[start:stop],
            weak_signature_indices,
            decimals=2,
        )
        for local_row in range(stop - start):
            row = start + local_row
            for column in range(1, columns):
                group = exact_lookup.get(bytes(exact_keys[local_row, column]))
                if group is not None and len(group) == 1:
                    resolved_ids[row, column] = group[0]
                    counters["exact_unique"] += 1
                    continue
                is_seen = bool(cached_features[row, column, seen_index] > 0.5)
                if not is_seen:
                    counters["cold_zero"] += 1
                    continue
                if group is None:
                    group = rounded_lookup.get(bytes(rounded_keys[local_row, column]))
                    counters["pending_rounded_seen"] += 1
                    if group is None:
                        group = weak_lookup.get(bytes(weak_keys[local_row, column]))
                        counters["pending_weak_seen"] += 1
                    if group is None:
                        group = _full_pool_identity_matches(
                            encoder=encoder,
                            candidate_values=candidate_values,
                            src=int(positives.src[row]),
                            query_time=int(positives.time[row]),
                            expected=np.asarray(
                                cached_features[row, column, match_indices],
                                dtype=np.float32,
                            ),
                            match_indices=match_indices,
                        )
                        counters["pending_full_pool_seen"] += 1
                else:
                    counters["pending_ambiguous_seen"] += 1
                if not group:
                    raise RuntimeError(
                        f"{label} row={row} column={column}: seen candidate has no signature group"
                    )
                if len(group) > max_resolution_group:
                    raise RuntimeError(
                        f"{label} row={row} column={column}: resolution group too large ({len(group)})"
                    )
                pending_by_row[row].append((column, group))
        if stop % 5_000 == 0 or stop == rows:
            print(
                f"[decay-recovery] {label} scan rows={stop}/{rows} "
                f"pending={sum(len(items) for items in pending_by_row.values())}",
                flush=True,
            )

    signature_pending_positions = sum(len(items) for items in pending_by_row.values())
    decay_ambiguous_by_row: dict[int, list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
    for row, items in pending_by_row.items():
        for column, group in items:
            group_ids = np.asarray(group, dtype=np.int32)
            group_decay = encoder.structure.index.cooccur_time_decay_scores(
                int(positives.src[row]),
                group_ids,
                int(positives.time[row]),
                source_history_limit=source_history_limit,
            )
            if np.allclose(group_decay, group_decay[0], rtol=1e-6, atol=1e-7):
                override_values[(row, column)] = float(group_decay[0])
                counters["resolved_equal_decay_tie"] += 1
            else:
                decay_ambiguous_by_row[row].append((column, group))
    pending_by_row = decay_ambiguous_by_row
    print(
        f"[decay-recovery] {label} decay-ambiguous "
        f"positions={sum(len(items) for items in pending_by_row.values())}/"
        f"{signature_pending_positions}",
        flush=True,
    )
    ambiguous_max_values: dict[tuple[int, int], float] = {}
    pending_rows = sorted(pending_by_row)
    for start in range(0, len(pending_rows), max(resolve_batch_rows, 1)):
        chunk_rows = pending_rows[start : start + max(resolve_batch_rows, 1)]
        unions: list[np.ndarray] = []
        positions: list[dict[int, int]] = []
        max_width = 1
        for row in chunk_rows:
            union = np.asarray(
                sorted(
                    {
                        candidate
                        for _, group in pending_by_row[row]
                        for candidate in group
                    }
                ),
                dtype=np.int32,
            )
            if union.size == 0:
                raise RuntimeError("empty candidate union during signature resolution")
            unions.append(union)
            positions.append({int(candidate): index for index, candidate in enumerate(union)})
            max_width = max(max_width, len(union))
        matrix = np.empty((len(chunk_rows), max_width), dtype=np.int32)
        for index, union in enumerate(unions):
            matrix[index] = int(union[0])
            matrix[index, : len(union)] = union
        queries = TestQueryArray(
            src=positives.src[np.asarray(chunk_rows, dtype=np.int64)].astype(np.int32, copy=False),
            time=positives.time[np.asarray(chunk_rows, dtype=np.int64)].astype(np.int32, copy=False),
            candidates=matrix,
        )
        generated_stats = encoder.stats.features_for_query_array(queries)
        generated_prior = encoder.candidate_prior.features_for_query_array(
            queries,
            generated_stats,
        )
        generated_target = encoder.target_window.features_for_query_array(queries)
        generated_identity = np.concatenate(
            (generated_stats, generated_prior, generated_target),
            axis=2,
        )
        for chunk_index, row in enumerate(chunk_rows):
            for column, group in pending_by_row[row]:
                candidate_positions = np.asarray(
                    [positions[chunk_index][candidate] for candidate in group],
                    dtype=np.int64,
                )
                actual = generated_identity[chunk_index, candidate_positions][
                    :,
                    match_indices,
                ]
                expected = np.asarray(cached_features[row, column, match_indices], dtype=np.float32)
                matches = np.flatnonzero(
                    np.all(
                        np.isclose(
                            actual,
                            expected[None, :],
                            rtol=2e-5,
                            atol=2e-6,
                        ),
                        axis=1,
                    )
                )
                if matches.size == 1:
                    resolved_ids[row, column] = group[int(matches[0])]
                    counters["resolved_by_query_features"] += 1
                    continue
                if matches.size == 0:
                    max_error = float(np.min(np.max(np.abs(actual - expected[None, :]), axis=1)))
                    raise RuntimeError(
                        f"{label} row={row} column={column}: no deterministic feature match "
                        f"group={len(group)} min_max_abs_error={max_error:.8g}"
                    )
                tied_ids = np.asarray(
                    [group[int(index)] for index in matches],
                    dtype=np.int32,
                )
                tied_decay = encoder.structure.index.cooccur_time_decay_scores(
                    int(positives.src[row]),
                    tied_ids,
                    int(positives.time[row]),
                    source_history_limit=source_history_limit,
                )
                if np.allclose(tied_decay, tied_decay[0], rtol=1e-6, atol=1e-7):
                    override_values[(row, column)] = float(tied_decay[0])
                    counters["resolved_equal_decay_tie"] += 1
                    continue
                tied_query = TestQueryArray(
                    src=np.asarray([int(positives.src[row])], dtype=np.int32),
                    time=np.asarray([int(positives.time[row])], dtype=np.int32),
                    candidates=tied_ids.reshape(1, -1),
                )
                structure_actual = encoder.structure.features_for_query_array(tied_query)[0][
                    :,
                    structure_match_local_indices,
                ]
                structure_expected = np.asarray(
                    cached_features[row, column, structure_match_indices],
                    dtype=np.float32,
                )
                structure_matches = np.flatnonzero(
                    np.all(
                        np.isclose(
                            structure_actual,
                            structure_expected[None, :],
                            rtol=2e-5,
                            atol=2e-6,
                        ),
                        axis=1,
                    )
                )
                if structure_matches.size == 1:
                    resolved_ids[row, column] = tied_ids[int(structure_matches[0])]
                    counters["resolved_by_query_features"] += 1
                    continue
                if structure_matches.size > 0:
                    tied_ids = tied_ids[structure_matches]
                    tied_decay = tied_decay[structure_matches]
                if not np.allclose(tied_decay, tied_decay[0], rtol=1e-6, atol=1e-7):
                    ambiguous_max_values[(row, column)] = float(np.max(tied_decay))
                    counters["ambiguous_conservative_max"] += 1
                    continue
                override_values[(row, column)] = float(tied_decay[0])
                counters["resolved_equal_decay_tie"] += 1
        completed = min(start + len(chunk_rows), len(pending_rows))
        if completed % 2_000 < len(chunk_rows) or completed == len(pending_rows):
            print(
                f"[decay-recovery] {label} disambiguated_rows={completed}/{len(pending_rows)}",
                flush=True,
            )

    decay = np.zeros((rows, columns), dtype=np.float32)
    for start in range(0, rows, max(score_batch_rows, 1)):
        stop = min(start + max(score_batch_rows, 1), rows)
        queries = TestQueryArray(
            src=positives.src[start:stop].astype(np.int32, copy=False),
            time=positives.time[start:stop].astype(np.int32, copy=False),
            candidates=resolved_ids[start:stop],
        )
        decay[start:stop] = encoder.structure.time_decay_features_for_queries(queries)[..., 0]
        if stop % 5_000 == 0 or stop == rows:
            print(f"[decay-recovery] {label} scored rows={stop}/{rows}", flush=True)
    for (row, column), value in override_values.items():
        decay[row, column] = value
    for (row, column), value in ambiguous_max_values.items():
        decay[row, column] = value
    if not np.all(np.isfinite(decay)) or np.any(decay < 0.0):
        raise RuntimeError(f"{label} recovered decay column is invalid")
    seen_mask = np.asarray(cached_features[..., seen_index] > 0.5)
    unresolved_seen = seen_mask & (resolved_ids < 0)
    for row, column in override_values:
        unresolved_seen[row, column] = False
    for row, column in ambiguous_max_values:
        unresolved_seen[row, column] = False
    ambiguous_max_limit = max(32, int(np.ceil(decay.size * 5e-4)))
    if len(ambiguous_max_values) > ambiguous_max_limit:
        raise RuntimeError(
            f"{label} conservative ambiguous fill exceeds safety limit: "
            f"actual={len(ambiguous_max_values)} limit={ambiguous_max_limit}"
        )
    if np.any(unresolved_seen):
        raise RuntimeError(f"{label} contains unresolved seen candidates")
    counters.update(
        {
            "rows": rows,
            "candidate_count": columns,
            "pending_rows": len(pending_rows),
            "pending_positions": sum(len(items) for items in pending_by_row.values()),
            "signature_pending_positions": signature_pending_positions,
            "unresolved_seen": int(unresolved_seen.sum()),
            "ambiguous_conservative_max_limit": ambiguous_max_limit,
            "ambiguous_conservative_max_share": float(
                len(ambiguous_max_values) / max(decay.size, 1)
            ),
            "ambiguous_conservative_max_mean": float(
                np.mean(tuple(ambiguous_max_values.values()))
                if ambiguous_max_values
                else 0.0
            ),
            "ambiguous_conservative_max_value": float(
                max(ambiguous_max_values.values(), default=0.0)
            ),
            "nonzero_decay_positions": int(np.count_nonzero(decay)),
            "nonzero_decay_share": float(np.count_nonzero(decay) / max(decay.size, 1)),
            "positive_nonzero_share": float(np.count_nonzero(decay[:, 0]) / max(rows, 1)),
            "decay_max": float(decay.max(initial=0.0)),
            "seconds": time.time() - started,
        }
    )
    del encoder, exact_lookup, rounded_lookup, weak_lookup, resolved_ids
    gc.collect()
    return decay, counters


def _signature_lookup(
    candidates: np.ndarray,
    features: np.ndarray,
    indices: tuple[int, ...],
    *,
    decimals: int | None,
) -> dict[bytes, tuple[int, ...]]:
    values: dict[bytes, list[int]] = defaultdict(list)
    for candidate, key in zip(
        candidates,
        _signature_keys(features, indices, decimals=decimals),
        strict=True,
    ):
        values[bytes(key)].append(int(candidate))
    return {key: tuple(group) for key, group in values.items()}


def _full_pool_identity_matches(
    *,
    encoder: HybridFeatureEncoder,
    candidate_values: np.ndarray,
    src: int,
    query_time: int,
    expected: np.ndarray,
    match_indices: tuple[int, ...],
) -> tuple[int, ...]:
    query = TestQueryArray(
        src=np.asarray([src], dtype=np.int32),
        time=np.asarray([query_time], dtype=np.int32),
        candidates=candidate_values.reshape(1, -1),
    )
    stats = encoder.stats.features_for_query_array(query)
    prior = encoder.candidate_prior.features_for_query_array(query, stats)
    target = encoder.target_window.features_for_query_array(query)
    identity_features = np.concatenate((stats, prior, target), axis=2)[0][:, match_indices]
    matches = np.flatnonzero(
        np.all(
            np.isclose(
                identity_features,
                expected[None, :],
                rtol=2e-5,
                atol=2e-6,
            ),
            axis=1,
        )
    )
    if matches.size == 0:
        max_error = float(
            np.min(np.max(np.abs(identity_features - expected[None, :]), axis=1))
        )
        raise RuntimeError(
            "full candidate pool has no deterministic identity match "
            f"src={src} time={query_time} min_max_abs_error={max_error:.8g}"
        )
    return tuple(int(candidate_values[index]) for index in matches)


def _signature_keys(
    features: np.ndarray,
    indices: tuple[int, ...],
    *,
    decimals: int | None,
) -> np.ndarray:
    selected = np.asarray(features, dtype=np.float32)[..., indices]
    if decimals is not None:
        selected = np.round(selected, decimals=decimals)
    selected = np.ascontiguousarray(selected, dtype=np.float32)
    return selected.view(
        np.dtype((np.void, selected.shape[-1] * selected.dtype.itemsize))
    )[..., 0]


def _validate_cache(manifest: dict[str, Any], train: np.ndarray, val: np.ndarray) -> None:
    if list(train.shape) != manifest["train"]["shape"]:
        raise ValueError("training cache shape does not match its manifest")
    if list(val.shape) != manifest["val"]["shape"]:
        raise ValueError("validation cache shape does not match its manifest")
    if train.dtype.str != manifest["train"]["dtype"] or val.dtype.str != manifest["val"]["dtype"]:
        raise ValueError("cache dtype does not match its manifest")
    if val.shape[1] != 100:
        raise ValueError("validation cache is not the required complete 100-candidate cache")


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _mrr(scores: np.ndarray) -> float:
    ranks = 1 + (scores[:, 1:] > scores[:, 0:1]).sum(axis=1)
    return float(np.mean(1.0 / ranks))


def _temporal_mrr(scores: np.ndarray, folds) -> dict[str, Any]:
    return {
        "full": _mrr(scores),
        "slices": [
            {
                "index": fold.index,
                "rows": [fold.holdout.start, fold.holdout.stop],
                "mrr": _mrr(scores[fold.holdout]),
            }
            for fold in folds
        ],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
