from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.core.io import read_interactions, read_test_queries
from jgrec.core.types import TestQueryArray
from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid.ranker import HybridFeatureEncoder, _sample_events

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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether stable cached features recover Dataset2 candidate identities."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cache-prefix", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-rows", type=int, default=256)
    args = parser.parse_args()
    started = time.time()

    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    config = state["config"]
    feature_names = tuple(str(name) for name in state["feature_names"])
    signature_indices = tuple(feature_names.index(name) for name in SIGNATURE_FEATURE_NAMES)
    train_features = np.load(
        args.cache_prefix.with_suffix(".train.npy"), mmap_mode="r", allow_pickle=False
    )
    val_features = np.load(
        args.cache_prefix.with_suffix(".val.npy"), mmap_mode="r", allow_pickle=False
    )
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
    if train_features.shape[0] != len(sampled_train):
        raise ValueError("cached training rows do not match reconstructed positives")
    if val_features.shape[0] != len(sampled_val):
        raise ValueError("cached validation rows do not match reconstructed positives")

    test_queries = read_test_queries(args.test_csv)
    candidate_values = np.unique(
        np.concatenate((interactions.dst, test_queries.candidates.reshape(-1)))
    ).astype(np.int32, copy=False)
    report = {
        "signature_feature_names": list(SIGNATURE_FEATURE_NAMES),
        "signature_feature_indices": list(signature_indices),
        "candidate_pool_size": len(candidate_values),
        "train": _diagnose_split(
            label="train",
            prefix=interactions[:context_end],
            positives=sampled_train,
            cached_features=train_features,
            candidate_values=candidate_values,
            state=state,
            signature_indices=signature_indices,
            train_seen_index=feature_names.index("candidate_train_seen"),
            batch_rows=args.batch_rows,
        ),
        "val": _diagnose_split(
            label="val",
            prefix=interactions[:train_end],
            positives=sampled_val,
            cached_features=val_features,
            candidate_values=candidate_values,
            state=state,
            signature_indices=signature_indices,
            train_seen_index=feature_names.index("candidate_train_seen"),
            batch_rows=args.batch_rows,
        ),
        "elapsed_seconds": time.time() - started,
    }
    report["exact_recovery_feasible"] = bool(
        report["train"]["missing_positions"] == 0
        and report["train"]["ambiguous_positions"] == 0
        and report["train"]["positive_identity_mismatches"] == 0
        and report["val"]["missing_positions"] == 0
        and report["val"]["ambiguous_positions"] == 0
        and report["val"]["positive_identity_mismatches"] == 0
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0 if report["exact_recovery_feasible"] else 2


def _diagnose_split(
    *,
    label: str,
    prefix,
    positives,
    cached_features: np.ndarray,
    candidate_values: np.ndarray,
    state: dict,
    signature_indices: tuple[int, ...],
    train_seen_index: int,
    batch_rows: int,
) -> dict:
    config = state["config"]
    deterministic_config = replace(
        config,
        auto_strategy_enabled=False,
        candidate_prior_enabled=True,
        target_window_enabled=True,
        structure_enabled=True,
        structure_cooccur_enabled=False,
        structure_transition_enabled=False,
        structure_future_only_transition_cooccur=True,
        structure_cooccur_time_decay_enabled=False,
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
    reference_query = TestQueryArray(
        src=np.asarray([-1], dtype=np.int32),
        time=np.asarray([int(prefix.time[-1]) + 1], dtype=np.int32),
        candidates=candidate_values.reshape(1, -1),
    )
    reference_features = encoder.features_for_query_array(reference_query)[0]
    lookup: dict[bytes, list[int]] = {}
    for candidate, key in zip(
        candidate_values,
        _signature_keys(reference_features, signature_indices),
        strict=True,
    ):
        lookup.setdefault(bytes(key), []).append(int(candidate))

    missing_positions = 0
    missing_seen_positions = 0
    ambiguous_positions = 0
    ambiguous_seen_positions = 0
    unique_positions = 0
    positive_identity_mismatches = 0
    max_group_size = max((len(values) for values in lookup.values()), default=0)
    group_size_counts: dict[int, int] = {}
    for start in range(0, len(cached_features), batch_rows):
        stop = min(start + batch_rows, len(cached_features))
        keys = _signature_keys(cached_features[start:stop], signature_indices)
        for local_row, row_keys in enumerate(keys):
            for column, key in enumerate(row_keys):
                candidates = lookup.get(bytes(key))
                is_seen = bool(
                    cached_features[start + local_row, column, train_seen_index] > 0.5
                )
                if candidates is None:
                    missing_positions += 1
                    missing_seen_positions += int(is_seen)
                    continue
                size = len(candidates)
                group_size_counts[size] = group_size_counts.get(size, 0) + 1
                if size == 1:
                    unique_positions += 1
                else:
                    ambiguous_positions += 1
                    ambiguous_seen_positions += int(is_seen)
                if column == 0 and int(positives.dst[start + local_row]) not in candidates:
                    positive_identity_mismatches += 1
        if stop % 5_000 == 0 or stop == len(cached_features):
            print(
                f"[candidate-signature] {label} rows={stop}/{len(cached_features)} "
                f"missing={missing_positions} ambiguous={ambiguous_positions}",
                flush=True,
            )
    total_positions = int(np.prod(cached_features.shape[:2]))
    return {
        "rows": len(cached_features),
        "candidate_count": cached_features.shape[1],
        "total_positions": total_positions,
        "unique_positions": unique_positions,
        "unique_position_share": unique_positions / max(total_positions, 1),
        "ambiguous_positions": ambiguous_positions,
        "ambiguous_seen_positions": ambiguous_seen_positions,
        "ambiguous_position_share": ambiguous_positions / max(total_positions, 1),
        "missing_positions": missing_positions,
        "missing_seen_positions": missing_seen_positions,
        "positive_identity_mismatches": positive_identity_mismatches,
        "signature_group_size_position_counts": {
            str(size): count for size, count in sorted(group_size_counts.items())
        },
        "reference_signature_count": len(lookup),
        "max_reference_group_size": max_group_size,
    }


def _signature_keys(
    features: np.ndarray, signature_indices: tuple[int, ...]
) -> np.ndarray:
    selected = np.ascontiguousarray(
        np.asarray(features, dtype=np.float32)[..., signature_indices]
    )
    return selected.view(np.dtype((np.void, selected.shape[-1] * selected.dtype.itemsize)))[
        ..., 0
    ]


if __name__ == "__main__":
    raise SystemExit(main())
