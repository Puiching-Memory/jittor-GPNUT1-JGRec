from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from evaluate_dataset2_two_hop_decay_full100 import (
    MATCH_FEATURE_NAMES,
    SIGNATURE_FEATURE_NAMES,
    WEAK_SIGNATURE_FEATURE_NAMES,
)
from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.core.io import read_interactions, read_test_queries
from jgrec.core.types import TestQueryArray
from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid.ranker import HybridFeatureEncoder, _sample_events


def main() -> int:
    checkpoint = "checkpoints/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722.pkl"
    cache = "cache/supervised_features/4baa722bf26e5d50356da26ac5f479cb54324ddb.train.npy"
    state = load_checkpoint_dataset(checkpoint, "dataset2")
    config = state["config"]
    names = tuple(state["feature_names"])
    interactions = read_interactions(Path("data/dataset2/train.csv")).sort_by_time()
    val_size = max(1, int(len(interactions) * config.val_ratio))
    train_end = len(interactions) - val_size
    context_end = int(train_end * config.context_ratio)
    rng = np.random.default_rng(config.seed)
    sampled = _sample_events(interactions[context_end:train_end], config.max_train_events, rng)
    prefix = interactions[:context_end]
    cached = np.load(cache, mmap_mode="r")[0, 1]
    deterministic = replace(
        config,
        auto_strategy_enabled=False,
        candidate_prior_enabled=True,
        candidate_prior_include_test_frequency=True,
        target_window_enabled=True,
        structure_enabled=False,
        source_profile_enabled=False,
        two_tower_enabled=False,
        gnn_enabled=False,
        seq_enabled=False,
        verbose=False,
    )
    encoder = HybridFeatureEncoder(
        id_map=NodeIdMap.from_interactions(prefix),
        recent_window=int(state["recent_window"]),
        candidate_prior_config=deterministic.candidate_prior_config(),
        target_window_config=deterministic.target_window_config(),
        structure_config=deterministic.structure_config(),
        source_profile_config=deterministic.source_profile_config(),
        two_tower_config=deterministic.two_tower_config(),
        graph_config=deterministic.graph_config(),
        sequence_config=deterministic.sequence_config(),
        dataset_profile=state["dataset_profile"],
    )
    encoder.fit(prefix, rng=np.random.default_rng(0), verbose=False)
    tests = read_test_queries(Path("data/dataset2/test.csv"))
    candidates = np.unique(
        np.concatenate((interactions.dst, tests.candidates.reshape(-1)))
    ).astype(np.int32)
    query = TestQueryArray(
        src=np.asarray([-1], dtype=np.int32),
        time=np.asarray([int(prefix.time[-1]) + 1], dtype=np.int32),
        candidates=candidates.reshape(1, -1),
    )
    reference = encoder.features_for_query_array(query)[0]
    for label, selected_names in (
        ("stable", SIGNATURE_FEATURE_NAMES),
        ("weak", WEAK_SIGNATURE_FEATURE_NAMES),
        ("match", MATCH_FEATURE_NAMES),
    ):
        indices = tuple(names.index(name) for name in selected_names)
        target = cached[list(indices)].astype(np.float64)
        actual = reference[:, list(indices)].astype(np.float64)
        scale = np.maximum(np.abs(target), 1e-6)
        relative = np.max(np.abs(actual - target) / scale, axis=1)
        order = np.argsort(relative)[:10]
        print(label, "top", [(int(candidates[i]), float(relative[i])) for i in order])
        best = int(order[0])
        print(
            label,
            "best_fields",
            [
                (name, float(target[j]), float(actual[best, j]), float(actual[best, j] - target[j]))
                for j, name in enumerate(selected_names)
                if abs(float(actual[best, j] - target[j])) > 1e-7
            ],
        )
        for decimals in (6, 5, 4, 3, 2):
            matches = np.all(np.round(actual, decimals) == np.round(target, decimals), axis=1)
            print(label, "decimals", decimals, "matches", int(matches.sum()))
    row_query = TestQueryArray(
        src=np.asarray([int(sampled.src[0])], dtype=np.int32),
        time=np.asarray([int(sampled.time[0])], dtype=np.int32),
        candidates=candidates.reshape(1, -1),
    )
    row_features = encoder.features_for_query_array(row_query)[0]
    match_indices = tuple(names.index(name) for name in MATCH_FEATURE_NAMES)
    target = cached[list(match_indices)].astype(np.float64)
    actual = row_features[:, list(match_indices)].astype(np.float64)
    scale = np.maximum(np.abs(target), 1e-6)
    relative = np.max(np.abs(actual - target) / scale, axis=1)
    order = np.argsort(relative)[:20]
    print("row_top", [(int(candidates[i]), float(relative[i])) for i in order])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
