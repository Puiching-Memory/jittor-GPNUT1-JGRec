from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from evaluate_dataset2_two_hop_decay_full100 import WEAK_SIGNATURE_FEATURE_NAMES
from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.core.io import read_interactions, read_test_queries
from jgrec.core.types import TestQueryArray
from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid.ranker import HybridFeatureEncoder, _sample_events


def main() -> int:
    checkpoint = Path("checkpoints/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722.pkl")
    cache_prefix = Path("cache/supervised_features/4baa722bf26e5d50356da26ac5f479cb54324ddb")
    state = load_checkpoint_dataset(checkpoint, "dataset2")
    config = state["config"]
    names = tuple(state["feature_names"])
    weak_indices = tuple(names.index(name) for name in WEAK_SIGNATURE_FEATURE_NAMES)
    seen_index = names.index("candidate_train_seen")
    interactions = read_interactions(Path("data/dataset2/train.csv")).sort_by_time()
    val_size = max(1, int(len(interactions) * config.val_ratio))
    train_end = len(interactions) - val_size
    context_end = int(train_end * config.context_ratio)
    rng = np.random.default_rng(config.seed)
    sampled_train = _sample_events(
        interactions[context_end:train_end], config.max_train_events, rng
    )
    sampled_val = _sample_events(interactions[train_end:], config.max_val_events, rng)
    tests = read_test_queries(Path("data/dataset2/test.csv"))
    candidates = np.unique(
        np.concatenate((interactions.dst, tests.candidates.reshape(-1)))
    ).astype(np.int32)
    for label, prefix, positives, cache_path in (
        ("train", interactions[:context_end], sampled_train, cache_prefix.with_suffix(".train.npy")),
        ("validation", interactions[:train_end], sampled_val, cache_prefix.with_suffix(".val.npy")),
    ):
        started = time.time()
        encoder = _encoder(prefix, state)
        reference_query = TestQueryArray(
            src=np.asarray([-1], dtype=np.int32),
            time=np.asarray([int(prefix.time[-1]) + 1], dtype=np.int32),
            candidates=candidates.reshape(1, -1),
        )
        reference = encoder.features_for_query_array(reference_query)[0][:, weak_indices]
        cached = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        print(label, "rows", len(positives), "reference", len(candidates), flush=True)
        for decimals in (6, 5, 4, 3, 2):
            lookup = _lookup(candidates, reference, decimals)
            missing = 0
            seen_positions = 0
            max_group = 0
            group_sum = 0
            for start in range(0, len(cached), 256):
                stop = min(start + 256, len(cached))
                keys = _keys(cached[start:stop, :, weak_indices], decimals)
                seen = cached[start:stop, :, seen_index] > 0.5
                seen[:, 0] = False
                for key in keys[seen]:
                    seen_positions += 1
                    group = lookup.get(bytes(key))
                    if group is None:
                        missing += 1
                    else:
                        size = len(group)
                        group_sum += size
                        max_group = max(max_group, size)
            print(
                label,
                "decimals",
                decimals,
                "seen",
                seen_positions,
                "missing",
                missing,
                "max_group",
                max_group,
                "mean_group",
                group_sum / max(seen_positions - missing, 1),
                flush=True,
            )
        print(label, "seconds", time.time() - started, flush=True)
    return 0


def _encoder(prefix, state) -> HybridFeatureEncoder:
    config = replace(
        state["config"],
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
        candidate_prior_config=config.candidate_prior_config(),
        target_window_config=config.target_window_config(),
        structure_config=config.structure_config(),
        source_profile_config=config.source_profile_config(),
        two_tower_config=config.two_tower_config(),
        graph_config=config.graph_config(),
        sequence_config=config.sequence_config(),
        dataset_profile=state["dataset_profile"],
    )
    encoder.fit(prefix, rng=np.random.default_rng(0), verbose=False)
    return encoder


def _lookup(candidates: np.ndarray, features: np.ndarray, decimals: int):
    values = defaultdict(list)
    for candidate, key in zip(candidates, _keys(features, decimals), strict=True):
        values[bytes(key)].append(int(candidate))
    return values


def _keys(features: np.ndarray, decimals: int) -> np.ndarray:
    selected = np.ascontiguousarray(np.round(features, decimals=decimals), dtype=np.float32)
    return selected.view(
        np.dtype((np.void, selected.shape[-1] * selected.dtype.itemsize))
    )[..., 0]


if __name__ == "__main__":
    raise SystemExit(main())
