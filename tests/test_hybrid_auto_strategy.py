import csv

import numpy as np

from jgrec.core.types import Interaction, InteractionTable
from jgrec.rankers.common.temporal_index import TemporalInteractionIndex
from jgrec.rankers.hybrid.auto_strategy import (
    NEW_LINK_COLD_MODE,
    REPEAT_MEMORY_MODE,
    choose_auto_strategy,
    profile_dataset,
    profile_dataset_paths,
)
from jgrec.rankers.hybrid.sampling import NegativeSamplingContext, sample_mixed_negatives


def _write_test(path, rows):
    header = ["src", "time", *(f"c{idx}" for idx in range(100))]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _write_train(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["src", "dst", "time"])
        writer.writerows(rows)


def _row(src: int, time: int, candidates: list[int]) -> list[int]:
    values = list(candidates)
    while len(values) < 100:
        values.append(10_000 + len(values))
    return [src, time, *values[:100]]


def test_auto_strategy_detects_new_link_cold_without_dataset_name(tmp_path):
    test_path = tmp_path / "arbitrary_name.csv"
    interactions = [
        *(Interaction(src=idx % 5, dst=idx, time=idx) for idx in range(1, 41)),
        *(Interaction(src=idx % 5, dst=idx + 1000, time=idx) for idx in range(41, 51)),
    ]
    _write_test(test_path, [_row(1, 100, list(range(2000, 2100)))])

    profile = profile_dataset(InteractionTable.from_events(interactions), test_path, val_ratio=0.2)
    strategy = choose_auto_strategy(profile)

    assert profile.holdout_pair_hit_rate == 0.0
    assert profile.candidate_unseen_dst_rate == 1.0
    assert strategy.mode == NEW_LINK_COLD_MODE
    assert strategy.test_candidate_negative_ratio == 0.60


def test_auto_strategy_detects_repeat_memory_without_dataset_name(tmp_path):
    test_path = tmp_path / "whatever.csv"
    interactions = [
        *(Interaction(src=1, dst=10, time=idx) for idx in range(1, 31)),
        *(Interaction(src=2, dst=20, time=idx) for idx in range(31, 51)),
    ]
    _write_test(test_path, [_row(1, 100, [10, 20] * 50)])

    profile = profile_dataset(InteractionTable.from_events(interactions), test_path, val_ratio=0.2)
    strategy = choose_auto_strategy(profile)

    assert profile.holdout_pair_hit_rate >= 0.25
    assert profile.candidate_unseen_dst_rate <= 0.20
    assert strategy.mode == REPEAT_MEMORY_MODE
    assert strategy.test_candidate_negative_ratio == 0.10


def test_profile_dataset_counts_test_candidates_and_min_time_once(tmp_path):
    test_path = tmp_path / "test.csv"
    interactions = [
        Interaction(src=1, dst=10, time=1),
        Interaction(src=2, dst=20, time=2),
    ]
    _write_test(
        test_path,
        [
            _row(1, 80, [10, 10, 20]),
            _row(2, 50, [10, 30]),
        ],
    )

    profile = profile_dataset(InteractionTable.from_events(interactions), test_path, val_ratio=0.5)

    assert profile.test_min_time == 50
    assert profile.test_candidate_total == 200
    assert profile.test_candidate_counts[10] == 3
    assert profile.test_candidate_counts[20] == 1
    assert profile.test_candidate_counts[30] == 1
    assert profile.candidate_unseen_dst_rate == 196 / 200


def test_path_profile_uses_time_order_not_file_order(tmp_path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    _write_train(
        train_path,
        [
            [2, 20, 2],
            [2, 30, 3],
            [2, 40, 4],
            [1, 10, 1],
            [1, 10, 101],
        ],
    )
    _write_test(test_path, [_row(1, 200, [10, 20] * 50)])

    profile = profile_dataset_paths(train_path, test_path, val_ratio=0.4)

    assert profile.holdout_pair_hit_rate == 0.5


def test_test_candidate_negative_sampling_can_use_test_only_destinations():
    interactions = [
        Interaction(src=1, dst=10, time=10),
        Interaction(src=1, dst=20, time=20),
        Interaction(src=2, dst=30, time=30),
    ]
    index = TemporalInteractionIndex()
    index.fit(InteractionTable.from_events(interactions))
    context = NegativeSamplingContext(
        index=index,
        dst_values=(10, 20, 30),
        test_candidate_values=np.asarray([999, 998], dtype=np.int64),
        test_candidate_weights=np.asarray([0.9, 0.1], dtype=np.float64),
    )

    negatives = sample_mixed_negatives(
        src=2,
        positive_dst=30,
        query_time=40,
        context=context,
        dst_pool=np.asarray([10, 20, 30], dtype=np.int64),
        num_negatives=2,
        rng=np.random.default_rng(1),
        hard_negative_ratio=0.0,
        popular_negative_ratio=0.0,
        test_candidate_negative_ratio=1.0,
    )

    assert set(negatives) & {998, 999}
