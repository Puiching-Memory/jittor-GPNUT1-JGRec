import numpy as np

from jgrec.core.types import InteractionTable
from jgrec.rankers.hybrid.source_sequence_cache import (
    build_causal_source_sequences,
    expanding_timestamp_abcd_folds,
)


def test_expanding_abcd_folds_keep_equal_timestamps_on_one_side():
    times = np.repeat(np.arange(10, dtype=np.int64), 10)

    folds = expanding_timestamp_abcd_folds(times)

    assert len(folds) == 3
    assert [fold.role for fold in folds] == [
        "selection",
        "selection",
        "gate",
    ]
    for fold in folds:
        train_stop = fold.train_rows[1]
        score_start, score_stop = fold.score_rows
        assert train_stop == score_start
        assert int(times[train_stop - 1]) < int(times[score_start])
        assert score_stop <= len(times)
    assert folds[-1].score_rows[1] == len(times)


def test_causal_sequences_exclude_equal_and_future_events():
    interactions = InteractionTable(
        src=np.array([7, 7, 8, 7, 7], dtype=np.int32),
        dst=np.array([11, 12, 21, 13, 14], dtype=np.int32),
        time=np.array([10, 20, 20, 30, 40], dtype=np.int32),
    )

    rows = build_causal_source_sequences(
        interactions,
        query_src=np.array([7, 7], dtype=np.int32),
        query_time=np.array([20, 35], dtype=np.int64),
        max_length=4,
    )

    assert rows.lengths.tolist() == [1, 3]
    assert rows.items[0].tolist() == [11, 0, 0, 0]
    assert rows.items[1].tolist() == [11, 12, 13, 0]
    assert np.all(rows.time_buckets[:, :3] >= 0)


def test_causal_sequences_truncate_to_recent_and_freeze_at_origin():
    interactions = InteractionTable(
        src=np.full(6, 3, dtype=np.int32),
        dst=np.arange(101, 107, dtype=np.int32),
        time=np.arange(10, 70, 10, dtype=np.int32),
    )

    rows = build_causal_source_sequences(
        interactions,
        query_src=np.array([3], dtype=np.int32),
        query_time=np.array([100], dtype=np.int64),
        max_length=3,
        history_time_limit=45,
    )

    assert rows.lengths.tolist() == [3]
    assert rows.items[0].tolist() == [102, 103, 104]


def test_candidate_and_history_ids_need_no_learned_or_statistical_mapping():
    interactions = InteractionTable(
        src=np.array([1000], dtype=np.int32),
        dst=np.array([42], dtype=np.int32),
        time=np.array([10], dtype=np.int32),
    )

    rows = build_causal_source_sequences(
        interactions,
        query_src=np.array([1000], dtype=np.int32),
        query_time=np.array([20], dtype=np.int64),
        max_length=2,
    )

    assert rows.items[0, 0] == 42

