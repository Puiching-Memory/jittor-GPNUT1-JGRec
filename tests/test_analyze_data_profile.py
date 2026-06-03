import csv
import importlib.util
import sys
from collections import deque
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_data_profile.py"
_SPEC = importlib.util.spec_from_file_location("analyze_data_profile_under_test", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
profile = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = profile
_SPEC.loader.exec_module(profile)

Event = profile.Event
Query = profile.Query
build_state = profile.build_state
candidate_distribution = profile.test_candidate_distribution
read_test = profile.read_test
read_train = profile.read_train
recent_hit_rank = profile.recent_hit_rank
time_drift = profile.time_drift
unseen_dst_analysis = profile.unseen_dst_analysis


def _write_csv(path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def test_read_train_uses_header_columns_and_stable_time_sort(tmp_path) -> None:
    path = tmp_path / "train.csv"
    _write_csv(
        path,
        [
            ["time", "dst", "src", "weight"],
            ["30", "9", "2", "0"],
            ["20", "8", "1", "0"],
            ["20", "10", "3", "0"],
            ["10", "7", "1", "0"],
        ],
    )

    assert read_train(path) == [
        Event(src=1, dst=7, time=10),
        Event(src=1, dst=8, time=20),
        Event(src=3, dst=10, time=20),
        Event(src=2, dst=9, time=30),
    ]


def test_read_test_parses_single_row_candidates(tmp_path) -> None:
    path = tmp_path / "test.csv"
    _write_csv(path, [["src", "time", "c1", "c2", "c3"], ["7", "11", "101", "102", "103"]])

    assert read_test(path) == [Query(src=7, time=11, candidates=(101, 102, 103))]


def test_read_test_rejects_inconsistent_row_width(tmp_path) -> None:
    path = tmp_path / "test.csv"
    _write_csv(path, [["src", "time", "c1", "c2", "c3"], ["7", "11", "101"]])

    with pytest.raises(ValueError):
        read_test(path)


def test_recent_hit_rank_and_time_drift_recent_rates() -> None:
    assert recent_hit_rank(deque([1, 2, 3, 2]), 2) == 1
    assert recent_hit_rank(deque([1, 2, 3, 2]), 1, limit=2) is None

    events = [
        Event(src=1, dst=1, time=10),
        Event(src=1, dst=2, time=20),
        Event(src=1, dst=1, time=30),
        Event(src=1, dst=2, time=40),
        Event(src=1, dst=1, time=50),
        Event(src=2, dst=4, time=60),
        Event(src=2, dst=5, time=70),
        Event(src=2, dst=6, time=80),
        Event(src=2, dst=5, time=90),
        Event(src=2, dst=6, time=100),
    ]

    deciles = time_drift(events)["deciles"]

    assert deciles[0]["recent_5_hit"] == 0.0
    assert deciles[2]["recent_1_hit"] == 0.0
    assert deciles[2]["recent_5_hit"] == 1.0
    assert deciles[8]["recent_10_hit"] == 1.0


def test_candidate_distribution_and_unseen_analysis_match_reference_counts() -> None:
    state = build_state(
        [
            Event(src=1, dst=10, time=1),
            Event(src=1, dst=11, time=2),
            Event(src=1, dst=10, time=3),
            Event(src=2, dst=12, time=4),
            Event(src=3, dst=13, time=5),
        ]
    )
    queries = [
        Query(src=1, time=10, candidates=(10, 11, 99, 2)),
        Query(src=2, time=10, candidates=(12, 99, 3, 10)),
    ]

    distribution = candidate_distribution(queries, state)
    rates = distribution["rates"]

    assert rates["known_dst_candidate_rate"] == pytest.approx(4 / 8)
    assert rates["unseen_dst_candidate_rate"] == pytest.approx(4 / 8)
    assert rates["pair_hit_candidate_rate"] == pytest.approx(3 / 8)
    assert rates["recent32_candidate_rate"] == pytest.approx(3 / 8)
    assert rates["top100_dst_candidate_rate"] == pytest.approx(4 / 8)
    assert rates["same_id_as_train_src_candidate_rate"] == pytest.approx(2 / 8)
    assert rates["query_with_pair_hit_rate"] == 1.0
    assert distribution["per_query"]["pair_hit"]["mean"] == pytest.approx(1.5)
    assert distribution["per_query"]["unseen_dst"]["max"] == 2.0

    unseen = unseen_dst_analysis(queries, state)

    assert unseen["unique_test_candidate_dst"] == 6
    assert unseen["unique_unseen_dst"] == 3
    assert unseen["unseen_candidate_events"] == 4
    assert unseen["unseen_candidate_event_rate"] == pytest.approx(0.5)
    assert unseen["unseen_unique_overlap_train_src"] == 2
    assert unseen["unseen_event_overlap_train_src_rate"] == pytest.approx(0.5)
    assert unseen["unseen_unique_inside_train_dst_id_range"] == 0
    assert unseen["unseen_min"] == 2
    assert unseen["unseen_max"] == 99
