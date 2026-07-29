from __future__ import annotations

import math

import numpy as np
import pytest

from jgrec.rankers.hybrid.two_hop_decay_proxy import (
    accumulate_required_cooccurrence_events,
    canonical_item_pair,
    passes_two_hop_proxy_gate,
    recent_unique_targets,
    tie_neutral_mrr,
    two_hop_scores,
)


def test_canonical_item_pair_is_order_independent() -> None:
    assert canonical_item_pair(20, 10) == (10, 20)
    assert canonical_item_pair(10, 20) == (10, 20)


def test_recent_unique_targets_keeps_latest_unique_items() -> None:
    assert recent_unique_targets(np.array([10, 20, 10, 30, 40]), 3).tolist() == [10, 30, 40]


def test_accumulate_events_matches_temporal_cooccurrence_semantics() -> None:
    required = {(10, 20), (10, 30), (20, 30)}
    output: dict[tuple[int, int], list[int]] = {}

    accumulate_required_cooccurrence_events(
        np.array([10, 20, 20, 30]),
        np.array([1, 3, 4, 7]),
        required,
        output,
        history_limit=2,
    )

    assert output == {
        (10, 20): [3],
        (10, 30): [7],
        (20, 30): [7],
    }


def test_two_hop_scores_exclude_current_and_future_events() -> None:
    raw, decayed = two_hop_scores(
        query_time=10,
        source_history=np.array([10, 30]),
        candidates=np.array([20, 40]),
        pair_event_times={
            (10, 20): np.array([3, 5, 10, 12]),
            (20, 30): np.array([7]),
        },
        tau=2.0,
    )

    assert raw.tolist() == [3.0, 0.0]
    assert decayed[0] == pytest.approx(
        math.exp(-7 / 2) + math.exp(-5 / 2) + math.exp(-3 / 2)
    )
    assert decayed[1] == 0.0


def test_tie_neutral_mrr_uses_average_rank_for_ties() -> None:
    scores = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
        ]
    )

    assert tie_neutral_mrr(scores) == pytest.approx((0.5 + 1.0 + 0.4) / 3)


def test_proxy_gate_requires_coverage_full_gain_and_every_slice_gain() -> None:
    assert passes_two_hop_proxy_gate(
        coverage=0.25,
        baseline_mrr=0.30,
        candidate_mrr=0.311,
        baseline_slice_mrrs=[0.28, 0.30, 0.32],
        candidate_slice_mrrs=[0.281, 0.302, 0.321],
    )
    assert not passes_two_hop_proxy_gate(
        coverage=0.19,
        baseline_mrr=0.30,
        candidate_mrr=0.32,
        baseline_slice_mrrs=[0.28, 0.30, 0.32],
        candidate_slice_mrrs=[0.29, 0.31, 0.33],
    )
    assert not passes_two_hop_proxy_gate(
        coverage=0.25,
        baseline_mrr=0.30,
        candidate_mrr=0.32,
        baseline_slice_mrrs=[0.28, 0.30, 0.32],
        candidate_slice_mrrs=[0.29, 0.31, 0.32],
    )
    assert not passes_two_hop_proxy_gate(
        coverage=0.25,
        baseline_mrr=0.30,
        candidate_mrr=0.309,
        baseline_slice_mrrs=[0.28, 0.30, 0.32],
        candidate_slice_mrrs=[0.29, 0.31, 0.33],
    )
