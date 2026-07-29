from __future__ import annotations

import numpy as np

from jgrec.rankers.hybrid.window_diversity import (
    select_temporally_robust_candidate_on_prefix,
)

CANDIDATE_ORDER = (
    "v1_champion",
    "v2_relative",
    "v1_v2_uniform",
)


def _scores_for_ranks(*ranks: int) -> np.ndarray:
    rows = {
        1: np.asarray([3.0, 2.0, 1.0]),
        2: np.asarray([2.0, 3.0, 1.0]),
        3: np.asarray([1.0, 3.0, 2.0]),
    }
    return np.stack([rows[rank] for rank in ranks], axis=0)


def test_temporal_robust_selection_rejects_candidate_that_hurts_second_slice() -> None:
    champion = _scores_for_ranks(1, 2, 1, 3, 1, 1)
    aggressive = _scores_for_ranks(1, 1, 2, 2, 1, 1)
    robust = _scores_for_ranks(1, 2, 1, 2, 1, 1)
    candidates = {
        "v1_champion": champion,
        "v2_relative": aggressive,
        "v1_v2_uniform": robust,
    }
    candidates_with_hidden_forward = {
        name: np.concatenate(
            (
                scores[:4],
                np.full_like(scores[4:], np.nan),
            ),
            axis=0,
        )
        for name, scores in candidates.items()
    }

    result = select_temporally_robust_candidate_on_prefix(
        candidates_with_hidden_forward,
        champion,
        first_slice_stop=2,
        selection_stop=4,
        candidate_complexity={
            "v1_champion": 1,
            "v2_relative": 1,
            "v1_v2_uniform": 2,
        },
        candidate_order=CANDIDATE_ORDER,
    )

    reports = {candidate.name: candidate for candidate in result.candidates}
    assert reports["v2_relative"].selection_mrr > reports[
        "v1_champion"
    ].selection_mrr
    assert reports["v2_relative"].eligible is False
    assert reports["v1_v2_uniform"].eligible is True
    assert result.selected_name == "v1_v2_uniform"


def test_temporal_robust_selection_ties_prefer_fewer_models_then_frozen_order() -> None:
    champion = _scores_for_ranks(1, 2, 1, 2, 1, 1)
    candidates = {
        name: champion.copy()
        for name in CANDIDATE_ORDER
    }

    result = select_temporally_robust_candidate_on_prefix(
        candidates,
        champion,
        first_slice_stop=2,
        selection_stop=4,
        candidate_complexity={
            "v1_champion": 1,
            "v2_relative": 1,
            "v1_v2_uniform": 2,
        },
        candidate_order=CANDIDATE_ORDER,
    )

    assert result.selected_name == "v1_champion"
