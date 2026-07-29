from __future__ import annotations

import numpy as np

from jgrec.cooccur_lift_transport_audit import (
    collapse_summary,
    first_layer_lift_intervention_summary,
    popularity_distribution_summary,
    probability_distribution_summary,
    time_support_summary,
    top1_change_summary,
)


def test_collapse_summary_distinguishes_zero_cells_and_zero_rows() -> None:
    short_lift = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    summary = collapse_summary(short_lift, chunk_rows=1)

    assert summary["exact_zero_cells"] == 5
    assert summary["exact_zero_cell_rate"] == 5 / 6
    assert summary["all_exact_zero_rows"] == 1
    assert summary["all_exact_zero_row_rate"] == 0.5


def test_probability_summary_reports_row_max_and_normalized_entropy() -> None:
    probabilities = np.asarray(
        [
            [0.5, 0.5],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )

    summary = probability_distribution_summary(
        probabilities,
        chunk_rows=1,
    )

    assert summary["row_max_probability"]["mean"] == 0.75
    assert summary["normalized_entropy"]["mean"] == 0.5
    assert summary["maximum_row_sum_error"] == 0.0


def test_popularity_summary_uses_common_training_destination_counts() -> None:
    candidates = np.asarray([[0, 1, 2], [2, 3, 0]], dtype=np.int32)
    destination_counts = np.asarray([0, 1, 3, 7], dtype=np.int64)

    summary = popularity_distribution_summary(
        candidates,
        destination_counts,
        chunk_rows=1,
    )

    assert summary["candidate_cells"] == 6
    assert summary["unseen_candidate_rate"] == 2 / 6
    assert summary["raw_train_dst_count"]["mean"] == 14 / 6
    assert summary["raw_train_dst_count"]["maximum"] == 7.0


def test_top1_change_summary_uses_candidate_order_for_ties() -> None:
    champion = np.asarray(
        [[0.6, 0.4], [0.5, 0.5], [0.1, 0.9]],
        dtype=np.float64,
    )
    candidate = np.asarray(
        [[0.4, 0.6], [0.5, 0.5], [0.2, 0.8]],
        dtype=np.float64,
    )

    summary = top1_change_summary(champion, candidate)

    assert summary == {
        "rows": 3,
        "top1_changed_rows": 1,
        "top1_change_rate": 1 / 3,
    }


def test_time_support_separates_frozen_origin_from_full_train_end() -> None:
    summary = time_support_summary(
        train_time=np.asarray([0, 100], dtype=np.int64),
        external_time=np.asarray([20, 100], dtype=np.int64),
        test_time=np.asarray([60, 120, 200], dtype=np.int64),
        frozen_training_time_max=0,
        short_window=50.0,
    )

    assert (
        summary[
            "test_rows_whose_short_window_starts_after_frozen_origin"
        ]
        == 1.0
    )
    assert (
        summary[
            "test_rows_whose_short_window_starts_after_full_train"
        ]
        == 1 / 3
    )


def test_first_layer_attribution_separates_full_and_short_channels() -> None:
    lift = np.asarray(
        [
            [[1.0, 0.0], [3.0, 0.0]],
            [[0.0, 2.0], [0.0, 4.0]],
        ],
        dtype=np.float32,
    )
    weight = np.zeros((1, 195), dtype=np.float32)
    weight[0, 63] = 1.0
    weight[0, 64] = 2.0

    summary = first_layer_lift_intervention_summary(
        lift,
        linear1_weight=weight,
        std=np.ones(195, dtype=np.float32),
        chunk_rows=1,
    )

    assert summary["full"]["preactivation_energy"] == 10.0
    assert summary["short"]["preactivation_energy"] == 80.0
    assert summary["full"]["exactly_zero_row_rate"] == 0.5
    assert summary["short"]["exactly_zero_row_rate"] == 0.5
    assert summary["full_energy_fraction_of_separate_sum"] == 1 / 9
