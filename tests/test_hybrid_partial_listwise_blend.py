from __future__ import annotations

import numpy as np
import pytest

from jgrec.rankers.hybrid.partial_listwise_blend import (
    ALLOWED_AUXILIARY_WEIGHTS,
    blend_partial_listwise,
    choose_forward_winner,
    descending_midrank_probabilities,
    evaluate_final_gate,
    evaluate_forward_gate,
    scan_auxiliary_weights,
    select_auxiliary_weight,
    selection_lock_sha256,
)


def test_partial_blend_uses_frozen_residual_formula_and_weight_grid() -> None:
    champion = np.array([[0.7, 0.2, 0.1]], dtype=np.float64)
    expert = np.array([[0.1, 0.6, 0.3]], dtype=np.float64)

    actual = blend_partial_listwise(champion, expert, auxiliary_weight=0.3)

    np.testing.assert_allclose(
        actual,
        0.7 * champion + 0.3 * expert,
    )
    with pytest.raises(ValueError, match="frozen auxiliary-weight grid"):
        blend_partial_listwise(champion, expert, auxiliary_weight=0.07)
    with pytest.raises(ValueError, match="same shape"):
        blend_partial_listwise(
            champion,
            np.ones((2, 3)),
            auxiliary_weight=0.3,
        )
    invalid = expert.copy()
    invalid[0, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        blend_partial_listwise(champion, invalid, auxiliary_weight=0.3)


def test_descending_midrank_transform_is_tie_neutral_and_row_normalized() -> None:
    scores = np.array(
        [
            [9.0, 4.0, 4.0, 1.0],
            [3.0, 3.0, 3.0, 3.0],
        ],
        dtype=np.float64,
    )

    probabilities = descending_midrank_probabilities(scores)

    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities[0, 0] > probabilities[0, 1]
    assert probabilities[0, 1] == pytest.approx(probabilities[0, 2])
    assert probabilities[0, 2] > probabilities[0, 3]
    np.testing.assert_allclose(probabilities[1], np.full(4, 0.25))


def test_slice0_selector_uses_smaller_weight_for_an_exact_metric_tie() -> None:
    champion = np.array(
        [
            [0.6, 0.3, 0.1],
            [0.6, 0.3, 0.1],
        ],
        dtype=np.float64,
    )
    expert = champion.copy()

    selection = select_auxiliary_weight(
        expert_name="listwise_mlp",
        champion_slice0=champion,
        expert_slice0=expert,
        candidate_manifest_sha256="candidates",
        weights=ALLOWED_AUXILIARY_WEIGHTS,
    )

    assert selection["selected_weight"] == pytest.approx(0.05)
    assert selection["selection_slice"] == "slice_0"
    assert len(selection["trials"]) == len(ALLOWED_AUXILIARY_WEIGHTS)
    assert all(trial["delta"] == pytest.approx(0.0) for trial in selection["trials"])
    assert selection["lock_sha256"] == selection_lock_sha256(selection)


def test_slice0_scan_preserves_all_rejected_weight_trials() -> None:
    champion = np.array([[0.51, 0.49]], dtype=np.float64)
    expert = np.array([[0.0, 1.0]], dtype=np.float64)

    scan = scan_auxiliary_weights(
        champion_slice0=champion,
        expert_slice0=expert,
        weights=(0.05, 0.1),
    )

    assert scan["baseline_mrr"] == pytest.approx(1.0)
    assert [trial["weight"] for trial in scan["trials"]] == [0.05, 0.1]
    assert all(trial["delta"] < 0.0 for trial in scan["trials"])
    assert scan["eligible_weight_count"] == 0


def test_forward_gate_keeps_weight_locked_and_rejects_slice1_regression() -> None:
    champion0 = np.array([[0.6, 0.3, 0.1]], dtype=np.float64)
    expert0 = np.array([[0.7, 0.2, 0.1]], dtype=np.float64)
    champion1 = np.array([[0.6, 0.3, 0.1]], dtype=np.float64)
    expert1 = np.array([[0.2, 0.7, 0.1]], dtype=np.float64)
    selection = select_auxiliary_weight(
        expert_name="listwise_mlp",
        champion_slice0=champion0,
        expert_slice0=expert0,
        candidate_manifest_sha256="candidates",
        weights=(0.5,),
    )

    report = evaluate_forward_gate(
        selection=selection,
        champion_slice0=champion0,
        expert_slice0=expert0,
        champion_slice1=champion1,
        expert_slice1=expert1,
        candidate_manifest_sha256="candidates",
        minimum_prefix_delta=0.0,
    )

    assert report["selected_weight"] == pytest.approx(0.5)
    assert report["slice_1_delta"] < 0.0
    assert report["passed"] is False


def test_forward_winner_prefers_prefix_mrr_then_smaller_weight_then_name() -> None:
    reports = [
        {
            "expert_name": "listwise_two_tower",
            "passed": True,
            "prefix_candidate_mrr": 0.55,
            "selected_weight": 0.2,
            "selection_lock_sha256": "tower",
        },
        {
            "expert_name": "listwise_mlp",
            "passed": True,
            "prefix_candidate_mrr": 0.55,
            "selected_weight": 0.1,
            "selection_lock_sha256": "mlp",
        },
    ]

    winner = choose_forward_winner(reports)

    assert winner["expert_name"] == "listwise_mlp"
    assert winner["selected_weight"] == pytest.approx(0.1)


def test_final_gate_requires_lock_hash_and_all_three_slices() -> None:
    champion = np.tile(
        np.array([[0.6, 0.3, 0.1]], dtype=np.float64),
        (6, 1),
    )
    expert = champion.copy()
    selection = select_auxiliary_weight(
        expert_name="listwise_mlp",
        champion_slice0=champion[:2],
        expert_slice0=expert[:2],
        candidate_manifest_sha256="candidates",
        weights=(0.3,),
    )
    forward = evaluate_forward_gate(
        selection=selection,
        champion_slice0=champion[:2],
        expert_slice0=expert[:2],
        champion_slice1=champion[2:4],
        expert_slice1=expert[2:4],
        candidate_manifest_sha256="candidates",
        minimum_prefix_delta=0.0,
    )

    with pytest.raises(ValueError, match="selection-lock hash"):
        evaluate_final_gate(
            selection=selection,
            forward_report=forward,
            champion_scores=champion,
            expert_scores=expert,
            slices=((0, 2), (2, 4), (4, 6)),
            expected_selection_lock_sha256="wrong",
            minimum_full_delta=0.0,
        )

    report = evaluate_final_gate(
        selection=selection,
        forward_report=forward,
        champion_scores=champion,
        expert_scores=expert,
        slices=((0, 2), (2, 4), (4, 6)),
        expected_selection_lock_sha256=selection["lock_sha256"],
        minimum_full_delta=0.0,
    )
    assert report["passed"] is True
    assert report["all_slices_non_decreasing"] is True
    assert report["full_delta"] == pytest.approx(0.0)
