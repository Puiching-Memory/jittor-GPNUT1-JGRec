from __future__ import annotations

import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from jgrec.cooccur_lift_successor_execution import (
    build_deterministic_replay_report,
    resolve_bugfixed_v1_fold_baseline,
    validate_successor_execution_contract,
)
from jgrec.rankers.hybrid.cooccur_lift_successor import (
    ConcatenatedFeatureView,
    CooccurLiftFullOnlyView,
    CooccurLiftGapAwareView,
    short_window_support,
)


def _fixtures() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.arange(2 * 3 * 63, dtype=np.float32).reshape(2, 3, 63)
    short_none = np.asarray(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        dtype=np.float32,
    )
    lift = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ],
        dtype=np.float32,
    )
    return base, short_none, lift


def test_full_only_view_removes_short_and_has_64_raw_columns() -> None:
    base, short_none, lift = _fixtures()

    view = CooccurLiftFullOnlyView(
        base,
        short_none_scores=short_none,
        gnn_short_column=59,
        lift_features=lift,
    )
    actual = view[:]

    assert view.shape == (2, 3, 64)
    np.testing.assert_array_equal(actual[..., 59], short_none)
    np.testing.assert_array_equal(actual[..., 63], lift[..., 0])


def test_gap_aware_view_broadcasts_row_support_and_has_66_raw_columns() -> None:
    base, short_none, lift = _fixtures()
    support = np.asarray([1.0, 0.0], dtype=np.float32)

    view = CooccurLiftGapAwareView(
        base,
        short_none_scores=short_none,
        gnn_short_column=59,
        lift_features=lift,
        short_window_supported=support,
    )
    actual = view[:]

    assert view.shape == (2, 3, 66)
    np.testing.assert_array_equal(actual[..., 63:65], lift)
    np.testing.assert_array_equal(actual[0, :, 65], 1.0)
    np.testing.assert_array_equal(actual[1, :, 65], 0.0)


def test_short_window_support_uses_strict_boundary() -> None:
    query_time = np.asarray([100, 101, 102], dtype=np.int64)
    availability_time = np.asarray([90, 90, 91], dtype=np.int64)

    actual = short_window_support(
        query_time,
        availability_time,
        short_window_seconds=11,
    )

    np.testing.assert_array_equal(actual, [1.0, 0.0, 0.0])


def test_concatenated_feature_view_preserves_numpy_index_order() -> None:
    left = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    right = 100 + np.arange(3 * 3 * 4, dtype=np.float32).reshape(3, 3, 4)
    view = ConcatenatedFeatureView((left, right))
    expected = np.concatenate((left, right), axis=0)

    assert view.shape == (5, 3, 4)
    np.testing.assert_array_equal(view[:], expected)
    np.testing.assert_array_equal(view[np.asarray([4, 0, 2])], expected[[4, 0, 2]])


def _execution_contract() -> dict:
    return {
        "schema_version": 1,
        "status": "frozen_before_successor_v2_metrics",
        "experiment_id": "dataset2_cooccur_lift_successor_v2_duel_20260729",
        "candidate_ids": [
            "cooccur_lift_full_only_v2",
            "cooccur_lift_gap_aware_v2",
        ],
        "training_device": "cpu",
        "internal_scoring_device": "cpu",
        "historical_near_v1_manifest_role": "diagnostic_only",
        "deterministic_replay_gate": {
            "runs": 2,
            "rtol": 0.00002,
            "atol": 0.000002,
            "tolerance_relaxation_authorized": False,
        },
        "external_authorized": False,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload.update(training_device="cuda"),
            "training device",
        ),
        (
            lambda payload: payload["deterministic_replay_gate"].update(runs=1),
            "two runs",
        ),
        (
            lambda payload: payload["deterministic_replay_gate"].update(
                rtol=0.1
            ),
            "tolerance",
        ),
        (
            lambda payload: payload.update(
                historical_near_v1_manifest_role="authoritative_replay"
            ),
            "diagnostic",
        ),
        (
            lambda payload: payload.update(external_authorized=True),
            "external",
        ),
    ],
)
def test_successor_execution_contract_rejects_nondeterministic_or_unfrozen_paths(
    mutation,
    message: str,
) -> None:
    contract = copy.deepcopy(_execution_contract())
    mutation(contract)

    with pytest.raises(ValueError, match=message):
        validate_successor_execution_contract(contract)


def test_bugfixed_v1_baseline_uses_cpu_replay_not_legacy_cuda_scores() -> None:
    first_predictions = {
        "score": np.asarray([[0.8, 0.2]], dtype=np.float32),
        "zero_short": np.asarray([[0.7, 0.3]], dtype=np.float32),
    }
    second_predictions = {
        name: values.copy() for name, values in first_predictions.items()
    }
    replay = build_deterministic_replay_report(
        first_state={"weight": np.asarray([1.0, 2.0], dtype=np.float32)},
        second_state={"weight": np.asarray([1.0, 2.0], dtype=np.float32)},
        first_losses=(2.0, 1.0),
        second_losses=(2.0, 1.0),
        first_predictions=first_predictions,
        second_predictions=second_predictions,
        rtol=0.00002,
        atol=0.000002,
    )
    prior = np.asarray([[0.4, 0.6]], dtype=np.float32)
    legacy_cuda_v1 = np.asarray([[0.5, 0.5]], dtype=np.float32)

    evidence = resolve_bugfixed_v1_fold_baseline(
        prior_baseline=prior,
        cpu_auxiliary=first_predictions["score"],
        legacy_cuda_v1=legacy_cuda_v1,
        replay=replay,
        weight=0.5,
    )

    np.testing.assert_allclose(evidence["baseline"], [[0.6, 0.4]])
    assert evidence["deterministic_replay"]["matched"] is True
    assert evidence["legacy_cuda_max_abs_error"] == pytest.approx(0.1)
    assert evidence["legacy_cuda_role"] == "diagnostic_only"


def test_bugfixed_v1_baseline_rejects_cpu_replay_drift_without_relaxing_gate() -> None:
    replay = build_deterministic_replay_report(
        first_state={"weight": np.asarray([1.0], dtype=np.float32)},
        second_state={"weight": np.asarray([1.0], dtype=np.float32)},
        first_losses=(1.0,),
        second_losses=(1.0,),
        first_predictions={
            "score": np.asarray([[0.8, 0.2]], dtype=np.float32)
        },
        second_predictions={
            "score": np.asarray([[0.7, 0.3]], dtype=np.float32)
        },
        rtol=0.00002,
        atol=0.000002,
    )

    with pytest.raises(ValueError, match="deterministic replay"):
        resolve_bugfixed_v1_fold_baseline(
            prior_baseline=np.asarray([[0.4, 0.6]], dtype=np.float32),
            cpu_auxiliary=np.asarray([[0.8, 0.2]], dtype=np.float32),
            legacy_cuda_v1=np.asarray([[0.5, 0.5]], dtype=np.float32),
            replay=replay,
            weight=0.5,
        )


def test_runner_binds_cache_report_only_to_gapped_training() -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "train_dataset2_cooccur_lift_successor_v2_duel.py"
    )
    tree = ast.parse(runner.read_text(encoding="utf-8"))
    calls = {
        call.func.id: {keyword.arg for keyword in call.keywords}
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id
        in {"_train_near_folds", "_train_gapped_folds"}
    }

    assert "expected_cache_report_sha256" not in calls["_train_near_folds"]
    assert "expected_cache_report_sha256" in calls["_train_gapped_folds"]
