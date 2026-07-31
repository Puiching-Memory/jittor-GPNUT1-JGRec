import sys
from types import ModuleType, SimpleNamespace

import numpy as np

from jgrec.core.types import TestQueryArray
from jgrec.rankers.hybrid.ranker import TemporalHybridRanker
from jgrec.rankers.hybrid.time_ramp import (
    apply_time_ramp,
    blend_query_scores,
    passes_time_ramp_gate,
    select_time_ramp_on_prefix,
    time_progress,
    time_ramp_weights,
)


def test_time_ramp_is_monotonic_with_exact_endpoints_and_constant_fallback():
    times = np.asarray([100, 125, 200], dtype=np.int64)

    progress = time_progress(times)
    weights = time_ramp_weights(times, power=2.0)

    np.testing.assert_allclose(progress, [0.0, 0.25, 1.0])
    np.testing.assert_allclose(weights, [0.0, 0.0625, 1.0])
    assert np.all(np.diff(weights) >= 0.0)
    np.testing.assert_allclose(
        time_ramp_weights(
            np.asarray([7, 7, 7], dtype=np.int64),
            power=1.0,
        ),
        0.0,
    )


def test_fixed_global_time_bounds_make_ramp_batch_and_order_invariant():
    times = np.asarray([100, 125, 175, 200], dtype=np.int64)
    full = time_ramp_weights(
        times,
        power=0.5,
        minimum_time=100,
        maximum_time=200,
    )
    scheduled = np.concatenate(
        (
            time_ramp_weights(
                times[[2, 0]],
                power=0.5,
                minimum_time=100,
                maximum_time=200,
            ),
            time_ramp_weights(
                times[[3, 1]],
                power=0.5,
                minimum_time=100,
                maximum_time=200,
            ),
        )
    )

    np.testing.assert_allclose(
        scheduled,
        full[[2, 0, 3, 1]],
    )


def test_ranker_time_ramp_uses_checkpoint_horizon_across_scheduled_batches(
    monkeypatch,
):
    champion_model = object()
    expert_model = object()

    class DummyEncoder:
        def features_for_queries(self, queries):
            return np.zeros((len(queries), 2, 1), dtype=np.float32)

    def fake_predict_logits(model, features, mean, std):
        del mean, std
        logits = [2.0, 0.0] if model is champion_model else [0.0, 2.0]
        return np.tile(logits, (features.shape[0], 1))

    fake_fusion = ModuleType("jgrec.rankers.hybrid.fusion")
    fake_fusion.predict_logits = fake_predict_logits
    monkeypatch.setitem(
        sys.modules,
        "jgrec.rankers.hybrid.fusion",
        fake_fusion,
    )
    ranker = TemporalHybridRanker()
    ranker.encoder = DummyEncoder()
    ranker.fusion = champion_model
    ranker.fusion_result = SimpleNamespace(
        feature_indices=(0,),
        mean=np.zeros(1),
        std=np.ones(1),
    )
    ranker.time_ramp_setwise_fusion = expert_model
    ranker.time_ramp_setwise_result = SimpleNamespace(
        feature_indices=(0, 1, 2),
        mean=np.zeros(3),
        std=np.ones(3),
    )
    ranker.time_ramp_config = {
        "power": 1.0,
        "minimum_time": 100.0,
        "maximum_time": 200.0,
    }
    queries = TestQueryArray(
        src=np.asarray([1, 2, 3, 4]),
        time=np.asarray([100, 125, 175, 200]),
        candidates=np.asarray([[10, 11]] * 4),
    )

    full = ranker.predict_batch(queries)
    first_order = np.asarray([2, 0])
    second_order = np.asarray([3, 1])
    scheduled = np.concatenate(
        (
            ranker.predict_batch(queries[first_order]),
            ranker.predict_batch(queries[second_order]),
        )
    )

    np.testing.assert_allclose(
        scheduled,
        full[np.concatenate((first_order, second_order))],
    )


def test_query_blend_has_exact_endpoints_and_is_candidate_permutation_equivariant():
    champion = np.asarray(
        [[0.7, 0.2, 0.1], [0.4, 0.3, 0.3]],
        dtype=np.float64,
    )
    expert = np.asarray(
        [[0.1, 0.6, 0.3], [0.2, 0.7, 0.1]],
        dtype=np.float64,
    )
    weights = np.asarray([0.0, 1.0], dtype=np.float64)

    blended = blend_query_scores(champion, expert, weights)
    permutation = np.asarray([2, 0, 1], dtype=np.int64)
    permuted = blend_query_scores(
        champion[:, permutation],
        expert[:, permutation],
        weights,
    )

    np.testing.assert_array_equal(blended[0], champion[0])
    np.testing.assert_array_equal(blended[1], expert[1])
    np.testing.assert_allclose(permuted, blended[:, permutation])


def test_prefix_selector_rejects_early_regression_and_never_scores_forward_rows():
    champion = np.asarray(
        [
            [0.60, 0.40],
            [0.55, 0.45],
            [0.40, 0.60],
            [0.40, 0.60],
            [0.10, 0.90],
            [0.10, 0.90],
        ],
        dtype=np.float64,
    )
    expert = np.asarray(
        [
            [0.30, 0.70],
            [0.30, 0.70],
            [0.80, 0.20],
            [0.80, 0.20],
            [0.90, 0.10],
            [0.90, 0.10],
        ],
        dtype=np.float64,
    )
    times = np.arange(6, dtype=np.int64)

    selection = select_time_ramp_on_prefix(
        champion,
        expert,
        times,
        powers=(0.5, 1.0, 2.0),
        first_slice_stop=2,
        selection_stop=4,
        minimum_prefix_delta=0.0,
    )

    assert selection.selected_power == 1.0
    assert selection.selection_rows == (0, 4)
    assert selection.forward_rows == (4, 6)
    assert selection.forward_metrics_read is False
    trials = {trial.power: trial for trial in selection.trials}
    assert trials[0.5].slice_deltas[0] < 0.0
    assert trials[0.5].eligible is False
    assert trials[1.0].eligible is True


def test_no_eligible_ramp_returns_exact_champion_fallback():
    champion = np.asarray(
        [[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.55, 0.45]],
        dtype=np.float64,
    )
    expert = champion[:, ::-1]
    times = np.arange(4, dtype=np.int64)

    selection = select_time_ramp_on_prefix(
        champion,
        expert,
        times,
        powers=(0.5, 1.0, 2.0),
        first_slice_stop=2,
        selection_stop=3,
        minimum_prefix_delta=0.001,
    )
    candidate = apply_time_ramp(
        champion,
        expert,
        times,
        power=selection.selected_power,
    )

    assert selection.selected_power is None
    np.testing.assert_array_equal(candidate, champion)


def test_gate_requires_minimum_full_gain_and_every_slice_non_decreasing():
    champion = np.asarray(
        [[0.4, 0.6], [0.6, 0.4]] * 3,
        dtype=np.float64,
    )
    improved = np.asarray(
        [[0.7, 0.3], [0.6, 0.4]] * 3,
        dtype=np.float64,
    )
    regressed = improved.copy()
    regressed[4] = [0.3, 0.7]
    regressed[5] = [0.3, 0.7]

    accepted = passes_time_ramp_gate(
        champion,
        improved,
        slice_stops=(2, 4),
        minimum_full_delta=0.2,
    )
    rejected = passes_time_ramp_gate(
        champion,
        regressed,
        slice_stops=(2, 4),
        minimum_full_delta=0.0,
    )

    assert accepted.passed is True
    assert all(delta >= 0.0 for delta in accepted.slice_deltas)
    assert rejected.passed is False
    assert rejected.slice_deltas[2] < 0.0
