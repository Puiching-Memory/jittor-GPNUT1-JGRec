from __future__ import annotations

import jittor as jt
import numpy as np
import pytest

from jgrec.rankers.hybrid import fusion
from jgrec.rankers.hybrid.fusion import (
    FusionConfig,
    _listwise_positive_loss,
    fit_fusion_mlp_listwise_fixed,
    fit_fusion_mlp_listwise_streaming,
)
from jgrec.rankers.hybrid.setwise import SetwiseFeatureView, setwise_context_features


def _reference_loss(logits: np.ndarray) -> float:
    row_max = logits.max(axis=1, keepdims=True)
    log_sum_exp = row_max[:, 0] + np.log(np.exp(logits - row_max).sum(axis=1))
    return float(np.mean(log_sum_exp - logits[:, 0]))


@pytest.mark.parametrize("candidate_count", [2, 3, 100])
def test_listwise_positive_loss_matches_numpy_reference(candidate_count: int) -> None:
    rng = np.random.default_rng(60 + candidate_count)
    logits = rng.normal(size=(4, candidate_count)).astype(np.float32)

    actual = float(_listwise_positive_loss(jt.array(logits, dtype=jt.float32)).item())

    assert actual == pytest.approx(_reference_loss(logits), abs=1e-6)


def test_listwise_positive_loss_is_invariant_to_per_query_shift() -> None:
    logits = np.asarray([[2.0, 1.0, -0.5], [-4.0, 0.5, 3.0]], dtype=np.float32)
    shifts = np.asarray([[100.0], [-37.0]], dtype=np.float32)

    original = float(_listwise_positive_loss(jt.array(logits, dtype=jt.float32)).item())
    shifted = float(_listwise_positive_loss(jt.array(logits + shifts, dtype=jt.float32)).item())

    assert shifted == pytest.approx(original, abs=1e-5)


def test_listwise_positive_loss_rewards_raising_candidate_zero() -> None:
    uniform = np.zeros((2, 4), dtype=np.float32)
    improved = uniform.copy()
    improved[:, 0] = 2.0

    uniform_loss = float(_listwise_positive_loss(jt.array(uniform, dtype=jt.float32)).item())
    improved_loss = float(_listwise_positive_loss(jt.array(improved, dtype=jt.float32)).item())

    assert uniform_loss == pytest.approx(np.log(4.0), abs=1e-6)
    assert improved_loss < uniform_loss


def test_listwise_positive_loss_supports_query_weights() -> None:
    logits = np.asarray(
        [
            [2.0, 0.0, -1.0],
            [0.0, 2.0, -1.0],
            [-1.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )
    weights = np.asarray([1.0, 2.0, 4.0], dtype=np.float32)
    row_max = logits.max(axis=1, keepdims=True)
    per_query = (
        row_max[:, 0]
        + np.log(np.exp(logits - row_max).sum(axis=1))
        - logits[:, 0]
    )
    expected = float(np.sum(per_query * weights) / np.sum(weights))

    actual = float(
        _listwise_positive_loss(
            jt.array(logits, dtype=jt.float32),
            jt.array(weights, dtype=jt.float32),
        ).item()
    )

    assert actual == pytest.approx(expected, abs=1e-6)


def test_listwise_positive_loss_rejects_invalid_query_weights() -> None:
    logits = jt.array(np.zeros((3, 4), dtype=np.float32), dtype=jt.float32)

    with pytest.raises(ValueError, match="one weight per query"):
        _listwise_positive_loss(
            logits,
            jt.array(np.ones((2,), dtype=np.float32), dtype=jt.float32),
        )


def test_fixed_listwise_trainer_evaluates_validation_only_after_all_epochs(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(60)
    train = rng.normal(size=(6, 3, 2)).astype(np.float32)
    validation = rng.normal(size=(2, 4, 2)).astype(np.float32)
    metric_calls: list[int] = []

    def fake_metrics(*_args: object, **_kwargs: object) -> tuple[float, float]:
        metric_calls.append(1)
        return 0.25, 0.5

    monkeypatch.setattr(fusion, "_metrics_from_model_streaming", fake_metrics)
    _model, result, epoch_losses = fit_fusion_mlp_listwise_fixed(
        train,
        validation,
        FusionConfig(epochs=2, batch_size=3, lr=0.001, hidden_dim=4),
        np.random.default_rng(60),
        verbose=False,
        candidate_name="listwise_test",
    )

    assert metric_calls == [1]
    assert result.best_val_ap == pytest.approx(0.25)
    assert result.best_val_mrr == pytest.approx(0.5)
    assert len(epoch_losses) == 2
    assert np.isfinite(epoch_losses).all()


def test_fixed_listwise_trainer_applies_query_weights_to_normalizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = np.asarray(
        [
            [[1.0], [1.0]],
            [[0.0], [0.0]],
        ],
        dtype=np.float32,
    )
    validation = train.copy()
    row_weights = np.asarray([0.75, 0.25], dtype=np.float32)
    monkeypatch.setattr(
        fusion,
        "_metrics_from_model_streaming",
        lambda *_args, **_kwargs: (0.25, 0.50),
    )

    _model, result, _losses = fit_fusion_mlp_listwise_fixed(
        train,
        validation,
        FusionConfig(epochs=1, batch_size=1, lr=0.001, hidden_dim=4),
        np.random.default_rng(60),
        verbose=False,
        candidate_name="weighted-normalizer-test",
        train_row_weights=row_weights,
    )

    np.testing.assert_allclose(result.mean, [0.75], rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(
        result.std,
        [np.sqrt(0.75 * 0.25)],
        rtol=0.0,
        atol=1e-7,
    )


def test_setwise_context_features_add_row_relative_mean_and_max_channels() -> None:
    values = np.asarray([[[1.0, 4.0], [3.0, 2.0]]], dtype=np.float32)

    transformed = setwise_context_features(values)

    expected = np.asarray(
        [
            [
                [1.0, 4.0, -1.0, 1.0, -2.0, 0.0],
                [3.0, 2.0, 1.0, -1.0, 0.0, -2.0],
            ]
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(transformed, expected)
    view = SetwiseFeatureView(values)
    assert view.shape == (1, 2, 6)
    np.testing.assert_array_equal(view[:], expected)


def test_streaming_listwise_trainer_early_stops_on_full_candidate_mrr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(61)
    train = rng.normal(size=(6, 3, 2)).astype(np.float32)
    validation = rng.normal(size=(2, 4, 2)).astype(np.float32)
    metric_values = iter(
        [
            (0.10, 0.20),
            (0.20, 0.50),
            (0.19, 0.40),
            (0.18, 0.30),
        ]
    )

    monkeypatch.setattr(
        fusion,
        "_metrics_from_model_streaming",
        lambda *_args, **_kwargs: next(metric_values),
    )
    _model, result, history = fit_fusion_mlp_listwise_streaming(
        train,
        validation,
        FusionConfig(
            epochs=8,
            batch_size=3,
            lr=0.001,
            hidden_dim=4,
            selection_metric="mrr",
            early_stop_patience=2,
        ),
        np.random.default_rng(60),
        verbose=False,
        candidate_name="setwise_early_stop_test",
    )

    assert result.best_val_mrr == pytest.approx(0.50)
    assert len(history) == 3
    assert history[0]["epoch"] == 1
    assert history[-1]["patience"] == 2


def test_streaming_listwise_trainer_shuffles_row_weights_with_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(62)
    train = rng.normal(size=(6, 3, 2)).astype(np.float32)
    validation = rng.normal(size=(2, 4, 2)).astype(np.float32)
    row_weights = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float32)
    expected_order = np.random.default_rng(60).permutation(6)
    observed: list[float] = []
    original_loss = fusion._listwise_positive_loss

    def capturing_loss(
        logits: jt.Var,
        query_weights: jt.Var | None = None,
    ) -> jt.Var:
        assert query_weights is not None
        observed.extend(
            np.asarray(query_weights.numpy(), dtype=np.float32).tolist()
        )
        return original_loss(logits, query_weights)

    monkeypatch.setattr(fusion, "_listwise_positive_loss", capturing_loss)
    monkeypatch.setattr(
        fusion,
        "_metrics_from_model_streaming",
        lambda *_args, **_kwargs: (0.25, 0.50),
    )
    fit_fusion_mlp_listwise_streaming(
        train,
        validation,
        FusionConfig(
            epochs=1,
            batch_size=3,
            lr=0.001,
            hidden_dim=4,
            selection_metric="mrr",
            early_stop_patience=2,
        ),
        np.random.default_rng(60),
        verbose=False,
        train_row_weights=row_weights,
    )

    np.testing.assert_array_equal(
        np.asarray(observed, dtype=np.float32),
        row_weights[expected_order],
    )
