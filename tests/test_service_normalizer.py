from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from jgrec.service_normalizer import (
    ServiceNormalizerResult,
    StreamingFeatureNormalizer,
    normalizer_drift_report,
    replace_result_normalizer,
)


@dataclass(frozen=True)
class _NormalizedResult:
    state: dict[str, np.ndarray]
    feature_indices: tuple[int, ...]
    mean: np.ndarray
    std: np.ndarray
    marker: str


def test_streaming_service_normalizer_matches_direct_moments() -> None:
    features = np.asarray(
        [
            [[1.0, 10.0, 7.0], [2.0, 20.0, 7.0]],
            [[3.0, 30.0, 7.0], [4.0, 40.0, 7.0]],
            [[5.0, 50.0, 7.0], [6.0, 60.0, 7.0]],
        ],
        dtype=np.float32,
    )
    normalizer = StreamingFeatureNormalizer()

    normalizer.update(features[:1])
    normalizer.update(features[1:])
    result = normalizer.finalize()

    flat = features.reshape((-1, features.shape[-1]))
    np.testing.assert_allclose(result.mean[:2], flat.mean(axis=0)[:2])
    np.testing.assert_allclose(result.std[:2], flat.std(axis=0)[:2])
    assert result.std[2] == pytest.approx(1.0)
    assert result.count == 6

    standardized = (features - result.mean) / result.std
    np.testing.assert_allclose(
        standardized.reshape((-1, 3)).mean(axis=0),
        np.zeros(3),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        standardized.reshape((-1, 3)).std(axis=0)[:2],
        np.ones(2),
        atol=1e-6,
    )


def test_streaming_service_normalizer_is_batch_order_invariant() -> None:
    rng = np.random.default_rng(60)
    features = rng.normal(
        loc=np.asarray([100.0, -4.0, 0.5]),
        scale=np.asarray([20.0, 0.2, 3.0]),
        size=(31, 7, 3),
    ).astype(np.float32)
    forward = StreamingFeatureNormalizer()
    reverse = StreamingFeatureNormalizer()

    for batch in np.array_split(features, 5):
        forward.update(batch)
    for batch in reversed(np.array_split(features, 5)):
        reverse.update(batch)

    forward_result = forward.finalize()
    reverse_result = reverse.finalize()
    np.testing.assert_allclose(
        forward_result.mean,
        reverse_result.mean,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        forward_result.std,
        reverse_result.std,
        rtol=1e-6,
        atol=1e-6,
    )


def test_streaming_service_normalizer_rejects_invalid_batches() -> None:
    normalizer = StreamingFeatureNormalizer()

    with pytest.raises(ValueError, match="at least two dimensions"):
        normalizer.update(np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        normalizer.update(np.asarray([[[1.0, np.nan]]], dtype=np.float32))

    normalizer.update(np.ones((2, 3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="feature dimension"):
        normalizer.update(np.ones((1, 3, 5), dtype=np.float32))


def test_normalizer_drift_report_quantifies_training_service_shift() -> None:
    service = StreamingFeatureNormalizer()
    service.update(
        np.asarray(
            [[[10.0, 4.0], [18.0, 8.0]]],
            dtype=np.float32,
        )
    )
    service_result = service.finalize()

    report = normalizer_drift_report(
        training_mean=np.asarray([10.0, 5.0], dtype=np.float32),
        training_std=np.asarray([2.0, 1.0], dtype=np.float32),
        service=service_result,
    )

    assert report["count"] == 2
    assert report["feature_dim"] == 2
    assert report["max_abs_mean_shift_in_training_std"] == pytest.approx(2.0)
    assert report["max_service_to_training_std_ratio"] == pytest.approx(2.0)
    assert report["min_service_to_training_std_ratio"] == pytest.approx(2.0)


def test_replace_result_normalizer_preserves_model_contract() -> None:
    state = {"weight": np.asarray([[1.0, 2.0]], dtype=np.float32)}
    original = _NormalizedResult(
        state=state,
        feature_indices=(3, 7),
        mean=np.asarray([10.0, 20.0], dtype=np.float32),
        std=np.asarray([2.0, 4.0], dtype=np.float32),
        marker="locked-head",
    )
    service = ServiceNormalizerResult(
        count=200,
        mean=np.asarray([12.0, 24.0], dtype=np.float32),
        std=np.asarray([3.0, 5.0], dtype=np.float32),
    )

    calibrated = replace_result_normalizer(original, service)

    assert calibrated is not original
    assert calibrated.state is state
    assert calibrated.feature_indices == original.feature_indices
    assert calibrated.marker == original.marker
    np.testing.assert_array_equal(calibrated.mean, service.mean)
    np.testing.assert_array_equal(calibrated.std, service.std)
    assert calibrated.mean is not service.mean
    assert calibrated.std is not service.std

    with pytest.raises(ValueError, match="feature dimension"):
        replace_result_normalizer(
            original,
            ServiceNormalizerResult(
                count=200,
                mean=np.asarray([1.0], dtype=np.float32),
                std=np.asarray([1.0], dtype=np.float32),
            ),
        )


def test_rejected_service_calibration_is_opt_in_by_default() -> None:
    from jgrec.rankers.hybrid.config import TrainingConfig  # noqa: PLC0415

    assert TrainingConfig().service_normalizer_calibration_enabled is False
