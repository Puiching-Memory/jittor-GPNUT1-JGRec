from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ServiceNormalizerResult:
    count: int
    mean: np.ndarray
    std: np.ndarray


class StreamingFeatureNormalizer:
    def __init__(self, *, minimum_std: float = 1e-6) -> None:
        minimum_std = float(minimum_std)
        if not np.isfinite(minimum_std) or minimum_std <= 0.0:
            raise ValueError("minimum_std must be finite and positive")
        self._minimum_std = minimum_std
        self._count = 0
        self._mean: np.ndarray | None = None
        self._m2: np.ndarray | None = None

    @property
    def count(self) -> int:
        return self._count

    def update(self, features: Any) -> None:
        values = np.asarray(features)
        if values.ndim < 2:
            raise ValueError("service normalizer features must have at least two dimensions")
        feature_dim = int(values.shape[-1])
        if feature_dim <= 0:
            raise ValueError("service normalizer feature dimension must be positive")
        flat = values.reshape((-1, feature_dim)).astype(
            np.float64,
            copy=False,
        )
        if flat.shape[0] == 0:
            return
        if not np.isfinite(flat).all():
            raise ValueError("service normalizer features must be finite")
        if self._mean is not None and self._mean.shape != (feature_dim,):
            raise ValueError("service normalizer feature dimension changed between batches")

        batch_count = int(flat.shape[0])
        batch_mean = flat.mean(axis=0)
        centered = flat - batch_mean
        batch_m2 = np.sum(centered * centered, axis=0)
        if self._mean is None or self._m2 is None:
            self._count = batch_count
            self._mean = batch_mean
            self._m2 = batch_m2
            return

        total_count = self._count + batch_count
        delta = batch_mean - self._mean
        self._mean = self._mean + delta * (float(batch_count) / float(total_count))
        self._m2 = self._m2 + batch_m2 + delta * delta * (float(self._count) * float(batch_count) / float(total_count))
        self._count = total_count

    def finalize(self) -> ServiceNormalizerResult:
        if self._count <= 0 or self._mean is None or self._m2 is None:
            raise ValueError("service normalizer received no feature rows")
        variance = np.maximum(self._m2 / float(self._count), 0.0)
        mean = self._mean.astype(np.float32)
        std = np.sqrt(variance).astype(np.float32)
        std[std < self._minimum_std] = 1.0
        return ServiceNormalizerResult(
            count=self._count,
            mean=mean,
            std=std,
        )


def replace_result_normalizer(
    result: Any,
    service: ServiceNormalizerResult,
) -> Any:
    if not is_dataclass(result):
        raise TypeError("normalized result must be a dataclass instance")
    training_mean = np.asarray(getattr(result, "mean", None))
    training_std = np.asarray(getattr(result, "std", None))
    service_mean = np.asarray(service.mean, dtype=np.float32)
    service_std = np.asarray(service.std, dtype=np.float32)
    if (
        training_mean.ndim != 1
        or training_std.shape != training_mean.shape
        or service_mean.shape != training_mean.shape
        or service_std.shape != training_mean.shape
    ):
        raise ValueError("service normalizer feature dimension does not match result")
    if (
        service.count <= 0
        or not np.isfinite(service_mean).all()
        or not np.isfinite(service_std).all()
        or np.any(service_std <= 0.0)
    ):
        raise ValueError("service normalizer must be finite, positive, and non-empty")
    return replace(
        result,
        mean=service_mean.copy(),
        std=service_std.copy(),
    )


def normalizer_drift_report(
    *,
    training_mean: Any,
    training_std: Any,
    service: ServiceNormalizerResult,
) -> dict[str, int | float]:
    old_mean = np.asarray(training_mean, dtype=np.float64)
    old_std = np.asarray(training_std, dtype=np.float64)
    new_mean = np.asarray(service.mean, dtype=np.float64)
    new_std = np.asarray(service.std, dtype=np.float64)
    expected_shape = new_mean.shape
    if old_mean.ndim != 1 or old_std.shape != expected_shape or new_mean.ndim != 1 or new_std.shape != expected_shape:
        raise ValueError("training and service normalizers must use equal vectors")
    if (
        not np.isfinite(old_mean).all()
        or not np.isfinite(old_std).all()
        or not np.isfinite(new_mean).all()
        or not np.isfinite(new_std).all()
        or np.any(old_std <= 0.0)
        or np.any(new_std <= 0.0)
    ):
        raise ValueError("training and service normalizers must be finite and positive")
    standardized_mean_shift = np.abs(new_mean - old_mean) / old_std
    std_ratio = new_std / old_std
    return {
        "count": int(service.count),
        "feature_dim": int(new_mean.shape[0]),
        "max_abs_mean_shift": float(np.max(np.abs(new_mean - old_mean), initial=0.0)),
        "max_abs_mean_shift_in_training_std": float(np.max(standardized_mean_shift, initial=0.0)),
        "max_service_to_training_std_ratio": float(np.max(std_ratio, initial=0.0)),
        "min_service_to_training_std_ratio": float(np.min(std_ratio, initial=np.inf)),
    }
