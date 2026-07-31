from __future__ import annotations

import numpy as np


def _in_batch_positive_destination_columns(
    destination_ids: np.ndarray,
    popularity_buckets: np.ndarray,
    recency_buckets: np.ndarray,
    time_buckets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    columns = tuple(
        np.asarray(values)
        for values in (
            destination_ids,
            popularity_buckets,
            recency_buckets,
            time_buckets,
        )
    )
    shape = columns[0].shape
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(
            "in-batch destination inputs must be non-empty matrices"
        )
    if any(values.shape != shape for values in columns[1:]):
        raise ValueError("in-batch destination inputs must align")
    return tuple(values[:, 0] for values in columns)


def _in_batch_positive_mask(positive_dst_ids: np.ndarray) -> np.ndarray:
    destination_ids = np.asarray(positive_dst_ids)
    if destination_ids.ndim != 1 or destination_ids.size == 0:
        raise ValueError("positive_dst_ids must be a non-empty vector")
    return destination_ids[:, None] == destination_ids[None, :]
