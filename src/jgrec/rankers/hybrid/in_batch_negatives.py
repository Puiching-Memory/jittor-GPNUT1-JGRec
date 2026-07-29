from __future__ import annotations

import numpy as np


def _in_batch_positive_mask(positive_dst_ids: np.ndarray) -> np.ndarray:
    destination_ids = np.asarray(positive_dst_ids)
    if destination_ids.ndim != 1 or destination_ids.size == 0:
        raise ValueError("positive_dst_ids must be a non-empty vector")
    return destination_ids[:, None] == destination_ids[None, :]
