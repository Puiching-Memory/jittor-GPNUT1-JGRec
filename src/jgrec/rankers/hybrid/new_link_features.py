from __future__ import annotations

import numpy as np

NEW_LINK_GROWTH_FEATURE_NAMES = (
    "target_short_vs_long_growth_log1p_ratio",
    "src_activity_x_target_short_vs_long_growth",
)
_REQUIRED_FEATURE_NAMES = (
    "src_activity",
    "target_pop_share_w001",
    "target_pop_share_w100",
)
_DENOMINATOR_EPSILON = 1e-12


def append_new_link_growth_features(
    features: np.ndarray,
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Append two leakage-safe growth transforms derived from cached features."""
    values = np.asarray(features)
    if values.ndim < 2 or values.shape[-1] != len(feature_names):
        raise ValueError("feature names must align with the final feature dimension")
    missing = tuple(name for name in _REQUIRED_FEATURE_NAMES if name not in feature_names)
    if missing:
        raise ValueError(f"required cached features are missing: {', '.join(missing)}")

    src_activity = values[..., feature_names.index("src_activity")].astype(np.float64, copy=False)
    short_share = values[..., feature_names.index("target_pop_share_w001")].astype(np.float64, copy=False)
    long_share = values[..., feature_names.index("target_pop_share_w100")].astype(np.float64, copy=False)
    ratio = short_share / np.maximum(long_share, _DENOMINATOR_EPSILON)
    growth = np.log1p(np.maximum(ratio, 0.0))

    output = np.empty((*values.shape[:-1], values.shape[-1] + 2), dtype=np.float32)
    output[..., : values.shape[-1]] = values
    output[..., -2] = growth.astype(np.float32, copy=False)
    output[..., -1] = (src_activity * growth).astype(np.float32, copy=False)
    return output, feature_names + NEW_LINK_GROWTH_FEATURE_NAMES
