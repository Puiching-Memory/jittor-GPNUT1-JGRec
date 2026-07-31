from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.rankers.common.temporal_index import TemporalInteractionIndex

COOCCUR_LIFT_FEATURE_NAMES = ("cooccur_lift_full", "cooccur_lift_short")
INTEGRATION_ID = "cooccur_lift_aux_expert_v1"
FROZEN_STATUS = "frozen_before_any_cooccur_lift_metric"
FROZEN_WEIGHTS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
FROZEN_FOLDS = (
    ("fold-0", (0, 79_909), (79_909, 118_816)),
    ("fold-1", (0, 118_816), (118_816, 159_804)),
    ("fold-2", (0, 159_804), (159_804, 200_000)),
)
HISTORY_LIMIT = 64
SHORT_WINDOW_RATIO = 0.05
BASE_FEATURE_COUNT = 63
AUGMENTED_FEATURE_COUNT = 65
APPENDED_FEATURE_INDICES = (63, 64)
CONTEXT_TRANSFORM_VERSION = 1
CONTEXT_FEATURE_COUNT = 195
BASE_SEED = 60
SEED_SALT = 30_013
SEED_STRIDE = 1_009
SETWISE_EPOCHS = 4
BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0
EARLY_STOP_PATIENCE = 0
SELECTION_METRIC = "mrr"

_FEATURE_FORMULA = {
    "cooccur_lift_full": "log1p(N_full) - log1p(pop_full)",
    "cooccur_lift_short": "log1p(N_short) - log1p(pop_short)",
}
_INDEX_BUILD = {
    "build_transitions": False,
    "build_cooccurs": True,
    "cooccur_history_limit": 256,
    "future_only_transition_cooccur": False,
    "cooccur_time_decay_ratio": 0.0,
}
_AUXILIARY_HEAD = {
    "type": "listwise_setwise",
    "context_transform_version": 1,
    "context_feature_count": 195,
    "feature_indices": "tuple(range(195))",
    "hidden_dim": "champion setwise_hidden_dim",
    "epochs": 4,
    "batch_size": 256,
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "early_stop_patience": 0,
    "selection_metric": "mrr",
}


class FrozenCooccurLiftConfigError(ValueError):
    """Raised when the preregistered cooccurrence-lift contract drifts."""


@dataclass(frozen=True)
class FrozenCooccurLiftConfig:
    integration_id: str
    weights: tuple[float, ...]
    folds: tuple[tuple[str, tuple[int, int], tuple[int, int]], ...]
    history_limit: int
    short_window_ratio: float
    cooccur_history_limit: int
    appended_feature_indices: tuple[int, int]
    context_feature_count: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    early_stop_patience: int
    selection_metric: str
    base_seed: int
    seed_salt: int

    @property
    def fold_boundaries(self) -> tuple[tuple[int, int, int], ...]:
        return tuple((train_rows[0], train_rows[1], score_rows[1]) for _, train_rows, score_rows in self.folds)

    @property
    def context_feature_indices(self) -> tuple[int, ...]:
        return tuple(range(self.context_feature_count))


class CooccurLiftAugmentedView:
    """Read-only 63-column cache overlay with two appended lift columns."""

    def __init__(
        self,
        source: Any,
        *,
        short_none_scores: np.ndarray,
        gnn_short_column: int,
        lift_features: np.ndarray,
    ) -> None:
        if len(source.shape) != 3 or int(source.shape[-1]) != BASE_FEATURE_COUNT:
            raise ValueError("source cache must have shape [rows, candidates, 63]")
        matrix_shape = tuple(int(value) for value in source.shape[:2])
        if tuple(short_none_scores.shape) != matrix_shape:
            raise ValueError("short_none scores must match source rows and candidates")
        if tuple(lift_features.shape) != (*matrix_shape, 2):
            raise ValueError("lift features must have shape [rows, candidates, 2]")
        column = int(gnn_short_column)
        if not 0 <= column < BASE_FEATURE_COUNT:
            raise ValueError("gnn_short_column is outside the 63 base columns")
        self._source = source
        self._short_none_scores = short_none_scores
        self._gnn_short_column = column
        self._lift_features = lift_features
        self.shape = (*matrix_shape, AUGMENTED_FEATURE_COUNT)
        self.ndim = 3
        self.size = int(np.prod(self.shape, dtype=np.int64))

    def __getitem__(self, key: Any) -> np.ndarray:
        base = np.array(self._source[key], dtype=np.float32, copy=True)
        base[..., self._gnn_short_column] = self._short_none_scores[key]
        lift = np.asarray(self._lift_features[key], dtype=np.float32)
        return np.concatenate((base, lift), axis=-1, dtype=np.float32)


def cooccur_lift_scores(
    index: TemporalInteractionIndex,
    src: int,
    candidates: np.ndarray,
    query_time: int,
    *,
    availability_time: int | None = None,
    short_window: float,
) -> np.ndarray:
    candidate_values = np.asarray(candidates)
    if candidate_values.ndim != 1:
        raise ValueError("candidates must be a one-dimensional array")
    if not np.issubdtype(candidate_values.dtype, np.integer):
        raise ValueError("candidates must contain integers")
    if not np.isfinite(short_window) or short_window <= 0:
        raise ValueError("short_window must be positive and finite")
    if not index.built_cooccurs or index.future_only:
        raise ValueError("lift requires causal cooccur_times, not future-only counts")
    if index.cooccur_decay_enabled:
        raise ValueError("cooccur time decay must remain disabled")

    query_time_int = int(query_time)
    availability_time_int = (
        query_time_int
        if availability_time is None
        else int(availability_time)
    )
    if availability_time_int > query_time_int:
        raise ValueError("availability_time must not exceed query_time")
    history = _latest_unique(
        index.source_view(int(src), availability_time_int).visible_dsts,
        HISTORY_LIMIT,
    )
    output = np.empty((len(candidate_values), 2), dtype=np.float32)
    lower_time = float(query_time_int) - float(short_window)
    short_supported = float(availability_time_int) > lower_time
    for position, candidate in enumerate(candidate_values):
        candidate_int = int(candidate)
        full_cooccurs = 0
        short_cooccurs = 0
        for seen_dst in history:
            times = index.cooccur_times.get((int(seen_dst), candidate_int))
            if times is None:
                continue
            upper = int(
                np.searchsorted(
                    times,
                    availability_time_int,
                    side="left",
                )
            )
            full_cooccurs += upper
            if short_supported:
                lower = int(np.searchsorted(times, lower_time, side="right"))
                short_cooccurs += max(upper - lower, 0)

        destination = index.destination_view(
            candidate_int,
            availability_time_int,
        )
        full_popularity = int(destination.cutoff)
        visible_times = destination.times[: destination.cutoff]
        short_popularity = (
            int(
                len(visible_times)
                - np.searchsorted(
                    visible_times,
                    lower_time,
                    side="right",
                )
            )
            if short_supported
            else 0
        )
        output[position, 0] = np.log1p(full_cooccurs) - np.log1p(full_popularity)
        output[position, 1] = np.log1p(short_cooccurs) - np.log1p(short_popularity)
    return output


def load_frozen_cooccur_lift_config(
    source: Path | str | Mapping[str, Any],
) -> FrozenCooccurLiftConfig:
    payload = _load_payload(source)
    _require_equal(payload, "status", FROZEN_STATUS)
    _require_equal(payload, "integration_id", INTEGRATION_ID)
    _require_equal(payload, "weights", list(FROZEN_WEIGHTS))
    _require_equal(payload, "feature_names", list(COOCCUR_LIFT_FEATURE_NAMES))
    _require_equal(payload, "feature_formula", _FEATURE_FORMULA)
    _require_equal(payload, "history_limit", HISTORY_LIMIT)
    _require_equal(payload, "short_window_ratio", SHORT_WINDOW_RATIO)
    _require_equal(payload, "index_build", _INDEX_BUILD)
    _require_equal(payload, "appended_feature_indices", [63, 64])
    _require_equal(payload, "base_feature_count", BASE_FEATURE_COUNT)
    _require_equal(payload, "augmented_feature_count", AUGMENTED_FEATURE_COUNT)
    _require_equal(payload, "auxiliary_head", _AUXILIARY_HEAD)
    _require_equal(payload, "base_seed", BASE_SEED)
    _require_equal(payload, "seed_salt", SEED_SALT)
    _require_equal(payload, "training_seed_formula", "base_seed + fold_index * 1009 + seed_salt")
    _require_equal(payload, "external_open_limit", 1)
    expected_folds = [
        {
            "fold_id": fold_id,
            "train_rows": list(train_rows),
            "score_rows": list(score_rows),
        }
        for fold_id, train_rows, score_rows in FROZEN_FOLDS
    ]
    if payload.get("folds") != expected_folds:
        raise FrozenCooccurLiftConfigError(f"folds differ from frozen boundaries {expected_folds}")
    prohibited = payload.get("prohibited")
    required_prohibitions = {
        "formula or window changes after metrics",
        "reusing cooccur_time_decay_score",
    }
    if not isinstance(prohibited, list) or not required_prohibitions.issubset(prohibited):
        raise FrozenCooccurLiftConfigError("cooccur_time_decay_score and post-metric drift must be prohibited")
    return FrozenCooccurLiftConfig(
        integration_id=INTEGRATION_ID,
        weights=FROZEN_WEIGHTS,
        folds=FROZEN_FOLDS,
        history_limit=HISTORY_LIMIT,
        short_window_ratio=SHORT_WINDOW_RATIO,
        cooccur_history_limit=_INDEX_BUILD["cooccur_history_limit"],
        appended_feature_indices=APPENDED_FEATURE_INDICES,
        context_feature_count=CONTEXT_FEATURE_COUNT,
        epochs=SETWISE_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        early_stop_patience=EARLY_STOP_PATIENCE,
        selection_metric=SELECTION_METRIC,
        base_seed=BASE_SEED,
        seed_salt=SEED_SALT,
    )


def training_seed(config: FrozenCooccurLiftConfig, fold_index: int) -> int:
    if not 0 <= fold_index < len(config.folds):
        raise IndexError(f"fold_index out of range: {fold_index}")
    return config.base_seed + fold_index * SEED_STRIDE + config.seed_salt


def _latest_unique(values: np.ndarray, limit: int) -> np.ndarray:
    selected: list[int] = []
    seen: set[int] = set()
    for value in np.asarray(values)[::-1]:
        value_int = int(value)
        if value_int in seen:
            continue
        seen.add(value_int)
        selected.append(value_int)
        if len(selected) >= limit:
            break
    selected.reverse()
    return np.asarray(selected, dtype=np.int32)


def _load_payload(source: Path | str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FrozenCooccurLiftConfigError(f"cannot read frozen config: {path}") from error
    if not isinstance(payload, dict):
        raise FrozenCooccurLiftConfigError("frozen config must be a JSON object")
    return payload


def _require_equal(
    payload: Mapping[str, Any],
    key: str,
    expected: Any,
) -> None:
    if payload.get(key) != expected:
        raise FrozenCooccurLiftConfigError(f"{key} differs from frozen value {expected!r}")
