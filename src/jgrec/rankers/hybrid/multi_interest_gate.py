from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np

from .segment_fusion import (
    QUERY_SEGMENT_FEATURE_NAMES,
    query_segment_features,
)

_PROXY_SUMMARY_STATS = ("mean", "max", "std")
EXPERT_SCORE_DESCRIPTOR_NAMES = (
    "champion_top_margin",
    "candidate_top_margin",
    "champion_entropy",
    "candidate_entropy",
    "expert_top1_agreement",
    "expert_abs_difference_mean",
    "expert_abs_difference_max",
)
MULTI_INTEREST_GATE_DESCRIPTOR_NAMES = (
    *QUERY_SEGMENT_FEATURE_NAMES,
    *EXPERT_SCORE_DESCRIPTOR_NAMES,
    *(
        f"proxy_{feature_index}_{stat}"
        for feature_index in range(9)
        for stat in _PROXY_SUMMARY_STATS
    ),
)


@dataclass(frozen=True)
class ConfidenceGateConfig:
    max_depth: int
    min_samples_leaf: int
    minimum_predicted_lift: float


@dataclass(frozen=True)
class ConfidenceGateOOFResult:
    champion_mrr: float
    candidate_mrr: float
    gated_mrr: float
    full_delta: float
    fold_deltas: tuple[float, ...]
    slice_deltas: tuple[float, ...]
    fold_coverage: tuple[float, ...]
    use_candidate: np.ndarray
    predicted_lift: np.ndarray


@dataclass(frozen=True)
class ConfidenceGateModel:
    model_bytes: bytes
    config: ConfidenceGateConfig
    descriptor_names: tuple[str, ...]


def route_query_experts(
    champion_scores: np.ndarray,
    candidate_scores: np.ndarray,
    use_candidate: np.ndarray,
) -> np.ndarray:
    champion = np.asarray(champion_scores)
    candidate = np.asarray(candidate_scores)
    selection = np.asarray(use_candidate, dtype=bool)
    if (
        champion.shape != candidate.shape
        or champion.ndim != 2
        or champion.shape[1] < 2
    ):
        raise ValueError(
            "expert scores must share a query-by-candidate shape"
        )
    if selection.shape != (champion.shape[0],):
        raise ValueError("query gate must contain one decision per query")
    return np.where(selection[:, None], candidate, champion)


def confidence_gate_descriptors(
    base_features: np.ndarray,
    feature_names: tuple[str, ...],
    proxy_features: np.ndarray,
    champion_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
) -> np.ndarray:
    base = np.asarray(base_features)
    proxy = np.asarray(proxy_features)
    champion = np.asarray(champion_probabilities, dtype=np.float64)
    candidate = np.asarray(candidate_probabilities, dtype=np.float64)
    if base.ndim != 3 or proxy.ndim != 3:
        raise ValueError("gate features must be query-by-candidate tensors")
    if base.shape[:2] != proxy.shape[:2] or proxy.shape[2] != 9:
        raise ValueError("multi-interest proxy features have an incompatible shape")
    expected_scores_shape = base.shape[:2]
    if (
        champion.shape != expected_scores_shape
        or candidate.shape != expected_scores_shape
    ):
        raise ValueError("expert probabilities must align with gate features")

    segment = query_segment_features(base, feature_names)
    score_descriptors = expert_score_descriptors(champion, candidate)
    proxy_descriptors = np.stack(
        (
            proxy.mean(axis=1),
            proxy.max(axis=1),
            proxy.std(axis=1),
        ),
        axis=-1,
    ).reshape(proxy.shape[0], -1)
    output = np.column_stack(
        (segment, score_descriptors, proxy_descriptors)
    )
    if output.shape[1] != len(MULTI_INTEREST_GATE_DESCRIPTOR_NAMES):
        raise RuntimeError("multi-interest gate descriptor schema mismatch")
    return np.asarray(output, dtype=np.float32)


def expert_score_descriptors(
    champion_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
) -> np.ndarray:
    champion = np.asarray(champion_probabilities, dtype=np.float64)
    candidate = np.asarray(candidate_probabilities, dtype=np.float64)
    if (
        champion.shape != candidate.shape
        or champion.ndim != 2
        or champion.shape[1] < 2
    ):
        raise ValueError(
            "expert probabilities must share a query-by-candidate shape"
        )
    difference = np.abs(champion - candidate)
    return np.asarray(
        np.column_stack(
            (
                _top_margin(champion),
                _top_margin(candidate),
                _entropy(champion),
                _entropy(candidate),
                np.argmax(champion, axis=1) == np.argmax(candidate, axis=1),
                difference.mean(axis=1),
                difference.max(axis=1),
            )
        ),
        dtype=np.float32,
    )


def blocked_temporal_folds(
    row_count: int,
    *,
    fold_count: int = 3,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    if fold_count < 2 or row_count < fold_count:
        raise ValueError("blocked OOF requires at least two non-empty folds")
    indices = np.arange(row_count, dtype=np.int64)
    held_out_folds = tuple(np.asarray(part) for part in np.array_split(indices, fold_count))
    return tuple(
        (
            np.concatenate(
                [
                    fold
                    for other_index, fold in enumerate(held_out_folds)
                    if other_index != fold_index
                ]
            ),
            test_indices,
        )
        for fold_index, test_indices in enumerate(held_out_folds)
    )


def blocked_temporal_oof_gate(
    descriptors: np.ndarray,
    champion_scores: np.ndarray,
    candidate_scores: np.ndarray,
    config: ConfidenceGateConfig,
    *,
    fold_count: int = 3,
    seed: int = 60,
) -> ConfidenceGateOOFResult:
    from sklearn.tree import DecisionTreeRegressor  # noqa: PLC0415

    inputs = np.asarray(descriptors, dtype=np.float32)
    champion = np.asarray(champion_scores, dtype=np.float64)
    candidate = np.asarray(candidate_scores, dtype=np.float64)
    if inputs.ndim != 2 or inputs.shape[0] != champion.shape[0]:
        raise ValueError("gate descriptors must contain one row per query")
    if champion.shape != candidate.shape or champion.ndim != 2:
        raise ValueError("expert scores must share a query-by-candidate shape")
    if config.max_depth <= 0 or config.min_samples_leaf <= 0:
        raise ValueError("confidence gate tree parameters must be positive")

    champion_rr = reciprocal_ranks(champion)
    candidate_rr = reciprocal_ranks(candidate)
    rewards = candidate_rr - champion_rr
    decisions = np.zeros(champion.shape[0], dtype=bool)
    predicted_lift = np.zeros(champion.shape[0], dtype=np.float64)
    fold_deltas: list[float] = []
    fold_coverage: list[float] = []
    folds = blocked_temporal_folds(
        champion.shape[0],
        fold_count=fold_count,
    )
    for fold_index, (train_indices, test_indices) in enumerate(folds):
        model = DecisionTreeRegressor(
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            random_state=seed + fold_index,
        )
        model.fit(inputs[train_indices], rewards[train_indices])
        fold_predictions = np.asarray(
            model.predict(inputs[test_indices]),
            dtype=np.float64,
        )
        fold_decisions = (
            fold_predictions >= config.minimum_predicted_lift
        )
        predicted_lift[test_indices] = fold_predictions
        decisions[test_indices] = fold_decisions
        routed_rr = np.where(
            fold_decisions,
            candidate_rr[test_indices],
            champion_rr[test_indices],
        )
        fold_deltas.append(
            float(routed_rr.mean() - champion_rr[test_indices].mean())
        )
        fold_coverage.append(float(fold_decisions.mean()))

    routed = route_query_experts(champion, candidate, decisions)
    champion_mrr = float(champion_rr.mean())
    candidate_mrr = float(candidate_rr.mean())
    gated_mrr = float(reciprocal_ranks(routed).mean())
    slice_deltas = tuple(
        float(
            reciprocal_ranks(routed[indices]).mean()
            - champion_rr[indices].mean()
        )
        for indices in np.array_split(
            np.arange(champion.shape[0], dtype=np.int64),
            3,
        )
    )
    return ConfidenceGateOOFResult(
        champion_mrr=champion_mrr,
        candidate_mrr=candidate_mrr,
        gated_mrr=gated_mrr,
        full_delta=gated_mrr - champion_mrr,
        fold_deltas=tuple(fold_deltas),
        slice_deltas=slice_deltas,
        fold_coverage=tuple(fold_coverage),
        use_candidate=decisions,
        predicted_lift=predicted_lift,
    )


def reciprocal_ranks(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("scores must contain grouped candidate queries")
    ranks = 1 + np.sum(values[:, 1:] > values[:, :1], axis=1)
    return 1.0 / ranks


def passes_stability_gate(
    *,
    full_delta: float,
    fold_deltas: tuple[float, ...],
    slice_deltas: tuple[float, ...],
    minimum_full_delta: float,
) -> bool:
    return bool(
        full_delta + 1e-12 >= minimum_full_delta
        and all(delta >= 0.0 for delta in fold_deltas)
        and all(delta >= 0.0 for delta in slice_deltas)
    )


def select_stable_high_confidence_trial(
    trials: list[dict[str, Any]],
    *,
    minimum_predicted_lift: float,
    maximum_coverage: float,
) -> int | None:
    if not 0.0 <= maximum_coverage <= 1.0:
        raise ValueError("maximum gate coverage must be between zero and one")
    eligible = [
        index
        for index, trial in enumerate(trials)
        if bool(trial["passed"])
        and float(trial["config"]["minimum_predicted_lift"])
        >= minimum_predicted_lift
        and float(trial["coverage"]) <= maximum_coverage
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda index: (
            min(float(value) for value in trials[index]["fold_deltas"]),
            float(trials[index]["full_delta"]),
            -float(trials[index]["coverage"]),
        ),
    )


def fit_confidence_gate(
    descriptors: np.ndarray,
    rewards: np.ndarray,
    config: ConfidenceGateConfig,
    *,
    descriptor_names: tuple[str, ...],
    seed: int,
) -> ConfidenceGateModel:
    from sklearn.tree import DecisionTreeRegressor  # noqa: PLC0415

    inputs = np.asarray(descriptors, dtype=np.float32)
    targets = np.asarray(rewards, dtype=np.float64)
    if inputs.ndim != 2 or inputs.shape[1] != len(descriptor_names):
        raise ValueError("confidence gate descriptor schema differs")
    if targets.shape != (inputs.shape[0],) or not np.all(np.isfinite(targets)):
        raise ValueError("confidence gate rewards must contain one finite value per query")
    model = DecisionTreeRegressor(
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        random_state=seed,
    )
    model.fit(inputs, targets)
    return ConfidenceGateModel(
        model_bytes=pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL),
        config=config,
        descriptor_names=tuple(descriptor_names),
    )


def predict_confidence_gate(
    model: ConfidenceGateModel,
    descriptors: np.ndarray,
    *,
    descriptor_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    inputs = np.asarray(descriptors, dtype=np.float32)
    if (
        tuple(descriptor_names) != model.descriptor_names
        or inputs.ndim != 2
        or inputs.shape[1] != len(model.descriptor_names)
    ):
        raise ValueError("confidence gate descriptor contract differs")
    estimator = pickle.loads(model.model_bytes)
    predicted_lift = np.asarray(estimator.predict(inputs), dtype=np.float64)
    return (
        predicted_lift >= model.config.minimum_predicted_lift,
        predicted_lift,
    )


def _top_margin(probabilities: np.ndarray) -> np.ndarray:
    top_two = np.partition(probabilities, -2, axis=1)[:, -2:]
    return top_two.max(axis=1) - top_two.min(axis=1)


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)
