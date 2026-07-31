from __future__ import annotations

import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .multi_interest_gate import reciprocal_ranks

_SELF_DESCRIPTOR_SUFFIXES = (
    "top1_score",
    "top_margin",
    "entropy",
    "top3_mass",
    "top5_mass",
)
_PAIR_DESCRIPTOR_SUFFIXES = (
    "top1_jaccard",
    "top3_jaccard",
    "top5_jaccard",
)


@dataclass(frozen=True)
class MultiExpertRoutingResult:
    scores: np.ndarray
    selected_experts: np.ndarray
    use_alternative: np.ndarray
    predicted_lift: np.ndarray


@dataclass(frozen=True)
class MultiExpertGateConfig:
    max_depth: int
    min_samples_leaf: int
    minimum_predicted_lift: float


@dataclass(frozen=True)
class MultiExpertGateModel:
    model_bytes: tuple[bytes, ...]
    config: MultiExpertGateConfig
    descriptor_names: tuple[str, ...]
    expert_order: tuple[str, ...]


@dataclass(frozen=True)
class MultiExpertForwardSelection:
    model: MultiExpertGateModel
    fallback_mrr: float
    selection_mrr: float
    delta: float
    coverage: float


def expert_top1_feature_deltas(
    expert_scores: Mapping[str, np.ndarray],
    candidate_features: np.ndarray,
    *,
    candidate_feature_names: tuple[str, ...],
    selected_feature_names: tuple[str, ...],
    fallback_expert: str,
    alternative_order: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Compare features attached to each expert's tie-neutral top1 set."""

    if (
        not alternative_order
        or fallback_expert in alternative_order
        or len(set(alternative_order)) != len(alternative_order)
    ):
        raise ValueError(
            "alternative expert order must contain unique non-fallback names"
        )
    expert_order = (fallback_expert, *alternative_order)
    scores = _validated_expert_scores(
        expert_scores,
        expert_order=expert_order,
        normalize=False,
    )

    features = np.asarray(candidate_features)
    if (
        features.ndim != 3
        or features.shape[:2] != scores[fallback_expert].shape
        or features.shape[2] != len(candidate_feature_names)
    ):
        raise ValueError(
            "candidate features must align with scores and feature names"
        )
    if len(set(candidate_feature_names)) != len(candidate_feature_names):
        raise ValueError("candidate feature names must be unique")
    if (
        not selected_feature_names
        or len(set(selected_feature_names)) != len(selected_feature_names)
    ):
        raise ValueError("selected feature names must be non-empty and unique")

    feature_index = {
        name: index for index, name in enumerate(candidate_feature_names)
    }
    missing = [
        name for name in selected_feature_names if name not in feature_index
    ]
    if missing:
        raise ValueError(f"selected candidate features are missing: {missing}")
    selected_indices = tuple(
        feature_index[name] for name in selected_feature_names
    )

    fallback_means = _top1_feature_means(
        scores[fallback_expert],
        features,
        selected_indices=selected_indices,
    )
    columns: list[np.ndarray] = []
    names: list[str] = []
    for alternative_name in alternative_order:
        alternative_means = _top1_feature_means(
            scores[alternative_name],
            features,
            selected_indices=selected_indices,
        )
        columns.append(alternative_means - fallback_means)
        names.extend(
            (
                f"{alternative_name}__vs__{fallback_expert}"
                f"_top1_delta__{feature_name}"
            )
            for feature_name in selected_feature_names
        )
    return (
        np.concatenate(columns, axis=1).astype(np.float32, copy=False),
        tuple(names),
    )


def multi_expert_score_descriptors(
    expert_scores: Mapping[str, np.ndarray],
    *,
    expert_order: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build tie-neutral query descriptors from aligned expert scores."""

    normalized = _validated_expert_scores(
        expert_scores,
        expert_order=expert_order,
    )
    columns: list[np.ndarray] = []
    names: list[str] = []
    for expert_name in expert_order:
        scores = normalized[expert_name]
        ordered = np.sort(scores, axis=1)
        columns.extend(
            (
                ordered[:, -1],
                ordered[:, -1] - ordered[:, -2],
                _entropy(scores),
                ordered[:, -min(3, scores.shape[1]) :].sum(axis=1),
                ordered[:, -min(5, scores.shape[1]) :].sum(axis=1),
            )
        )
        names.extend(
            f"{expert_name}_{suffix}"
            for suffix in _SELF_DESCRIPTOR_SUFFIXES
        )

    for left_name, right_name in combinations(expert_order, 2):
        left = normalized[left_name]
        right = normalized[right_name]
        pair_prefix = f"{left_name}__{right_name}"
        for top_k, suffix in zip(
            (1, 3, 5),
            _PAIR_DESCRIPTOR_SUFFIXES,
            strict=True,
        ):
            columns.append(
                _top_k_jaccard(left, right, top_k=top_k)
            )
            names.append(f"{pair_prefix}_{suffix}")

        left_top = _top_k_mask(left, top_k=1)
        right_top = _top_k_mask(right, top_k=1)
        columns.extend(
            (
                left.max(axis=1) - _masked_row_mean(left, right_top),
                right.max(axis=1) - _masked_row_mean(right, left_top),
                _mean_rank_of_mask(left, right_top),
                _mean_rank_of_mask(right, left_top),
                np.mean(np.abs(left - right), axis=1, dtype=np.float64),
                np.abs(left - right).max(axis=1),
            )
        )
        names.extend(
            (
                f"{pair_prefix}_{left_name}_own_top_preference",
                f"{pair_prefix}_{right_name}_own_top_preference",
                f"{pair_prefix}_{left_name}_rank_of_{right_name}_top1",
                f"{pair_prefix}_{right_name}_rank_of_{left_name}_top1",
                f"{pair_prefix}_mean_abs_difference",
                f"{pair_prefix}_max_abs_difference",
            )
        )

    return (
        np.column_stack(columns).astype(np.float32, copy=False),
        tuple(names),
    )


def route_multi_expert(
    fallback_scores: np.ndarray,
    alternative_scores: Mapping[str, np.ndarray],
    predicted_lifts: np.ndarray,
    *,
    expert_order: tuple[str, ...],
    minimum_predicted_lift: float,
) -> MultiExpertRoutingResult:
    """Route each query to its best confident expert or exact fallback."""

    fallback = _score_matrix(fallback_scores, label="fallback")
    alternatives = _validated_expert_scores(
        alternative_scores,
        expert_order=expert_order,
        expected_shape=fallback.shape,
        normalize=False,
    )
    lifts = np.asarray(predicted_lifts, dtype=np.float32)
    expected_lift_shape = (fallback.shape[0], len(expert_order))
    if lifts.shape != expected_lift_shape:
        raise ValueError(
            "predicted lifts must contain one value per query and expert"
        )
    if not np.all(np.isfinite(lifts)):
        raise ValueError("predicted lifts must be finite")
    if not np.isfinite(minimum_predicted_lift):
        raise ValueError("minimum predicted lift must be finite")

    best_indices = np.argmax(lifts, axis=1)
    best_lifts = lifts[np.arange(lifts.shape[0]), best_indices]
    use_alternative = best_lifts >= minimum_predicted_lift
    routed = fallback.copy()
    selected = np.full(fallback.shape[0], "current_gate", dtype=object)
    for expert_index, expert_name in enumerate(expert_order):
        mask = use_alternative & (best_indices == expert_index)
        routed[mask] = alternatives[expert_name][mask]
        selected[mask] = expert_name
    return MultiExpertRoutingResult(
        scores=routed,
        selected_experts=selected.astype(str),
        use_alternative=use_alternative,
        predicted_lift=best_lifts,
    )


def fit_multi_expert_gate(
    descriptors: np.ndarray,
    rewards: np.ndarray,
    config: MultiExpertGateConfig,
    *,
    descriptor_names: tuple[str, ...],
    expert_order: tuple[str, ...],
    seed: int,
) -> MultiExpertGateModel:
    """Fit one shallow reward regressor per alternative expert."""

    from sklearn.tree import DecisionTreeRegressor  # noqa: PLC0415

    inputs = _descriptor_matrix(
        descriptors,
        descriptor_names=descriptor_names,
    )
    targets = np.asarray(rewards, dtype=np.float64)
    expected_shape = (inputs.shape[0], len(expert_order))
    if targets.shape != expected_shape or not np.all(np.isfinite(targets)):
        raise ValueError("rewards must contain one finite value per expert")
    _validate_gate_contract(config, expert_order=expert_order)

    model_bytes: list[bytes] = []
    for expert_index in range(len(expert_order)):
        estimator = DecisionTreeRegressor(
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            random_state=seed + expert_index,
        )
        estimator.fit(inputs, targets[:, expert_index])
        model_bytes.append(
            pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL)
        )
    return MultiExpertGateModel(
        model_bytes=tuple(model_bytes),
        config=config,
        descriptor_names=tuple(descriptor_names),
        expert_order=tuple(expert_order),
    )


def predict_multi_expert_gate(
    model: MultiExpertGateModel,
    descriptors: np.ndarray,
    *,
    descriptor_names: tuple[str, ...],
) -> np.ndarray:
    """Predict one relative reward for every query and alternative expert."""

    if tuple(descriptor_names) != model.descriptor_names:
        raise ValueError("multi-expert descriptor contract differs")
    inputs = _descriptor_matrix(
        descriptors,
        descriptor_names=descriptor_names,
    )
    if len(model.model_bytes) != len(model.expert_order):
        raise ValueError("multi-expert model count differs from expert order")
    predictions = [
        np.asarray(
            pickle.loads(model_bytes).predict(inputs),
            dtype=np.float32,
        )
        for model_bytes in model.model_bytes
    ]
    output = np.column_stack(predictions).astype(np.float32, copy=False)
    if not np.all(np.isfinite(output)):
        raise FloatingPointError("multi-expert predicted lift is non-finite")
    return output


def select_multi_expert_config_on_forward_slice(
    descriptors: np.ndarray,
    fallback_scores: np.ndarray,
    alternative_scores: Mapping[str, np.ndarray],
    *,
    configs: tuple[MultiExpertGateConfig, ...],
    descriptor_names: tuple[str, ...],
    expert_order: tuple[str, ...],
    train_rows: tuple[int, int],
    selection_rows: tuple[int, int],
    minimum_selection_delta: float,
    maximum_coverage: float,
    seed: int,
) -> MultiExpertForwardSelection | None:
    """Fit before the selection slice and choose without reading later rows."""

    if not configs:
        raise ValueError("at least one multi-expert config is required")
    if not 0.0 <= maximum_coverage <= 1.0:
        raise ValueError("maximum coverage must be between zero and one")
    _validate_forward_rows(train_rows, selection_rows)

    inputs = np.asarray(descriptors)
    fallback_values = np.asarray(fallback_scores)
    alternatives_values = {
        name: np.asarray(values)
        for name, values in alternative_scores.items()
    }
    train_slice = slice(*train_rows)
    selection_slice = slice(*selection_rows)
    train_descriptors = inputs[train_slice]
    selection_descriptors = inputs[selection_slice]
    train_fallback = fallback_values[train_slice]
    selection_fallback = fallback_values[selection_slice]
    train_alternatives = {
        name: values[train_slice]
        for name, values in alternatives_values.items()
    }
    selection_alternatives = {
        name: values[selection_slice]
        for name, values in alternatives_values.items()
    }

    fallback_train_rr = reciprocal_ranks(train_fallback)
    rewards = np.column_stack(
        [
            reciprocal_ranks(train_alternatives[name])
            - fallback_train_rr
            for name in expert_order
        ]
    )
    fallback_mrr = float(np.mean(reciprocal_ranks(selection_fallback)))
    best: MultiExpertForwardSelection | None = None
    best_key: tuple[float, float, int] | None = None
    for config_index, config in enumerate(configs):
        model = fit_multi_expert_gate(
            train_descriptors,
            rewards,
            config,
            descriptor_names=descriptor_names,
            expert_order=expert_order,
            seed=seed,
        )
        predicted_lifts = predict_multi_expert_gate(
            model,
            selection_descriptors,
            descriptor_names=descriptor_names,
        )
        routed = route_multi_expert(
            selection_fallback,
            selection_alternatives,
            predicted_lifts,
            expert_order=expert_order,
            minimum_predicted_lift=config.minimum_predicted_lift,
        )
        selection_mrr = float(np.mean(reciprocal_ranks(routed.scores)))
        delta = selection_mrr - fallback_mrr
        coverage = float(np.mean(routed.use_alternative))
        if (
            delta + 1e-12 < minimum_selection_delta
            or coverage > maximum_coverage + 1e-12
        ):
            continue
        key = (delta, -coverage, -config_index)
        if best_key is None or key > best_key:
            best_key = key
            best = MultiExpertForwardSelection(
                model=model,
                fallback_mrr=fallback_mrr,
                selection_mrr=selection_mrr,
                delta=delta,
                coverage=coverage,
            )
    return best


def _validated_expert_scores(
    expert_scores: Mapping[str, np.ndarray],
    *,
    expert_order: tuple[str, ...],
    expected_shape: tuple[int, int] | None = None,
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    if not expert_order or len(set(expert_order)) != len(expert_order):
        raise ValueError("expert order must contain unique names")
    if set(expert_scores) != set(expert_order):
        raise ValueError("expert scores must exactly match expert order")
    output: dict[str, np.ndarray] = {}
    shape = expected_shape
    for expert_name in expert_order:
        scores = _score_matrix(
            expert_scores[expert_name],
            label=expert_name,
        )
        if shape is None:
            shape = scores.shape
        if scores.shape != shape:
            raise ValueError("all expert score matrices must align")
        output[expert_name] = (
            _normalize_rows(scores) if normalize else scores
        )
    return output


def _descriptor_matrix(
    descriptors: np.ndarray,
    *,
    descriptor_names: tuple[str, ...],
) -> np.ndarray:
    inputs = np.asarray(descriptors, dtype=np.float32)
    if (
        inputs.ndim != 2
        or inputs.shape[1] != len(descriptor_names)
        or not np.all(np.isfinite(inputs))
    ):
        raise ValueError("descriptors must match their finite schema")
    return inputs


def _validate_gate_contract(
    config: MultiExpertGateConfig,
    *,
    expert_order: tuple[str, ...],
) -> None:
    if (
        config.max_depth < 1
        or config.min_samples_leaf < 1
        or not np.isfinite(config.minimum_predicted_lift)
    ):
        raise ValueError("multi-expert gate config is invalid")
    if not expert_order or len(set(expert_order)) != len(expert_order):
        raise ValueError("expert order must contain unique names")


def _validate_forward_rows(
    train_rows: tuple[int, int],
    selection_rows: tuple[int, int],
) -> None:
    train_start, train_stop = train_rows
    selection_start, selection_stop = selection_rows
    if not (
        0 <= train_start < train_stop <= selection_start < selection_stop
    ):
        raise ValueError("forward rows must be ordered and non-empty")


def _score_matrix(values: np.ndarray, *, label: str) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float32)
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError(f"{label} scores must be query-by-candidate")
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"{label} scores must be finite")
    if np.any(scores < 0.0):
        raise ValueError(f"{label} scores must be non-negative")
    if np.any(scores.sum(axis=1) <= 0.0):
        raise ValueError(f"{label} scores must have positive row mass")
    return scores


def _normalize_rows(scores: np.ndarray) -> np.ndarray:
    values = scores.astype(np.float64)
    return values / np.sum(values, axis=1, keepdims=True)


def _entropy(scores: np.ndarray) -> np.ndarray:
    safe_scores = np.maximum(scores, np.finfo(np.float64).tiny)
    return -np.sum(
        np.where(scores > 0.0, scores * np.log(safe_scores), 0.0),
        axis=1,
        dtype=np.float64,
    )


def _top_k_mask(scores: np.ndarray, *, top_k: int) -> np.ndarray:
    k = min(top_k, scores.shape[1])
    threshold = np.partition(scores, scores.shape[1] - k, axis=1)[
        :, scores.shape[1] - k
    ]
    return scores >= threshold[:, None]


def _top_k_jaccard(
    left: np.ndarray,
    right: np.ndarray,
    *,
    top_k: int,
) -> np.ndarray:
    left_mask = _top_k_mask(left, top_k=top_k)
    right_mask = _top_k_mask(right, top_k=top_k)
    intersection = np.sum(left_mask & right_mask, axis=1)
    union = np.sum(left_mask | right_mask, axis=1)
    return intersection / union


def _masked_row_mean(
    values: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    return np.sum(
        values * mask,
        axis=1,
        dtype=np.float64,
    ) / np.sum(mask, axis=1)


def _top1_feature_means(
    scores: np.ndarray,
    candidate_features: np.ndarray,
    *,
    selected_indices: tuple[int, ...],
) -> np.ndarray:
    mask = _top_k_mask(scores, top_k=1)
    counts = np.sum(mask, axis=1)
    output = np.empty(
        (scores.shape[0], len(selected_indices)),
        dtype=np.float64,
    )
    for output_index, feature_index in enumerate(selected_indices):
        values = candidate_features[:, :, feature_index]
        if not np.all(np.isfinite(values)):
            raise ValueError("selected candidate features must be finite")
        output[:, output_index] = np.sum(
            values * mask,
            axis=1,
            dtype=np.float64,
        ) / counts
    return output


def _mean_rank_of_mask(
    ranking_scores: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    order = np.argsort(-ranking_scores, axis=1, kind="stable")
    ordered_scores = np.take_along_axis(ranking_scores, order, axis=1)
    positions = np.arange(ranking_scores.shape[1], dtype=np.int32)
    group_starts = np.zeros_like(order, dtype=np.int32)
    group_starts[:, 0] = 0
    group_starts[:, 1:] = np.where(
        ordered_scores[:, 1:] != ordered_scores[:, :-1],
        positions[1:],
        0,
    )
    ordered_ranks = np.maximum.accumulate(group_starts, axis=1)
    ranks = np.empty_like(ordered_ranks)
    np.put_along_axis(ranks, order, ordered_ranks, axis=1)
    denominator = max(1, ranking_scores.shape[1] - 1)
    return _masked_row_mean(
        ranks.astype(np.float32) / denominator,
        selected,
    )
