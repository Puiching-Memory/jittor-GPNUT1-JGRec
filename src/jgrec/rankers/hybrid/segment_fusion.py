from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np

QUERY_SEGMENT_FEATURE_NAMES = (
    "repeat_candidate_fraction",
    "repeat_strength_max",
    "repeat_rate_max",
    "source_activity",
    "source_recency",
    "target_unseen_fraction",
    "target_popularity_mean",
    "target_popularity_max",
    "prior_strength_mean",
    "prior_strength_max",
    "prior_strength_top_gap",
    "memory_recent_hit_fraction",
    "memory_recent_hit_max",
    "memory_short_max",
    "memory_long_max",
    "memory_short_minus_long",
)

_REQUIRED_FEATURE_NAMES = (
    "pair_strength",
    "repeat_rate",
    "dst_popularity",
    "recent_hit",
    "src_activity",
    "src_recency",
    "candidate_train_seen",
    "candidate_test_freq",
    "pair_decay_short",
    "pair_decay_long",
)


@dataclass(frozen=True)
class SegmentGateResult:
    model_bytes: bytes
    descriptor_names: tuple[str, ...]
    candidate_weights: tuple[float, ...]
    global_weight: float
    name: str
    max_depth: int
    min_samples_leaf: int


@dataclass(frozen=True)
class MRRPolicyTree:
    estimator: object
    leaf_weight_indices: dict[int, int]

    def predict(self, descriptors: np.ndarray) -> np.ndarray:
        leaf_ids = np.asarray(self.estimator.apply(descriptors), dtype=np.int64)
        try:
            return np.asarray(
                [self.leaf_weight_indices[int(leaf_id)] for leaf_id in leaf_ids],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError("segment policy tree emitted an unknown leaf") from error


def query_segment_features(features: np.ndarray, feature_names: tuple[str, ...]) -> np.ndarray:
    values = np.asarray(features)
    if values.ndim != 3 or values.shape[1] < 2:
        raise ValueError("segment fusion requires a query-by-candidate feature tensor")
    indices = {name: index for index, name in enumerate(feature_names)}
    missing = [name for name in _REQUIRED_FEATURE_NAMES if name not in indices]
    if missing:
        raise ValueError(f"segment fusion features are missing: {', '.join(missing)}")

    pair_strength = values[..., indices["pair_strength"]]
    repeat_rate = values[..., indices["repeat_rate"]]
    dst_popularity = values[..., indices["dst_popularity"]]
    recent_hit = values[..., indices["recent_hit"]]
    src_activity = values[..., indices["src_activity"]]
    src_recency = values[..., indices["src_recency"]]
    candidate_train_seen = values[..., indices["candidate_train_seen"]]
    candidate_test_freq = values[..., indices["candidate_test_freq"]]
    pair_decay_short = values[..., indices["pair_decay_short"]]
    pair_decay_long = values[..., indices["pair_decay_long"]]

    sorted_prior = np.sort(candidate_test_freq, axis=1)
    prior_gap = sorted_prior[:, -1] - sorted_prior[:, -2]
    short_max = pair_decay_short.max(axis=1)
    long_max = pair_decay_long.max(axis=1)
    output = np.column_stack(
        (
            np.mean(pair_strength > 0.0, axis=1),
            pair_strength.max(axis=1),
            repeat_rate.max(axis=1),
            src_activity.mean(axis=1),
            src_recency.mean(axis=1),
            np.mean(candidate_train_seen <= 0.0, axis=1),
            dst_popularity.mean(axis=1),
            dst_popularity.max(axis=1),
            candidate_test_freq.mean(axis=1),
            candidate_test_freq.max(axis=1),
            prior_gap,
            np.mean(recent_hit > 0.0, axis=1),
            recent_hit.max(axis=1),
            short_max,
            long_max,
            short_max - long_max,
        )
    )
    return np.asarray(output, dtype=np.float32)


def best_query_weights(
    mlp_probs: np.ndarray,
    lgbm_probs: np.ndarray,
    *,
    candidate_weights: tuple[float, ...],
    global_weight: float,
) -> np.ndarray:
    mlp = np.asarray(mlp_probs, dtype=np.float64)
    lgbm = np.asarray(lgbm_probs, dtype=np.float64)
    if mlp.shape != lgbm.shape or mlp.ndim != 2 or mlp.shape[1] < 2:
        raise ValueError("expert probabilities must have matching query-by-candidate shapes")
    weights = tuple(float(weight) for weight in candidate_weights)
    if not weights or global_weight not in weights:
        raise ValueError("candidate_weights must include the current global weight")

    preference_order = sorted(range(len(weights)), key=lambda index: (abs(weights[index] - global_weight), index))
    best_rr = np.full(mlp.shape[0], -1.0, dtype=np.float64)
    best_weight = np.full(mlp.shape[0], global_weight, dtype=np.float64)
    for index in preference_order:
        weight = weights[index]
        blended = weight * mlp + (1.0 - weight) * lgbm
        ranks = 1 + (blended[:, 1:] > blended[:, 0:1]).sum(axis=1)
        reciprocal_rank = 1.0 / ranks
        improved = reciprocal_rank > best_rr
        best_rr[improved] = reciprocal_rank[improved]
        best_weight[improved] = weight
    return best_weight


def fit_segment_gate(
    descriptors: np.ndarray,
    oracle_weights: np.ndarray,
    *,
    candidate_weights: tuple[float, ...],
    global_weight: float,
    max_depth: int,
    min_samples_leaf: int,
    seed: int,
    name: str,
    sample_weight: np.ndarray | None = None,
) -> SegmentGateResult:
    from sklearn.tree import DecisionTreeClassifier  # noqa: PLC0415

    inputs = np.asarray(descriptors, dtype=np.float32)
    targets = np.asarray(oracle_weights, dtype=np.float64)
    if inputs.ndim != 2 or inputs.shape[1] != len(QUERY_SEGMENT_FEATURE_NAMES):
        raise ValueError("segment descriptors have an incompatible shape")
    if targets.shape != (inputs.shape[0],):
        raise ValueError("oracle weights must contain one value per query")
    allowed = np.asarray(candidate_weights, dtype=np.float64)
    if allowed.ndim != 1 or allowed.size == 0 or np.unique(allowed).size != allowed.size:
        raise ValueError("candidate_weights must contain unique scalar values")
    matches = np.isclose(
        targets[:, None],
        allowed[None, :],
        rtol=0.0,
        atol=1e-12,
    )
    if not np.all(matches.sum(axis=1) == 1):
        raise ValueError("oracle weights must come from candidate_weights")
    encoded_targets = matches.argmax(axis=1).astype(np.int32, copy=False)

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
    )
    model.fit(inputs, encoded_targets, sample_weight=sample_weight)
    return SegmentGateResult(
        model_bytes=pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL),
        descriptor_names=QUERY_SEGMENT_FEATURE_NAMES,
        candidate_weights=tuple(float(weight) for weight in candidate_weights),
        global_weight=float(global_weight),
        name=name,
        max_depth=int(max_depth),
        min_samples_leaf=int(min_samples_leaf),
    )


def fit_segment_policy_gate(
    descriptors: np.ndarray,
    reciprocal_rank_rewards: np.ndarray,
    *,
    candidate_weights: tuple[float, ...],
    global_weight: float,
    max_depth: int,
    min_samples_leaf: int,
    seed: int,
    name: str,
    sample_weight: np.ndarray | None = None,
) -> SegmentGateResult:
    from sklearn.tree import DecisionTreeRegressor  # noqa: PLC0415

    inputs = np.asarray(descriptors, dtype=np.float32)
    rewards = np.asarray(reciprocal_rank_rewards, dtype=np.float64)
    allowed = np.asarray(candidate_weights, dtype=np.float64)
    if inputs.ndim != 2 or inputs.shape[1] != len(QUERY_SEGMENT_FEATURE_NAMES):
        raise ValueError("segment descriptors have an incompatible shape")
    if allowed.ndim != 1 or allowed.size == 0 or np.unique(allowed).size != allowed.size:
        raise ValueError("candidate_weights must contain unique scalar values")
    if rewards.shape != (inputs.shape[0], allowed.size) or not np.all(np.isfinite(rewards)):
        raise ValueError("rewards must contain one finite value per query and candidate weight")
    global_matches = np.flatnonzero(np.isclose(allowed, global_weight, rtol=0.0, atol=1e-12))
    if global_matches.size != 1:
        raise ValueError("candidate_weights must include the current global weight")
    if sample_weight is not None:
        row_weights = np.asarray(sample_weight, dtype=np.float64)
        if row_weights.shape != (inputs.shape[0],) or np.any(row_weights < 0.0):
            raise ValueError("sample_weight must contain one non-negative value per query")
    else:
        row_weights = np.ones(inputs.shape[0], dtype=np.float64)

    global_index = int(global_matches[0])
    estimator = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
    )
    estimator.fit(
        inputs,
        rewards - rewards[:, global_index : global_index + 1],
        sample_weight=sample_weight,
    )
    leaf_ids = np.asarray(estimator.apply(inputs), dtype=np.int64)
    preference_order = sorted(
        range(allowed.size),
        key=lambda index: (abs(float(allowed[index]) - global_weight), index),
    )
    leaf_weight_indices: dict[int, int] = {}
    for leaf_id in np.unique(leaf_ids):
        mask = leaf_ids == leaf_id
        leaf_rewards = (rewards[mask] * row_weights[mask, None]).sum(axis=0)
        best_index = global_index
        best_reward = float(leaf_rewards[global_index])
        for index in preference_order:
            reward = float(leaf_rewards[index])
            if reward > best_reward + 1e-12:
                best_index = index
                best_reward = reward
        leaf_weight_indices[int(leaf_id)] = int(best_index)

    policy = MRRPolicyTree(
        estimator=estimator,
        leaf_weight_indices=leaf_weight_indices,
    )
    return SegmentGateResult(
        model_bytes=pickle.dumps(policy, protocol=pickle.HIGHEST_PROTOCOL),
        descriptor_names=QUERY_SEGMENT_FEATURE_NAMES,
        candidate_weights=tuple(float(weight) for weight in candidate_weights),
        global_weight=float(global_weight),
        name=name,
        max_depth=int(max_depth),
        min_samples_leaf=int(min_samples_leaf),
    )


def predict_segment_weights(
    result: SegmentGateResult,
    features: np.ndarray,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    if result.descriptor_names != QUERY_SEGMENT_FEATURE_NAMES:
        raise ValueError("segment gate descriptor contract does not match this runtime")
    descriptors = query_segment_features(features, feature_names)
    model = pickle.loads(result.model_bytes)
    class_indices = np.asarray(model.predict(descriptors), dtype=np.int64)
    allowed = np.asarray(result.candidate_weights, dtype=np.float64)
    if (
        class_indices.shape != (descriptors.shape[0],)
        or np.any(class_indices < 0)
        or np.any(class_indices >= allowed.size)
    ):
        raise ValueError("segment gate emitted an invalid MLP weight")
    return allowed[class_indices]


def blend_expert_probabilities(
    mlp_probs: np.ndarray,
    lgbm_probs: np.ndarray,
    mlp_weight: float | np.ndarray,
) -> np.ndarray:
    mlp = np.asarray(mlp_probs, dtype=np.float64)
    lgbm = np.asarray(lgbm_probs, dtype=np.float64)
    if mlp.shape != lgbm.shape or mlp.ndim != 2:
        raise ValueError("expert probabilities must have matching query-by-candidate shapes")
    weights = np.asarray(mlp_weight, dtype=np.float64)
    if weights.ndim == 0:
        weights = np.full(mlp.shape[0], float(weights), dtype=np.float64)
    if weights.shape != (mlp.shape[0],) or not np.all(np.isfinite(weights)):
        raise ValueError("MLP weights must contain one finite value per query")
    if np.any((weights < 0.0) | (weights > 1.0)):
        raise ValueError("MLP weights must be between zero and one")
    return weights[:, None] * mlp + (1.0 - weights[:, None]) * lgbm
