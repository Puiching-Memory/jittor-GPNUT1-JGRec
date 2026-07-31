from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MULTI_INTEREST_FAMILIES = (("temporal2", 2), ("cluster2", 2), ("cluster4", 4))
MULTI_INTEREST_FEATURE_NAMES = tuple(
    f"multi_interest_{family}_{stat}"
    for family, _ in MULTI_INTEREST_FAMILIES
    for stat in ("max", "top2", "coverage")
)
ACTIVITY_ADAPTIVE_FEATURE_NAMES = (
    "multi_interest_decay_cosine",
    "multi_interest_recent16_cosine",
    "multi_interest_recent64_cosine",
    "multi_interest_full_cosine",
    "multi_interest_adaptive_cluster_weighted_max",
    "multi_interest_adaptive_cluster_weighted_top2",
    "multi_interest_adaptive_cluster_weighted_coverage",
    "multi_interest_adaptive_cluster_best_support",
    "multi_interest_adaptive_cluster_best_age",
    "multi_interest_adaptive_cluster_best_last_hit",
)
MULTI_INTEREST_V2_FEATURE_NAMES = (
    MULTI_INTEREST_FEATURE_NAMES + ACTIVITY_ADAPTIVE_FEATURE_NAMES
)
_ADAPTIVE_STATE_SHAPES = {
    "decay1": 1,
    "hierarchical3": 3,
    "adaptive_cluster4": 4,
}
_ADAPTIVE_METADATA_KEYS = (
    "adaptive_cluster_support",
    "adaptive_cluster_age",
    "adaptive_cluster_last_hit",
    "adaptive_cluster_weight",
)


@dataclass(frozen=True)
class AdaptiveClusterInterests:
    centers: np.ndarray
    support: np.ndarray
    age: np.ndarray
    last_hit: np.ndarray
    weights: np.ndarray
    activity: int
    half_life_events: float


def temporal_interest_centers(history_embeddings: np.ndarray) -> np.ndarray:
    """Return normalized recent and older centroids for an ordered history."""

    history = _embedding_matrix(history_embeddings)
    if history.shape[0] == 0:
        return np.zeros((2, history.shape[1]), dtype=np.float32)
    split = max(1, history.shape[0] // 2)
    recent = history[-split:]
    older = history[:-split]
    if older.shape[0] == 0:
        older = recent
    return _normalize_rows(
        np.stack((recent.mean(axis=0), older.mean(axis=0))).astype(
            np.float32,
            copy=False,
        )
    )


def cluster_interest_centers(
    history_embeddings: np.ndarray,
    *,
    k: int,
    iterations: int = 5,
) -> np.ndarray:
    """Deterministic cosine K-means initialized from recent/farthest points."""

    history = _normalize_rows(_embedding_matrix(history_embeddings))
    if k < 1:
        raise ValueError("interest count must be positive")
    if iterations < 1:
        raise ValueError("K-means iterations must be positive")
    if history.shape[0] == 0:
        return np.zeros((k, history.shape[1]), dtype=np.float32)

    selected = [history.shape[0] - 1]
    while len(selected) < min(k, history.shape[0]):
        similarities = history @ history[selected].T
        nearest = similarities.max(axis=1)
        nearest[selected] = np.inf
        selected.append(int(np.argmin(nearest)))
    while len(selected) < k:
        selected.append(selected[len(selected) % len(selected)])
    centers = history[np.asarray(selected, dtype=np.int64)].copy()

    for _ in range(iterations):
        assignment = np.argmax(history @ centers.T, axis=1)
        updated = centers.copy()
        for cluster_index in range(k):
            members = history[assignment == cluster_index]
            if members.shape[0]:
                updated[cluster_index] = members.mean(axis=0)
        centers = _normalize_rows(updated)
    return centers


def exponential_interest_center(
    history_embeddings: np.ndarray,
    history_times: np.ndarray,
    *,
    reference_time: int | float | None = None,
    half_life_events: float = 64.0,
) -> np.ndarray:
    """Return a normalized center with source-local exponential time decay."""

    history = _embedding_matrix(history_embeddings)
    event_ages = _event_equivalent_ages(
        history_times,
        expected_rows=history.shape[0],
        reference_time=reference_time,
    )
    if half_life_events <= 0.0:
        raise ValueError("half-life must be positive")
    if history.shape[0] == 0:
        return np.zeros(history.shape[1], dtype=np.float32)
    weights = np.exp2(-event_ages / float(half_life_events))
    center = np.sum(history * weights[:, None], axis=0)
    center /= max(float(weights.sum()), 1e-12)
    return _normalize_vector(center)


def hierarchical_interest_centers(
    history_embeddings: np.ndarray,
    *,
    windows: tuple[int | None, ...] = (16, 64, None),
) -> np.ndarray:
    """Return normalized recent-window centers in the supplied order."""

    history = _embedding_matrix(history_embeddings)
    if not windows:
        raise ValueError("at least one interest window is required")
    centers = np.zeros((len(windows), history.shape[1]), dtype=np.float32)
    for index, window in enumerate(windows):
        if window is not None and window < 1:
            raise ValueError("interest windows must be positive")
        values = history if window is None else history[-window:]
        if values.shape[0]:
            centers[index] = _normalize_vector(values.mean(axis=0))
    return centers


def activity_adaptive_cluster_interests(
    history_embeddings: np.ndarray,
    history_times: np.ndarray,
    *,
    k: int,
    base_half_life_events: float = 64.0,
    minimum_half_life_events: float = 8.0,
    iterations: int = 5,
) -> AdaptiveClusterInterests:
    """Fit deterministic weighted clusters and attach recency metadata."""

    history = _normalize_rows(_embedding_matrix(history_embeddings))
    if k < 1:
        raise ValueError("interest count must be positive")
    if iterations < 1:
        raise ValueError("K-means iterations must be positive")
    if base_half_life_events <= 0.0:
        raise ValueError("base half-life must be positive")
    if minimum_half_life_events <= 0.0:
        raise ValueError("minimum half-life must be positive")
    if minimum_half_life_events > base_half_life_events:
        raise ValueError("minimum half-life cannot exceed base half-life")

    activity = int(history.shape[0])
    half_life = _activity_half_life(
        activity,
        base=base_half_life_events,
        minimum=minimum_half_life_events,
    )
    event_ages = _event_equivalent_ages(
        history_times,
        expected_rows=activity,
        reference_time=None,
    )
    empty_metadata = np.zeros(k, dtype=np.float32)
    if activity == 0:
        return AdaptiveClusterInterests(
            centers=np.zeros((k, history.shape[1]), dtype=np.float32),
            support=empty_metadata.copy(),
            age=empty_metadata.copy(),
            last_hit=empty_metadata.copy(),
            weights=empty_metadata.copy(),
            activity=0,
            half_life_events=half_life,
        )

    event_weights = np.exp2(-event_ages / half_life).astype(
        np.float32,
        copy=False,
    )
    centers = _initial_cluster_centers(history, k)
    for _ in range(iterations):
        assignment = np.argmax(history @ centers.T, axis=1)
        updated = centers.copy()
        for cluster_index in range(k):
            mask = assignment == cluster_index
            if np.any(mask):
                member_weights = event_weights[mask]
                updated[cluster_index] = np.average(
                    history[mask],
                    axis=0,
                    weights=member_weights,
                )
        centers = _normalize_rows(updated)

    assignment = np.argmax(history @ centers.T, axis=1)
    support = np.zeros(k, dtype=np.float32)
    age = np.zeros(k, dtype=np.float32)
    last_hit = np.zeros(k, dtype=np.float32)
    total_mass = max(float(event_weights.sum()), 1e-12)
    maximum_age = max(float(event_ages.max()), 1.0)
    for cluster_index in range(k):
        mask = assignment == cluster_index
        if not np.any(mask):
            continue
        member_weights = event_weights[mask]
        member_ages = event_ages[mask]
        support[cluster_index] = float(member_weights.sum()) / total_mass
        age[cluster_index] = min(
            float(np.average(member_ages, weights=member_weights)) / maximum_age,
            1.0,
        )
        last_hit[cluster_index] = float(
            np.exp2(-float(member_ages.min()) / half_life)
        )
    routing_weights = np.sqrt(support) * last_hit
    return AdaptiveClusterInterests(
        centers=centers,
        support=support,
        age=age,
        last_hit=last_hit,
        weights=routing_weights.astype(np.float32, copy=False),
        activity=activity,
        half_life_events=half_life,
    )


def interest_affinity_features(
    candidate_embeddings: np.ndarray,
    interest_centers: np.ndarray,
) -> np.ndarray:
    """Return max cosine, second cosine, and positive-interest coverage."""

    candidates = _normalize_rows(_embedding_matrix(candidate_embeddings))
    centers = _normalize_rows(_embedding_matrix(interest_centers))
    if candidates.shape[1] != centers.shape[1]:
        raise ValueError("candidate and interest embedding dimensions differ")
    if centers.shape[0] == 0:
        return np.zeros((candidates.shape[0], 3), dtype=np.float32)

    similarities = candidates @ centers.T
    ordered = np.sort(similarities, axis=1)
    maximum = ordered[:, -1]
    second = ordered[:, -2] if centers.shape[0] > 1 else maximum
    coverage = np.maximum(similarities, 0.0).mean(axis=1)
    return np.stack((maximum, second, coverage), axis=1).astype(
        np.float32,
        copy=False,
    )


def adaptive_interest_affinity_features(
    candidate_embeddings: np.ndarray,
    interest_centers: np.ndarray,
    support: np.ndarray,
    age: np.ndarray,
    last_hit: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return weighted affinity and best-matching cluster metadata."""

    candidates = _normalize_rows(_embedding_matrix(candidate_embeddings))
    centers = _normalize_rows(_embedding_matrix(interest_centers))
    if candidates.shape[1] != centers.shape[1]:
        raise ValueError("candidate and interest embedding dimensions differ")
    metadata = tuple(
        _cluster_metadata_vector(values, centers.shape[0])
        for values in (support, age, last_hit, weights)
    )
    if centers.shape[0] == 0:
        return np.zeros((candidates.shape[0], 6), dtype=np.float32)

    similarities = candidates @ centers.T
    weighted = _recency_weighted_similarities(similarities, metadata[3])
    ordered = np.sort(weighted, axis=1)
    maximum = ordered[:, -1]
    second = ordered[:, -2] if centers.shape[0] > 1 else maximum
    coverage = np.maximum(weighted, 0.0).mean(axis=1)
    best = np.argmax(similarities, axis=1)
    return np.stack(
        (
            maximum,
            second,
            coverage,
            metadata[0][best],
            metadata[1][best],
            metadata[2][best],
        ),
        axis=1,
    ).astype(np.float32, copy=False)


def activity_adaptive_features_for_candidate_batch(
    candidate_embeddings: np.ndarray,
    decay_centers: np.ndarray,
    hierarchical_centers: np.ndarray,
    cluster_centers: np.ndarray,
    support: np.ndarray,
    age: np.ndarray,
    last_hit: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Vectorize the ten adaptive channels over query candidate sets."""

    candidates = np.asarray(candidate_embeddings, dtype=np.float32)
    decay = np.asarray(decay_centers, dtype=np.float32)
    hierarchical = np.asarray(hierarchical_centers, dtype=np.float32)
    clusters = np.asarray(cluster_centers, dtype=np.float32)
    if candidates.ndim != 3:
        raise ValueError("candidate embeddings must be query-by-candidate vectors")
    batch_rows, candidate_count, embedding_dim = candidates.shape
    expected_centers = (
        (decay, (batch_rows, 1, embedding_dim), "decay"),
        (hierarchical, (batch_rows, 3, embedding_dim), "hierarchical"),
        (clusters, (batch_rows, 4, embedding_dim), "adaptive cluster"),
    )
    for values, expected, name in expected_centers:
        if values.shape != expected:
            raise ValueError(
                f"{name} center shape differs: {values.shape} != {expected}"
            )
    metadata = tuple(
        np.asarray(values, dtype=np.float32)
        for values in (support, age, last_hit, weights)
    )
    for values in metadata:
        if values.shape != (batch_rows, 4):
            raise ValueError("adaptive cluster metadata shape differs")
        if not np.all(np.isfinite(values)):
            raise ValueError("adaptive cluster metadata must be finite")

    candidates = _normalize_last_axis(candidates)
    decay = _normalize_last_axis(decay)
    hierarchical = _normalize_last_axis(hierarchical)
    clusters = _normalize_last_axis(clusters)
    output = np.zeros(
        (batch_rows, candidate_count, len(ACTIVITY_ADAPTIVE_FEATURE_NAMES)),
        dtype=np.float32,
    )
    output[..., 0] = np.einsum(
        "bcd,bkd->bck",
        candidates,
        decay,
        optimize=True,
    )[..., 0]
    output[..., 1:4] = np.einsum(
        "bcd,bkd->bck",
        candidates,
        hierarchical,
        optimize=True,
    )
    cluster_similarity = np.einsum(
        "bcd,bkd->bck",
        candidates,
        clusters,
        optimize=True,
    )
    weighted = _recency_weighted_similarities(
        cluster_similarity,
        metadata[3][:, None, :],
    )
    ordered = np.sort(weighted, axis=2)
    output[..., 4] = ordered[..., -1]
    output[..., 5] = ordered[..., -2]
    output[..., 6] = np.maximum(weighted, 0.0).mean(axis=2)
    best = np.argmax(cluster_similarity, axis=2)
    for metadata_index in range(3):
        output[..., 7 + metadata_index] = np.take_along_axis(
            metadata[metadata_index][:, None, :],
            best[..., None],
            axis=2,
        )[..., 0]
    return output


def append_multi_interest_features(
    base_features: np.ndarray,
    queries,
    id_map,
    state: dict[str, np.ndarray] | None,
) -> np.ndarray:
    """Append frozen proxy features, preserving old checkpoints unchanged."""

    if state is None:
        return base_features
    proxy = multi_interest_features_for_query_array(queries, id_map, state)
    base = np.asarray(base_features, dtype=np.float32)
    if base.shape[:2] != proxy.shape[:2]:
        raise ValueError("base and multi-interest query shapes differ")
    return np.concatenate((base, proxy), axis=-1, dtype=np.float32)


def multi_interest_features_for_query_array(
    queries,
    id_map,
    state: dict[str, np.ndarray],
) -> np.ndarray:
    """Compute the frozen v1 or activity-adaptive v2 query proxy."""

    item_embeddings = _normalize_rows(
        _embedding_matrix(state["item_embeddings"])
    )
    if item_embeddings.shape[0] != id_map.num_dst:
        raise ValueError("multi-interest item map differs from checkpoint")
    src_ids = id_map.src_ids(queries.src)
    dst_ids = id_map.dst_ids(queries.candidates)
    valid_src = src_ids >= 0
    valid_dst = dst_ids >= 0
    candidate_vectors = np.zeros(
        (
            len(queries),
            queries.candidate_count,
            item_embeddings.shape[1],
        ),
        dtype=np.float32,
    )
    candidate_vectors[valid_dst] = item_embeddings[dst_ids[valid_dst]]
    adaptive = _adaptive_state_enabled(state)
    feature_names = (
        MULTI_INTEREST_V2_FEATURE_NAMES
        if adaptive
        else MULTI_INTEREST_FEATURE_NAMES
    )
    output = np.zeros(
        (
            len(queries),
            queries.candidate_count,
            len(feature_names),
        ),
        dtype=np.float32,
    )
    output_column = 0
    for family, interest_count in MULTI_INTEREST_FAMILIES:
        all_centers = np.asarray(state[family], dtype=np.float32)
        expected_shape = (
            id_map.num_src,
            interest_count,
            item_embeddings.shape[1],
        )
        if all_centers.shape != expected_shape:
            raise ValueError(
                f"multi-interest {family} shape differs: "
                f"{all_centers.shape} != {expected_shape}"
            )
        batch_centers = np.zeros(
            (len(queries), interest_count, item_embeddings.shape[1]),
            dtype=np.float32,
        )
        batch_centers[valid_src] = all_centers[src_ids[valid_src]]
        batch_centers = _normalize_last_axis(batch_centers)
        similarities = np.einsum(
            "bcd,bkd->bck",
            candidate_vectors,
            batch_centers,
            optimize=True,
        )
        ordered = np.sort(similarities, axis=2)
        output[..., output_column] = ordered[..., -1]
        output[..., output_column + 1] = ordered[..., -2]
        output[..., output_column + 2] = np.maximum(
            similarities,
            0.0,
        ).mean(axis=2)
        output_column += 3
    if adaptive:
        output_column = _append_adaptive_query_features(
            output,
            output_column=output_column,
            candidate_vectors=candidate_vectors,
            src_ids=src_ids,
            valid_src=valid_src,
            valid_dst=valid_dst,
            state=state,
            num_src=id_map.num_src,
            embedding_dim=item_embeddings.shape[1],
        )
    if output_column != len(feature_names):
        raise AssertionError("multi-interest feature schema is inconsistent")
    return output


def _append_adaptive_query_features(
    output: np.ndarray,
    *,
    output_column: int,
    candidate_vectors: np.ndarray,
    src_ids: np.ndarray,
    valid_src: np.ndarray,
    valid_dst: np.ndarray,
    state: dict[str, np.ndarray],
    num_src: int,
    embedding_dim: int,
) -> int:
    batch_rows = candidate_vectors.shape[0]
    center_batches: dict[str, np.ndarray] = {}
    for name, count in _ADAPTIVE_STATE_SHAPES.items():
        values = np.asarray(state[name], dtype=np.float32)
        expected = (num_src, count, embedding_dim)
        if values.shape != expected:
            raise ValueError(
                f"multi-interest {name} shape differs: {values.shape} != {expected}"
            )
        batch = np.zeros((batch_rows, count, embedding_dim), dtype=np.float32)
        batch[valid_src] = values[src_ids[valid_src]]
        center_batches[name] = _normalize_last_axis(batch)

    metadata_batches: list[np.ndarray] = []
    for key in _ADAPTIVE_METADATA_KEYS:
        values = np.asarray(state[key], dtype=np.float32)
        expected = (num_src, 4)
        if values.shape != expected:
            raise ValueError(
                f"multi-interest {key} shape differs: {values.shape} != {expected}"
            )
        batch = np.zeros((batch_rows, 4), dtype=np.float32)
        batch[valid_src] = values[src_ids[valid_src]]
        metadata_batches.append(batch)
    adaptive_features = activity_adaptive_features_for_candidate_batch(
        candidate_vectors,
        center_batches["decay1"],
        center_batches["hierarchical3"],
        center_batches["adaptive_cluster4"],
        *metadata_batches,
    )
    adaptive_features *= valid_dst[..., None]
    output[
        ...,
        output_column : output_column + len(ACTIVITY_ADAPTIVE_FEATURE_NAMES),
    ] = adaptive_features
    output_column += len(ACTIVITY_ADAPTIVE_FEATURE_NAMES)
    return output_column


def _embedding_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("embeddings must be finite")
    return matrix


def _adaptive_state_enabled(state: dict[str, np.ndarray]) -> bool:
    keys = tuple(_ADAPTIVE_STATE_SHAPES) + _ADAPTIVE_METADATA_KEYS
    present = tuple(key in state for key in keys)
    if any(present) and not all(present):
        missing = [key for key, exists in zip(keys, present, strict=True) if not exists]
        raise ValueError(
            "incomplete activity-adaptive multi-interest state: "
            + ", ".join(missing)
        )
    return all(present)


def _activity_half_life(
    activity: int,
    *,
    base: float,
    minimum: float,
) -> float:
    activity_ratio = max(float(activity) / 64.0, 1.0)
    return max(float(base) / np.sqrt(activity_ratio), float(minimum))


def _event_equivalent_ages(
    history_times: np.ndarray,
    *,
    expected_rows: int,
    reference_time: int | float | None,
) -> np.ndarray:
    times = np.asarray(history_times, dtype=np.float64)
    if times.ndim != 1 or times.shape[0] != expected_rows:
        raise ValueError("history times must align with history embeddings")
    if not np.all(np.isfinite(times)):
        raise ValueError("history times must be finite")
    if expected_rows == 0:
        return np.zeros(0, dtype=np.float64)
    if np.any(np.diff(times) < 0.0):
        raise ValueError("history times must be ordered")
    reference = float(times[-1]) if reference_time is None else float(reference_time)
    if reference < float(times[-1]):
        raise ValueError("reference time cannot precede history")
    positive_gaps = np.diff(times)
    positive_gaps = positive_gaps[positive_gaps > 0.0]
    time_per_event = float(np.median(positive_gaps)) if positive_gaps.size else 1.0
    return np.maximum(reference - times, 0.0) / max(time_per_event, 1.0)


def _initial_cluster_centers(history: np.ndarray, k: int) -> np.ndarray:
    selected = [history.shape[0] - 1]
    while len(selected) < min(k, history.shape[0]):
        similarities = history @ history[selected].T
        nearest = similarities.max(axis=1)
        nearest[selected] = np.inf
        selected.append(int(np.argmin(nearest)))
    while len(selected) < k:
        selected.append(selected[len(selected) % len(selected)])
    return history[np.asarray(selected, dtype=np.int64)].copy()


def _cluster_metadata_vector(values: np.ndarray, count: int) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (count,):
        raise ValueError(
            f"cluster metadata shape differs: {vector.shape} != {(count,)}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("cluster metadata must be finite")
    return vector


def _recency_weighted_similarities(
    similarities: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    return np.where(
        similarities >= 0.0,
        similarities * weights,
        similarities,
    )


def _normalize_vector(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return np.zeros_like(vector)
    return (vector / norm).astype(np.float32, copy=False)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(
        matrix,
        np.maximum(norms, 1e-12),
        out=np.zeros_like(matrix),
        where=norms > 0.0,
    )


def _normalize_last_axis(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return np.divide(
        matrix,
        np.maximum(norms, 1e-12),
        out=np.zeros_like(matrix),
        where=norms > 0.0,
    )
