from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from .candidate_set_transformer import _load_state, _snapshot_state

CHECKPOINT_FORMAT = "jgrec-residual-advantage-router"
CHECKPOINT_VERSION = 1

ROUTER_FEATURE_NAMES = (
    "default_margin_z",
    "default_entropy",
    "default_top_probability",
    "default_score_std",
    "default_score_range",
    "short_residual_rms",
    "short_residual_max_abs",
    "short_residual_default_top1",
    "medium_residual_rms",
    "medium_residual_max_abs",
    "medium_residual_default_top1",
    "long_residual_rms",
    "long_residual_max_abs",
    "long_residual_default_top1",
    "short_medium_residual_cosine",
    "short_medium_residual_l2",
    "short_medium_max_candidate_agreement",
    "short_long_residual_cosine",
    "short_long_residual_l2",
    "short_long_max_candidate_agreement",
    "medium_long_residual_cosine",
    "medium_long_residual_l2",
    "medium_long_max_candidate_agreement",
    "medium_switch_max_abs",
    "medium_switch_rms",
    "medium_switch_changed_fraction",
    "medium_switch_changes_top1",
    "medium_alternative_margin_z",
    "long_switch_max_abs",
    "long_switch_rms",
    "long_switch_changed_fraction",
    "long_switch_changes_top1",
    "long_alternative_margin_z",
    "short_gap_days",
    "medium_gap_days",
    "long_gap_days",
    "medium_minus_short_gap_days",
    "long_minus_short_gap_days",
)


@dataclass(frozen=True)
class BoundedTopKAlternative:
    scores: np.ndarray
    delta: np.ndarray
    topk_mask: np.ndarray
    top_k: int
    cap: float


@dataclass(frozen=True)
class HighConfidenceRoutingResult:
    scores: np.ndarray
    route_index: np.ndarray
    route_mask: np.ndarray
    confidence: np.ndarray
    predicted_advantages: np.ndarray
    quota: int


@dataclass(frozen=True)
class RouterTemporalSplit:
    train_rows: tuple[int, int]
    selection_rows: tuple[int, int]
    gate_rows: tuple[int, int]
    train_time_max: int
    selection_time_min: int
    selection_time_max: int
    gate_time_min: int
    gate_time_max: int


@dataclass(frozen=True)
class ResidualAdvantageRouterConfig:
    input_dim: int
    hidden_dim: int = 32
    dropout: float = 0.05

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("residual router dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("residual router dropout must be in [0, 1)")


@dataclass(frozen=True)
class ResidualAdvantageRouterTrainingConfig:
    epochs: int = 20
    batch_size: int = 512
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    reward_scale: float = 10.0
    nonzero_weight: float = 4.0
    seed: int = 60

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("residual router training sizes must be positive")
        if (
            self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or not math.isfinite(self.reward_scale)
            or self.reward_scale <= 0.0
            or not math.isfinite(self.nonzero_weight)
            or self.nonzero_weight < 1.0
        ):
            raise ValueError("residual router optimizer settings are invalid")


@dataclass(frozen=True)
class ResidualAdvantageRouterFitResult:
    model_config: ResidualAdvantageRouterConfig
    training_config: ResidualAdvantageRouterTrainingConfig
    mean: np.ndarray
    std: np.ndarray
    state: dict[str, np.ndarray]
    history: tuple[dict[str, float | int], ...]
    training_rows: int
    nonzero_targets: int
    feature_names: tuple[str, ...]
    trainable_frameworks: tuple[str, ...] = ("jittor",)
    non_jittor_trainable_models: tuple[str, ...] = ()


class ResidualAdvantageRouter(jt.nn.Module):
    def __init__(self, config: ResidualAdvantageRouterConfig) -> None:
        super().__init__()
        self.config = config
        self.input = jt.nn.Linear(config.input_dim, config.hidden_dim)
        self.hidden = jt.nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = jt.nn.Dropout(config.dropout)
        self.output = jt.nn.Linear(config.hidden_dim, 2)

    def execute(self, features: jt.Var) -> jt.Var:
        if len(features.shape) != 2:
            raise ValueError("residual router features must be 2D")
        hidden = jt.nn.relu(self.input(features))
        hidden = jt.nn.relu(self.hidden(self.dropout(hidden)))
        return self.output(self.dropout(hidden))


def bounded_topk_alternative(
    default_scores: Any,
    short_residual: Any,
    alternative_residual: Any,
    *,
    top_k: int,
    cap: float,
) -> BoundedTopKAlternative:
    """Apply an alternative residual only inside default short top-k."""
    default = _score_matrix(default_scores, label="default short scores")
    short = _aligned_matrix(
        short_residual,
        default,
        label="short residual",
    )
    alternative = _aligned_matrix(
        alternative_residual,
        default,
        label="alternative residual",
    )
    width = int(top_k)
    if width != top_k or not 1 <= width < default.shape[1]:
        raise ValueError("top_k must be an integer below candidate count")
    limit = float(cap)
    if not math.isfinite(limit) or not 0.0 <= limit <= 0.10:
        raise ValueError("switch cap must be in [0, 0.10]")

    top_indices = np.argsort(-default, axis=1, kind="stable")[:, :width]
    mask = np.zeros(default.shape, dtype=bool)
    np.put_along_axis(mask, top_indices, True, axis=1)
    raw = np.where(mask, alternative - short, 0.0).astype(
        np.float32,
        copy=False,
    )
    centered = np.where(
        mask,
        raw - np.sum(raw, axis=1, keepdims=True) / float(width),
        0.0,
    )
    maximum = np.max(np.abs(centered), axis=1, keepdims=True)
    scale = np.minimum(
        1.0,
        limit / np.maximum(maximum, np.finfo(np.float32).tiny),
    )
    projected = np.where(mask, centered * scale, 0.0).astype(
        np.float32,
        copy=False,
    )
    scores = np.asarray(default + projected, dtype=np.float32)
    actual_delta = np.asarray(scores - default, dtype=np.float32)
    actual_delta[~mask] = 0.0
    scores[~mask] = default[~mask]
    return BoundedTopKAlternative(
        scores=scores,
        delta=actual_delta,
        topk_mask=mask,
        top_k=width,
        cap=limit,
    )


def hard_high_confidence_route(
    default_scores: Any,
    alternative_scores: tuple[Any, Any],
    predicted_advantages: Any,
    *,
    minimum_confidence: float,
    maximum_route_fraction: float,
) -> HighConfidenceRoutingResult:
    """Default to short and route only the strongest eligible rows."""
    default = _score_matrix(default_scores, label="route default scores")
    if len(alternative_scores) != 2:
        raise ValueError("router requires medium and long alternatives")
    alternatives = tuple(
        _aligned_matrix(value, default, label="route alternative")
        for value in alternative_scores
    )
    predictions = np.asarray(predicted_advantages, dtype=np.float32)
    if (
        predictions.shape != (default.shape[0], 2)
        or not np.isfinite(predictions).all()
    ):
        raise ValueError("predicted route advantages must have shape [rows, 2]")
    threshold = float(minimum_confidence)
    fraction = float(maximum_route_fraction)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("minimum route confidence must be non-negative")
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("maximum route fraction must be in [0, 1]")

    best = np.argmax(predictions, axis=1)
    rows = np.arange(default.shape[0])
    best_advantage = predictions[rows, best]
    other_advantage = predictions[rows, 1 - best]
    confidence = best_advantage - np.maximum(other_advantage, 0.0)
    eligible = np.flatnonzero(
        (best_advantage > 0.0) & (confidence >= threshold)
    )
    quota = math.floor(default.shape[0] * fraction)
    selected = eligible[
        np.argsort(-confidence[eligible], kind="stable")
    ][:quota]

    route_index = np.zeros(default.shape[0], dtype=np.int8)
    route_index[selected] = (best[selected] + 1).astype(np.int8)
    route_mask = route_index > 0
    scores = np.array(default, copy=True)
    for alternative_index, values in enumerate(alternatives, start=1):
        chosen = route_index == alternative_index
        scores[chosen] = values[chosen]
    return HighConfidenceRoutingResult(
        scores=scores,
        route_index=route_index,
        route_mask=route_mask,
        confidence=np.asarray(confidence, dtype=np.float32),
        predicted_advantages=predictions.copy(),
        quota=quota,
    )


def route_reward_targets(
    default_scores: Any,
    alternative_scores: tuple[Any, Any],
    positive_indices: Any,
) -> np.ndarray:
    default = _score_matrix(default_scores, label="reward default scores")
    if len(alternative_scores) != 2:
        raise ValueError("reward targets require two alternatives")
    alternatives = tuple(
        _aligned_matrix(value, default, label="reward alternative")
        for value in alternative_scores
    )
    positives = np.asarray(positive_indices)
    if (
        positives.shape != (default.shape[0],)
        or positives.dtype.kind not in "iu"
        or np.any(positives < 0)
        or np.any(positives >= default.shape[1])
    ):
        raise ValueError("reward positive indices are invalid")
    default_rr = _reciprocal_ranks(default, positives)
    return np.column_stack(
        [
            _reciprocal_ranks(values, positives) - default_rr
            for values in alternatives
        ]
    ).astype(np.float32)


def audit_bounded_topk_route(
    default_scores: Any,
    alternatives: tuple[BoundedTopKAlternative, BoundedTopKAlternative],
    routed: HighConfidenceRoutingResult,
    *,
    cap: float,
    maximum_route_fraction: float,
    tolerance: float = 2e-6,
) -> dict[str, float | int | bool]:
    default = _score_matrix(default_scores, label="route audit default")
    if len(alternatives) != 2:
        raise ValueError("route audit requires two alternatives")
    limit = float(cap)
    fraction = float(maximum_route_fraction)
    allowed_error = float(tolerance)
    if (
        not math.isfinite(limit)
        or not 0.0 <= limit <= 0.10
        or not math.isfinite(fraction)
        or not 0.0 <= fraction <= 1.0
        or not math.isfinite(allowed_error)
        or allowed_error < 0.0
    ):
        raise ValueError("route audit bounds are invalid")
    route_index = np.asarray(routed.route_index)
    route_mask = np.asarray(routed.route_mask, dtype=bool)
    scores = _aligned_matrix(
        routed.scores,
        default,
        label="routed scores",
    )
    if (
        route_index.shape != (default.shape[0],)
        or route_mask.shape != route_index.shape
        or np.any((route_index < 0) | (route_index > 2))
        or not np.array_equal(route_mask, route_index > 0)
    ):
        raise ValueError("route audit metadata does not align")

    outside_exact = True
    row_centered = True
    maximum = 0.0
    routed_matches = True
    for alternative_index, alternative in enumerate(
        alternatives,
        start=1,
    ):
        if (
            alternative.scores.shape != default.shape
            or alternative.delta.shape != default.shape
            or alternative.topk_mask.shape != default.shape
        ):
            raise ValueError("route audit alternative does not align")
        outside_exact = bool(
            outside_exact
            and np.array_equal(
                alternative.scores[~alternative.topk_mask],
                default[~alternative.topk_mask],
            )
        )
        actual = alternative.scores - default
        maximum = max(maximum, float(np.max(np.abs(actual))))
        row_centered = bool(
            row_centered
            and np.max(np.abs(np.sum(actual, axis=1))) <= allowed_error
        )
        selected = route_index == alternative_index
        routed_matches = bool(
            routed_matches
            and np.array_equal(scores[selected], alternative.scores[selected])
        )
    unrouted = route_index == 0
    unrouted_exact = bool(np.array_equal(scores[unrouted], default[unrouted]))
    maximum_rows = math.floor(default.shape[0] * fraction)
    routed_rows = int(np.sum(route_mask))
    cap_passed = maximum <= limit + allowed_error
    passed = bool(
        outside_exact
        and row_centered
        and routed_matches
        and unrouted_exact
        and cap_passed
        and routed_rows <= maximum_rows
    )
    return {
        "passed": passed,
        "topk_outside_exact": outside_exact,
        "row_centered": row_centered,
        "routed_rows_match_alternative": routed_matches,
        "unrouted_rows_exact": unrouted_exact,
        "cap_passed": cap_passed,
        "max_absolute_switch": maximum,
        "cap": limit,
        "routed_rows": routed_rows,
        "maximum_routed_rows": maximum_rows,
        "route_fraction": float(np.mean(route_mask)),
        "medium_rows": int(np.sum(route_index == 1)),
        "long_rows": int(np.sum(route_index == 2)),
        "tolerance": allowed_error,
    }


def router_summary_features(
    default_scores: Any,
    residuals: Any,
    gap_days: Any,
    alternative_scores: tuple[Any, Any],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build candidate-permutation-invariant, label-free row features."""
    default = _score_matrix(default_scores, label="router default scores")
    expert_residuals = np.asarray(residuals, dtype=np.float32)
    gaps = np.asarray(gap_days, dtype=np.float32)
    if (
        expert_residuals.shape != (3, *default.shape)
        or gaps.shape != (3, default.shape[0])
        or not np.isfinite(expert_residuals).all()
        or not np.isfinite(gaps).all()
    ):
        raise ValueError("router residual and gap arrays do not align")
    if len(alternative_scores) != 2:
        raise ValueError("router features require two alternatives")
    alternatives = tuple(
        _aligned_matrix(value, default, label="router alternative")
        for value in alternative_scores
    )

    default_std = np.maximum(
        np.std(default, axis=1, dtype=np.float64),
        1e-6,
    )
    default_ordered = np.sort(default, axis=1)
    probabilities = _row_softmax(
        (default - default.mean(axis=1, keepdims=True))
        / default_std[:, None]
    )
    default_top = np.argmax(default, axis=1)
    rows = np.arange(default.shape[0])
    columns: list[np.ndarray] = [
        (default_ordered[:, -1] - default_ordered[:, -2]) / default_std,
        -np.sum(
            probabilities * np.log(np.maximum(probabilities, 1e-30)),
            axis=1,
        )
        / math.log(default.shape[1]),
        np.max(probabilities, axis=1),
        default_std,
        default_ordered[:, -1] - default_ordered[:, 0],
    ]
    for expert in expert_residuals:
        columns.extend(
            (
                np.sqrt(np.mean(np.square(expert), axis=1)),
                np.max(np.abs(expert), axis=1),
                expert[rows, default_top],
            )
        )
    for left, right in ((0, 1), (0, 2), (1, 2)):
        left_values = expert_residuals[left]
        right_values = expert_residuals[right]
        columns.extend(
            (
                _row_cosine(left_values, right_values),
                np.linalg.norm(left_values - right_values, axis=1),
                (
                    np.argmax(left_values, axis=1)
                    == np.argmax(right_values, axis=1)
                ).astype(np.float32),
            )
        )
    for alternative in alternatives:
        delta = alternative - default
        changed = delta != 0.0
        alternative_std = np.maximum(
            np.std(alternative, axis=1, dtype=np.float64),
            1e-6,
        )
        ordered = np.sort(alternative, axis=1)
        columns.extend(
            (
                np.max(np.abs(delta), axis=1),
                np.sqrt(np.mean(np.square(delta), axis=1)),
                np.mean(changed, axis=1),
                (
                    np.argmax(alternative, axis=1) != default_top
                ).astype(np.float32),
                (ordered[:, -1] - ordered[:, -2]) / alternative_std,
            )
        )
    columns.extend(
        (
            gaps[0],
            gaps[1],
            gaps[2],
            gaps[1] - gaps[0],
            gaps[2] - gaps[0],
        )
    )
    features = np.column_stack(columns).astype(np.float32, copy=False)
    if (
        features.shape[1] != len(ROUTER_FEATURE_NAMES)
        or not np.isfinite(features).all()
    ):
        raise FloatingPointError("router summary features are invalid")
    return features, ROUTER_FEATURE_NAMES


def router_candidate_support_features(
    candidate_features: Any,
    feature_names: tuple[str, ...],
    default_scores: Any,
    alternative_scores: tuple[Any, Any],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Describe default/alternative top candidates without using their IDs."""
    default = _score_matrix(
        default_scores,
        label="candidate support default scores",
    )
    raw = np.asarray(candidate_features, dtype=np.float32)
    names = tuple(str(name) for name in feature_names)
    if (
        raw.ndim != 3
        or raw.shape[:2] != default.shape
        or raw.shape[2] <= 0
        or raw.shape[2] != len(names)
        or not np.isfinite(raw).all()
        or len(set(names)) != len(names)
    ):
        raise ValueError("candidate support features do not align")
    if len(alternative_scores) != 2:
        raise ValueError("candidate support requires two alternatives")
    alternatives = tuple(
        _aligned_matrix(
            values,
            default,
            label="candidate support alternative",
        )
        for values in alternative_scores
    )
    rows = np.arange(default.shape[0])
    default_top = np.argmax(default, axis=1)
    default_top_features = raw[rows, default_top]
    columns = [default_top_features]
    output_names = [f"default_top__{name}" for name in names]
    for route_name, alternative in zip(
        ("medium", "long"),
        alternatives,
        strict=True,
    ):
        alternative_top = np.argmax(alternative, axis=1)
        delta = alternative - default
        promoted = np.argmax(delta, axis=1)
        demoted = np.argmin(delta, axis=1)
        columns.extend(
            (
                raw[rows, alternative_top] - default_top_features,
                raw[rows, promoted] - raw[rows, demoted],
            )
        )
        output_names.extend(
            [
                f"{route_name}_top_minus_default__{name}"
                for name in names
            ]
        )
        output_names.extend(
            [
                f"{route_name}_promoted_minus_demoted__{name}"
                for name in names
            ]
        )
    output = np.concatenate(columns, axis=1).astype(
        np.float32,
        copy=False,
    )
    if not np.isfinite(output).all():
        raise FloatingPointError("candidate support summary is non-finite")
    return output, tuple(output_names)


def fit_residual_advantage_router(
    features: Any,
    rewards: Any,
    *,
    model_config: ResidualAdvantageRouterConfig,
    training_config: ResidualAdvantageRouterTrainingConfig,
    feature_names: tuple[str, ...] | None = None,
    verbose: bool = True,
) -> tuple[ResidualAdvantageRouter, ResidualAdvantageRouterFitResult]:
    values = np.asarray(features, dtype=np.float32)
    targets = np.asarray(rewards, dtype=np.float32)
    if (
        values.ndim != 2
        or values.shape[0] <= 1
        or values.shape[1] != model_config.input_dim
        or targets.shape != (values.shape[0], 2)
        or not np.isfinite(values).all()
        or not np.isfinite(targets).all()
    ):
        raise ValueError("residual router training arrays are invalid")
    names = (
        tuple(str(name) for name in feature_names)
        if feature_names is not None
        else tuple(f"feature_{index}" for index in range(values.shape[1]))
    )
    if len(names) != values.shape[1] or len(set(names)) != len(names):
        raise ValueError("residual router feature names do not align")
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    normalized = np.asarray((values - mean) / std, dtype=np.float32)
    scaled_targets = np.asarray(
        targets * training_config.reward_scale,
        dtype=np.float32,
    )
    weights = np.where(
        targets != 0.0,
        training_config.nonzero_weight,
        1.0,
    ).astype(np.float32)
    jt.set_seed(training_config.seed)
    rng = np.random.default_rng(training_config.seed)
    model = ResidualAdvantageRouter(model_config)
    optimizer = jt.nn.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(1, training_config.epochs + 1):
        model.train()
        order = rng.permutation(values.shape[0])
        losses = []
        for start in range(0, values.shape[0], training_config.batch_size):
            indices = order[start : start + training_config.batch_size]
            prediction = model(
                jt.array(normalized[indices], dtype=jt.float32)
            )
            difference = prediction - jt.array(
                scaled_targets[indices],
                dtype=jt.float32,
            )
            loss = (
                difference
                * difference
                * jt.array(weights[indices], dtype=jt.float32)
            ).mean()
            optimizer.step(loss)
            losses.append(float(loss.item()))
        mean_loss = float(np.mean(losses))
        if not math.isfinite(mean_loss):
            raise FloatingPointError(
                f"non-finite residual router loss at epoch {epoch}"
            )
        history.append({"epoch": epoch, "train_loss": mean_loss})
        if verbose:
            print(
                f"[residual-router] epoch={epoch} "
                f"train_loss={mean_loss:.6f}",
                flush=True,
            )
    result = ResidualAdvantageRouterFitResult(
        model_config=model_config,
        training_config=training_config,
        mean=mean,
        std=std,
        state=_snapshot_state(model),
        history=tuple(history),
        training_rows=int(values.shape[0]),
        nonzero_targets=int(np.count_nonzero(targets)),
        feature_names=names,
    )
    return model, result


def predict_residual_advantages(
    model: ResidualAdvantageRouter,
    features: Any,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    reward_scale: float,
    batch_size: int,
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    means = np.asarray(mean, dtype=np.float32)
    scales = np.asarray(std, dtype=np.float32)
    scale = float(reward_scale)
    if (
        values.ndim != 2
        or values.shape[1] != model.config.input_dim
        or means.shape != (values.shape[1],)
        or scales.shape != means.shape
        or np.any(scales <= 0.0)
        or not math.isfinite(scale)
        or scale <= 0.0
        or batch_size <= 0
    ):
        raise ValueError("residual router prediction arrays are invalid")
    normalized = np.asarray((values - means) / scales, dtype=np.float32)
    output = np.empty((values.shape[0], 2), dtype=np.float32)
    model.eval()
    with jt.no_grad():
        for start in range(0, values.shape[0], batch_size):
            stop = min(start + batch_size, values.shape[0])
            output[start:stop] = (
                model(jt.array(normalized[start:stop], dtype=jt.float32))
                .numpy()
                / scale
            )
    return output


def save_residual_advantage_router_checkpoint(
    path: Path,
    model: ResidualAdvantageRouter,
    result: ResidualAdvantageRouterFitResult,
) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"residual router checkpoint exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "model_config": asdict(result.model_config),
        "training_config": asdict(result.training_config),
        "history": list(result.history),
        "training_rows": result.training_rows,
        "nonzero_targets": result.nonzero_targets,
        "feature_names": list(result.feature_names),
        "trainable_frameworks": list(result.trainable_frameworks),
        "non_jittor_trainable_models": list(
            result.non_jittor_trainable_models
        ),
    }
    payload: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "normalizer_mean": np.asarray(result.mean, dtype=np.float32),
        "normalizer_std": np.asarray(result.std, dtype=np.float32),
    }
    payload.update(
        {
            f"state__{key}": np.asarray(value, dtype=np.float32)
            for key, value in _snapshot_state(model).items()
        }
    )
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_residual_advantage_router_checkpoint(
    path: Path,
) -> tuple[ResidualAdvantageRouter, ResidualAdvantageRouterFitResult]:
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        if (
            metadata.get("format") != CHECKPOINT_FORMAT
            or metadata.get("version") != CHECKPOINT_VERSION
            or metadata.get("trainable_frameworks") != ["jittor"]
            or metadata.get("non_jittor_trainable_models") != []
        ):
            raise ValueError("residual router checkpoint provenance differs")
        model_config = ResidualAdvantageRouterConfig(
            **metadata["model_config"]
        )
        training_config = ResidualAdvantageRouterTrainingConfig(
            **metadata["training_config"]
        )
        feature_names = tuple(
            str(name) for name in metadata.get("feature_names", [])
        )
        if (
            len(feature_names) != model_config.input_dim
            or len(set(feature_names)) != len(feature_names)
        ):
            raise ValueError("residual router checkpoint features differ")
        state = {
            key.removeprefix("state__"): np.asarray(
                payload[key],
                dtype=np.float32,
            ).copy()
            for key in payload.files
            if key.startswith("state__")
        }
        result = ResidualAdvantageRouterFitResult(
            model_config=model_config,
            training_config=training_config,
            mean=np.asarray(
                payload["normalizer_mean"],
                dtype=np.float32,
            ).copy(),
            std=np.asarray(
                payload["normalizer_std"],
                dtype=np.float32,
            ).copy(),
            state=state,
            history=tuple(metadata["history"]),
            training_rows=int(metadata["training_rows"]),
            nonzero_targets=int(metadata["nonzero_targets"]),
            feature_names=feature_names,
        )
    model = ResidualAdvantageRouter(model_config)
    _load_state(model, state)
    return model, result


def timestamp_router_split(query_times: Any) -> RouterTemporalSplit:
    """Create strict 60/20/20 timestamp groups."""
    times = np.asarray(query_times)
    if (
        times.ndim != 1
        or times.size < 5
        or not np.issubdtype(times.dtype, np.integer)
    ):
        raise ValueError("router split needs a non-empty integer time vector")
    times64 = times.astype(np.int64, copy=False)
    if np.any(np.diff(times64) < 0):
        raise ValueError("router split times must be non-decreasing")
    train_stop = _timestamp_boundary(times64, 0.6)
    selection_stop = _timestamp_boundary(times64, 0.8)
    if not 0 < train_stop < selection_stop < times64.size:
        raise ValueError("timestamps are too coarse for router split")
    if (
        times64[train_stop - 1] >= times64[train_stop]
        or times64[selection_stop - 1] >= times64[selection_stop]
    ):
        raise RuntimeError("router split divided an equal timestamp")
    return RouterTemporalSplit(
        train_rows=(0, train_stop),
        selection_rows=(train_stop, selection_stop),
        gate_rows=(selection_stop, int(times64.size)),
        train_time_max=int(times64[train_stop - 1]),
        selection_time_min=int(times64[train_stop]),
        selection_time_max=int(times64[selection_stop - 1]),
        gate_time_min=int(times64[selection_stop]),
        gate_time_max=int(times64[-1]),
    )


def _row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1, dtype=np.float64)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.asarray(
        numerator / np.maximum(denominator, 1e-12),
        dtype=np.float32,
    )


def _row_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values.astype(np.float64) - np.max(
        values,
        axis=1,
        keepdims=True,
    )
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=1, keepdims=True)


def _reciprocal_ranks(
    scores: np.ndarray,
    positive_indices: np.ndarray,
) -> np.ndarray:
    positives = scores[
        np.arange(scores.shape[0]),
        positive_indices,
    ][:, None]
    greater = np.sum(scores > positives, axis=1)
    equal = np.sum(scores == positives, axis=1)
    return 1.0 / (greater + (equal + 1.0) / 2.0)


def _timestamp_boundary(times: np.ndarray, fraction: float) -> int:
    target = min(max(int(times.size * fraction), 1), times.size - 1)
    return int(np.searchsorted(times, times[target], side="left"))


def _score_matrix(values: Any, *, label: str) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float32)
    if (
        scores.ndim != 2
        or scores.shape[0] <= 0
        or scores.shape[1] <= 1
        or not np.isfinite(scores).all()
    ):
        raise ValueError(f"{label} must be a finite score matrix")
    return scores


def _aligned_matrix(
    values: Any,
    reference: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    matrix = _score_matrix(values, label=label)
    if matrix.shape != reference.shape:
        raise ValueError(f"{label} does not align with default scores")
    return matrix
