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

CHECKPOINT_FORMAT = "jgrec-oof-utility-router"
CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class UtilityTargets:
    labels: np.ndarray
    changed: np.ndarray
    gain: np.ndarray
    magnitude: np.ndarray
    available: np.ndarray


@dataclass(frozen=True)
class UtilityPrediction:
    change_probability: np.ndarray
    gain_probability_given_change: np.ndarray
    expected_gain: np.ndarray
    expected_loss: np.ndarray
    expected_utility: np.ndarray


@dataclass(frozen=True)
class UtilityRoutingResult:
    scores: np.ndarray
    route_index: np.ndarray
    route_mask: np.ndarray
    selected_utility: np.ndarray
    selected_change_probability: np.ndarray
    quota: int


@dataclass(frozen=True)
class OOFUtilityRouterConfig:
    input_dim: int
    hidden_dim: int = 64
    dropout: float = 0.05

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("utility router dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("utility router dropout must be in [0, 1)")


@dataclass(frozen=True)
class OOFUtilityRouterTrainingConfig:
    epochs: int = 20
    batch_size: int = 512
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    reward_scale: float = 10.0
    change_positive_weight: float = 8.0
    loss_direction_weight: float = 2.0
    magnitude_weight: float = 0.5
    hard_negative_weight: float = 4.0
    regret_weight: float = 2.0
    seed: int = 60

    def __post_init__(self) -> None:
        positive = (
            self.learning_rate,
            self.reward_scale,
            self.change_positive_weight,
            self.loss_direction_weight,
            self.hard_negative_weight,
            self.regret_weight,
        )
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("utility router training sizes must be positive")
        if (
            any(not math.isfinite(value) or value <= 0.0 for value in positive)
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0.0
            or not math.isfinite(self.magnitude_weight)
            or self.magnitude_weight < 0.0
        ):
            raise ValueError("utility router optimizer settings are invalid")


@dataclass(frozen=True)
class OOFUtilityRouterFitResult:
    model_config: OOFUtilityRouterConfig
    training_config: OOFUtilityRouterTrainingConfig
    mean: np.ndarray
    std: np.ndarray
    state: dict[str, np.ndarray]
    history: tuple[dict[str, float | int], ...]
    training_rows: int
    gain_rows: int
    no_change_rows: int
    loss_rows: int
    hard_negative_rows: int
    feature_names: tuple[str, ...]
    trainable_frameworks: tuple[str, ...] = ("jittor",)
    non_jittor_trainable_models: tuple[str, ...] = ()


class OOFUtilityRouter(jt.nn.Module):
    """A pure-Jittor hurdle model for sparse route utility."""

    def __init__(self, config: OOFUtilityRouterConfig) -> None:
        super().__init__()
        self.config = config
        self.input = jt.nn.Linear(config.input_dim, config.hidden_dim)
        self.hidden = jt.nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = jt.nn.Dropout(config.dropout)
        self.output = jt.nn.Linear(config.hidden_dim, 4)

    def execute(self, features: jt.Var) -> jt.Var:
        if (
            len(features.shape) != 2
            or features.shape[1] != self.config.input_dim
        ):
            raise ValueError("utility router features do not align")
        hidden = jt.nn.relu(self.input(features))
        hidden = jt.nn.relu(self.hidden(self.dropout(hidden)))
        return self.output(self.dropout(hidden))


def build_utility_targets(
    rewards: Any,
    *,
    available: Any | None = None,
    zero_tolerance: float = 0.0,
) -> UtilityTargets:
    values = np.asarray(rewards, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("utility rewards must be a finite vector")
    if available is None:
        availability = np.ones(values.shape, dtype=bool)
    else:
        availability = np.asarray(available, dtype=bool)
        if availability.shape != values.shape:
            raise ValueError("utility availability does not align")
    tolerance = float(zero_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("utility zero tolerance must be non-negative")

    changed = availability & (np.abs(values) > tolerance)
    gain = changed & (values > 0.0)
    loss = changed & ~gain
    labels = np.full(values.shape, -1, dtype=np.int8)
    labels[availability] = 0
    labels[gain] = 1
    labels[loss] = 2
    magnitude = np.where(changed, np.abs(values), 0.0).astype(
        np.float32,
        copy=False,
    )
    return UtilityTargets(
        labels=labels,
        changed=changed,
        gain=gain,
        magnitude=magnitude,
        available=availability,
    )


def action_utility_features(
    default_scores: Any,
    short_residual: Any,
    action_residual: Any,
    short_gap_days: Any,
    action_gap_days: Any,
    action_scores: Any,
    candidate_features: Any,
    candidate_feature_names: tuple[str, ...],
    *,
    action_index: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build label-free features for one route action.

    Every candidate-derived value is selected by a score/residual operation, so
    applying the same candidate permutation to all inputs leaves the row
    representation unchanged.
    """

    default = _score_matrix(default_scores, label="utility default")
    short = _aligned_matrix(
        short_residual,
        default,
        label="utility short residual",
    )
    action_res = _aligned_matrix(
        action_residual,
        default,
        label="utility action residual",
    )
    action = _aligned_matrix(
        action_scores,
        default,
        label="utility action scores",
    )
    raw = np.asarray(candidate_features, dtype=np.float32)
    raw_names = tuple(str(name) for name in candidate_feature_names)
    short_gap = np.asarray(short_gap_days, dtype=np.float32)
    action_gap = np.asarray(action_gap_days, dtype=np.float32)
    if (
        raw.ndim != 3
        or raw.shape[:2] != default.shape
        or raw.shape[2] != len(raw_names)
        or not raw_names
        or len(set(raw_names)) != len(raw_names)
        or short_gap.shape != (default.shape[0],)
        or action_gap.shape != short_gap.shape
        or not np.isfinite(raw).all()
        or not np.isfinite(short_gap).all()
        or not np.isfinite(action_gap).all()
        or action_index not in (0, 1)
    ):
        raise ValueError("utility action feature inputs do not align")

    rows = np.arange(default.shape[0])
    default_top = np.argmax(default, axis=1)
    action_top = np.argmax(action, axis=1)
    switch = np.asarray(action - default, dtype=np.float32)
    promoted = np.argmax(switch, axis=1)
    demoted = np.argmin(switch, axis=1)
    default_std = np.maximum(
        np.std(default, axis=1, dtype=np.float64),
        1e-6,
    )
    action_std = np.maximum(
        np.std(action, axis=1, dtype=np.float64),
        1e-6,
    )
    default_ordered = np.sort(default, axis=1)
    action_ordered = np.sort(action, axis=1)
    probabilities = _row_softmax(
        (default - default.mean(axis=1, keepdims=True))
        / default_std[:, None]
    )
    changed = switch != 0.0
    top5_overlap = _topk_overlap(default, action, 5)
    top10_overlap = _topk_overlap(default, action, 10)
    row_columns = [
        (default_ordered[:, -1] - default_ordered[:, -2]) / default_std,
        -np.sum(
            probabilities * np.log(np.maximum(probabilities, 1e-30)),
            axis=1,
        )
        / math.log(default.shape[1]),
        np.max(probabilities, axis=1),
        default_std,
        default_ordered[:, -1] - default_ordered[:, 0],
        np.sqrt(np.mean(np.square(short), axis=1)),
        np.max(np.abs(short), axis=1),
        short[rows, default_top],
        np.sqrt(np.mean(np.square(action_res), axis=1)),
        np.max(np.abs(action_res), axis=1),
        action_res[rows, default_top],
        _row_cosine(short, action_res),
        np.linalg.norm(short - action_res, axis=1),
        (np.argmax(short, axis=1) == np.argmax(action_res, axis=1)).astype(
            np.float32
        ),
        np.max(np.abs(switch), axis=1),
        np.sqrt(np.mean(np.square(switch), axis=1)),
        np.mean(changed, axis=1),
        (action_top != default_top).astype(np.float32),
        (action_ordered[:, -1] - action_ordered[:, -2]) / action_std,
        _row_cosine(default, action),
        top5_overlap,
        top10_overlap,
        short_gap,
        action_gap,
        action_gap - short_gap,
        np.log1p(np.maximum(action_gap, 0.0)),
        np.full(default.shape[0], action_index == 0, dtype=np.float32),
        np.full(default.shape[0], action_index == 1, dtype=np.float32),
    ]
    row_names = (
        "default_margin_z",
        "default_entropy",
        "default_top_probability",
        "default_score_std",
        "default_score_range",
        "short_residual_rms",
        "short_residual_max_abs",
        "short_residual_default_top1",
        "action_residual_rms",
        "action_residual_max_abs",
        "action_residual_default_top1",
        "short_action_residual_cosine",
        "short_action_residual_l2",
        "short_action_max_candidate_agreement",
        "switch_max_abs",
        "switch_rms",
        "switch_changed_fraction",
        "switch_changes_top1",
        "action_margin_z",
        "default_action_score_cosine",
        "default_action_top5_overlap",
        "default_action_top10_overlap",
        "short_gap_days",
        "action_gap_days",
        "action_minus_short_gap_days",
        "log1p_action_gap_days",
        "action_is_medium",
        "action_is_long",
    )
    default_top_raw = raw[rows, default_top]
    action_top_delta = raw[rows, action_top] - default_top_raw
    promoted_demoted_delta = raw[rows, promoted] - raw[rows, demoted]
    output = np.concatenate(
        (
            np.column_stack(row_columns),
            default_top_raw,
            action_top_delta,
            promoted_demoted_delta,
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    names = (
        row_names
        + tuple(f"default_top__{name}" for name in raw_names)
        + tuple(f"action_top_minus_default__{name}" for name in raw_names)
        + tuple(f"promoted_minus_demoted__{name}" for name in raw_names)
    )
    if (
        output.shape != (default.shape[0], len(names))
        or len(set(names)) != len(names)
        or not np.isfinite(output).all()
    ):
        raise FloatingPointError("utility action features are invalid")
    return output, names


def utility_hurdle_loss(
    raw_outputs: jt.Var,
    scaled_rewards: jt.Var,
    *,
    change_positive_weight: float,
    loss_direction_weight: float,
    magnitude_weight: float,
    hard_negative_weights: jt.Var | None = None,
) -> tuple[jt.Var, dict[str, jt.Var]]:
    if (
        len(raw_outputs.shape) != 2
        or raw_outputs.shape[1] != 4
        or scaled_rewards.shape != (raw_outputs.shape[0],)
    ):
        raise ValueError("utility hurdle loss inputs do not align")
    if (
        change_positive_weight <= 0.0
        or loss_direction_weight <= 0.0
        or magnitude_weight < 0.0
    ):
        raise ValueError("utility hurdle loss weights are invalid")
    if hard_negative_weights is None:
        row_weight = jt.ones((raw_outputs.shape[0],), dtype=jt.float32)
    else:
        if hard_negative_weights.shape != scaled_rewards.shape:
            raise ValueError("hard-negative weights do not align")
        row_weight = hard_negative_weights

    changed = (jt.abs(scaled_rewards) > 0.0).float32()
    gain = (scaled_rewards > 0.0).float32()
    loss = changed * (1.0 - gain)
    change_logit = raw_outputs[:, 0]
    direction_logit = raw_outputs[:, 1]
    predicted_gain = jt.nn.softplus(raw_outputs[:, 2])
    predicted_loss = jt.nn.softplus(raw_outputs[:, 3])

    change_bce = jt.nn.softplus(change_logit) - changed * change_logit
    change_weight = (
        1.0 + changed * (float(change_positive_weight) - 1.0)
    ) * row_weight
    change_loss = (change_bce * change_weight).sum() / change_weight.sum()

    direction_bce = (
        jt.nn.softplus(direction_logit) - gain * direction_logit
    )
    direction_weight = changed * (
        1.0 + loss * (float(loss_direction_weight) - 1.0)
    ) * row_weight
    direction_loss = (
        direction_bce * direction_weight
    ).sum() / jt.maximum(direction_weight.sum(), 1.0)

    target_magnitude = jt.abs(scaled_rewards)
    gain_error = (predicted_gain - target_magnitude) ** 2
    loss_error = (predicted_loss - target_magnitude) ** 2
    magnitude_rows = gain * gain_error + loss * loss_error
    magnitude_denominator = (changed * row_weight).sum()
    magnitude_loss = (
        magnitude_rows * row_weight
    ).sum() / jt.maximum(magnitude_denominator, 1.0)

    total = (
        change_loss
        + direction_loss
        + float(magnitude_weight) * magnitude_loss
    )
    return total, {
        "change": change_loss,
        "direction": direction_loss,
        "magnitude": magnitude_loss,
    }


def fit_oof_utility_router(
    features: Any,
    rewards: Any,
    *,
    model_config: OOFUtilityRouterConfig,
    training_config: OOFUtilityRouterTrainingConfig,
    feature_names: tuple[str, ...] | None = None,
    hard_negative_mask: Any | None = None,
    verbose: bool = True,
) -> tuple[OOFUtilityRouter, OOFUtilityRouterFitResult]:
    values = np.asarray(features, dtype=np.float32)
    targets = np.asarray(rewards, dtype=np.float32)
    if (
        values.ndim != 2
        or values.shape[0] <= 1
        or values.shape[1] != model_config.input_dim
        or targets.shape != (values.shape[0],)
        or not np.isfinite(values).all()
        or not np.isfinite(targets).all()
    ):
        raise ValueError("utility router training arrays are invalid")
    if hard_negative_mask is None:
        hard = np.zeros(targets.shape, dtype=bool)
    else:
        hard = np.asarray(hard_negative_mask, dtype=bool)
        if hard.shape != targets.shape:
            raise ValueError("utility hard-negative mask does not align")
    names = (
        tuple(str(name) for name in feature_names)
        if feature_names is not None
        else tuple(f"feature_{index}" for index in range(values.shape[1]))
    )
    if (
        len(names) != values.shape[1]
        or len(set(names)) != len(names)
        or any("positive" in name.lower() for name in names)
    ):
        raise ValueError("utility router feature names do not align")

    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    normalized = np.asarray((values - mean) / std, dtype=np.float32)
    scaled = np.asarray(
        targets * training_config.reward_scale,
        dtype=np.float32,
    )
    hard_weights = np.where(
        hard,
        training_config.hard_negative_weight,
        1.0,
    ).astype(np.float32)

    jt.set_seed(training_config.seed)
    rng = np.random.default_rng(training_config.seed)
    model = OOFUtilityRouter(model_config)
    optimizer = jt.nn.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(1, training_config.epochs + 1):
        model.train()
        order = rng.permutation(values.shape[0])
        epoch_parts: dict[str, list[float]] = {
            "loss": [],
            "change": [],
            "direction": [],
            "magnitude": [],
        }
        for start in range(0, values.shape[0], training_config.batch_size):
            indices = order[start : start + training_config.batch_size]
            raw = model(jt.array(normalized[indices], dtype=jt.float32))
            total, parts = utility_hurdle_loss(
                raw,
                jt.array(scaled[indices], dtype=jt.float32),
                change_positive_weight=(
                    training_config.change_positive_weight
                ),
                loss_direction_weight=(
                    training_config.loss_direction_weight
                ),
                magnitude_weight=training_config.magnitude_weight,
                hard_negative_weights=jt.array(
                    hard_weights[indices],
                    dtype=jt.float32,
                ),
            )
            optimizer.step(total)
            epoch_parts["loss"].append(float(total.item()))
            for key, value in parts.items():
                epoch_parts[key].append(float(value.item()))
        row: dict[str, float | int] = {"epoch": epoch}
        for key, entries in epoch_parts.items():
            row[key] = float(np.mean(entries))
            if not math.isfinite(float(row[key])):
                raise FloatingPointError(
                    f"non-finite utility router {key} at epoch {epoch}"
                )
        history.append(row)
        if verbose:
            print(
                f"[utility-router] epoch={epoch} "
                f"loss={row['loss']:.6f} "
                f"change={row['change']:.6f} "
                f"direction={row['direction']:.6f} "
                f"magnitude={row['magnitude']:.6f}",
                flush=True,
            )

    utility_targets = build_utility_targets(targets)
    result = OOFUtilityRouterFitResult(
        model_config=model_config,
        training_config=training_config,
        mean=mean,
        std=std,
        state=_snapshot_state(model),
        history=tuple(history),
        training_rows=int(values.shape[0]),
        gain_rows=int(np.sum(utility_targets.labels == 1)),
        no_change_rows=int(np.sum(utility_targets.labels == 0)),
        loss_rows=int(np.sum(utility_targets.labels == 2)),
        hard_negative_rows=int(np.sum(hard)),
        feature_names=names,
    )
    return model, result


def predict_oof_utility(
    model: OOFUtilityRouter,
    features: Any,
    *,
    result: OOFUtilityRouterFitResult,
    batch_size: int,
) -> UtilityPrediction:
    values = np.asarray(features, dtype=np.float32)
    if (
        values.ndim != 2
        or values.shape[1] != model.config.input_dim
        or values.shape[1] != result.model_config.input_dim
        or result.mean.shape != (values.shape[1],)
        or result.std.shape != result.mean.shape
        or np.any(result.std <= 0.0)
        or batch_size <= 0
    ):
        raise ValueError("utility router prediction arrays are invalid")
    normalized = np.asarray(
        (values - result.mean) / result.std,
        dtype=np.float32,
    )
    raw_output = np.empty((values.shape[0], 4), dtype=np.float32)
    model.eval()
    with jt.no_grad():
        for start in range(0, values.shape[0], batch_size):
            stop = min(start + batch_size, values.shape[0])
            raw_output[start:stop] = model(
                jt.array(normalized[start:stop], dtype=jt.float32)
            ).numpy()
    return _decode_utility_output(
        raw_output,
        reward_scale=result.training_config.reward_scale,
        regret_weight=result.training_config.regret_weight,
    )


def route_by_expected_utility(
    default_scores: Any,
    alternative_scores: tuple[Any, Any],
    expected_utility: Any,
    change_probability: Any,
    *,
    available: Any | None = None,
    minimum_utility: float,
    minimum_change_probability: float,
    maximum_route_fraction: float,
) -> UtilityRoutingResult:
    default = _score_matrix(default_scores, label="utility route default")
    if len(alternative_scores) != 2:
        raise ValueError("utility route requires medium and long alternatives")
    alternatives = tuple(
        _aligned_matrix(value, default, label="utility route alternative")
        for value in alternative_scores
    )
    utility = np.asarray(expected_utility, dtype=np.float32)
    change = np.asarray(change_probability, dtype=np.float32)
    availability = (
        np.ones(utility.shape, dtype=bool)
        if available is None
        else np.asarray(available, dtype=bool)
    )
    min_utility = float(minimum_utility)
    min_change = float(minimum_change_probability)
    fraction = float(maximum_route_fraction)
    if (
        utility.shape != (default.shape[0], 2)
        or change.shape != utility.shape
        or availability.shape != utility.shape
        or not np.isfinite(utility).all()
        or not np.isfinite(change).all()
        or np.any((change < 0.0) | (change > 1.0))
        or not math.isfinite(min_utility)
        or not math.isfinite(min_change)
        or not 0.0 <= min_change <= 1.0
        or not math.isfinite(fraction)
        or not 0.0 <= fraction <= 1.0
    ):
        raise ValueError("utility route inputs are invalid")

    eligible_actions = (
        availability
        & (utility >= min_utility)
        & (change >= min_change)
    )
    masked_utility = np.where(eligible_actions, utility, -np.inf)
    best_action = np.argmax(masked_utility, axis=1)
    rows = np.arange(default.shape[0])
    best_utility = masked_utility[rows, best_action]
    best_change = change[rows, best_action]
    eligible_rows = np.flatnonzero(np.isfinite(best_utility))
    quota = math.floor(default.shape[0] * fraction)
    selected = eligible_rows[
        np.argsort(-best_utility[eligible_rows], kind="stable")
    ][:quota]

    route_index = np.zeros(default.shape[0], dtype=np.int8)
    route_index[selected] = (best_action[selected] + 1).astype(np.int8)
    route_mask = route_index > 0
    scores = np.array(default, copy=True)
    for action_index, action_scores in enumerate(alternatives, start=1):
        chosen = route_index == action_index
        scores[chosen] = action_scores[chosen]
    selected_utility = np.zeros(default.shape[0], dtype=np.float32)
    selected_change = np.zeros(default.shape[0], dtype=np.float32)
    selected_utility[selected] = best_utility[selected]
    selected_change[selected] = best_change[selected]
    return UtilityRoutingResult(
        scores=scores,
        route_index=route_index,
        route_mask=route_mask,
        selected_utility=selected_utility,
        selected_change_probability=selected_change,
        quota=quota,
    )


def save_oof_utility_router_checkpoint(
    path: Path,
    model: OOFUtilityRouter,
    result: OOFUtilityRouterFitResult,
) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"utility router checkpoint exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "model_config": asdict(result.model_config),
        "training_config": asdict(result.training_config),
        "history": list(result.history),
        "training_rows": result.training_rows,
        "gain_rows": result.gain_rows,
        "no_change_rows": result.no_change_rows,
        "loss_rows": result.loss_rows,
        "hard_negative_rows": result.hard_negative_rows,
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


def load_oof_utility_router_checkpoint(
    path: Path,
) -> tuple[OOFUtilityRouter, OOFUtilityRouterFitResult]:
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        if (
            metadata.get("format") != CHECKPOINT_FORMAT
            or metadata.get("version") != CHECKPOINT_VERSION
            or metadata.get("trainable_frameworks") != ["jittor"]
            or metadata.get("non_jittor_trainable_models") != []
        ):
            raise ValueError("utility router checkpoint provenance differs")
        model_config = OOFUtilityRouterConfig(**metadata["model_config"])
        training_config = OOFUtilityRouterTrainingConfig(
            **metadata["training_config"]
        )
        feature_names = tuple(
            str(name) for name in metadata.get("feature_names", [])
        )
        if (
            len(feature_names) != model_config.input_dim
            or len(set(feature_names)) != len(feature_names)
            or any("positive" in name.lower() for name in feature_names)
        ):
            raise ValueError("utility router checkpoint features differ")
        state = {
            key.removeprefix("state__"): np.asarray(
                payload[key],
                dtype=np.float32,
            ).copy()
            for key in payload.files
            if key.startswith("state__")
        }
        result = OOFUtilityRouterFitResult(
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
            gain_rows=int(metadata["gain_rows"]),
            no_change_rows=int(metadata["no_change_rows"]),
            loss_rows=int(metadata["loss_rows"]),
            hard_negative_rows=int(metadata["hard_negative_rows"]),
            feature_names=feature_names,
        )
    model = OOFUtilityRouter(model_config)
    _load_state(model, state)
    return model, result


def _decode_utility_output(
    raw_output: np.ndarray,
    *,
    reward_scale: float,
    regret_weight: float,
) -> UtilityPrediction:
    raw = np.asarray(raw_output, dtype=np.float32)
    scale = float(reward_scale)
    regret = float(regret_weight)
    if (
        raw.ndim != 2
        or raw.shape[1] != 4
        or not np.isfinite(raw).all()
        or not math.isfinite(scale)
        or scale <= 0.0
        or not math.isfinite(regret)
        or regret <= 0.0
    ):
        raise ValueError("utility output decode inputs are invalid")
    change = _sigmoid(raw[:, 0])
    gain_given_change = _sigmoid(raw[:, 1])
    expected_gain = np.logaddexp(0.0, raw[:, 2]) / scale
    expected_loss = np.logaddexp(0.0, raw[:, 3]) / scale
    utility = change * (
        gain_given_change * expected_gain
        - regret * (1.0 - gain_given_change) * expected_loss
    )
    return UtilityPrediction(
        change_probability=np.asarray(change, dtype=np.float32),
        gain_probability_given_change=np.asarray(
            gain_given_change,
            dtype=np.float32,
        ),
        expected_gain=np.asarray(expected_gain, dtype=np.float32),
        expected_loss=np.asarray(expected_loss, dtype=np.float32),
        expected_utility=np.asarray(utility, dtype=np.float32),
    )


def _topk_overlap(
    left: np.ndarray,
    right: np.ndarray,
    top_k: int,
) -> np.ndarray:
    width = min(int(top_k), left.shape[1])
    left_top = np.argsort(-left, axis=1, kind="stable")[:, :width]
    right_top = np.argsort(-right, axis=1, kind="stable")[:, :width]
    overlap = np.empty(left.shape[0], dtype=np.float32)
    for row in range(left.shape[0]):
        overlap[row] = np.intersect1d(
            left_top[row],
            right_top[row],
            assume_unique=True,
        ).size / float(width)
    return overlap


def _row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1, dtype=np.float64)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.asarray(
        numerator / np.maximum(denominator, 1e-12),
        dtype=np.float32,
    )


def _row_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=1, keepdims=True)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    output = np.empty(values.shape, dtype=np.float64)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output.astype(np.float32)


def _score_matrix(values: Any, *, label: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if (
        matrix.ndim != 2
        or matrix.shape[0] <= 0
        or matrix.shape[1] <= 1
        or not np.isfinite(matrix).all()
    ):
        raise ValueError(f"{label} must be a finite 2D matrix")
    return matrix


def _aligned_matrix(
    values: Any,
    reference: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.shape != reference.shape or not np.isfinite(matrix).all():
        raise ValueError(f"{label} does not align")
    return matrix
