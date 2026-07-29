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
from .high_confidence_topk_router import BoundedTopKAlternative

CHECKPOINT_FORMAT = "jgrec-joint-oof-lambdamrr"
CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class JointOOFLambdaMRRConfig:
    row_input_dim: int
    candidate_input_dim: int
    hidden_dim: int = 64
    dropout: float = 0.05

    def __post_init__(self) -> None:
        if (
            self.row_input_dim <= 0
            or self.candidate_input_dim <= 0
            or self.hidden_dim <= 0
        ):
            raise ValueError("joint model dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("joint model dropout must be in [0, 1)")


@dataclass(frozen=True)
class JointOOFLambdaMRRTrainingConfig:
    epochs: int = 8
    batch_size: int = 256
    learning_rate: float = 0.0005
    weight_decay: float = 0.0001
    reward_scale: float = 10.0
    nonzero_weight: float = 16.0
    route_loss_weight: float = 1.0
    rank_loss_weight: float = 0.1
    seed: int = 60

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("joint training sizes must be positive")
        finite_positive = (
            self.learning_rate,
            self.reward_scale,
            self.nonzero_weight,
            self.route_loss_weight,
            self.rank_loss_weight,
        )
        if (
            any(not math.isfinite(value) or value <= 0.0 for value in finite_positive)
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0.0
            or self.nonzero_weight < 1.0
        ):
            raise ValueError("joint optimizer settings are invalid")


@dataclass(frozen=True)
class JointOOFLambdaMRRFitResult:
    model_config: JointOOFLambdaMRRConfig
    training_config: JointOOFLambdaMRRTrainingConfig
    top_k: int
    cap: float
    row_mean: np.ndarray
    row_std: np.ndarray
    candidate_mean: np.ndarray
    candidate_std: np.ndarray
    state: dict[str, np.ndarray]
    history: tuple[dict[str, float | int], ...]
    training_rows: int
    nonzero_targets: int
    row_feature_names: tuple[str, ...]
    candidate_feature_names: tuple[str, ...]
    trainable_frameworks: tuple[str, ...] = ("jittor",)
    non_jittor_trainable_models: tuple[str, ...] = ()


class JointOOFLambdaMRRModel(jt.nn.Module):
    """Share row context between OOF routing and candidate residual heads."""

    def __init__(self, config: JointOOFLambdaMRRConfig) -> None:
        super().__init__()
        self.config = config
        self.row_input = jt.nn.Linear(
            config.row_input_dim,
            config.hidden_dim,
        )
        self.row_hidden = jt.nn.Linear(
            config.hidden_dim,
            config.hidden_dim,
        )
        self.route_output = jt.nn.Linear(config.hidden_dim, 2)
        self.candidate_input = jt.nn.Linear(
            config.candidate_input_dim,
            config.hidden_dim,
        )
        self.candidate_context = jt.nn.Linear(
            config.hidden_dim,
            config.hidden_dim,
        )
        self.candidate_output = jt.nn.Linear(config.hidden_dim, 2)
        self.dropout = jt.nn.Dropout(config.dropout)

    def execute(
        self,
        row_features: jt.Var,
        candidate_features: jt.Var,
    ) -> tuple[jt.Var, jt.Var]:
        if (
            len(row_features.shape) != 2
            or row_features.shape[1] != self.config.row_input_dim
        ):
            raise ValueError("joint row features do not align")
        if (
            len(candidate_features.shape) != 3
            or candidate_features.shape[0] != row_features.shape[0]
            or candidate_features.shape[2]
            != self.config.candidate_input_dim
        ):
            raise ValueError("joint candidate features do not align")

        row_hidden = jt.nn.relu(self.row_input(row_features))
        row_hidden = jt.nn.relu(
            self.row_hidden(self.dropout(row_hidden))
        )
        row_hidden = self.dropout(row_hidden)
        route_advantages = self.route_output(row_hidden)

        candidate_shape = candidate_features.shape
        candidate_hidden = self.candidate_input(
            candidate_features.reshape(
                (-1, self.config.candidate_input_dim)
            )
        ).reshape(
            (
                candidate_shape[0],
                candidate_shape[1],
                self.config.hidden_dim,
            )
        )
        context = self.candidate_context(row_hidden).unsqueeze(1)
        candidate_hidden = jt.nn.relu(candidate_hidden + context)
        candidate_residuals = self.candidate_output(
            self.dropout(candidate_hidden).reshape(
                (-1, self.config.hidden_dim)
            )
        ).reshape((candidate_shape[0], candidate_shape[1], 2))
        return route_advantages, candidate_residuals


def bounded_joint_topk_alternatives(
    default_scores: Any,
    base_alternative_scores: tuple[Any, Any],
    candidate_residuals: Any,
    *,
    top_k: int,
    cap: float,
) -> tuple[BoundedTopKAlternative, BoundedTopKAlternative]:
    """Project horizon plus learned corrections inside the default top-k."""
    default = _score_matrix(default_scores, label="joint default scores")
    if len(base_alternative_scores) != 2:
        raise ValueError("joint correction requires medium and long bases")
    bases = tuple(
        _aligned_matrix(value, default, label="joint base alternative")
        for value in base_alternative_scores
    )
    residuals = np.asarray(candidate_residuals, dtype=np.float32)
    if (
        residuals.shape != (*default.shape, 2)
        or not np.isfinite(residuals).all()
    ):
        raise ValueError("joint candidate residuals do not align")
    width = int(top_k)
    limit = float(cap)
    if width != top_k or not 1 <= width < default.shape[1]:
        raise ValueError("joint top_k must be below candidate count")
    if not math.isfinite(limit) or not 0.0 <= limit <= 0.10:
        raise ValueError("joint correction cap must be in [0, 0.10]")

    top_indices = np.argsort(-default, axis=1, kind="stable")[:, :width]
    mask = np.zeros(default.shape, dtype=bool)
    np.put_along_axis(mask, top_indices, True, axis=1)
    output = []
    for route_index, base in enumerate(bases):
        raw = np.where(
            mask,
            base - default + residuals[:, :, route_index],
            0.0,
        ).astype(np.float32, copy=False)
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
        delta = np.where(mask, centered * scale, 0.0).astype(
            np.float32,
            copy=False,
        )
        scores = np.asarray(default + delta, dtype=np.float32)
        scores[~mask] = default[~mask]
        actual_delta = np.asarray(scores - default, dtype=np.float32)
        actual_delta[~mask] = 0.0
        output.append(
            BoundedTopKAlternative(
                scores=scores,
                delta=actual_delta,
                topk_mask=mask.copy(),
                top_k=width,
                cap=limit,
            )
        )
    return output[0], output[1]


def joint_router_lambdamrr_loss(
    route_predictions: jt.Var,
    route_targets: jt.Var,
    adjusted_group_scores: jt.Var,
    pair_weights: jt.Var,
    *,
    route_loss_weight: float,
    rank_loss_weight: float,
    route_sample_weights: jt.Var | None = None,
) -> tuple[jt.Var, jt.Var, jt.Var]:
    """Combine route reward regression and route-specific LambdaMRR."""
    if (
        len(route_predictions.shape) != 2
        or route_predictions.shape[1] != 2
        or route_targets.shape != route_predictions.shape
    ):
        raise ValueError("joint route targets do not align")
    if (
        len(adjusted_group_scores.shape) != 3
        or adjusted_group_scores.shape[:2]
        != route_predictions.shape
        or adjusted_group_scores.shape[2] <= 1
        or pair_weights.shape
        != (
            adjusted_group_scores.shape[0],
            adjusted_group_scores.shape[1],
            adjusted_group_scores.shape[2] - 1,
        )
    ):
        raise ValueError("joint LambdaMRR groups do not align")
    if route_loss_weight < 0.0 or rank_loss_weight < 0.0:
        raise ValueError("joint loss weights must be non-negative")

    route_squared_error = (
        (route_predictions - route_targets)
        * (route_predictions - route_targets)
    )
    if route_sample_weights is None:
        route_loss = route_squared_error.mean()
    else:
        if route_sample_weights.shape != route_predictions.shape:
            raise ValueError("joint route sample weights do not align")
        route_loss = (
            route_squared_error * route_sample_weights
        ).sum() / route_sample_weights.sum()
    margins = (
        adjusted_group_scores[:, :, :1]
        - adjusted_group_scores[:, :, 1:]
    )
    rank_loss = (
        jt.nn.softplus(-margins) * pair_weights
    ).sum() / pair_weights.sum()
    total = (
        route_loss * route_loss_weight
        + rank_loss * rank_loss_weight
    )
    return total, route_loss, rank_loss


def fit_joint_oof_lambdamrr(
    row_features: Any,
    candidate_features: Any,
    default_scores: Any,
    base_alternative_scores: tuple[Any, Any],
    route_targets: Any,
    *,
    top_k: int,
    cap: float,
    model_config: JointOOFLambdaMRRConfig,
    training_config: JointOOFLambdaMRRTrainingConfig,
    row_feature_names: tuple[str, ...] | None = None,
    candidate_feature_names: tuple[str, ...] | None = None,
    verbose: bool = True,
) -> tuple[JointOOFLambdaMRRModel, JointOOFLambdaMRRFitResult]:
    rows = np.asarray(row_features, dtype=np.float32)
    default = _score_matrix(default_scores, label="joint training default")
    if len(base_alternative_scores) != 2:
        raise ValueError("joint training needs medium and long alternatives")
    bases = tuple(
        _aligned_matrix(value, default, label="joint training alternative")
        for value in base_alternative_scores
    )
    targets = np.asarray(route_targets, dtype=np.float32)
    candidate_shape = tuple(int(value) for value in candidate_features.shape)
    if (
        rows.ndim != 2
        or rows.shape[0] <= 1
        or rows.shape[1] != model_config.row_input_dim
        or candidate_shape
        != (
            rows.shape[0],
            default.shape[1],
            model_config.candidate_input_dim,
        )
        or default.shape[0] != rows.shape[0]
        or targets.shape != (rows.shape[0], 2)
        or not np.isfinite(rows).all()
        or not np.isfinite(targets).all()
    ):
        raise ValueError("joint training arrays do not align")
    width = int(top_k)
    limit = float(cap)
    if width != top_k or not 1 <= width < default.shape[1]:
        raise ValueError("joint training top_k is invalid")
    if not math.isfinite(limit) or not 0.0 < limit <= 0.10:
        raise ValueError("joint training cap must be in (0, 0.10]")
    row_names = _feature_names(
        row_feature_names,
        rows.shape[1],
        prefix="row",
    )
    candidate_names = _feature_names(
        candidate_feature_names,
        candidate_shape[2],
        prefix="candidate",
    )

    hard_negatives = _hard_negative_indices(
        default,
        top_k=width,
    )
    group_indices = np.concatenate(
        (
            np.zeros((default.shape[0], 1), dtype=np.int64),
            hard_negatives,
        ),
        axis=1,
    )
    top_indices = np.argsort(-default, axis=1, kind="stable")[:, :width]
    topk_mask = np.zeros(default.shape, dtype=bool)
    np.put_along_axis(topk_mask, top_indices, True, axis=1)
    group_active = _gather_candidates_2d(
        topk_mask,
        group_indices,
    ).astype(np.float32)
    pair_weights = np.stack(
        [
            _lambda_mrr_pair_weights(base, hard_negatives)
            for base in bases
        ],
        axis=1,
    )
    pair_weights *= group_active[:, None, 1:]
    if float(np.sum(pair_weights, dtype=np.float64)) <= 0.0:
        raise ValueError("joint LambdaMRR pair weights are empty")

    row_mean = rows.mean(axis=0, dtype=np.float64).astype(np.float32)
    row_std = rows.std(axis=0, dtype=np.float64).astype(np.float32)
    row_std[row_std < 1e-6] = 1.0
    normalized_rows = np.asarray(
        (rows - row_mean) / row_std,
        dtype=np.float32,
    )
    candidate_mean, candidate_std = _group_feature_normalizer(
        candidate_features,
        group_indices,
        batch_size=training_config.batch_size,
    )
    scaled_targets = np.asarray(
        targets * training_config.reward_scale,
        dtype=np.float32,
    )
    route_weights = np.where(
        targets != 0.0,
        training_config.nonzero_weight,
        1.0,
    ).astype(np.float32)

    jt.set_seed(training_config.seed)
    rng = np.random.default_rng(training_config.seed)
    model = JointOOFLambdaMRRModel(model_config)
    optimizer = jt.nn.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(1, training_config.epochs + 1):
        model.train()
        order = rng.permutation(rows.shape[0])
        total_values: list[float] = []
        route_values: list[float] = []
        rank_values: list[float] = []
        for start in range(0, rows.shape[0], training_config.batch_size):
            indices = order[start : start + training_config.batch_size]
            batch_candidates = np.asarray(
                candidate_features[indices],
                dtype=np.float32,
            )
            selected_candidates = _gather_candidates(
                batch_candidates,
                group_indices[indices],
            )
            normalized_candidates = np.asarray(
                (selected_candidates - candidate_mean) / candidate_std,
                dtype=np.float32,
            )
            default_group = _gather_candidates_2d(
                default[indices],
                group_indices[indices],
            )
            base_group = np.stack(
                [
                    _gather_candidates_2d(
                        base[indices],
                        group_indices[indices],
                    )
                    for base in bases
                ],
                axis=1,
            )
            route_predictions, candidate_residuals = model(
                jt.array(normalized_rows[indices], dtype=jt.float32),
                jt.array(normalized_candidates, dtype=jt.float32),
            )
            adjusted = _bounded_joint_group_scores(
                jt.array(default_group, dtype=jt.float32),
                jt.array(base_group, dtype=jt.float32),
                candidate_residuals,
                jt.array(group_active[indices], dtype=jt.float32),
                cap=limit,
            )
            total, route_loss, rank_loss = joint_router_lambdamrr_loss(
                route_predictions,
                jt.array(scaled_targets[indices], dtype=jt.float32),
                adjusted,
                jt.array(pair_weights[indices], dtype=jt.float32),
                route_loss_weight=training_config.route_loss_weight,
                rank_loss_weight=training_config.rank_loss_weight,
                route_sample_weights=jt.array(
                    route_weights[indices],
                    dtype=jt.float32,
                ),
            )
            optimizer.step(total)
            total_values.append(float(total.item()))
            route_values.append(float(route_loss.item()))
            rank_values.append(float(rank_loss.item()))
        row = {
            "epoch": epoch,
            "loss": float(np.mean(total_values)),
            "route_loss": float(np.mean(route_values)),
            "rank_loss": float(np.mean(rank_values)),
            "batches": len(total_values),
        }
        if not all(
            math.isfinite(float(row[key]))
            for key in ("loss", "route_loss", "rank_loss")
        ):
            raise FloatingPointError(
                f"non-finite joint loss at epoch {epoch}"
            )
        history.append(row)
        if verbose:
            print(
                f"[joint-oof-lambdamrr] epoch={epoch} "
                f"loss={row['loss']:.6f} "
                f"route={row['route_loss']:.6f} "
                f"rank={row['rank_loss']:.6f}",
                flush=True,
            )
    result = JointOOFLambdaMRRFitResult(
        model_config=model_config,
        training_config=training_config,
        top_k=width,
        cap=limit,
        row_mean=row_mean,
        row_std=row_std,
        candidate_mean=candidate_mean,
        candidate_std=candidate_std,
        state=_snapshot_state(model),
        history=tuple(history),
        training_rows=int(rows.shape[0]),
        nonzero_targets=int(np.count_nonzero(targets)),
        row_feature_names=row_names,
        candidate_feature_names=candidate_names,
    )
    return model, result


def predict_joint_oof_lambdamrr(
    model: JointOOFLambdaMRRModel,
    row_features: Any,
    candidate_features: Any,
    *,
    row_mean: Any,
    row_std: Any,
    candidate_mean: Any,
    candidate_std: Any,
    reward_scale: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(row_features, dtype=np.float32)
    row_means = np.asarray(row_mean, dtype=np.float32)
    row_scales = np.asarray(row_std, dtype=np.float32)
    candidate_means = np.asarray(candidate_mean, dtype=np.float32)
    candidate_scales = np.asarray(candidate_std, dtype=np.float32)
    candidate_shape = tuple(int(value) for value in candidate_features.shape)
    scale = float(reward_scale)
    if (
        rows.ndim != 2
        or rows.shape[1] != model.config.row_input_dim
        or candidate_shape
        != (
            rows.shape[0],
            candidate_shape[1],
            model.config.candidate_input_dim,
        )
        or row_means.shape != (rows.shape[1],)
        or row_scales.shape != row_means.shape
        or candidate_means.shape != (candidate_shape[2],)
        or candidate_scales.shape != candidate_means.shape
        or np.any(row_scales <= 0.0)
        or np.any(candidate_scales <= 0.0)
        or not math.isfinite(scale)
        or scale <= 0.0
        or batch_size <= 0
    ):
        raise ValueError("joint prediction arrays do not align")
    normalized_rows = np.asarray(
        (rows - row_means) / row_scales,
        dtype=np.float32,
    )
    route_output = np.empty((rows.shape[0], 2), dtype=np.float32)
    residual_output = np.empty(
        (rows.shape[0], candidate_shape[1], 2),
        dtype=np.float32,
    )
    model.eval()
    with jt.no_grad():
        for start in range(0, rows.shape[0], batch_size):
            stop = min(start + batch_size, rows.shape[0])
            candidates = np.asarray(
                candidate_features[start:stop],
                dtype=np.float32,
            )
            normalized_candidates = np.asarray(
                (candidates - candidate_means) / candidate_scales,
                dtype=np.float32,
            )
            routes, residuals = model(
                jt.array(normalized_rows[start:stop], dtype=jt.float32),
                jt.array(normalized_candidates, dtype=jt.float32),
            )
            route_output[start:stop] = routes.numpy() / scale
            residual_output[start:stop] = residuals.numpy()
    if (
        not np.isfinite(route_output).all()
        or not np.isfinite(residual_output).all()
    ):
        raise FloatingPointError("joint predictions are non-finite")
    return route_output, residual_output


def save_joint_oof_lambdamrr_checkpoint(
    path: Path,
    model: JointOOFLambdaMRRModel,
    result: JointOOFLambdaMRRFitResult,
) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"joint checkpoint exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "model_config": asdict(result.model_config),
        "training_config": asdict(result.training_config),
        "top_k": result.top_k,
        "cap": result.cap,
        "history": list(result.history),
        "training_rows": result.training_rows,
        "nonzero_targets": result.nonzero_targets,
        "row_feature_names": list(result.row_feature_names),
        "candidate_feature_names": list(result.candidate_feature_names),
        "trainable_frameworks": list(result.trainable_frameworks),
        "non_jittor_trainable_models": list(
            result.non_jittor_trainable_models
        ),
    }
    payload: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "row_mean": np.asarray(result.row_mean, dtype=np.float32),
        "row_std": np.asarray(result.row_std, dtype=np.float32),
        "candidate_mean": np.asarray(
            result.candidate_mean,
            dtype=np.float32,
        ),
        "candidate_std": np.asarray(
            result.candidate_std,
            dtype=np.float32,
        ),
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


def load_joint_oof_lambdamrr_checkpoint(
    path: Path,
) -> tuple[JointOOFLambdaMRRModel, JointOOFLambdaMRRFitResult]:
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        if (
            metadata.get("format") != CHECKPOINT_FORMAT
            or metadata.get("version") != CHECKPOINT_VERSION
            or metadata.get("trainable_frameworks") != ["jittor"]
            or metadata.get("non_jittor_trainable_models") != []
        ):
            raise ValueError("joint checkpoint provenance differs")
        model_config = JointOOFLambdaMRRConfig(
            **metadata["model_config"]
        )
        training_config = JointOOFLambdaMRRTrainingConfig(
            **metadata["training_config"]
        )
        state = {
            key.removeprefix("state__"): np.asarray(
                payload[key],
                dtype=np.float32,
            ).copy()
            for key in payload.files
            if key.startswith("state__")
        }
        result = JointOOFLambdaMRRFitResult(
            model_config=model_config,
            training_config=training_config,
            top_k=int(metadata["top_k"]),
            cap=float(metadata["cap"]),
            row_mean=np.asarray(
                payload["row_mean"],
                dtype=np.float32,
            ).copy(),
            row_std=np.asarray(
                payload["row_std"],
                dtype=np.float32,
            ).copy(),
            candidate_mean=np.asarray(
                payload["candidate_mean"],
                dtype=np.float32,
            ).copy(),
            candidate_std=np.asarray(
                payload["candidate_std"],
                dtype=np.float32,
            ).copy(),
            state=state,
            history=tuple(metadata["history"]),
            training_rows=int(metadata["training_rows"]),
            nonzero_targets=int(metadata["nonzero_targets"]),
            row_feature_names=tuple(metadata["row_feature_names"]),
            candidate_feature_names=tuple(
                metadata["candidate_feature_names"]
            ),
        )
    model = JointOOFLambdaMRRModel(model_config)
    _load_state(model, state)
    return model, result


def _bounded_joint_group_scores(
    default_group_scores: jt.Var,
    base_group_scores: jt.Var,
    candidate_residuals: jt.Var,
    active_mask: jt.Var,
    *,
    cap: float,
) -> jt.Var:
    if (
        len(default_group_scores.shape) != 2
        or base_group_scores.shape
        != (
            default_group_scores.shape[0],
            2,
            default_group_scores.shape[1],
        )
        or candidate_residuals.shape
        != (
            default_group_scores.shape[0],
            default_group_scores.shape[1],
            2,
        )
        or active_mask.shape != default_group_scores.shape
    ):
        raise ValueError("joint bounded group tensors do not align")
    limit = float(cap)
    if not math.isfinite(limit) or not 0.0 < limit <= 0.10:
        raise ValueError("joint bounded group cap is invalid")
    active = active_mask.unsqueeze(1)
    combined = (
        base_group_scores
        - default_group_scores.unsqueeze(1)
        + candidate_residuals.permute(0, 2, 1)
    ) * active
    count = jt.maximum(active.sum(dim=2, keepdims=True), 1.0)
    centered = (
        combined - combined.sum(dim=2, keepdims=True) / count
    ) * active
    maximum = jt.abs(centered).max(dim=2, keepdims=True)
    projection = jt.minimum(
        jt.ones_like(maximum),
        limit / jt.maximum(maximum, 1e-12),
    )
    return default_group_scores.unsqueeze(1) + centered * projection


def _group_feature_normalizer(
    candidate_features: Any,
    group_indices: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    feature_sum: np.ndarray | None = None
    feature_sq_sum: np.ndarray | None = None
    count = 0
    for start in range(0, group_indices.shape[0], batch_size):
        stop = min(start + batch_size, group_indices.shape[0])
        candidates = np.asarray(
            candidate_features[start:stop],
            dtype=np.float32,
        )
        selected = _gather_candidates(
            candidates,
            group_indices[start:stop],
        ).astype(np.float64, copy=False)
        flat = selected.reshape((-1, selected.shape[-1]))
        batch_sum = np.sum(flat, axis=0)
        batch_sq_sum = np.sum(flat * flat, axis=0)
        if feature_sum is None:
            feature_sum = batch_sum
            feature_sq_sum = batch_sq_sum
        else:
            feature_sum += batch_sum
            feature_sq_sum += batch_sq_sum
        count += flat.shape[0]
    if count <= 0 or feature_sum is None or feature_sq_sum is None:
        raise ValueError("joint candidate normalizer received no rows")
    mean64 = feature_sum / count
    variance64 = np.maximum(
        feature_sq_sum / count - mean64 * mean64,
        0.0,
    )
    mean = mean64.astype(np.float32)
    std = np.sqrt(variance64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def _feature_names(
    values: tuple[str, ...] | None,
    width: int,
    *,
    prefix: str,
) -> tuple[str, ...]:
    names = (
        tuple(str(value) for value in values)
        if values is not None
        else tuple(f"{prefix}_{index}" for index in range(width))
    )
    if len(names) != width or len(set(names)) != len(names):
        raise ValueError(f"joint {prefix} feature names do not align")
    return names


def _gather_candidates(
    values: np.ndarray,
    candidate_indices: np.ndarray,
) -> np.ndarray:
    rows = np.arange(values.shape[0])[:, None]
    return values[rows, candidate_indices]


def _gather_candidates_2d(
    values: np.ndarray,
    candidate_indices: np.ndarray,
) -> np.ndarray:
    rows = np.arange(values.shape[0])[:, None]
    return values[rows, candidate_indices]


def _hard_negative_indices(
    scores: np.ndarray,
    *,
    top_k: int,
) -> np.ndarray:
    order = np.argsort(-scores[:, 1:], axis=1, kind="stable")
    return order[:, :top_k].astype(np.int64, copy=False) + 1


def _lambda_mrr_pair_weights(
    scores: np.ndarray,
    hard_negative_indices: np.ndarray,
) -> np.ndarray:
    negatives = np.asarray(hard_negative_indices)
    positive_scores = scores[:, :1]
    positive_ranks = 1 + np.sum(
        scores > positive_scores,
        axis=1,
        dtype=np.int32,
    )
    negative_scores = np.take_along_axis(scores, negatives, axis=1)
    negative_ranks = 1 + np.sum(
        scores[:, :, None] > negative_scores[:, None, :],
        axis=1,
        dtype=np.int32,
    )
    return np.abs(
        1.0 / positive_ranks[:, None] - 1.0 / negative_ranks
    ).astype(np.float32, copy=False)


def _score_matrix(values: Any, *, label: str) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float32)
    if (
        scores.ndim != 2
        or scores.shape[1] < 2
        or not np.isfinite(scores).all()
    ):
        raise ValueError(f"{label} must be finite query-by-candidate")
    return scores


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
