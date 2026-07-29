from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from .candidate_set_transformer import (
    _load_state,
    _snapshot_state,
    candidate_set_listwise_loss,
)

CHECKPOINT_VERSION = 1
CORRECTION_FORMAT = "jgrec-topk-id-correction"
ROUTER_FORMAT = "jgrec-confidence-router"
ROUTER_FEATURE_NAMES = (
    "base_top1_margin_z",
    "base_entropy",
    "proposed_top1_margin_z",
    "top1_changed",
    "maximum_absolute_correction",
    "mean_absolute_topk_correction",
    "changed_topk_fraction",
    "base_top1_support_log1p",
    "proposed_top1_support_log1p",
    "proposed_vs_base_top1_support_delta",
    "maximum_topk_support_log1p",
    "mean_topk_support_log1p",
)


@dataclass(frozen=True)
class TopKIDCorrectionConfig:
    num_items: int
    embedding_dim: int = 32
    cap: float = 0.10
    dropout: float = 0.10

    def __post_init__(self) -> None:
        if self.num_items <= 0 or self.embedding_dim <= 0:
            raise ValueError("top-k correction dimensions must be positive")
        if not math.isfinite(self.cap) or not 0.0 <= self.cap <= 0.10:
            raise ValueError("top-k correction cap must be in [0, 0.10]")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("top-k correction dropout must be in [0, 1)")


@dataclass(frozen=True)
class TopKIDCorrectionTrainingConfig:
    epochs: int = 3
    batch_size: int = 512
    learning_rate: float = 0.01
    weight_decay: float = 0.001
    seed: int = 60

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("top-k correction training sizes must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("top-k correction optimizer is invalid")


@dataclass(frozen=True)
class TopKIDCorrectionFitResult:
    model_config: TopKIDCorrectionConfig
    training_config: TopKIDCorrectionTrainingConfig
    state: dict[str, np.ndarray]
    history: tuple[dict[str, float | int], ...]
    training_rows: int
    trainable_frameworks: tuple[str, ...] = ("jittor",)
    non_jittor_trainable_models: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConfidenceRouterConfig:
    input_dim: int
    hidden_dim: int = 16
    dropout: float = 0.05

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("confidence router dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("confidence router dropout must be in [0, 1)")


@dataclass(frozen=True)
class ConfidenceRouterTrainingConfig:
    epochs: int = 8
    batch_size: int = 512
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    seed: int = 60

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("confidence router training sizes must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("confidence router optimizer is invalid")


@dataclass(frozen=True)
class ConfidenceRouterFitResult:
    model_config: ConfidenceRouterConfig
    training_config: ConfidenceRouterTrainingConfig
    mean: np.ndarray
    std: np.ndarray
    state: dict[str, np.ndarray]
    history: tuple[dict[str, float | int], ...]
    training_rows: int
    positive_rows: int
    trainable_frameworks: tuple[str, ...] = ("jittor",)
    non_jittor_trainable_models: tuple[str, ...] = ()

    def predict(
        self,
        model: ConfidenceRouter,
        features: Any,
        *,
        batch_size: int,
    ) -> np.ndarray:
        return predict_confidence_router(
            model,
            features,
            mean=self.mean,
            std=self.std,
            batch_size=batch_size,
        )


@dataclass(frozen=True)
class SparseRoutingConfig:
    maximum_route_fraction: float
    minimum_probability: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.maximum_route_fraction <= 1.0:
            raise ValueError("maximum route fraction must be in [0, 1]")
        if not 0.0 <= self.minimum_probability <= 1.0:
            raise ValueError("minimum route probability must be in [0, 1]")


@dataclass(frozen=True)
class SparseRoutingResult:
    scores: np.ndarray
    route_mask: np.ndarray
    route_probabilities: np.ndarray
    quota: int


class TopKIDCorrection(jt.nn.Module):
    def __init__(self, config: TopKIDCorrectionConfig) -> None:
        super().__init__()
        self.config = config
        self.item_embedding = jt.nn.Embedding(
            config.num_items + 1,
            config.embedding_dim,
        )
        self.dropout = jt.nn.Dropout(config.dropout)
        self.output = jt.nn.Linear(config.embedding_dim, 1)
        self.output.weight.assign(jt.zeros_like(self.output.weight))
        if self.output.bias is not None:
            self.output.bias.assign(jt.zeros_like(self.output.bias))

    def execute(
        self,
        base_logits: jt.Var,
        candidate_ids: jt.Var,
        topk_mask: jt.Var,
    ) -> jt.Var:
        if (
            len(base_logits.shape) != 2
            or candidate_ids.shape != base_logits.shape
            or topk_mask.shape != base_logits.shape
        ):
            raise ValueError("top-k correction tensors must align")
        raw = self.output(
            self.dropout(self.item_embedding(candidate_ids))
        ).reshape(base_logits.shape)
        return topk_bounded_correction_scores(
            base_logits,
            raw,
            topk_mask,
            cap=self.config.cap,
        )


class ConfidenceRouter(jt.nn.Module):
    def __init__(self, config: ConfidenceRouterConfig) -> None:
        super().__init__()
        self.config = config
        self.input = jt.nn.Linear(config.input_dim, config.hidden_dim)
        self.dropout = jt.nn.Dropout(config.dropout)
        self.output = jt.nn.Linear(config.hidden_dim, 1)

    def execute(self, features: jt.Var) -> jt.Var:
        if len(features.shape) != 2:
            raise ValueError("confidence router features must be 2D")
        hidden = jt.nn.relu(self.input(features))
        return self.output(self.dropout(hidden)).reshape((-1,))


def topk_mask_from_scores(
    base_scores: Any,
    *,
    top_k: int,
) -> np.ndarray:
    base = _score_matrix(base_scores, label="top-k base")
    width = int(top_k)
    if width != top_k or not 1 <= width < base.shape[1]:
        raise ValueError("top_k must be an integer below candidate count")
    indices = np.argsort(-base, axis=1, kind="stable")[:, :width]
    mask = np.zeros(base.shape, dtype=bool)
    np.put_along_axis(mask, indices, True, axis=1)
    return mask


def topk_bounded_correction_scores(
    base_logits: jt.Var,
    raw_residual_logits: jt.Var,
    topk_mask: jt.Var,
    *,
    cap: float,
) -> jt.Var:
    if (
        len(base_logits.shape) != 2
        or raw_residual_logits.shape != base_logits.shape
        or topk_mask.shape != base_logits.shape
    ):
        raise ValueError("top-k bounded correction tensors must align")
    limit = float(cap)
    if not math.isfinite(limit) or not 0.0 <= limit <= 0.10:
        raise ValueError("top-k correction cap must be in [0, 0.10]")
    mask = topk_mask.float32()
    counts = mask.sum(dim=1, keepdims=True)
    centered = raw_residual_logits - (
        (raw_residual_logits * mask).sum(dim=1, keepdims=True)
        / jt.maximum(counts, 1.0)
    )
    residual = limit * jt.tanh(centered) * mask
    return base_logits + residual


def fit_topk_id_correction_fixed(
    base_logits: Any,
    candidate_ids: Any,
    topk_mask: Any,
    positive_indices: np.ndarray,
    *,
    model_config: TopKIDCorrectionConfig,
    training_config: TopKIDCorrectionTrainingConfig,
    verbose: bool = True,
) -> tuple[TopKIDCorrection, TopKIDCorrectionFitResult]:
    _validate_correction_arrays(
        base_logits,
        candidate_ids,
        topk_mask,
        positive_indices,
        label="top-k correction training",
    )
    jt.set_seed(training_config.seed)
    rng = np.random.default_rng(training_config.seed)
    positives = np.asarray(positive_indices, dtype=np.int32)
    model = TopKIDCorrection(model_config)
    optimizer = jt.nn.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(1, training_config.epochs + 1):
        model.train()
        order = rng.permutation(int(base_logits.shape[0]))
        losses: list[float] = []
        for start in range(
            0,
            int(base_logits.shape[0]),
            training_config.batch_size,
        ):
            indices = order[start : start + training_config.batch_size]
            scores = model(
                jt.array(
                    np.asarray(base_logits[indices], dtype=np.float32),
                    dtype=jt.float32,
                ),
                jt.array(
                    np.asarray(candidate_ids[indices], dtype=np.int32),
                    dtype=jt.int32,
                ),
                jt.array(
                    np.asarray(topk_mask[indices], dtype=np.float32),
                    dtype=jt.float32,
                ),
            )
            loss = candidate_set_listwise_loss(
                scores,
                jt.array(positives[indices], dtype=jt.int32),
            )
            optimizer.step(loss)
            losses.append(float(loss.item()))
        mean_loss = float(np.mean(losses))
        if not math.isfinite(mean_loss):
            raise FloatingPointError(
                f"non-finite top-k correction loss at epoch {epoch}"
            )
        history.append({"epoch": epoch, "train_loss": mean_loss})
        if verbose:
            print(
                "[topk-id-correction] "
                f"epoch={epoch} train_loss={mean_loss:.6f}",
                flush=True,
            )
    result = TopKIDCorrectionFitResult(
        model_config=model_config,
        training_config=training_config,
        state=_snapshot_state(model),
        history=tuple(history),
        training_rows=int(base_logits.shape[0]),
    )
    return model, result


def predict_topk_id_correction(
    model: TopKIDCorrection,
    base_logits: Any,
    candidate_ids: Any,
    topk_mask: Any,
    *,
    batch_size: int,
) -> np.ndarray:
    positives = np.zeros(int(base_logits.shape[0]), dtype=np.int32)
    _validate_correction_arrays(
        base_logits,
        candidate_ids,
        topk_mask,
        positives,
        label="top-k correction prediction",
    )
    if batch_size <= 0:
        raise ValueError("top-k correction prediction batch must be positive")
    output = np.empty(base_logits.shape, dtype=np.float32)
    model.eval()
    with jt.no_grad():
        for start in range(0, int(base_logits.shape[0]), batch_size):
            stop = min(start + batch_size, int(base_logits.shape[0]))
            output[start:stop] = model(
                jt.array(
                    np.asarray(base_logits[start:stop], dtype=np.float32),
                    dtype=jt.float32,
                ),
                jt.array(
                    np.asarray(candidate_ids[start:stop], dtype=np.int32),
                    dtype=jt.int32,
                ),
                jt.array(
                    np.asarray(topk_mask[start:stop], dtype=np.float32),
                    dtype=jt.float32,
                ),
            ).numpy()
    return output


def confidence_router_features(
    base_scores: Any,
    proposed_scores: Any,
    candidate_ids: Any,
    item_support: Any,
    topk_mask: Any,
) -> tuple[np.ndarray, tuple[str, ...]]:
    base = _score_matrix(base_scores, label="router base")
    proposed = _score_matrix(proposed_scores, label="router proposal")
    candidates = np.asarray(candidate_ids)
    mask = np.asarray(topk_mask, dtype=bool)
    support = np.asarray(item_support)
    if (
        proposed.shape != base.shape
        or candidates.shape != base.shape
        or mask.shape != base.shape
        or candidates.dtype.kind not in "iu"
        or support.ndim != 1
        or support.dtype.kind not in "iuf"
        or np.any(candidates < 0)
        or np.any(candidates >= support.shape[0])
        or np.any(mask.sum(axis=1) < 1)
    ):
        raise ValueError("confidence router inputs do not align")
    if not np.all(proposed[~mask] == base[~mask]):
        raise ValueError("router proposal changed candidates outside top-k")
    base_std = np.maximum(base.std(axis=1), 1e-6)
    proposed_std = np.maximum(proposed.std(axis=1), 1e-6)
    base_ordered = np.sort(base, axis=1)
    proposed_ordered = np.sort(proposed, axis=1)
    base_top = np.argmax(base, axis=1)
    proposed_top = np.argmax(proposed, axis=1)
    rows = np.arange(base.shape[0])
    correction = proposed - base
    absolute = np.abs(correction)
    masked_count = mask.sum(axis=1)
    changed = (absolute > 0.0) & mask
    candidate_support = np.log1p(
        np.maximum(support[candidates], 0.0)
    )
    base_support = candidate_support[rows, base_top]
    proposed_support = candidate_support[rows, proposed_top]
    masked_support = np.where(mask, candidate_support, -np.inf)
    maximum_support = np.max(masked_support, axis=1)
    mean_support = np.sum(
        np.where(mask, candidate_support, 0.0),
        axis=1,
    ) / masked_count
    probabilities = _row_softmax(
        (base - base.mean(axis=1, keepdims=True))
        / base_std[:, None]
    )
    entropy = -np.sum(
        probabilities
        * np.log(np.maximum(probabilities, np.finfo(np.float64).tiny)),
        axis=1,
    ) / math.log(base.shape[1])
    features = np.column_stack(
        (
            (base_ordered[:, -1] - base_ordered[:, -2]) / base_std,
            entropy,
            (
                proposed_ordered[:, -1]
                - proposed_ordered[:, -2]
            )
            / proposed_std,
            (base_top != proposed_top).astype(np.float64),
            absolute.max(axis=1),
            absolute.sum(axis=1) / masked_count,
            changed.sum(axis=1) / masked_count,
            base_support,
            proposed_support,
            proposed_support - base_support,
            maximum_support,
            mean_support,
        )
    ).astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise FloatingPointError("confidence router features are non-finite")
    return features, ROUTER_FEATURE_NAMES


def correction_improvement_labels(
    base_scores: Any,
    proposed_scores: Any,
    positive_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    base = _score_matrix(base_scores, label="label base")
    proposed = _score_matrix(proposed_scores, label="label proposal")
    positives = _positive_indices(
        positive_indices,
        rows=base.shape[0],
        candidates=base.shape[1],
    )
    if proposed.shape != base.shape:
        raise ValueError("correction label score matrices do not align")
    base_rr = _reciprocal_ranks(base, positives)
    proposed_rr = _reciprocal_ranks(proposed, positives)
    rewards = (proposed_rr - base_rr).astype(np.float32)
    return (rewards > 0.0).astype(np.float32), rewards


def fit_confidence_router(
    features: Any,
    labels: np.ndarray,
    *,
    model_config: ConfidenceRouterConfig,
    training_config: ConfidenceRouterTrainingConfig,
    verbose: bool = True,
) -> tuple[ConfidenceRouter, ConfidenceRouterFitResult]:
    values = np.asarray(features, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.float32)
    if (
        values.ndim != 2
        or values.shape[0] <= 1
        or values.shape[1] != model_config.input_dim
        or targets.shape != (values.shape[0],)
        or not np.all(np.isfinite(values))
        or not np.all((targets == 0.0) | (targets == 1.0))
    ):
        raise ValueError("confidence router training arrays are invalid")
    positive_rows = int(targets.sum())
    negative_rows = int(targets.shape[0] - positive_rows)
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    normalized = (values - mean) / std
    jt.set_seed(training_config.seed)
    rng = np.random.default_rng(training_config.seed)
    model = ConfidenceRouter(model_config)
    if positive_rows == 0 or negative_rows == 0:
        constant_logit = 20.0 if positive_rows else -20.0
        model.output.weight.assign(jt.zeros_like(model.output.weight))
        if model.output.bias is not None:
            model.output.bias.assign(
                jt.ones_like(model.output.bias) * constant_logit
            )
        history = (
            {
                "epoch": 0,
                "train_loss": 0.0,
                "degenerate_label": int(positive_rows > 0),
            },
        )
        return model, ConfidenceRouterFitResult(
            model_config=model_config,
            training_config=training_config,
            mean=mean,
            std=std,
            state=_snapshot_state(model),
            history=history,
            training_rows=values.shape[0],
            positive_rows=positive_rows,
        )
    positive_weight = float(negative_rows / positive_rows)
    optimizer = jt.nn.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(1, training_config.epochs + 1):
        model.train()
        order = rng.permutation(values.shape[0])
        losses: list[float] = []
        for start in range(
            0,
            values.shape[0],
            training_config.batch_size,
        ):
            indices = order[start : start + training_config.batch_size]
            batch_targets = jt.array(targets[indices], dtype=jt.float32)
            logits = model(
                jt.array(normalized[indices], dtype=jt.float32)
            )
            losses_per_row = (
                jt.maximum(logits, 0.0)
                - logits * batch_targets
                + jt.log(1.0 + jt.exp(-jt.abs(logits)))
            )
            weights = (
                batch_targets * positive_weight
                + (1.0 - batch_targets)
            )
            loss = (losses_per_row * weights).mean()
            optimizer.step(loss)
            losses.append(float(loss.item()))
        mean_loss = float(np.mean(losses))
        if not math.isfinite(mean_loss):
            raise FloatingPointError(
                f"non-finite confidence router loss at epoch {epoch}"
            )
        history.append({"epoch": epoch, "train_loss": mean_loss})
        if verbose:
            print(
                "[confidence-router] "
                f"epoch={epoch} train_loss={mean_loss:.6f}",
                flush=True,
            )
    result = ConfidenceRouterFitResult(
        model_config=model_config,
        training_config=training_config,
        mean=mean,
        std=std,
        state=_snapshot_state(model),
        history=tuple(history),
        training_rows=values.shape[0],
        positive_rows=positive_rows,
    )
    return model, result


def predict_confidence_router(
    model: ConfidenceRouter,
    features: Any,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    means = np.asarray(mean, dtype=np.float32)
    scales = np.asarray(std, dtype=np.float32)
    if (
        values.ndim != 2
        or values.shape[1] != model.config.input_dim
        or means.shape != (values.shape[1],)
        or scales.shape != means.shape
        or np.any(scales <= 0.0)
        or batch_size <= 0
    ):
        raise ValueError("confidence router prediction arrays are invalid")
    normalized = (values - means) / scales
    output = np.empty(values.shape[0], dtype=np.float32)
    model.eval()
    with jt.no_grad():
        for start in range(0, values.shape[0], batch_size):
            stop = min(start + batch_size, values.shape[0])
            output[start:stop] = jt.sigmoid(
                model(jt.array(normalized[start:stop], dtype=jt.float32))
            ).numpy()
    return output


def hard_confidence_route(
    base_scores: Any,
    proposed_scores: Any,
    route_probabilities: Any,
    *,
    config: SparseRoutingConfig,
) -> SparseRoutingResult:
    base = _score_matrix(base_scores, label="route base")
    proposed = _score_matrix(proposed_scores, label="route proposal")
    probabilities = np.asarray(route_probabilities, dtype=np.float32)
    if (
        proposed.shape != base.shape
        or probabilities.shape != (base.shape[0],)
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
    ):
        raise ValueError("confidence route inputs do not align")
    quota = math.floor(base.shape[0] * config.maximum_route_fraction)
    eligible = np.flatnonzero(
        probabilities >= config.minimum_probability
    )
    order = eligible[
        np.argsort(-probabilities[eligible], kind="stable")
    ]
    selected = order[:quota]
    route_mask = np.zeros(base.shape[0], dtype=bool)
    route_mask[selected] = True
    scores = base.copy()
    scores[route_mask] = proposed[route_mask]
    return SparseRoutingResult(
        scores=scores,
        route_mask=route_mask,
        route_probabilities=probabilities.copy(),
        quota=quota,
    )


def sparse_correction_audit(
    base_scores: Any,
    proposed_scores: Any,
    routed_scores: Any,
    topk_mask: Any,
    route_mask: Any,
    *,
    cap: float,
    maximum_route_fraction: float,
    tolerance: float = 2e-6,
) -> dict[str, float | int | bool]:
    base = _score_matrix(base_scores, label="audit base")
    proposed = _score_matrix(proposed_scores, label="audit proposal")
    routed = _score_matrix(routed_scores, label="audit routed")
    candidate_mask = np.asarray(topk_mask, dtype=bool)
    rows = np.asarray(route_mask, dtype=bool)
    if (
        proposed.shape != base.shape
        or routed.shape != base.shape
        or candidate_mask.shape != base.shape
        or rows.shape != (base.shape[0],)
    ):
        raise ValueError("sparse correction audit arrays do not align")
    maximum_rows = math.floor(
        base.shape[0] * maximum_route_fraction
    )
    absolute = np.abs(proposed - base)
    max_violation = float(np.max(absolute - float(cap)))
    outside_exact = bool(
        np.array_equal(proposed[~candidate_mask], base[~candidate_mask])
    )
    unrouted_exact = bool(
        np.array_equal(routed[~rows], base[~rows])
    )
    routed_matches = bool(
        np.array_equal(routed[rows], proposed[rows])
    )
    passed = bool(
        max_violation <= tolerance
        and outside_exact
        and unrouted_exact
        and routed_matches
        and int(rows.sum()) <= maximum_rows
    )
    return {
        "passed": passed,
        "topk_outside_exact": outside_exact,
        "unrouted_rows_exact": unrouted_exact,
        "routed_rows_match_proposal": routed_matches,
        "routed_rows": int(rows.sum()),
        "maximum_routed_rows": maximum_rows,
        "route_fraction": float(rows.mean()),
        "max_absolute_residual": float(np.max(absolute)),
        "max_bound_violation": max_violation,
        "cap": float(cap),
        "tolerance": float(tolerance),
    }


def save_topk_id_correction_checkpoint(
    path: Path,
    model: TopKIDCorrection,
    result: TopKIDCorrectionFitResult,
) -> None:
    metadata = {
        "format": CORRECTION_FORMAT,
        "version": CHECKPOINT_VERSION,
        "model_config": asdict(result.model_config),
        "training_config": asdict(result.training_config),
        "history": list(result.history),
        "training_rows": result.training_rows,
        "trainable_frameworks": list(result.trainable_frameworks),
        "non_jittor_trainable_models": list(
            result.non_jittor_trainable_models
        ),
    }
    _save_state_checkpoint(path, metadata, _snapshot_state(model))


def load_topk_id_correction_checkpoint(
    path: Path,
) -> tuple[TopKIDCorrection, TopKIDCorrectionFitResult]:
    metadata, state = _load_state_checkpoint(
        path,
        expected_format=CORRECTION_FORMAT,
    )
    model_config = TopKIDCorrectionConfig(**metadata["model_config"])
    training_config = TopKIDCorrectionTrainingConfig(
        **metadata["training_config"]
    )
    result = TopKIDCorrectionFitResult(
        model_config=model_config,
        training_config=training_config,
        state=state,
        history=tuple(metadata["history"]),
        training_rows=int(metadata["training_rows"]),
    )
    model = TopKIDCorrection(model_config)
    _load_state(model, state)
    return model, result


def save_confidence_router_checkpoint(
    path: Path,
    model: ConfidenceRouter,
    result: ConfidenceRouterFitResult,
) -> None:
    metadata = {
        "format": ROUTER_FORMAT,
        "version": CHECKPOINT_VERSION,
        "model_config": asdict(result.model_config),
        "training_config": asdict(result.training_config),
        "history": list(result.history),
        "training_rows": result.training_rows,
        "positive_rows": result.positive_rows,
        "trainable_frameworks": list(result.trainable_frameworks),
        "non_jittor_trainable_models": list(
            result.non_jittor_trainable_models
        ),
    }
    arrays = {
        "normalizer_mean": result.mean,
        "normalizer_std": result.std,
    }
    _save_state_checkpoint(
        path,
        metadata,
        _snapshot_state(model),
        arrays=arrays,
    )


def load_confidence_router_checkpoint(
    path: Path,
) -> tuple[ConfidenceRouter, ConfidenceRouterFitResult]:
    metadata, state, arrays = _load_state_checkpoint(
        path,
        expected_format=ROUTER_FORMAT,
        load_arrays=("normalizer_mean", "normalizer_std"),
    )
    model_config = ConfidenceRouterConfig(**metadata["model_config"])
    training_config = ConfidenceRouterTrainingConfig(
        **metadata["training_config"]
    )
    result = ConfidenceRouterFitResult(
        model_config=model_config,
        training_config=training_config,
        mean=arrays["normalizer_mean"],
        std=arrays["normalizer_std"],
        state=state,
        history=tuple(metadata["history"]),
        training_rows=int(metadata["training_rows"]),
        positive_rows=int(metadata["positive_rows"]),
    )
    model = ConfidenceRouter(model_config)
    _load_state(model, state)
    return model, result


def _validate_correction_arrays(
    base_logits: Any,
    candidate_ids: Any,
    topk_mask: Any,
    positive_indices: np.ndarray,
    *,
    label: str,
) -> None:
    if (
        len(base_logits.shape) != 2
        or int(base_logits.shape[0]) <= 0
        or int(base_logits.shape[1]) <= 1
        or candidate_ids.shape != base_logits.shape
        or topk_mask.shape != base_logits.shape
    ):
        raise ValueError(f"{label} arrays do not align")
    mask = np.asarray(topk_mask)
    if np.any(mask.sum(axis=1) < 1):
        raise ValueError(f"{label} top-k mask is empty")
    _positive_indices(
        positive_indices,
        rows=int(base_logits.shape[0]),
        candidates=int(base_logits.shape[1]),
    )


def _positive_indices(
    values: np.ndarray,
    *,
    rows: int,
    candidates: int,
) -> np.ndarray:
    positives = np.asarray(values)
    if (
        positives.shape != (rows,)
        or positives.dtype.kind not in "iu"
        or np.any(positives < 0)
        or np.any(positives >= candidates)
    ):
        raise ValueError("positive indices are invalid")
    return positives.astype(np.int64, copy=False)


def _reciprocal_ranks(
    scores: np.ndarray,
    positives: np.ndarray,
) -> np.ndarray:
    positive_scores = scores[
        np.arange(scores.shape[0]),
        positives,
    ][:, None]
    greater = np.sum(scores > positive_scores, axis=1)
    equal = np.sum(scores == positive_scores, axis=1)
    return 1.0 / (greater + (equal + 1.0) / 2.0)


def _row_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted, dtype=np.float64)
    return exponent / exponent.sum(axis=1, keepdims=True)


def _score_matrix(values: Any, *, label: str) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float32)
    if (
        scores.ndim != 2
        or scores.shape[0] <= 0
        or scores.shape[1] <= 1
        or not np.all(np.isfinite(scores))
    ):
        raise ValueError(f"{label} scores must be a finite 2D matrix")
    return scores


def _save_state_checkpoint(
    path: Path,
    metadata: dict[str, Any],
    state: dict[str, np.ndarray],
    *,
    arrays: dict[str, np.ndarray] | None = None,
) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"checkpoint already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        **{
            f"state__{key}": np.asarray(value, dtype=np.float32)
            for key, value in state.items()
        },
    }
    if arrays is not None:
        payload.update(
            {
                key: np.asarray(value, dtype=np.float32)
                for key, value in arrays.items()
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


def _load_state_checkpoint(
    path: Path,
    *,
    expected_format: str,
    load_arrays: tuple[str, ...] = (),
) -> (
    tuple[dict[str, Any], dict[str, np.ndarray]]
    | tuple[
        dict[str, Any],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ]
):
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        if (
            metadata.get("format") != expected_format
            or metadata.get("version") != CHECKPOINT_VERSION
            or metadata.get("trainable_frameworks") != ["jittor"]
            or metadata.get("non_jittor_trainable_models") != []
        ):
            raise ValueError("checkpoint provenance or version differs")
        state = {
            key.removeprefix("state__"): np.asarray(
                payload[key],
                dtype=np.float32,
            ).copy()
            for key in payload.files
            if key.startswith("state__")
        }
        arrays = {
            key: np.asarray(payload[key], dtype=np.float32).copy()
            for key in load_arrays
        }
    if load_arrays:
        return metadata, state, arrays
    return metadata, state
