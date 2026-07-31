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

BOUNDED_ID_RESIDUAL_CHECKPOINT_FORMAT = "jgrec-bounded-id-residual"
BOUNDED_ID_RESIDUAL_CHECKPOINT_VERSION = 2


@dataclass(frozen=True)
class BoundedIDResidualConfig:
    num_items: int
    embedding_dim: int = 32
    cap: float = 0.05
    dropout: float = 0.10

    def __post_init__(self) -> None:
        if self.num_items <= 0 or self.embedding_dim <= 0:
            raise ValueError("bounded ID dimensions must be positive")
        if not math.isfinite(self.cap) or self.cap < 0.0:
            raise ValueError("bounded ID cap must be finite and non-negative")
        if self.cap > 0.10:
            raise ValueError("bounded ID cap must be at most 0.10")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("bounded ID dropout must be in [0, 1)")


@dataclass(frozen=True)
class BoundedIDResidualTrainingConfig:
    epochs: int = 3
    batch_size: int = 512
    learning_rate: float = 0.01
    weight_decay: float = 0.001
    seed: int = 60

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("bounded ID training sizes must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("bounded ID optimizer configuration is invalid")


@dataclass(frozen=True)
class BoundedIDResidualFitResult:
    model_config: BoundedIDResidualConfig
    training_config: BoundedIDResidualTrainingConfig
    state: dict[str, np.ndarray]
    history: tuple[dict[str, float | int], ...]
    training_rows: int
    trainable_frameworks: tuple[str, ...] = ("jittor",)
    non_jittor_trainable_models: tuple[str, ...] = ()


def bounded_id_residual_scores(
    base_logits: jt.Var,
    raw_residual_logits: jt.Var,
    *,
    cap: float,
) -> jt.Var:
    """Add a candidate-ID residual with an absolute logit-space bound."""
    if (
        len(base_logits.shape) != 2
        or raw_residual_logits.shape != base_logits.shape
    ):
        raise ValueError("bounded ID logits must be aligned 2D matrices")
    limit = float(cap)
    if not math.isfinite(limit) or limit < 0.0:
        raise ValueError("bounded ID cap must be finite and non-negative")
    if limit > 0.10:
        raise ValueError("bounded ID cap must be at most 0.10")
    raw_centered = raw_residual_logits - raw_residual_logits.mean(
        dim=1,
        keepdims=True,
    )
    residual = limit * jt.tanh(raw_centered)
    return base_logits + residual


class BoundedIDResidual(jt.nn.Module):
    """Candidate-ID-only residual that cannot override its frozen base."""

    def __init__(self, config: BoundedIDResidualConfig) -> None:
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
    ) -> jt.Var:
        if (
            len(base_logits.shape) != 2
            or candidate_ids.shape != base_logits.shape
        ):
            raise ValueError(
                "bounded ID base logits and candidate IDs must align"
            )
        raw = self.output(
            self.dropout(self.item_embedding(candidate_ids))
        ).reshape(base_logits.shape)
        return bounded_id_residual_scores(
            base_logits,
            raw,
            cap=self.config.cap,
        )


def fit_bounded_id_residual_fixed(
    base_logits: Any,
    candidate_ids: Any,
    positive_indices: np.ndarray,
    *,
    model_config: BoundedIDResidualConfig,
    training_config: BoundedIDResidualTrainingConfig,
    verbose: bool = True,
) -> tuple[BoundedIDResidual, BoundedIDResidualFitResult]:
    _validate_arrays(
        base_logits,
        candidate_ids,
        positive_indices,
        label="bounded ID training",
    )
    jt.set_seed(int(training_config.seed))
    rng = np.random.default_rng(training_config.seed)
    positives = np.asarray(positive_indices, dtype=np.int32)
    model = BoundedIDResidual(model_config)
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
            indices = order[
                start : start + training_config.batch_size
            ]
            scores = model(
                jt.array(
                    np.asarray(base_logits[indices], dtype=np.float32),
                    dtype=jt.float32,
                ),
                jt.array(
                    np.asarray(candidate_ids[indices], dtype=np.int32),
                    dtype=jt.int32,
                ),
            )
            loss = candidate_set_listwise_loss(
                scores,
                jt.array(positives[indices], dtype=jt.int32),
            )
            optimizer.step(loss)
            losses.append(float(loss.item()))
        train_loss = float(np.mean(losses))
        if not math.isfinite(train_loss):
            raise FloatingPointError(
                f"non-finite bounded ID loss at epoch {epoch}"
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
            }
        )
        if verbose:
            print(
                "[bounded-id-residual] "
                f"cap={model_config.cap:.3f} epoch={epoch} "
                f"train_loss={train_loss:.6f}",
                flush=True,
            )
    state = _snapshot_state(model)
    return model, BoundedIDResidualFitResult(
        model_config=model_config,
        training_config=training_config,
        state=state,
        history=tuple(history),
        training_rows=int(base_logits.shape[0]),
    )


def predict_bounded_id_residual_logits(
    model: BoundedIDResidual,
    base_logits: Any,
    candidate_ids: Any,
    *,
    batch_size: int,
) -> np.ndarray:
    positives = np.zeros(int(base_logits.shape[0]), dtype=np.int32)
    _validate_arrays(
        base_logits,
        candidate_ids,
        positives,
        label="bounded ID prediction",
    )
    if batch_size <= 0:
        raise ValueError("bounded ID prediction batch_size must be positive")
    result = np.empty(base_logits.shape, dtype=np.float32)
    model.eval()
    with jt.no_grad():
        for start in range(0, int(base_logits.shape[0]), batch_size):
            stop = min(start + batch_size, int(base_logits.shape[0]))
            result[start:stop] = np.asarray(
                model(
                    jt.array(
                        np.asarray(
                            base_logits[start:stop],
                            dtype=np.float32,
                        ),
                        dtype=jt.float32,
                    ),
                    jt.array(
                        np.asarray(
                            candidate_ids[start:stop],
                            dtype=np.int32,
                        ),
                        dtype=jt.int32,
                    ),
                ).numpy(),
                dtype=np.float32,
            )
    return result


def bounded_id_residual_audit(
    base_logits: np.ndarray,
    scores: np.ndarray,
    *,
    cap: float,
    tolerance: float = 2e-6,
) -> dict[str, float | bool]:
    base = np.asarray(base_logits, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    if base.shape != values.shape or base.ndim != 2:
        raise ValueError("bounded ID audit arrays must align")
    limit = float(cap)
    if not math.isfinite(limit) or not 0.0 <= limit <= 0.10:
        raise ValueError("bounded ID audit cap must be in [0, 0.10]")
    allowed = np.full_like(base, limit)
    absolute = np.abs(values - base)
    violation = absolute - allowed
    positive_scale = allowed > 0.0
    ratios = np.zeros_like(absolute)
    np.divide(
        absolute,
        allowed,
        out=ratios,
        where=positive_scale,
    )
    max_violation = float(np.max(violation))
    return {
        "passed": bool(max_violation <= float(tolerance)),
        "max_absolute_residual": float(np.max(absolute)),
        "max_allowed_residual": float(np.max(allowed)),
        "max_bound_ratio": float(np.max(ratios)),
        "max_violation": max_violation,
        "tolerance": float(tolerance),
    }


def save_bounded_id_residual_checkpoint(
    path: Path,
    model: BoundedIDResidual,
    result: BoundedIDResidualFitResult,
) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(
            f"bounded ID checkpoint already exists: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": BOUNDED_ID_RESIDUAL_CHECKPOINT_FORMAT,
        "version": BOUNDED_ID_RESIDUAL_CHECKPOINT_VERSION,
        "model_config": asdict(result.model_config),
        "training_config": asdict(result.training_config),
        "history": list(result.history),
        "training_rows": result.training_rows,
        "trainable_frameworks": list(result.trainable_frameworks),
        "non_jittor_trainable_models": list(
            result.non_jittor_trainable_models
        ),
    }
    payload: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(
            json.dumps(metadata, sort_keys=True)
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


def load_bounded_id_residual_checkpoint(
    path: Path,
) -> tuple[BoundedIDResidual, BoundedIDResidualFitResult]:
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        _validate_checkpoint_metadata(metadata)
        model_config = BoundedIDResidualConfig(
            **metadata["model_config"]
        )
        training_config = BoundedIDResidualTrainingConfig(
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
        result = BoundedIDResidualFitResult(
            model_config=model_config,
            training_config=training_config,
            state=state,
            history=tuple(
                {
                    str(key): value
                    for key, value in row.items()
                }
                for row in metadata["history"]
            ),
            training_rows=int(metadata["training_rows"]),
            trainable_frameworks=tuple(
                metadata["trainable_frameworks"]
            ),
            non_jittor_trainable_models=tuple(
                metadata["non_jittor_trainable_models"]
            ),
        )
    model = BoundedIDResidual(model_config)
    _load_state(model, state)
    return model, result


def _validate_arrays(
    base_logits: Any,
    candidate_ids: Any,
    positive_indices: np.ndarray,
    *,
    label: str,
) -> None:
    if (
        len(base_logits.shape) != 2
        or int(base_logits.shape[0]) <= 0
        or int(base_logits.shape[1]) <= 1
        or candidate_ids.shape != base_logits.shape
    ):
        raise ValueError(f"{label} arrays do not align")
    positives = np.asarray(positive_indices)
    if (
        positives.shape != (int(base_logits.shape[0]),)
        or not np.issubdtype(positives.dtype, np.integer)
        or np.any(positives < 0)
        or np.any(positives >= int(base_logits.shape[1]))
    ):
        raise ValueError(f"{label} positive indices are invalid")


def _validate_checkpoint_metadata(metadata: Any) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("invalid bounded ID checkpoint metadata")
    if metadata.get("format") != BOUNDED_ID_RESIDUAL_CHECKPOINT_FORMAT:
        raise ValueError("unsupported bounded ID checkpoint")
    if metadata.get("version") != BOUNDED_ID_RESIDUAL_CHECKPOINT_VERSION:
        raise ValueError("unsupported bounded ID checkpoint version")
    if metadata.get("trainable_frameworks") != ["jittor"]:
        raise ValueError("bounded ID checkpoint is not pure Jittor")
    if metadata.get("non_jittor_trainable_models") != []:
        raise ValueError("bounded ID checkpoint contains non-Jittor models")
