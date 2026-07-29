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
from .sequence_model import TIME_DELTA_BUCKETS
from .source_sequence_cache import SourceSequenceRows

BOUNDED_SOURCE_DECODER_CHECKPOINT_FORMAT = "jgrec-bounded-source-decoder"
BOUNDED_SOURCE_DECODER_CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class BoundedSourceDecoderConfig:
    num_items: int
    embedding_dim: int = 32
    heads: int = 4
    source_max_length: int = 64
    time_bucket_count: int = TIME_DELTA_BUCKETS
    cap: float = 0.05
    support_tau: float = 20.0
    dropout: float = 0.10

    def __post_init__(self) -> None:
        if (
            self.num_items <= 0
            or self.embedding_dim <= 0
            or self.heads <= 0
            or self.source_max_length <= 0
            or self.time_bucket_count <= 0
        ):
            raise ValueError("bounded source decoder dimensions must be positive")
        if self.embedding_dim % self.heads != 0:
            raise ValueError("embedding_dim must be divisible by heads")
        if not math.isfinite(self.cap) or not 0.0 <= self.cap <= 0.10:
            raise ValueError("bounded source decoder cap must be in [0, 0.10]")
        if not math.isfinite(self.support_tau) or self.support_tau <= 0.0:
            raise ValueError("support_tau must be finite and positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("bounded source decoder dropout must be in [0, 1)")


@dataclass(frozen=True)
class BoundedSourceDecoderTrainingConfig:
    epochs: int = 3
    batch_size: int = 128
    learning_rate: float = 0.003
    weight_decay: float = 0.01
    seed: int = 60

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("bounded source training sizes must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("bounded source optimizer configuration is invalid")


@dataclass(frozen=True)
class BoundedSourceDecoderFitResult:
    model_config: BoundedSourceDecoderConfig
    training_config: BoundedSourceDecoderTrainingConfig
    state: dict[str, np.ndarray]
    history: tuple[dict[str, float | int], ...]
    training_rows: int
    trainable_frameworks: tuple[str, ...] = ("jittor",)
    non_jittor_trainable_models: tuple[str, ...] = ()


def support_shrinkage(
    candidate_support: jt.Var,
    *,
    tau: float,
) -> jt.Var:
    if len(candidate_support.shape) != 2:
        raise ValueError("candidate support must be a 2D matrix")
    strength = float(tau)
    if not math.isfinite(strength) or strength <= 0.0:
        raise ValueError("support shrinkage tau must be finite and positive")
    support = jt.maximum(candidate_support.float32(), 0.0)
    return jt.sqrt(support / (support + strength))


def bounded_source_residual_scores(
    base_logits: jt.Var,
    raw_residual_logits: jt.Var,
    shrinkage: jt.Var,
    has_history: jt.Var,
    *,
    cap: float,
) -> jt.Var:
    if (
        len(base_logits.shape) != 2
        or raw_residual_logits.shape != base_logits.shape
        or shrinkage.shape != base_logits.shape
        or has_history.shape != (base_logits.shape[0],)
    ):
        raise ValueError("bounded source residual tensors must align")
    limit = float(cap)
    if not math.isfinite(limit) or not 0.0 <= limit <= 0.10:
        raise ValueError("bounded source residual cap must be in [0, 0.10]")
    if limit == 0.0:
        return base_logits
    visible = has_history.float32().reshape((-1, 1))
    scaled = raw_residual_logits * shrinkage * visible
    centered = scaled - scaled.mean(dim=1, keepdims=True)
    maximum = jt.abs(centered).max(dim=1, keepdims=True)
    projection = jt.minimum(
        jt.ones_like(maximum),
        limit / jt.maximum(maximum, 1e-12),
    )
    return base_logits + centered * projection


class BoundedSourceDecoder(jt.nn.Module):
    """Source-history interaction residual with an exact frozen-base fallback."""

    def __init__(self, config: BoundedSourceDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.head_dim = config.embedding_dim // config.heads
        self.scale = math.sqrt(self.head_dim)
        self.item_embedding = jt.nn.Embedding(
            config.num_items + 1,
            config.embedding_dim,
        )
        self.time_embedding = jt.nn.Embedding(
            config.time_bucket_count + 1,
            config.embedding_dim,
        )
        self.position_embedding = jt.nn.Embedding(
            config.source_max_length,
            config.embedding_dim,
        )
        self.source_norm = jt.nn.LayerNorm(config.embedding_dim)
        self.candidate_norm = jt.nn.LayerNorm(config.embedding_dim)
        self.query = jt.nn.Linear(config.embedding_dim, config.embedding_dim)
        self.key = jt.nn.Linear(config.embedding_dim, config.embedding_dim)
        self.value = jt.nn.Linear(config.embedding_dim, config.embedding_dim)
        self.context_output = jt.nn.Linear(
            config.embedding_dim,
            config.embedding_dim,
        )
        self.attention_dropout = jt.nn.Dropout(config.dropout)
        self.output_dropout = jt.nn.Dropout(config.dropout)
        self.interaction_norm = jt.nn.LayerNorm(config.embedding_dim * 2)
        self.residual_head = jt.nn.Linear(config.embedding_dim * 2, 1)
        self.residual_head.weight.assign(
            jt.zeros_like(self.residual_head.weight)
        )
        if self.residual_head.bias is not None:
            self.residual_head.bias.assign(
                jt.zeros_like(self.residual_head.bias)
            )

    def execute(
        self,
        base_logits: jt.Var,
        candidate_ids: jt.Var,
        source_items: jt.Var,
        source_time_buckets: jt.Var,
        source_lengths: jt.Var,
        candidate_support: jt.Var,
    ) -> jt.Var:
        self._validate_inputs(
            base_logits,
            candidate_ids,
            source_items,
            source_time_buckets,
            source_lengths,
            candidate_support,
        )
        source_length = int(source_items.shape[1])
        positions = jt.arange(source_length).reshape((1, -1))
        source_mask = positions < source_lengths.unsqueeze(1)
        source_hidden = self.source_norm(
            self.item_embedding(source_items)
            + self.time_embedding(source_time_buckets)
            + self.position_embedding(positions)
        )
        candidate_hidden = self.candidate_norm(
            self.item_embedding(candidate_ids)
        )
        query = self._split_heads(self.query(candidate_hidden))
        key = self._split_heads(self.key(source_hidden)).permute(0, 1, 3, 2)
        value = self._split_heads(self.value(source_hidden))
        attention_logits = jt.matmul(query, key) / self.scale
        key_mask = source_mask.float32().unsqueeze(1).unsqueeze(1)
        attention_logits = attention_logits + (1.0 - key_mask) * -1e9
        attention = jt.nn.softmax(attention_logits, dim=-1)
        context = jt.matmul(self.attention_dropout(attention), value)
        context = context.permute(0, 2, 1, 3).reshape(
            (
                base_logits.shape[0],
                base_logits.shape[1],
                self.config.embedding_dim,
            )
        )
        context = self.output_dropout(self.context_output(context))
        interaction = self.interaction_norm(
            jt.concat((context, context * candidate_hidden), dim=-1)
        )
        raw = self.residual_head(interaction).reshape(base_logits.shape)
        shrinkage = support_shrinkage(
            candidate_support,
            tau=self.config.support_tau,
        )
        has_history = source_lengths > 0
        return bounded_source_residual_scores(
            base_logits,
            raw,
            shrinkage,
            has_history,
            cap=self.config.cap,
        )

    def _split_heads(self, values: jt.Var) -> jt.Var:
        return values.reshape(
            (
                values.shape[0],
                values.shape[1],
                self.config.heads,
                self.head_dim,
            )
        ).permute(0, 2, 1, 3)

    def _validate_inputs(
        self,
        base_logits: jt.Var,
        candidate_ids: jt.Var,
        source_items: jt.Var,
        source_time_buckets: jt.Var,
        source_lengths: jt.Var,
        candidate_support: jt.Var,
    ) -> None:
        if (
            len(base_logits.shape) != 2
            or candidate_ids.shape != base_logits.shape
            or candidate_support.shape != base_logits.shape
        ):
            raise ValueError("bounded source candidate tensors must align")
        if (
            source_items.shape
            != (base_logits.shape[0], self.config.source_max_length)
            or source_time_buckets.shape != source_items.shape
            or source_lengths.shape != (base_logits.shape[0],)
        ):
            raise ValueError("bounded source sequence tensors must align")


def fit_bounded_source_decoder_fixed(
    base_logits: Any,
    candidate_ids: Any,
    sequences: SourceSequenceRows,
    candidate_support: Any,
    positive_indices: np.ndarray,
    *,
    model_config: BoundedSourceDecoderConfig,
    training_config: BoundedSourceDecoderTrainingConfig,
    verbose: bool = True,
) -> tuple[BoundedSourceDecoder, BoundedSourceDecoderFitResult]:
    _validate_arrays(
        base_logits,
        candidate_ids,
        sequences,
        candidate_support,
        positive_indices,
        model_config=model_config,
        label="bounded source training",
    )
    jt.set_seed(training_config.seed)
    rng = np.random.default_rng(training_config.seed)
    positives = np.asarray(positive_indices, dtype=np.int32)
    model = BoundedSourceDecoder(model_config)
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
            scores = _model_batch(
                model,
                base_logits,
                candidate_ids,
                sequences,
                candidate_support,
                indices,
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
                f"non-finite bounded source loss at epoch {epoch}"
            )
        history.append({"epoch": epoch, "train_loss": train_loss})
        if verbose:
            print(
                "[bounded-source-decoder] "
                f"cap={model_config.cap:.3f} epoch={epoch} "
                f"train_loss={train_loss:.6f}",
                flush=True,
            )
    result = BoundedSourceDecoderFitResult(
        model_config=model_config,
        training_config=training_config,
        state=_snapshot_state(model),
        history=tuple(history),
        training_rows=int(base_logits.shape[0]),
    )
    return model, result


def predict_bounded_source_decoder_logits(
    model: BoundedSourceDecoder,
    base_logits: Any,
    candidate_ids: Any,
    sequences: SourceSequenceRows,
    candidate_support: Any,
    *,
    batch_size: int,
) -> np.ndarray:
    positives = np.zeros(int(base_logits.shape[0]), dtype=np.int32)
    _validate_arrays(
        base_logits,
        candidate_ids,
        sequences,
        candidate_support,
        positives,
        model_config=model.config,
        label="bounded source prediction",
    )
    if batch_size <= 0:
        raise ValueError("bounded source prediction batch must be positive")
    output = np.empty(base_logits.shape, dtype=np.float32)
    model.eval()
    with jt.no_grad():
        for start in range(0, int(base_logits.shape[0]), batch_size):
            stop = min(start + batch_size, int(base_logits.shape[0]))
            indices = slice(start, stop)
            output[start:stop] = _model_batch(
                model,
                base_logits,
                candidate_ids,
                sequences,
                candidate_support,
                indices,
            ).numpy()
    return output


def bounded_source_decoder_audit(
    base_logits: Any,
    scores: Any,
    source_lengths: Any,
    *,
    cap: float,
    tolerance: float = 2e-6,
) -> dict[str, float | int | bool]:
    base = np.asarray(base_logits, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    lengths = np.asarray(source_lengths)
    if (
        base.ndim != 2
        or values.shape != base.shape
        or lengths.shape != (base.shape[0],)
    ):
        raise ValueError("bounded source audit arrays must align")
    limit = float(cap)
    if not math.isfinite(limit) or not 0.0 <= limit <= 0.10:
        raise ValueError("bounded source audit cap must be in [0, 0.10]")
    residual = values - base
    maximum = float(np.max(np.abs(residual)))
    row_mean = float(np.max(np.abs(np.mean(residual, axis=1))))
    empty = lengths <= 0
    empty_exact = bool(np.array_equal(values[empty], base[empty]))
    passed = bool(
        maximum <= limit + tolerance
        and row_mean <= tolerance
        and empty_exact
        and np.isfinite(values).all()
    )
    return {
        "passed": passed,
        "max_absolute_residual": maximum,
        "cap": limit,
        "max_cap_violation": maximum - limit,
        "max_absolute_row_mean": row_mean,
        "empty_history_rows": int(np.sum(empty)),
        "empty_history_exact": empty_exact,
        "finite": bool(np.isfinite(values).all()),
        "tolerance": float(tolerance),
    }


def save_bounded_source_decoder_checkpoint(
    path: Path,
    model: BoundedSourceDecoder,
    result: BoundedSourceDecoderFitResult,
) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(
            f"bounded source checkpoint already exists: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": BOUNDED_SOURCE_DECODER_CHECKPOINT_FORMAT,
        "version": BOUNDED_SOURCE_DECODER_CHECKPOINT_VERSION,
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
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
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


def load_bounded_source_decoder_checkpoint(
    path: Path,
) -> tuple[BoundedSourceDecoder, BoundedSourceDecoderFitResult]:
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        _validate_checkpoint_metadata(metadata)
        model_config = BoundedSourceDecoderConfig(**metadata["model_config"])
        training_config = BoundedSourceDecoderTrainingConfig(
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
        result = BoundedSourceDecoderFitResult(
            model_config=model_config,
            training_config=training_config,
            state=state,
            history=tuple(
                {str(key): value for key, value in row.items()}
                for row in metadata["history"]
            ),
            training_rows=int(metadata["training_rows"]),
            trainable_frameworks=tuple(metadata["trainable_frameworks"]),
            non_jittor_trainable_models=tuple(
                metadata["non_jittor_trainable_models"]
            ),
        )
    model = BoundedSourceDecoder(model_config)
    _load_state(model, state)
    return model, result


def _model_batch(
    model: BoundedSourceDecoder,
    base_logits: Any,
    candidate_ids: Any,
    sequences: SourceSequenceRows,
    candidate_support: Any,
    indices: Any,
) -> jt.Var:
    return model(
        jt.array(
            np.asarray(base_logits[indices], dtype=np.float32),
            dtype=jt.float32,
        ),
        jt.array(
            np.asarray(candidate_ids[indices], dtype=np.int32),
            dtype=jt.int32,
        ),
        jt.array(
            np.asarray(sequences.items[indices], dtype=np.int32),
            dtype=jt.int32,
        ),
        jt.array(
            np.asarray(sequences.time_buckets[indices], dtype=np.int32),
            dtype=jt.int32,
        ),
        jt.array(
            np.asarray(sequences.lengths[indices], dtype=np.int32),
            dtype=jt.int32,
        ),
        jt.array(
            np.asarray(candidate_support[indices], dtype=np.float32),
            dtype=jt.float32,
        ),
    )


def _validate_arrays(
    base_logits: Any,
    candidate_ids: Any,
    sequences: SourceSequenceRows,
    candidate_support: Any,
    positive_indices: np.ndarray,
    *,
    model_config: BoundedSourceDecoderConfig,
    label: str,
) -> None:
    rows = int(base_logits.shape[0])
    if (
        len(base_logits.shape) != 2
        or rows <= 0
        or int(base_logits.shape[1]) <= 1
        or candidate_ids.shape != base_logits.shape
        or candidate_support.shape != base_logits.shape
        or sequences.items.shape
        != (rows, model_config.source_max_length)
        or sequences.time_buckets.shape != sequences.items.shape
        or sequences.lengths.shape != (rows,)
    ):
        raise ValueError(f"{label} arrays do not align")
    positives = np.asarray(positive_indices)
    if (
        positives.shape != (rows,)
        or not np.issubdtype(positives.dtype, np.integer)
        or np.any(positives < 0)
        or np.any(positives >= int(base_logits.shape[1]))
    ):
        raise ValueError(f"{label} positive indices are invalid")


def _validate_checkpoint_metadata(metadata: Any) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("invalid bounded source checkpoint metadata")
    if metadata.get("format") != BOUNDED_SOURCE_DECODER_CHECKPOINT_FORMAT:
        raise ValueError("unsupported bounded source checkpoint")
    if metadata.get("version") != BOUNDED_SOURCE_DECODER_CHECKPOINT_VERSION:
        raise ValueError("unsupported bounded source checkpoint version")
    if metadata.get("trainable_frameworks") != ["jittor"]:
        raise ValueError("bounded source checkpoint is not pure Jittor")
    if metadata.get("non_jittor_trainable_models") != []:
        raise ValueError("bounded source checkpoint contains non-Jittor models")
