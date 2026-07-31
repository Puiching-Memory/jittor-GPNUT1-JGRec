from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from .candidate_set_transformer import (
    _load_state,
    _normalize,
    _snapshot_state,
    _streaming_normalizer,
    candidate_set_listwise_loss,
)
from .oof_stacking import tie_neutral_mrr
from .source_conditioned_cst import (
    SourceConditionedCandidateSetTransformer,
    SourceConditionedCSTConfig,
)
from .source_sequence_cache import SourceSequenceRows

SOURCE_CONDITIONED_CHECKPOINT_FORMAT = "jgrec-source-conditioned-cst"
SOURCE_CONDITIONED_CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class SourceConditionedTrainingConfig:
    epochs: int = 6
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    seed: int = 60
    early_stop_patience: int = 2

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer configuration is invalid")
        if self.early_stop_patience < 0:
            raise ValueError("early_stop_patience must be non-negative")


@dataclass(frozen=True)
class SourceConditionedFitResult:
    model_config: SourceConditionedCSTConfig
    training_config: SourceConditionedTrainingConfig
    best_val_mrr: float
    best_epoch: int
    state: dict[str, np.ndarray]
    mean: np.ndarray
    std: np.ndarray
    history: tuple[dict[str, float | int], ...]
    selection_mode: str = "validation_best"
    training_rows: int = 0
    trainable_frameworks: tuple[str, ...] = ("jittor",)
    non_jittor_trainable_models: tuple[str, ...] = ()


def fit_source_conditioned_cst(
    train_features: Any,
    train_candidates: Any,
    train_sequences: SourceSequenceRows,
    train_positive_indices: np.ndarray,
    validation_features: Any,
    validation_candidates: Any,
    validation_sequences: SourceSequenceRows,
    validation_positive_indices: np.ndarray,
    *,
    model_config: SourceConditionedCSTConfig,
    training_config: SourceConditionedTrainingConfig,
    verbose: bool = True,
) -> tuple[
    SourceConditionedCandidateSetTransformer,
    SourceConditionedFitResult,
]:
    _validate_arrays(
        train_features,
        train_candidates,
        train_sequences,
        train_positive_indices,
        model_config=model_config,
        label="training",
    )
    _validate_arrays(
        validation_features,
        validation_candidates,
        validation_sequences,
        validation_positive_indices,
        model_config=model_config,
        label="validation",
    )
    if (
        int(train_features.shape[1]) != int(validation_features.shape[1])
        or int(train_features.shape[2])
        != int(validation_features.shape[2])
    ):
        raise ValueError("training and validation feature shapes differ")
    return _fit(
        train_features,
        train_candidates,
        train_sequences,
        np.asarray(train_positive_indices, dtype=np.int32),
        validation_features=validation_features,
        validation_candidates=validation_candidates,
        validation_sequences=validation_sequences,
        validation_positive_indices=np.asarray(
            validation_positive_indices,
            dtype=np.int32,
        ),
        model_config=model_config,
        training_config=training_config,
        selection_mode="validation_best",
        verbose=verbose,
    )


def fit_source_conditioned_cst_fixed(
    train_features: Any,
    train_candidates: Any,
    train_sequences: SourceSequenceRows,
    train_positive_indices: np.ndarray,
    *,
    model_config: SourceConditionedCSTConfig,
    training_config: SourceConditionedTrainingConfig,
    verbose: bool = True,
) -> tuple[
    SourceConditionedCandidateSetTransformer,
    SourceConditionedFitResult,
]:
    _validate_arrays(
        train_features,
        train_candidates,
        train_sequences,
        train_positive_indices,
        model_config=model_config,
        label="fixed training",
    )
    fixed_config = replace(training_config, early_stop_patience=0)
    return _fit(
        train_features,
        train_candidates,
        train_sequences,
        np.asarray(train_positive_indices, dtype=np.int32),
        validation_features=None,
        validation_candidates=None,
        validation_sequences=None,
        validation_positive_indices=None,
        model_config=model_config,
        training_config=fixed_config,
        selection_mode="fixed_full",
        verbose=verbose,
    )


def predict_source_conditioned_logits(
    model: SourceConditionedCandidateSetTransformer,
    features: Any,
    candidates: Any,
    sequences: SourceSequenceRows,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    positives = np.zeros(int(features.shape[0]), dtype=np.int32)
    _validate_arrays(
        features,
        candidates,
        sequences,
        positives,
        model_config=model.config,
        label="prediction",
    )
    if batch_size <= 0:
        raise ValueError("prediction batch_size must be positive")
    result = np.empty(features.shape[:2], dtype=np.float32)
    model.eval()
    with jt.no_grad():
        for start in range(0, int(features.shape[0]), int(batch_size)):
            stop = min(start + int(batch_size), int(features.shape[0]))
            result[start:stop] = np.asarray(
                _forward_batch(
                    model,
                    features,
                    candidates,
                    sequences,
                    np.arange(start, stop, dtype=np.int64),
                    mean,
                    std,
                ).numpy(),
                dtype=np.float32,
            )
    return result


def save_source_conditioned_checkpoint(
    path: Path,
    model: SourceConditionedCandidateSetTransformer,
    result: SourceConditionedFitResult,
) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(
            f"source-conditioned checkpoint already exists: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": SOURCE_CONDITIONED_CHECKPOINT_FORMAT,
        "version": SOURCE_CONDITIONED_CHECKPOINT_VERSION,
        "model_config": asdict(result.model_config),
        "training_config": asdict(result.training_config),
        "best_val_mrr": result.best_val_mrr,
        "best_epoch": result.best_epoch,
        "history": list(result.history),
        "selection_mode": result.selection_mode,
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
        "mean": np.asarray(result.mean, dtype=np.float32),
        "std": np.asarray(result.std, dtype=np.float32),
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


def load_source_conditioned_checkpoint(
    path: Path,
) -> tuple[
    SourceConditionedCandidateSetTransformer,
    SourceConditionedFitResult,
]:
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        _validate_checkpoint_metadata(metadata)
        model_config = SourceConditionedCSTConfig(
            **metadata["model_config"]
        )
        training_config = SourceConditionedTrainingConfig(
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
        result = SourceConditionedFitResult(
            model_config=model_config,
            training_config=training_config,
            best_val_mrr=float(metadata["best_val_mrr"]),
            best_epoch=int(metadata["best_epoch"]),
            state=state,
            mean=np.asarray(payload["mean"], dtype=np.float32).copy(),
            std=np.asarray(payload["std"], dtype=np.float32).copy(),
            history=tuple(
                {
                    str(key): value
                    for key, value in row.items()
                }
                for row in metadata["history"]
            ),
            selection_mode=str(metadata["selection_mode"]),
            training_rows=int(metadata["training_rows"]),
            trainable_frameworks=tuple(
                metadata["trainable_frameworks"]
            ),
            non_jittor_trainable_models=tuple(
                metadata["non_jittor_trainable_models"]
            ),
        )
    model = SourceConditionedCandidateSetTransformer(model_config)
    _load_state(model, state)
    return model, result


def _fit(
    train_features: Any,
    train_candidates: Any,
    train_sequences: SourceSequenceRows,
    train_positive_indices: np.ndarray,
    *,
    validation_features: Any | None,
    validation_candidates: Any | None,
    validation_sequences: SourceSequenceRows | None,
    validation_positive_indices: np.ndarray | None,
    model_config: SourceConditionedCSTConfig,
    training_config: SourceConditionedTrainingConfig,
    selection_mode: str,
    verbose: bool,
) -> tuple[
    SourceConditionedCandidateSetTransformer,
    SourceConditionedFitResult,
]:
    jt.set_seed(int(training_config.seed))
    rng = np.random.default_rng(training_config.seed)
    mean, std = _streaming_normalizer(
        train_features,
        batch_size=training_config.batch_size,
    )
    model = SourceConditionedCandidateSetTransformer(model_config)
    optimizer = jt.nn.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    best_mrr = -math.inf
    best_epoch = 0
    best_state = _snapshot_state(model)
    patience = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, training_config.epochs + 1):
        model.train()
        order = rng.permutation(int(train_features.shape[0]))
        losses: list[float] = []
        for start in range(
            0,
            int(train_features.shape[0]),
            training_config.batch_size,
        ):
            indices = order[
                start : start + training_config.batch_size
            ]
            logits = _forward_batch(
                model,
                train_features,
                train_candidates,
                train_sequences,
                indices,
                mean,
                std,
            )
            positives = jt.array(
                train_positive_indices[indices],
                dtype=jt.int32,
            )
            loss = candidate_set_listwise_loss(logits, positives)
            optimizer.step(loss)
            losses.append(float(loss.item()))
        train_loss = float(np.mean(losses))
        if not math.isfinite(train_loss):
            raise FloatingPointError(
                f"non-finite source-conditioned loss at epoch {epoch}"
            )
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
        }
        if validation_features is not None:
            assert validation_candidates is not None
            assert validation_sequences is not None
            assert validation_positive_indices is not None
            validation_logits = predict_source_conditioned_logits(
                model,
                validation_features,
                validation_candidates,
                validation_sequences,
                mean=mean,
                std=std,
                batch_size=training_config.batch_size,
            )
            val_mrr = tie_neutral_mrr(
                validation_logits,
                validation_positive_indices,
            )
            if val_mrr >= best_mrr:
                best_mrr = float(val_mrr)
                best_epoch = epoch
                best_state = _snapshot_state(model)
                patience = 0
            else:
                patience += 1
            row.update(
                {
                    "val_mrr": float(val_mrr),
                    "best_val_mrr": float(best_mrr),
                    "patience": patience,
                }
            )
        else:
            best_epoch = epoch
            best_state = _snapshot_state(model)
        history.append(row)
        if verbose:
            print(
                "[source-conditioned-cst] "
                f"variant={model_config.variant} epoch={epoch} "
                f"train_loss={train_loss:.6f}"
                + (
                    f" val_mrr={row['val_mrr']:.6f} "
                    f"best_val_mrr={best_mrr:.6f}"
                    if validation_features is not None
                    else ""
                ),
                flush=True,
            )
        if (
            validation_features is not None
            and training_config.early_stop_patience > 0
            and patience >= training_config.early_stop_patience
        ):
            break
    _load_state(model, best_state)
    return model, SourceConditionedFitResult(
        model_config=model_config,
        training_config=training_config,
        best_val_mrr=(
            float(best_mrr)
            if validation_features is not None
            else math.nan
        ),
        best_epoch=int(best_epoch),
        state=best_state,
        mean=mean,
        std=std,
        history=tuple(history),
        selection_mode=selection_mode,
        training_rows=int(train_features.shape[0]),
    )


def _forward_batch(
    model: SourceConditionedCandidateSetTransformer,
    features: Any,
    candidates: Any,
    sequences: SourceSequenceRows,
    indices: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> jt.Var:
    normalized = _normalize(
        np.asarray(features[indices], dtype=np.float32),
        mean,
        std,
    )
    return model(
        jt.array(normalized, dtype=jt.float32),
        jt.array(
            np.asarray(candidates[indices], dtype=np.int32),
            dtype=jt.int32,
        ),
        jt.array(
            np.asarray(sequences.items[indices], dtype=np.int32),
            dtype=jt.int32,
        ),
        jt.array(
            np.asarray(
                sequences.time_buckets[indices],
                dtype=np.int32,
            ),
            dtype=jt.int32,
        ),
        jt.array(
            np.asarray(sequences.lengths[indices], dtype=np.int32),
            dtype=jt.int32,
        ),
    )


def _validate_arrays(
    features: Any,
    candidates: Any,
    sequences: SourceSequenceRows,
    positive_indices: np.ndarray,
    *,
    model_config: SourceConditionedCSTConfig,
    label: str,
) -> None:
    if (
        len(features.shape) != 3
        or int(features.shape[0]) <= 0
        or int(features.shape[2]) != model_config.input_dim
    ):
        raise ValueError(f"{label} features differ from model config")
    if candidates.shape != features.shape[:2]:
        raise ValueError(f"{label} candidates do not align")
    rows = int(features.shape[0])
    expected_sequence_shape = (rows, model_config.source_max_length)
    if (
        sequences.items.shape != expected_sequence_shape
        or sequences.time_buckets.shape != expected_sequence_shape
        or sequences.lengths.shape != (rows,)
    ):
        raise ValueError(f"{label} source sequences do not align")
    positives = np.asarray(positive_indices)
    if (
        positives.shape != (rows,)
        or not np.issubdtype(positives.dtype, np.integer)
        or np.any(positives < 0)
        or np.any(positives >= int(features.shape[1]))
    ):
        raise ValueError(f"{label} positive indices are invalid")


def _validate_checkpoint_metadata(metadata: Any) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("invalid source-conditioned checkpoint metadata")
    if metadata.get("format") != SOURCE_CONDITIONED_CHECKPOINT_FORMAT:
        raise ValueError("unsupported source-conditioned checkpoint")
    if metadata.get("version") != SOURCE_CONDITIONED_CHECKPOINT_VERSION:
        raise ValueError("unsupported source-conditioned checkpoint version")
    if metadata.get("trainable_frameworks") != ["jittor"]:
        raise ValueError("checkpoint trainable framework is not pure Jittor")
    if metadata.get("non_jittor_trainable_models") != []:
        raise ValueError("checkpoint contains non-Jittor trainable models")
