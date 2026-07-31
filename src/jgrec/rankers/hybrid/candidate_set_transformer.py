from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

ALLOWED_FEATURE_PROVENANCE = frozenset(
    {"numpy_deterministic", "jittor"}
)
CANDIDATE_SET_CHECKPOINT_FORMAT = "jgrec-candidate-set-transformer"
CANDIDATE_SET_CHECKPOINT_VERSION = 1
CANDIDATE_SET_ENSEMBLE_FORMAT = "jgrec-candidate-set-transformer-ensemble"
CANDIDATE_SET_ENSEMBLE_VERSION = 1


def candidate_set_listwise_loss(
    logits: jt.Var,
    positive_indices: jt.Var,
    candidate_mask: jt.Var | None = None,
) -> jt.Var:
    """Softmax cross-entropy over the valid candidates of each query."""
    if len(logits.shape) != 2 or logits.shape[1] <= 0:
        raise ValueError(
            "candidate-set logits must have shape [queries, candidates]"
        )
    if (
        len(positive_indices.shape) != 1
        or positive_indices.shape[0] != logits.shape[0]
    ):
        raise ValueError(
            "candidate-set loss requires one positive index per query"
        )
    if candidate_mask is not None:
        if candidate_mask.shape != logits.shape:
            raise ValueError(
                "candidate-set mask must match the logits shape"
            )
        logits = jt.where(
            candidate_mask,
            logits,
            jt.full_like(logits, -1e9),
        )
    positions = jt.arange(logits.shape[1]).reshape((1, -1))
    positive_one_hot = (
        positions == positive_indices.unsqueeze(1)
    ).float()
    log_probabilities = jt.nn.log_softmax(logits, dim=1)
    return -(
        log_probabilities * positive_one_hot
    ).sum(dim=1).mean()


def candidate_relative_features(
    features: jt.Var,
    *,
    mode: str,
) -> jt.Var:
    """Build deterministic row-relative channels inside the Jittor graph."""
    if mode == "none":
        return features
    if mode != "mean_max":
        raise ValueError(
            f"unsupported candidate relative context mode: {mode}"
        )
    row_mean = features.mean(dim=1, keepdims=True)
    row_max = features.max(dim=1, keepdims=True)
    return jt.concat(
        [
            features,
            features - row_mean,
            features - row_max,
        ],
        dim=-1,
    )


@dataclass(frozen=True)
class CandidateSetTransformerConfig:
    input_dim: int
    model_dim: int = 64
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1
    feedforward_multiplier: int = 4
    relative_context: str = "none"
    pointwise_residual_dim: int = 0

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("candidate-set input_dim must be positive")
        if self.model_dim <= 0:
            raise ValueError("candidate-set model_dim must be positive")
        if self.heads <= 0 or self.model_dim % self.heads != 0:
            raise ValueError(
                "candidate-set model_dim must be divisible by heads"
            )
        if self.layers <= 0:
            raise ValueError("candidate-set layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("candidate-set dropout must be in [0, 1)")
        if self.feedforward_multiplier <= 0:
            raise ValueError(
                "candidate-set feedforward_multiplier must be positive"
            )
        if self.relative_context not in {"none", "mean_max"}:
            raise ValueError(
                "candidate-set relative_context must be none or mean_max"
            )
        if self.pointwise_residual_dim < 0:
            raise ValueError(
                "candidate-set pointwise_residual_dim must be non-negative"
            )


@dataclass(frozen=True)
class CandidateSetTrainingConfig:
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    seed: int = 60
    early_stop_patience: int = 0

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("candidate-set epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("candidate-set batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError(
                "candidate-set learning_rate must be positive"
            )
        if self.weight_decay < 0.0:
            raise ValueError(
                "candidate-set weight_decay must be non-negative"
            )
        if self.early_stop_patience < 0:
            raise ValueError(
                "candidate-set early_stop_patience must be non-negative"
            )


@dataclass(frozen=True)
class CandidateSetFitResult:
    model_config: CandidateSetTransformerConfig
    training_config: CandidateSetTrainingConfig
    best_val_mrr: float
    state: dict[str, np.ndarray]
    mean: np.ndarray
    std: np.ndarray
    feature_names: tuple[str, ...]
    feature_provenance: tuple[str, ...]
    history: tuple[dict[str, float | int], ...]
    selection_mode: str = "validation_best"
    training_rows: int = 0
    trainable_frameworks: tuple[str, ...] = ("jittor",)
    non_jittor_trainable_models: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateSetEnsembleCheckpoint:
    models: tuple[CandidateSetTransformer, ...]
    results: tuple[CandidateSetFitResult, ...]
    weights: tuple[float, ...]
    trainable_frameworks: tuple[str, ...] = ("jittor",)
    non_jittor_trainable_models: tuple[str, ...] = ()


def snapshot_candidate_set_ensemble(
    ensemble: CandidateSetEnsembleCheckpoint,
) -> dict[str, Any]:
    """Convert a pure-Jittor ensemble into contest-checkpoint state."""
    experts = tuple(
        zip(ensemble.models, ensemble.results, strict=True)
    )
    weights = _validate_ensemble_contract(experts, ensemble.weights)
    if (
        ensemble.trainable_frameworks != ("jittor",)
        or ensemble.non_jittor_trainable_models
    ):
        raise ValueError(
            "candidate-set ensemble has non-Jittor trainable provenance"
        )
    return {
        "format": CANDIDATE_SET_ENSEMBLE_FORMAT,
        "version": CANDIDATE_SET_ENSEMBLE_VERSION,
        "weights": weights,
        "blend": "fixed_probability",
        "trainable_frameworks": ("jittor",),
        "non_jittor_trainable_models": (),
        "experts": tuple(
            {
                "metadata": _fit_result_metadata(result),
                "state": _snapshot_state(model),
                "mean": np.asarray(
                    result.mean,
                    dtype=np.float32,
                ).copy(),
                "std": np.asarray(
                    result.std,
                    dtype=np.float32,
                ).copy(),
            }
            for model, result in experts
        ),
    }


def hydrate_candidate_set_ensemble(
    snapshot: dict[str, Any],
) -> CandidateSetEnsembleCheckpoint:
    """Restore contest-checkpoint state without legacy ML imports."""
    metadata = {
        "format": snapshot.get("format"),
        "version": snapshot.get("version"),
        "weights": list(snapshot.get("weights", ())),
        "blend": snapshot.get("blend"),
        "trainable_frameworks": list(
            snapshot.get("trainable_frameworks", ())
        ),
        "non_jittor_trainable_models": list(
            snapshot.get("non_jittor_trainable_models", ())
        ),
        "experts": list(snapshot.get("experts", ())),
    }
    _validate_ensemble_metadata(metadata)
    models: list[CandidateSetTransformer] = []
    results: list[CandidateSetFitResult] = []
    for expert in metadata["experts"]:
        if not isinstance(expert, dict):
            raise ValueError(
                "candidate-set ensemble expert state is invalid"
            )
        state = {
            str(key): np.asarray(value, dtype=np.float32).copy()
            for key, value in expert["state"].items()
        }
        result = _fit_result_from_metadata(
            expert["metadata"],
            state=state,
            mean=np.asarray(
                expert["mean"],
                dtype=np.float32,
            ).copy(),
            std=np.asarray(
                expert["std"],
                dtype=np.float32,
            ).copy(),
        )
        model = CandidateSetTransformer(result.model_config)
        _load_state(model, state)
        models.append(model)
        results.append(result)
    experts = tuple(zip(models, results, strict=True))
    weights = _validate_ensemble_contract(
        experts,
        tuple(float(value) for value in metadata["weights"]),
    )
    return CandidateSetEnsembleCheckpoint(
        models=tuple(models),
        results=tuple(results),
        weights=weights,
    )


def fit_candidate_set_transformer(
    train_features: Any,
    train_positive_indices: np.ndarray,
    validation_features: Any,
    validation_positive_indices: np.ndarray,
    *,
    model_config: CandidateSetTransformerConfig,
    training_config: CandidateSetTrainingConfig,
    feature_names: tuple[str, ...],
    feature_provenance: tuple[str, ...],
    verbose: bool = True,
) -> tuple[CandidateSetTransformer, CandidateSetFitResult]:
    """Fit a pure-Jittor candidate-set ranker on grouped features."""
    _validate_training_arrays(
        train_features,
        train_positive_indices,
        validation_features,
        validation_positive_indices,
        model_config=model_config,
    )
    names, provenance = _validate_feature_contract(
        feature_names,
        feature_provenance,
        input_dim=model_config.input_dim,
    )
    train_positive_indices = np.asarray(
        train_positive_indices,
        dtype=np.int32,
    )
    validation_positive_indices = np.asarray(
        validation_positive_indices,
        dtype=np.int32,
    )
    jt.set_seed(int(training_config.seed))
    rng = np.random.default_rng(training_config.seed)
    mean, std = _streaming_normalizer(
        train_features,
        batch_size=training_config.batch_size,
    )
    model = CandidateSetTransformer(model_config)
    optimizer = jt.nn.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    best_mrr = -math.inf
    best_state = _snapshot_state(model)
    patience = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, training_config.epochs + 1):
        model.train()
        order = rng.permutation(train_features.shape[0])
        losses: list[float] = []
        for start in range(
            0,
            train_features.shape[0],
            training_config.batch_size,
        ):
            batch_indices = order[
                start : start + training_config.batch_size
            ]
            batch = _normalize(
                np.asarray(train_features[batch_indices], dtype=np.float32),
                mean,
                std,
            )
            logits = model(jt.array(batch, dtype=jt.float32))
            positives = jt.array(
                train_positive_indices[batch_indices],
                dtype=jt.int32,
            )
            loss = candidate_set_listwise_loss(logits, positives)
            optimizer.step(loss)
            losses.append(float(loss.item()))
        train_loss = float(np.mean(losses))
        if not math.isfinite(train_loss):
            raise FloatingPointError(
                f"non-finite candidate-set loss at epoch {epoch}"
            )
        val_scores = predict_candidate_set_logits(
            model,
            validation_features,
            mean=mean,
            std=std,
            batch_size=training_config.batch_size,
        )
        val_mrr = _ranking_mrr(
            val_scores,
            validation_positive_indices,
        )
        if val_mrr >= best_mrr:
            best_mrr = val_mrr
            best_state = _snapshot_state(model)
            patience = 0
        else:
            patience += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_mrr": val_mrr,
                "patience": patience,
            }
        )
        if verbose:
            print(
                "[candidate-set-transformer] "
                f"epoch={epoch} train_loss={train_loss:.6f} "
                f"val_mrr={val_mrr:.6f} best_val_mrr={best_mrr:.6f}",
                flush=True,
            )
        if (
            training_config.early_stop_patience > 0
            and patience >= training_config.early_stop_patience
        ):
            break
    _load_state(model, best_state)
    return model, CandidateSetFitResult(
        model_config=model_config,
        training_config=training_config,
        best_val_mrr=float(best_mrr),
        state=best_state,
        mean=mean,
        std=std,
        feature_names=names,
        feature_provenance=provenance,
        history=tuple(history),
        selection_mode="validation_best",
        training_rows=int(train_features.shape[0]),
    )


def fit_candidate_set_transformer_fixed(
    train_features: Any,
    train_positive_indices: np.ndarray,
    *,
    model_config: CandidateSetTransformerConfig,
    training_config: CandidateSetTrainingConfig,
    feature_names: tuple[str, ...],
    feature_provenance: tuple[str, ...],
    verbose: bool = True,
) -> tuple[CandidateSetTransformer, CandidateSetFitResult]:
    """Train for fixed epochs on every row, without validation selection."""
    if len(train_features.shape) != 3 or int(train_features.shape[0]) <= 0:
        raise ValueError(
            "fixed candidate-set training features must be non-empty and 3D"
        )
    if int(train_features.shape[-1]) != model_config.input_dim:
        raise ValueError(
            "fixed candidate-set training feature dimension differs"
        )
    positives = np.asarray(train_positive_indices, dtype=np.int32)
    if positives.shape != (int(train_features.shape[0]),):
        raise ValueError(
            "fixed candidate-set training requires one positive per row"
        )
    if np.any(positives < 0) or np.any(
        positives >= int(train_features.shape[1])
    ):
        raise ValueError(
            "fixed candidate-set training positive index is out of range"
        )
    names, provenance = _validate_feature_contract(
        feature_names,
        feature_provenance,
        input_dim=model_config.input_dim,
    )
    jt.set_seed(int(training_config.seed))
    rng = np.random.default_rng(training_config.seed)
    mean, std = _streaming_normalizer(
        train_features,
        batch_size=training_config.batch_size,
    )
    model = CandidateSetTransformer(model_config)
    optimizer = jt.nn.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
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
            batch_indices = order[
                start : start + training_config.batch_size
            ]
            batch = _normalize(
                np.asarray(
                    train_features[batch_indices],
                    dtype=np.float32,
                ),
                mean,
                std,
            )
            logits = model(jt.array(batch, dtype=jt.float32))
            positive_var = jt.array(
                positives[batch_indices],
                dtype=jt.int32,
            )
            loss = candidate_set_listwise_loss(logits, positive_var)
            optimizer.step(loss)
            losses.append(float(loss.item()))
        train_loss = float(np.mean(losses))
        if not math.isfinite(train_loss):
            raise FloatingPointError(
                f"non-finite fixed candidate-set loss at epoch {epoch}"
            )
        history.append({"epoch": epoch, "train_loss": train_loss})
        if verbose:
            print(
                "[candidate-set-transformer:fixed] "
                f"epoch={epoch} train_loss={train_loss:.6f}",
                flush=True,
            )
    state = _snapshot_state(model)
    return model, CandidateSetFitResult(
        model_config=model_config,
        training_config=training_config,
        best_val_mrr=math.nan,
        state=state,
        mean=mean,
        std=std,
        feature_names=names,
        feature_provenance=provenance,
        history=tuple(history),
        selection_mode="fixed_full",
        training_rows=int(train_features.shape[0]),
    )


def predict_candidate_set_logits(
    model: CandidateSetTransformer,
    features: Any,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    if len(features.shape) != 3:
        raise ValueError(
            "candidate-set prediction features must have three dimensions"
        )
    if features.shape[-1] != model.config.input_dim:
        raise ValueError(
            "candidate-set prediction feature dimension differs"
        )
    if batch_size <= 0:
        raise ValueError(
            "candidate-set prediction batch_size must be positive"
        )
    result = np.empty(features.shape[:2], dtype=np.float32)
    model.eval()
    with jt.no_grad():
        for start in range(0, features.shape[0], batch_size):
            end = min(start + batch_size, features.shape[0])
            batch = _normalize(
                np.asarray(features[start:end], dtype=np.float32),
                mean,
                std,
            )
            scores = model(jt.array(batch, dtype=jt.float32))
            result[start:end] = np.asarray(
                scores.numpy(),
                dtype=np.float32,
            )
    return result


def save_candidate_set_checkpoint(
    path: Path,
    model: CandidateSetTransformer,
    result: CandidateSetFitResult,
) -> None:
    """Atomically save a NumPy-only, pure-Jittor model artifact."""
    target = Path(path)
    if target.exists():
        raise FileExistsError(
            f"candidate-set checkpoint already exists: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": CANDIDATE_SET_CHECKPOINT_FORMAT,
        "version": CANDIDATE_SET_CHECKPOINT_VERSION,
        "model_config": asdict(result.model_config),
        "training_config": asdict(result.training_config),
        "best_val_mrr": result.best_val_mrr,
        "feature_names": list(result.feature_names),
        "feature_provenance": list(result.feature_provenance),
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
            json.dumps(metadata, sort_keys=True),
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


def load_candidate_set_checkpoint(
    path: Path,
) -> tuple[CandidateSetTransformer, CandidateSetFitResult]:
    """Load a pure-Jittor artifact without importing other ML libraries."""
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        _validate_checkpoint_metadata(metadata)
        model_config = CandidateSetTransformerConfig(
            **metadata["model_config"]
        )
        training_config = CandidateSetTrainingConfig(
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
        result = CandidateSetFitResult(
            model_config=model_config,
            training_config=training_config,
            best_val_mrr=float(metadata["best_val_mrr"]),
            state=state,
            mean=np.asarray(payload["mean"], dtype=np.float32).copy(),
            std=np.asarray(payload["std"], dtype=np.float32).copy(),
            feature_names=tuple(metadata["feature_names"]),
            feature_provenance=tuple(metadata["feature_provenance"]),
            history=tuple(
                {
                    str(key): value
                    for key, value in row.items()
                }
                for row in metadata["history"]
            ),
            selection_mode=str(
                metadata.get("selection_mode", "validation_best")
            ),
            training_rows=int(metadata.get("training_rows", 0)),
            trainable_frameworks=tuple(
                metadata["trainable_frameworks"]
            ),
            non_jittor_trainable_models=tuple(
                metadata["non_jittor_trainable_models"]
            ),
        )
    _validate_feature_contract(
        result.feature_names,
        result.feature_provenance,
        input_dim=model_config.input_dim,
    )
    model = CandidateSetTransformer(model_config)
    _load_state(model, state)
    return model, result


def save_candidate_set_ensemble_checkpoint(
    path: Path,
    experts: tuple[
        tuple[CandidateSetTransformer, CandidateSetFitResult],
        ...,
    ],
    *,
    weights: tuple[float, ...],
) -> None:
    """Save multiple Jittor experts plus a fixed probability blend."""
    normalized_weights = _validate_ensemble_contract(experts, weights)
    target = Path(path)
    if target.exists():
        raise FileExistsError(
            f"candidate-set ensemble checkpoint already exists: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": CANDIDATE_SET_ENSEMBLE_FORMAT,
        "version": CANDIDATE_SET_ENSEMBLE_VERSION,
        "weights": list(normalized_weights),
        "blend": "fixed_probability",
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
        "experts": [
            _fit_result_metadata(result)
            for _model, result in experts
        ],
    }
    payload: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(
            json.dumps(metadata, sort_keys=True)
        )
    }
    for index, (model, result) in enumerate(experts):
        prefix = f"expert_{index}__"
        payload[f"{prefix}mean"] = np.asarray(
            result.mean,
            dtype=np.float32,
        )
        payload[f"{prefix}std"] = np.asarray(
            result.std,
            dtype=np.float32,
        )
        payload.update(
            {
                f"{prefix}state__{key}": np.asarray(
                    value,
                    dtype=np.float32,
                )
                for key, value in _snapshot_state(model).items()
            }
        )
    _save_npz_atomic(target, payload)


def load_candidate_set_ensemble_checkpoint(
    path: Path,
) -> CandidateSetEnsembleCheckpoint:
    """Load a fixed ensemble without importing external ML libraries."""
    models: list[CandidateSetTransformer] = []
    results: list[CandidateSetFitResult] = []
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        _validate_ensemble_metadata(metadata)
        for index, expert_metadata in enumerate(metadata["experts"]):
            prefix = f"expert_{index}__"
            state = {
                key.removeprefix(f"{prefix}state__"): np.asarray(
                    payload[key],
                    dtype=np.float32,
                ).copy()
                for key in payload.files
                if key.startswith(f"{prefix}state__")
            }
            result = _fit_result_from_metadata(
                expert_metadata,
                state=state,
                mean=np.asarray(
                    payload[f"{prefix}mean"],
                    dtype=np.float32,
                ).copy(),
                std=np.asarray(
                    payload[f"{prefix}std"],
                    dtype=np.float32,
                ).copy(),
            )
            model = CandidateSetTransformer(result.model_config)
            _load_state(model, state)
            models.append(model)
            results.append(result)
    experts = tuple(zip(models, results, strict=True))
    weights = _validate_ensemble_contract(
        experts,
        tuple(float(value) for value in metadata["weights"]),
    )
    return CandidateSetEnsembleCheckpoint(
        models=tuple(models),
        results=tuple(results),
        weights=weights,
    )


def predict_candidate_set_ensemble_probabilities(
    ensemble: CandidateSetEnsembleCheckpoint,
    features: Any,
    *,
    batch_size: int,
) -> np.ndarray:
    probabilities = np.zeros(features.shape[:2], dtype=np.float32)
    for weight, model, result in zip(
        ensemble.weights,
        ensemble.models,
        ensemble.results,
        strict=True,
    ):
        logits = predict_candidate_set_logits(
            model,
            features,
            mean=result.mean,
            std=result.std,
            batch_size=batch_size,
        )
        shifted = logits - logits.max(axis=1, keepdims=True)
        expert_probabilities = np.exp(shifted)
        expert_probabilities /= expert_probabilities.sum(
            axis=1,
            keepdims=True,
        )
        probabilities += np.float32(weight) * expert_probabilities
    return probabilities


def compare_candidate_set_to_baseline(
    candidate_scores: np.ndarray,
    baseline_scores: np.ndarray,
    *,
    positive_indices: np.ndarray,
) -> dict[str, Any]:
    """Compare independent rankings; baseline scores never enter the model."""
    candidate = np.asarray(candidate_scores, dtype=np.float64)
    baseline = np.asarray(baseline_scores, dtype=np.float64)
    positives = np.asarray(positive_indices, dtype=np.int32)
    if candidate.shape != baseline.shape:
        raise ValueError(
            "candidate-set and baseline scores must have the same shape"
        )
    if (
        candidate.ndim != 2
        or candidate.shape[0] < 3
        or candidate.shape[1] < 2
    ):
        raise ValueError(
            "candidate-set comparison requires at least three queries "
            "and two candidates"
        )
    if positives.shape != (candidate.shape[0],):
        raise ValueError(
            "candidate-set comparison requires one positive per query"
        )
    if not np.all(np.isfinite(candidate)) or not np.all(
        np.isfinite(baseline)
    ):
        raise ValueError(
            "candidate-set comparison scores must be finite"
        )
    candidate_metrics = _ranking_mrr_three_slices(
        candidate,
        positives,
    )
    baseline_metrics = _ranking_mrr_three_slices(
        baseline,
        positives,
    )
    return {
        "protocol": "comparison_only_no_blend",
        "candidate": candidate_metrics,
        "baseline": baseline_metrics,
        "delta_vs_baseline": {
            key: candidate_metrics[key] - baseline_metrics[key]
            for key in candidate_metrics
        },
    }


class CandidateSetTransformer(jt.nn.Module):
    """Permutation-equivariant scorer over all candidates in a query."""

    def __init__(self, config: CandidateSetTransformerConfig) -> None:
        super().__init__()
        self.config = config
        projection_input_dim = config.input_dim * (
            3 if config.relative_context == "mean_max" else 1
        )
        self.input_projection = jt.nn.Linear(
            projection_input_dim,
            config.model_dim,
        )
        self.blocks = jt.nn.ModuleList(
            [
                _CandidateTransformerBlock(config)
                for _ in range(config.layers)
            ]
        )
        self.output_norm = jt.nn.LayerNorm(config.model_dim)
        self.score_head = jt.nn.Linear(config.model_dim, 1)
        if config.pointwise_residual_dim > 0:
            second_dim = max(config.pointwise_residual_dim // 2, 1)
            self.pointwise_score_head = jt.nn.Sequential(
                jt.nn.Linear(
                    projection_input_dim,
                    config.pointwise_residual_dim,
                ),
                jt.nn.ReLU(),
                jt.nn.Linear(
                    config.pointwise_residual_dim,
                    second_dim,
                ),
                jt.nn.ReLU(),
                jt.nn.Linear(second_dim, 1),
            )
            self.interaction_scale = jt.nn.Parameter(
                jt.array([0.1], dtype=jt.float32)
            )
        else:
            self.pointwise_score_head = None
            self.interaction_scale = None

    def execute(
        self,
        features: jt.Var,
        candidate_mask: jt.Var | None = None,
    ) -> jt.Var:
        if len(features.shape) != 3:
            raise ValueError(
                "candidate-set features must have shape "
                "[queries, candidates, features]"
            )
        if features.shape[-1] != self.config.input_dim:
            raise ValueError(
                "candidate-set feature dimension differs from model config"
            )
        if candidate_mask is not None and candidate_mask.shape != features.shape[:2]:
            raise ValueError(
                "candidate-set mask must have shape [queries, candidates]"
            )
        contextual_features = candidate_relative_features(
            features,
            mode=self.config.relative_context,
        )
        hidden = self.input_projection(contextual_features)
        for block in self.blocks:
            hidden = block(hidden, candidate_mask)
        hidden = self.output_norm(hidden)
        scores = self.score_head(hidden).reshape(features.shape[:2])
        if self.pointwise_score_head is not None:
            scores = (
                self.pointwise_score_head(contextual_features).reshape(
                    features.shape[:2]
                )
                + self.interaction_scale * scores
            )
        if candidate_mask is not None:
            scores = jt.where(
                candidate_mask,
                scores,
                jt.full_like(scores, -1e9),
            )
        return scores

    def pointwise_scores(self, features: jt.Var) -> jt.Var:
        if self.pointwise_score_head is None:
            raise RuntimeError(
                "candidate-set pointwise residual is disabled"
            )
        if (
            len(features.shape) != 3
            or features.shape[-1] != self.config.input_dim
        ):
            raise ValueError(
                "candidate-set pointwise features differ from model config"
            )
        contextual_features = candidate_relative_features(
            features,
            mode=self.config.relative_context,
        )
        return self.pointwise_score_head(contextual_features).reshape(
            features.shape[:2]
        )


class _CandidateTransformerBlock(jt.nn.Module):
    def __init__(self, config: CandidateSetTransformerConfig) -> None:
        super().__init__()
        self.attention_norm = jt.nn.LayerNorm(config.model_dim)
        self.attention = _CandidateSelfAttention(config)
        self.feedforward_norm = jt.nn.LayerNorm(config.model_dim)
        inner_dim = config.model_dim * config.feedforward_multiplier
        self.feedforward = jt.nn.Sequential(
            jt.nn.Linear(config.model_dim, inner_dim),
            jt.nn.ReLU(),
            jt.nn.Dropout(config.dropout),
            jt.nn.Linear(inner_dim, config.model_dim),
            jt.nn.Dropout(config.dropout),
        )

    def execute(
        self,
        hidden: jt.Var,
        candidate_mask: jt.Var | None = None,
    ) -> jt.Var:
        hidden = hidden + self.attention(
            self.attention_norm(hidden),
            candidate_mask,
        )
        return hidden + self.feedforward(self.feedforward_norm(hidden))


class _CandidateSelfAttention(jt.nn.Module):
    def __init__(self, config: CandidateSetTransformerConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.model_dim // config.heads
        self.scale = math.sqrt(self.head_dim)
        self.query = jt.nn.Linear(config.model_dim, config.model_dim)
        self.key = jt.nn.Linear(config.model_dim, config.model_dim)
        self.value = jt.nn.Linear(config.model_dim, config.model_dim)
        self.output = jt.nn.Linear(config.model_dim, config.model_dim)
        self.attention_dropout = jt.nn.Dropout(config.dropout)
        self.output_dropout = jt.nn.Dropout(config.dropout)

    def execute(
        self,
        hidden: jt.Var,
        candidate_mask: jt.Var | None = None,
    ) -> jt.Var:
        batch_size, candidate_count, model_dim = hidden.shape
        query = self._split_heads(self.query(hidden))
        key = self._split_heads(self.key(hidden)).permute(0, 1, 3, 2)
        value = self._split_heads(self.value(hidden))
        attention_scores = jt.matmul(query, key) / self.scale
        if candidate_mask is not None:
            key_mask = candidate_mask.float().unsqueeze(1).unsqueeze(1)
            attention_scores = (
                attention_scores
                + (1.0 - key_mask) * -1e9
            )
        attention = jt.nn.softmax(attention_scores, dim=-1)
        attended = jt.matmul(self.attention_dropout(attention), value)
        attended = attended.permute(0, 2, 1, 3).reshape(
            (batch_size, candidate_count, model_dim)
        )
        return self.output_dropout(self.output(attended))

    def _split_heads(self, values: jt.Var) -> jt.Var:
        return values.reshape(
            (
                values.shape[0],
                values.shape[1],
                self.heads,
                self.head_dim,
            )
        ).permute(0, 2, 1, 3)


def _validate_training_arrays(
    train_features: Any,
    train_positive_indices: np.ndarray,
    validation_features: Any,
    validation_positive_indices: np.ndarray,
    *,
    model_config: CandidateSetTransformerConfig,
) -> None:
    if (
        len(train_features.shape) != 3
        or len(validation_features.shape) != 3
    ):
        raise ValueError(
            "candidate-set training features must have shape "
            "[queries, candidates, features]"
        )
    if train_features.shape[0] <= 0 or validation_features.shape[0] <= 0:
        raise ValueError(
            "candidate-set training and validation must be non-empty"
        )
    if (
        train_features.shape[-1] != model_config.input_dim
        or validation_features.shape[-1] != model_config.input_dim
    ):
        raise ValueError(
            "candidate-set training feature dimension differs from config"
        )
    for label, features, positives in (
        ("train", train_features, train_positive_indices),
        ("validation", validation_features, validation_positive_indices),
    ):
        values = np.asarray(positives)
        if values.shape != (features.shape[0],):
            raise ValueError(
                f"candidate-set {label} positives must have one index "
                "per query"
            )
        if np.any(values < 0) or np.any(values >= features.shape[1]):
            raise ValueError(
                f"candidate-set {label} positive index is out of range"
            )


def _validate_feature_contract(
    feature_names: tuple[str, ...],
    feature_provenance: tuple[str, ...],
    *,
    input_dim: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    names = tuple(str(name) for name in feature_names)
    provenance = tuple(str(value) for value in feature_provenance)
    if len(names) != input_dim or len(provenance) != input_dim:
        raise ValueError(
            "candidate-set feature contract must describe every input"
        )
    forbidden = tuple(
        sorted(set(provenance) - ALLOWED_FEATURE_PROVENANCE)
    )
    if forbidden:
        raise ValueError(
            "candidate-set feature provenance contains non-Jittor "
            f"trainable sources: {', '.join(forbidden)}"
        )
    return names, provenance


def _streaming_normalizer(
    features: Any,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    feature_sum = np.zeros(features.shape[-1], dtype=np.float64)
    feature_sq_sum = np.zeros(features.shape[-1], dtype=np.float64)
    count = 0
    for start in range(0, features.shape[0], batch_size):
        end = min(start + batch_size, features.shape[0])
        flat = np.asarray(
            features[start:end],
            dtype=np.float32,
        ).reshape((-1, features.shape[-1])).astype(
            np.float64,
            copy=False,
        )
        feature_sum += flat.sum(axis=0)
        feature_sq_sum += (flat * flat).sum(axis=0)
        count += flat.shape[0]
    mean64 = feature_sum / count
    variance64 = np.maximum(
        feature_sq_sum / count - mean64 * mean64,
        0.0,
    )
    mean = mean64.astype(np.float32)
    std = np.sqrt(variance64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def _normalize(
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return ((features - mean) / std).astype(np.float32, copy=False)


def _ranking_mrr(
    scores: np.ndarray,
    positive_indices: np.ndarray,
) -> float:
    positive_scores = np.take_along_axis(
        scores,
        positive_indices[:, None],
        axis=1,
    )
    ranks = 1 + (scores > positive_scores).sum(axis=1)
    return float(np.mean(1.0 / ranks))


def _ranking_mrr_three_slices(
    scores: np.ndarray,
    positive_indices: np.ndarray,
) -> dict[str, float]:
    base_size, remainder = divmod(scores.shape[0], 3)
    sizes = tuple(
        base_size + (1 if index < remainder else 0)
        for index in range(3)
    )
    stops = (sizes[0], sizes[0] + sizes[1], scores.shape[0])
    starts = (0, stops[0], stops[1])
    return {
        "full": _ranking_mrr(scores, positive_indices),
        **{
            f"slice_{index}": _ranking_mrr(
                scores[start:stop],
                positive_indices[start:stop],
            )
            for index, (start, stop) in enumerate(
                zip(starts, stops, strict=True)
            )
        },
    }


def _snapshot_state(
    model: CandidateSetTransformer,
) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(value.numpy(), dtype=np.float32).copy()
        for key, value in model.state_dict().items()
    }


def _load_state(
    model: CandidateSetTransformer,
    state: dict[str, np.ndarray],
) -> None:
    model.load_state_dict(
        {
            key: jt.array(value, dtype=jt.float32)
            for key, value in state.items()
        }
    )


def _fit_result_metadata(
    result: CandidateSetFitResult,
) -> dict[str, Any]:
    return {
        "model_config": asdict(result.model_config),
        "training_config": asdict(result.training_config),
        "best_val_mrr": result.best_val_mrr,
        "feature_names": list(result.feature_names),
        "feature_provenance": list(result.feature_provenance),
        "history": list(result.history),
        "trainable_frameworks": list(result.trainable_frameworks),
        "non_jittor_trainable_models": list(
            result.non_jittor_trainable_models
        ),
    }


def _fit_result_from_metadata(
    metadata: dict[str, Any],
    *,
    state: dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
) -> CandidateSetFitResult:
    result = CandidateSetFitResult(
        model_config=CandidateSetTransformerConfig(
            **metadata["model_config"]
        ),
        training_config=CandidateSetTrainingConfig(
            **metadata["training_config"]
        ),
        best_val_mrr=float(metadata["best_val_mrr"]),
        state=state,
        mean=mean,
        std=std,
        feature_names=tuple(metadata["feature_names"]),
        feature_provenance=tuple(metadata["feature_provenance"]),
        history=tuple(
            {
                str(key): value
                for key, value in row.items()
            }
            for row in metadata["history"]
        ),
        selection_mode=str(
            metadata.get("selection_mode", "validation_best")
        ),
        training_rows=int(metadata.get("training_rows", 0)),
        trainable_frameworks=tuple(
            metadata["trainable_frameworks"]
        ),
        non_jittor_trainable_models=tuple(
            metadata["non_jittor_trainable_models"]
        ),
    )
    _validate_feature_contract(
        result.feature_names,
        result.feature_provenance,
        input_dim=result.model_config.input_dim,
    )
    if (
        result.trainable_frameworks != ("jittor",)
        or result.non_jittor_trainable_models
    ):
        raise ValueError(
            "candidate-set expert has non-Jittor trainable provenance"
        )
    return result


def _validate_ensemble_contract(
    experts: Any,
    weights: tuple[float, ...],
) -> tuple[float, ...]:
    if len(experts) < 2 or len(experts) != len(weights):
        raise ValueError(
            "candidate-set ensemble needs matching experts and weights"
        )
    normalized = tuple(float(value) for value in weights)
    if (
        any(not math.isfinite(value) or value <= 0.0 for value in normalized)
        or not math.isclose(sum(normalized), 1.0, abs_tol=1e-7)
    ):
        raise ValueError(
            "candidate-set ensemble weights must be positive and sum to one"
        )
    first_result = experts[0][1]
    for model, result in experts:
        if model.config != result.model_config:
            raise ValueError(
                "candidate-set ensemble model and result configs differ"
            )
        if (
            result.feature_names != first_result.feature_names
            or result.feature_provenance
            != first_result.feature_provenance
            or result.model_config.input_dim
            != first_result.model_config.input_dim
        ):
            raise ValueError(
                "candidate-set ensemble expert feature contracts differ"
            )
        if (
            result.trainable_frameworks != ("jittor",)
            or result.non_jittor_trainable_models
        ):
            raise ValueError(
                "candidate-set ensemble contains non-Jittor training"
            )
    return normalized


def _save_npz_atomic(
    target: Path,
    payload: dict[str, np.ndarray],
) -> None:
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


def _validate_checkpoint_metadata(metadata: Any) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("invalid candidate-set checkpoint metadata")
    if metadata.get("format") != CANDIDATE_SET_CHECKPOINT_FORMAT:
        raise ValueError("unsupported candidate-set checkpoint format")
    if metadata.get("version") != CANDIDATE_SET_CHECKPOINT_VERSION:
        raise ValueError("unsupported candidate-set checkpoint version")
    if metadata.get("trainable_frameworks") != ["jittor"]:
        raise ValueError(
            "candidate-set checkpoint has non-Jittor trainable frameworks"
        )
    if metadata.get("non_jittor_trainable_models") != []:
        raise ValueError(
            "candidate-set checkpoint has non-Jittor trainable models"
        )


def _validate_ensemble_metadata(metadata: Any) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("invalid candidate-set ensemble metadata")
    if metadata.get("format") != CANDIDATE_SET_ENSEMBLE_FORMAT:
        raise ValueError("unsupported candidate-set ensemble format")
    if metadata.get("version") != CANDIDATE_SET_ENSEMBLE_VERSION:
        raise ValueError("unsupported candidate-set ensemble version")
    if metadata.get("blend") != "fixed_probability":
        raise ValueError("unsupported candidate-set ensemble blend")
    if metadata.get("trainable_frameworks") != ["jittor"]:
        raise ValueError(
            "candidate-set ensemble has non-Jittor trainable frameworks"
        )
    if metadata.get("non_jittor_trainable_models") != []:
        raise ValueError(
            "candidate-set ensemble has non-Jittor trainable models"
        )
    if not isinstance(metadata.get("experts"), list):
        raise ValueError("candidate-set ensemble experts are invalid")
