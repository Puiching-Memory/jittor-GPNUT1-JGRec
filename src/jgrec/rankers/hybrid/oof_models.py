from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from jgrec.rankers.hybrid.candidate_set_transformer import (
    ALLOWED_FEATURE_PROVENANCE,
    CandidateSetEnsembleCheckpoint,
    candidate_relative_features,
    candidate_set_listwise_loss,
    hydrate_candidate_set_ensemble,
    predict_candidate_set_logits,
    snapshot_candidate_set_ensemble,
)
from jgrec.rankers.hybrid.oof_stacking import (
    STABLE_EXPERT_LOGIT_FEATURE_VERSION,
    stable_expert_logit_feature_names,
    stable_expert_logit_features,
    tie_neutral_mrr,
)

_MLP_CHECKPOINT_FORMAT = "jgrec-pure-jittor-candidate-set-mlp"
_MLP_CHECKPOINT_VERSION = 1
_OOF_STACKING_FORMAT = "jgrec-pure-jittor-oof-stacking"
_OOF_STACKING_VERSION = 2


@dataclass(frozen=True)
class CandidateSetMLPConfig:
    input_dim: int
    hidden_dim: int = 128
    dropout: float = 0.05
    relative_context: str = "mean_max"

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("candidate-set MLP input_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("candidate-set MLP hidden_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("candidate-set MLP dropout must be in [0, 1)")
        if self.relative_context not in {"none", "mean_max"}:
            raise ValueError(
                "candidate-set MLP relative_context must be none or mean_max"
            )


@dataclass(frozen=True)
class CandidateSetMLPTrainingConfig:
    epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    seed: int = 60
    early_stop_patience: int = 0

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("candidate-set MLP epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("candidate-set MLP batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError(
                "candidate-set MLP learning_rate must be positive"
            )
        if self.weight_decay < 0.0:
            raise ValueError(
                "candidate-set MLP weight_decay must be non-negative"
            )
        if self.early_stop_patience < 0:
            raise ValueError(
                "candidate-set MLP early_stop_patience must be non-negative"
            )


@dataclass(frozen=True)
class CandidateSetMLPFitResult:
    model_config: CandidateSetMLPConfig
    training_config: CandidateSetMLPTrainingConfig
    selection_mode: str
    training_rows: int
    best_val_mrr: float | None
    state: dict[str, np.ndarray]
    mean: np.ndarray
    std: np.ndarray
    feature_names: tuple[str, ...]
    feature_provenance: tuple[str, ...]
    history: tuple[dict[str, float | int], ...]
    trainable_frameworks: tuple[str, ...] = ("jittor",)
    non_jittor_trainable_models: tuple[str, ...] = ()


@dataclass(frozen=True)
class PureJittorOOFStackingCheckpoint:
    expert_names: tuple[str, ...]
    cst_experts: CandidateSetEnsembleCheckpoint
    setwise_mlp: tuple[CandidateSetMLP, CandidateSetMLPFitResult]
    meta_mlp: tuple[CandidateSetMLP, CandidateSetMLPFitResult]
    meta_weight: float
    trainable_frameworks: tuple[str, ...] = ("jittor",)
    non_jittor_trainable_models: tuple[str, ...] = ()


class CandidateSetMLP(jt.nn.Module):
    """Pointwise MLP with deterministic candidate-set relative context."""

    def __init__(self, config: CandidateSetMLPConfig) -> None:
        super().__init__()
        self.config = config
        context_multiplier = 3 if config.relative_context == "mean_max" else 1
        context_dim = config.input_dim * context_multiplier
        second_dim = max(config.hidden_dim // 2, 1)
        self.linear1 = jt.nn.Linear(context_dim, config.hidden_dim)
        self.linear2 = jt.nn.Linear(config.hidden_dim, second_dim)
        self.output = jt.nn.Linear(second_dim, 1)
        self.dropout = jt.nn.Dropout(config.dropout)

    def execute(self, features: jt.Var) -> jt.Var:
        if len(features.shape) != 3:
            raise ValueError(
                "candidate-set MLP features require "
                "[queries, candidates, features]"
            )
        if features.shape[-1] != self.config.input_dim:
            raise ValueError(
                "candidate-set MLP feature dimension differs from config"
            )
        context = candidate_relative_features(
            features,
            mode=self.config.relative_context,
        )
        shape = context.shape[:-1]
        hidden = context.reshape((-1, context.shape[-1]))
        hidden = self.dropout(jt.nn.relu(self.linear1(hidden)))
        hidden = self.dropout(jt.nn.relu(self.linear2(hidden)))
        return self.output(hidden).reshape(shape)


def fit_candidate_set_mlp(
    train_features: Any,
    train_positive_indices: np.ndarray,
    *,
    model_config: CandidateSetMLPConfig,
    training_config: CandidateSetMLPTrainingConfig,
    feature_names: tuple[str, ...],
    feature_provenance: tuple[str, ...],
    validation_features: Any | None = None,
    validation_positive_indices: np.ndarray | None = None,
    verbose: bool = True,
) -> tuple[CandidateSetMLP, CandidateSetMLPFitResult]:
    """Fit a pure-Jittor setwise MLP, with optional validation selection."""
    positives = _validate_features_and_positives(
        train_features,
        train_positive_indices,
        model_config.input_dim,
        "training",
    )
    names, provenance = _validate_feature_contract(
        feature_names,
        feature_provenance,
        model_config.input_dim,
    )
    has_validation = validation_features is not None
    if has_validation != (validation_positive_indices is not None):
        raise ValueError(
            "candidate-set MLP validation features and positives "
            "must be supplied together"
        )
    validation_positives: np.ndarray | None = None
    if has_validation:
        validation_positives = _validate_features_and_positives(
            validation_features,
            validation_positive_indices,
            model_config.input_dim,
            "validation",
        )

    jt.set_seed(int(training_config.seed))
    rng = np.random.default_rng(training_config.seed)
    mean, std = _streaming_normalizer(
        train_features,
        training_config.batch_size,
    )
    model = CandidateSetMLP(model_config)
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
        order = rng.permutation(int(train_features.shape[0]))
        losses: list[float] = []
        for start in range(
            0,
            int(train_features.shape[0]),
            training_config.batch_size,
        ):
            indices = order[start : start + training_config.batch_size]
            batch = _normalize(
                np.asarray(train_features[indices], dtype=np.float32),
                mean,
                std,
            )
            logits = model(jt.array(batch, dtype=jt.float32))
            positive_var = jt.array(positives[indices], dtype=jt.int32)
            loss = candidate_set_listwise_loss(logits, positive_var)
            optimizer.step(loss)
            losses.append(float(loss.item()))
        train_loss = float(np.mean(losses))
        if not math.isfinite(train_loss):
            raise FloatingPointError(
                f"non-finite candidate-set MLP loss at epoch {epoch}"
            )
        entry: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
        }
        if has_validation:
            val_scores = predict_candidate_set_mlp_logits(
                model,
                validation_features,
                mean=mean,
                std=std,
                batch_size=training_config.batch_size,
            )
            val_mrr = _ranking_mrr(val_scores, validation_positives)
            if val_mrr >= best_mrr:
                best_mrr = val_mrr
                best_state = _snapshot_state(model)
                patience = 0
            else:
                patience += 1
            entry.update({"val_mrr": val_mrr, "patience": patience})
        else:
            best_state = _snapshot_state(model)
        history.append(entry)
        if verbose:
            suffix = (
                ""
                if not has_validation
                else (
                    f" val_mrr={entry['val_mrr']:.6f}"
                    f" best_val_mrr={best_mrr:.6f}"
                )
            )
            print(
                "[candidate-set-mlp] "
                f"epoch={epoch} train_loss={train_loss:.6f}{suffix}",
                flush=True,
            )
        if (
            has_validation
            and training_config.early_stop_patience > 0
            and patience >= training_config.early_stop_patience
        ):
            break
    _load_state(model, best_state)
    return model, CandidateSetMLPFitResult(
        model_config=model_config,
        training_config=training_config,
        selection_mode=(
            "validation_best" if has_validation else "fixed_full"
        ),
        training_rows=int(train_features.shape[0]),
        best_val_mrr=(float(best_mrr) if has_validation else None),
        state=best_state,
        mean=mean,
        std=std,
        feature_names=names,
        feature_provenance=provenance,
        history=tuple(history),
    )


def predict_candidate_set_mlp_logits(
    model: CandidateSetMLP,
    features: Any,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    if len(features.shape) != 3:
        raise ValueError(
            "candidate-set MLP prediction requires three dimensions"
        )
    if int(features.shape[-1]) != model.config.input_dim:
        raise ValueError(
            "candidate-set MLP prediction feature dimension differs"
        )
    if batch_size <= 0:
        raise ValueError(
            "candidate-set MLP prediction batch_size must be positive"
        )
    scores = np.empty(features.shape[:2], dtype=np.float32)
    model.eval()
    with jt.no_grad():
        for start in range(0, int(features.shape[0]), int(batch_size)):
            stop = min(start + int(batch_size), int(features.shape[0]))
            batch = _normalize(
                np.asarray(features[start:stop], dtype=np.float32),
                mean,
                std,
            )
            logits = model(jt.array(batch, dtype=jt.float32))
            scores[start:stop] = np.asarray(
                logits.numpy(),
                dtype=np.float32,
            )
    return scores


def snapshot_candidate_set_mlp(
    model: CandidateSetMLP,
    result: CandidateSetMLPFitResult,
) -> dict[str, Any]:
    """Convert a pure-Jittor MLP into contest-checkpoint state."""
    if (
        result.trainable_frameworks != ("jittor",)
        or result.non_jittor_trainable_models
    ):
        raise ValueError(
            "candidate-set MLP snapshot has non-Jittor provenance"
        )
    return {
        "format": _MLP_CHECKPOINT_FORMAT,
        "version": _MLP_CHECKPOINT_VERSION,
        "metadata": _mlp_result_metadata(result),
        "state": _snapshot_state(model),
        "mean": np.asarray(result.mean, dtype=np.float32).copy(),
        "std": np.asarray(result.std, dtype=np.float32).copy(),
    }


def hydrate_candidate_set_mlp(
    snapshot: dict[str, Any],
) -> tuple[CandidateSetMLP, CandidateSetMLPFitResult]:
    """Restore contest-checkpoint state without external ML imports."""
    if (
        snapshot.get("format") != _MLP_CHECKPOINT_FORMAT
        or snapshot.get("version") != _MLP_CHECKPOINT_VERSION
    ):
        raise ValueError("candidate-set MLP snapshot format differs")
    metadata = snapshot.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("candidate-set MLP snapshot metadata is invalid")
    state = {
        str(key): np.asarray(value, dtype=np.float32).copy()
        for key, value in snapshot["state"].items()
    }
    result = _mlp_result_from_metadata(
        metadata,
        state=state,
        mean=np.asarray(snapshot["mean"], dtype=np.float32).copy(),
        std=np.asarray(snapshot["std"], dtype=np.float32).copy(),
    )
    model = CandidateSetMLP(result.model_config)
    _load_state(model, state)
    return model, result


def snapshot_pure_jittor_oof_stacking(
    checkpoint: PureJittorOOFStackingCheckpoint,
) -> dict[str, Any]:
    """Convert the complete mixed-expert stack into contest state."""
    _validate_oof_stacking_checkpoint(checkpoint)
    setwise_model, setwise_result = checkpoint.setwise_mlp
    meta_model, meta_result = checkpoint.meta_mlp
    return {
        "format": _OOF_STACKING_FORMAT,
        "version": _OOF_STACKING_VERSION,
        "stable_feature_version": STABLE_EXPERT_LOGIT_FEATURE_VERSION,
        "expert_names": checkpoint.expert_names,
        "meta_weight": float(checkpoint.meta_weight),
        "trainable_frameworks": ("jittor",),
        "non_jittor_trainable_models": (),
        "cst_experts": snapshot_candidate_set_ensemble(
            checkpoint.cst_experts
        ),
        "setwise_mlp": snapshot_candidate_set_mlp(
            setwise_model,
            setwise_result,
        ),
        "meta_mlp": snapshot_candidate_set_mlp(
            meta_model,
            meta_result,
        ),
    }


def hydrate_pure_jittor_oof_stacking(
    snapshot: dict[str, Any],
) -> PureJittorOOFStackingCheckpoint:
    """Restore a stack without importing LightGBM or sklearn."""
    if (
        snapshot.get("format") != _OOF_STACKING_FORMAT
        or snapshot.get("version") != _OOF_STACKING_VERSION
        or snapshot.get("stable_feature_version")
        != STABLE_EXPERT_LOGIT_FEATURE_VERSION
    ):
        raise ValueError("pure-Jittor OOF stacking snapshot format differs")
    if (
        tuple(snapshot.get("trainable_frameworks", ())) != ("jittor",)
        or tuple(snapshot.get("non_jittor_trainable_models", ()))
    ):
        raise ValueError("OOF stacking snapshot is not pure Jittor")
    checkpoint = PureJittorOOFStackingCheckpoint(
        expert_names=tuple(snapshot["expert_names"]),
        cst_experts=hydrate_candidate_set_ensemble(
            snapshot["cst_experts"]
        ),
        setwise_mlp=hydrate_candidate_set_mlp(
            snapshot["setwise_mlp"]
        ),
        meta_mlp=hydrate_candidate_set_mlp(snapshot["meta_mlp"]),
        meta_weight=float(snapshot["meta_weight"]),
    )
    _validate_oof_stacking_checkpoint(checkpoint)
    return checkpoint


def predict_pure_jittor_oof_stacking_scores(
    checkpoint: PureJittorOOFStackingCheckpoint,
    features: Any,
    *,
    batch_size: int,
) -> np.ndarray:
    """Run all full-data experts and the OOF meta model."""
    _validate_oof_stacking_checkpoint(checkpoint)
    if len(features.shape) != 3 or int(features.shape[0]) <= 0:
        raise ValueError("OOF stacking prediction features must be 3D")
    if batch_size <= 0:
        raise ValueError("OOF stacking prediction batch_size must be positive")
    raw_dim = checkpoint.cst_experts.results[0].model_config.input_dim
    if int(features.shape[-1]) != raw_dim:
        raise ValueError("OOF stacking prediction feature width differs")
    setwise_model, setwise_result = checkpoint.setwise_mlp
    meta_model, meta_result = checkpoint.meta_mlp
    output = np.empty(features.shape[:2], dtype=np.float32)
    for start in range(0, int(features.shape[0]), int(batch_size)):
        stop = min(start + int(batch_size), int(features.shape[0]))
        batch = features[start:stop]
        expert_logits = [
            predict_candidate_set_logits(
                model,
                batch,
                mean=result.mean,
                std=result.std,
                batch_size=batch_size,
            )
            for model, result in zip(
                checkpoint.cst_experts.models,
                checkpoint.cst_experts.results,
                strict=True,
            )
        ]
        expert_logits.append(
            predict_candidate_set_mlp_logits(
                setwise_model,
                batch,
                mean=setwise_result.mean,
                std=setwise_result.std,
                batch_size=batch_size,
            )
        )
        stable = stable_expert_logit_features(
            np.stack(expert_logits, axis=0)
        )
        meta_logits = predict_candidate_set_mlp_logits(
            meta_model,
            stable,
            mean=meta_result.mean,
            std=meta_result.std,
            batch_size=batch_size,
        )
        meta_percentile = stable_expert_logit_features(
            meta_logits[None, ...]
        )[..., 0]
        consensus = stable[..., -5]
        output[start:stop] = (
            checkpoint.meta_weight * meta_percentile
            + (1.0 - checkpoint.meta_weight) * consensus
        )
    return output


def save_candidate_set_mlp_checkpoint(
    path: Path,
    model: CandidateSetMLP,
    result: CandidateSetMLPFitResult,
) -> None:
    """Save a self-contained checkpoint with explicit pure-Jittor provenance."""
    target = Path(path)
    if target.exists():
        raise FileExistsError(
            f"candidate-set MLP checkpoint already exists: {target}"
        )
    if (
        result.trainable_frameworks != ("jittor",)
        or result.non_jittor_trainable_models
    ):
        raise ValueError(
            "candidate-set MLP checkpoint has non-Jittor trainable provenance"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": _MLP_CHECKPOINT_FORMAT,
        "version": _MLP_CHECKPOINT_VERSION,
        **_mlp_result_metadata(result),
    }
    arrays: dict[str, np.ndarray] = {
        "metadata": np.asarray(
            json.dumps(metadata, sort_keys=True),
            dtype=np.str_,
        ),
        "mean": np.asarray(result.mean, dtype=np.float32),
        "std": np.asarray(result.std, dtype=np.float32),
    }
    arrays.update(
        {
            f"state::{key}": value
            for key, value in _snapshot_state(model).items()
        }
    )
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def load_candidate_set_mlp_checkpoint(
    path: Path,
) -> tuple[CandidateSetMLP, CandidateSetMLPFitResult]:
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if (
            metadata.get("format") != _MLP_CHECKPOINT_FORMAT
            or metadata.get("version") != _MLP_CHECKPOINT_VERSION
        ):
            raise ValueError("candidate-set MLP checkpoint format differs")
        if (
            tuple(metadata.get("trainable_frameworks", ())) != ("jittor",)
            or metadata.get("non_jittor_trainable_models")
        ):
            raise ValueError(
                "candidate-set MLP checkpoint is not pure Jittor"
            )
        state = {
            key.removeprefix("state::"): np.asarray(
                archive[key],
                dtype=np.float32,
            ).copy()
            for key in archive.files
            if key.startswith("state::")
        }
        mean = np.asarray(archive["mean"], dtype=np.float32).copy()
        std = np.asarray(archive["std"], dtype=np.float32).copy()
    result = _mlp_result_from_metadata(
        metadata,
        state=state,
        mean=mean,
        std=std,
    )
    model = CandidateSetMLP(result.model_config)
    _load_state(model, state)
    return model, result


def _mlp_result_metadata(
    result: CandidateSetMLPFitResult,
) -> dict[str, Any]:
    return {
        "model_config": asdict(result.model_config),
        "training_config": asdict(result.training_config),
        "selection_mode": result.selection_mode,
        "training_rows": result.training_rows,
        "best_val_mrr": result.best_val_mrr,
        "feature_names": list(result.feature_names),
        "feature_provenance": list(result.feature_provenance),
        "history": list(result.history),
        "trainable_frameworks": list(result.trainable_frameworks),
        "non_jittor_trainable_models": list(
            result.non_jittor_trainable_models
        ),
    }


def _mlp_result_from_metadata(
    metadata: dict[str, Any],
    *,
    state: dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
) -> CandidateSetMLPFitResult:
    result = CandidateSetMLPFitResult(
        model_config=CandidateSetMLPConfig(**metadata["model_config"]),
        training_config=CandidateSetMLPTrainingConfig(
            **metadata["training_config"]
        ),
        selection_mode=str(metadata["selection_mode"]),
        training_rows=int(metadata["training_rows"]),
        best_val_mrr=(
            None
            if metadata["best_val_mrr"] is None
            else float(metadata["best_val_mrr"])
        ),
        state=state,
        mean=mean,
        std=std,
        feature_names=tuple(metadata["feature_names"]),
        feature_provenance=tuple(metadata["feature_provenance"]),
        history=tuple(
            {str(key): value for key, value in row.items()}
            for row in metadata["history"]
        ),
        trainable_frameworks=tuple(metadata["trainable_frameworks"]),
        non_jittor_trainable_models=tuple(
            metadata["non_jittor_trainable_models"]
        ),
    )
    _validate_feature_contract(
        result.feature_names,
        result.feature_provenance,
        result.model_config.input_dim,
    )
    if (
        result.trainable_frameworks != ("jittor",)
        or result.non_jittor_trainable_models
    ):
        raise ValueError("candidate-set MLP result is not pure Jittor")
    return result


def _validate_oof_stacking_checkpoint(
    checkpoint: PureJittorOOFStackingCheckpoint,
) -> None:
    if (
        checkpoint.trainable_frameworks != ("jittor",)
        or checkpoint.non_jittor_trainable_models
    ):
        raise ValueError("OOF stacking checkpoint is not pure Jittor")
    names = tuple(checkpoint.expert_names)
    cst_results = checkpoint.cst_experts.results
    if (
        len(names) != len(cst_results) + 1
        or not names
        or names[-1] != "setwise_mlp"
        or len(set(names)) != len(names)
    ):
        raise ValueError("OOF stacking expert order is invalid")
    if not math.isfinite(checkpoint.meta_weight) or not (
        0.0 <= checkpoint.meta_weight <= 1.0
    ):
        raise ValueError("OOF stacking meta_weight must be in [0, 1]")
    if not cst_results:
        raise ValueError("OOF stacking requires at least one CST expert")
    raw_names = cst_results[0].feature_names
    raw_provenance = cst_results[0].feature_provenance
    for result in cst_results:
        if (
            result.feature_names != raw_names
            or result.feature_provenance != raw_provenance
            or result.trainable_frameworks != ("jittor",)
            or result.non_jittor_trainable_models
        ):
            raise ValueError("OOF stacking CST feature contract differs")
    _, setwise_result = checkpoint.setwise_mlp
    if (
        setwise_result.feature_names != raw_names
        or setwise_result.feature_provenance != raw_provenance
        or setwise_result.trainable_frameworks != ("jittor",)
        or setwise_result.non_jittor_trainable_models
    ):
        raise ValueError("OOF stacking Setwise MLP feature contract differs")
    _, meta_result = checkpoint.meta_mlp
    expected_meta_names = stable_expert_logit_feature_names(names)
    if (
        meta_result.feature_names != expected_meta_names
        or meta_result.model_config.input_dim != len(expected_meta_names)
        or meta_result.trainable_frameworks != ("jittor",)
        or meta_result.non_jittor_trainable_models
    ):
        raise ValueError("OOF stacking meta feature contract differs")


def _validate_features_and_positives(
    features: Any,
    positive_indices: Any,
    input_dim: int,
    label: str,
) -> np.ndarray:
    if len(features.shape) != 3 or int(features.shape[0]) <= 0:
        raise ValueError(
            f"candidate-set MLP {label} features must be non-empty and 3D"
        )
    if int(features.shape[-1]) != input_dim:
        raise ValueError(
            f"candidate-set MLP {label} feature dimension differs"
        )
    positives = np.asarray(positive_indices, dtype=np.int32)
    if positives.shape != (int(features.shape[0]),):
        raise ValueError(
            f"candidate-set MLP {label} positives must have one row each"
        )
    if np.any(positives < 0) or np.any(positives >= int(features.shape[1])):
        raise ValueError(
            f"candidate-set MLP {label} positive index is out of range"
        )
    return positives


def _validate_feature_contract(
    feature_names: tuple[str, ...],
    feature_provenance: tuple[str, ...],
    input_dim: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    names = tuple(str(name) for name in feature_names)
    provenance = tuple(str(value) for value in feature_provenance)
    if len(names) != input_dim or len(provenance) != input_dim:
        raise ValueError(
            "candidate-set MLP feature contract does not match input_dim"
        )
    unsupported = set(provenance) - ALLOWED_FEATURE_PROVENANCE
    if unsupported:
        raise ValueError(
            "candidate-set MLP has unsupported feature provenance: "
            f"{sorted(unsupported)}"
        )
    return names, provenance


def _streaming_normalizer(
    features: Any,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    total = 0
    feature_sum: np.ndarray | None = None
    feature_sq_sum: np.ndarray | None = None
    for start in range(0, int(features.shape[0]), int(batch_size)):
        stop = min(start + int(batch_size), int(features.shape[0]))
        batch = np.asarray(
            features[start:stop],
            dtype=np.float64,
        ).reshape((-1, int(features.shape[-1])))
        batch_sum = batch.sum(axis=0)
        batch_sq_sum = np.square(batch).sum(axis=0)
        feature_sum = (
            batch_sum if feature_sum is None else feature_sum + batch_sum
        )
        feature_sq_sum = (
            batch_sq_sum
            if feature_sq_sum is None
            else feature_sq_sum + batch_sq_sum
        )
        total += int(batch.shape[0])
    if total <= 0 or feature_sum is None or feature_sq_sum is None:
        raise ValueError("candidate-set MLP normalizer received no rows")
    mean64 = feature_sum / float(total)
    variance = np.maximum(
        feature_sq_sum / float(total) - np.square(mean64),
        0.0,
    )
    mean = mean64.astype(np.float32)
    std = np.sqrt(variance).astype(np.float32)
    std[std < np.float32(1e-6)] = np.float32(1.0)
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
    return tie_neutral_mrr(scores, positive_indices)


def _snapshot_state(model: CandidateSetMLP) -> dict[str, np.ndarray]:
    return {
        str(key): np.asarray(value.numpy(), dtype=np.float32).copy()
        for key, value in model.state_dict().items()
    }


def _load_state(
    model: CandidateSetMLP,
    state: dict[str, np.ndarray],
) -> None:
    model.load_state_dict(
        {
            key: jt.array(value, dtype=jt.float32)
            for key, value in state.items()
        }
    )
