from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import jittor as jt
import numpy as np
from sklearn.metrics import average_precision_score

from jgrec.core.memory import release_memory
from jgrec.logging import log, track


@dataclass(frozen=True)
class FusionConfig:
    epochs: int = 5
    batch_size: int = 512
    lr: float = 0.001
    weight_decay: float = 0.0
    hidden_dim: int = 64
    selection_metric: str = "mrr"
    early_stop_patience: int = 10
    context_transform_version: int = 0


@dataclass(frozen=True)
class FusionResult:
    best_val_ap: float
    best_val_mrr: float
    state: dict[str, np.ndarray]
    mean: np.ndarray
    std: np.ndarray
    feature_indices: tuple[int, ...]
    candidate_name: str


class FusionMLP(jt.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        hidden_dim = max(int(hidden_dim), 1)
        self.linear1 = jt.nn.Linear(input_dim, hidden_dim)
        self.linear2 = jt.nn.Linear(hidden_dim, max(hidden_dim // 2, 1))
        self.linear3 = jt.nn.Linear(max(hidden_dim // 2, 1), 1)

    def execute(self, features: jt.Var) -> jt.Var:
        original_shape = features.shape[:-1]
        x = features.reshape((-1, features.shape[-1]))
        x = jt.nn.relu(self.linear1(x))
        x = jt.nn.relu(self.linear2(x))
        x = self.linear3(x)
        return x.reshape(original_shape)


def _listwise_positive_loss(
    logits: jt.Var,
    query_weights: jt.Var | None = None,
) -> jt.Var:
    """Return query-level softmax cross-entropy for a positive at candidate zero."""
    if len(logits.shape) != 2 or logits.shape[1] <= 0:
        raise ValueError("listwise fusion logits must have shape [queries, candidates]")
    per_query = -jt.nn.log_softmax(logits, dim=1)[:, 0]
    if query_weights is None:
        return per_query.mean()
    if len(query_weights.shape) != 1 or query_weights.shape[0] != logits.shape[0]:
        raise ValueError("listwise fusion requires one weight per query")
    return (per_query * query_weights).sum() / query_weights.sum()


def fit_fusion_mlp(
    train_features: np.ndarray,
    val_features: np.ndarray,
    config: FusionConfig,
    rng: np.random.Generator,
    verbose: bool,
    feature_indices: tuple[int, ...] | None = None,
    candidate_name: str = "all",
) -> tuple[FusionMLP, FusionResult]:
    if train_features.size == 0 or val_features.size == 0:
        raise ValueError("fusion reranker requires non-empty train and validation features")

    if feature_indices is None:
        feature_indices = tuple(range(train_features.shape[-1]))
    _set_jittor_seed_from_rng(rng)
    mean, std = _feature_normalizer(
        train_features,
        feature_indices,
        context_transform_version=config.context_transform_version,
        batch_size=config.batch_size,
    )
    selection_metric = config.selection_metric.lower()
    if selection_metric not in {"ap", "mrr"}:
        raise ValueError(f"unsupported fusion selection metric: {config.selection_metric}")

    model = FusionMLP(input_dim=mean.shape[0], hidden_dim=config.hidden_dim)
    _initialize_fusion_mlp_from_rng(model, rng)
    optimizer = jt.nn.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_ap, best_mrr = _metrics_from_model(
        model,
        val_features,
        mean,
        std,
        config.batch_size,
        feature_indices,
        context_transform_version=config.context_transform_version,
    )
    best_score = _selected_metric(best_ap, best_mrr, selection_metric)
    best_state = _snapshot_state(model)
    train_size = train_features.shape[0]
    patience_counter = 0

    epochs = range(1, config.epochs + 1)
    for epoch in track(epochs, description=f"fusion:{candidate_name}", total=config.epochs, enabled=verbose):
        order = rng.permutation(train_size)
        losses: list[float] = []
        for start in range(0, train_size, config.batch_size):
            batch_idx = order[start : start + config.batch_size]
            batch_features = _normalize(
                _prepare_fusion_features(
                    train_features[batch_idx],
                    feature_indices,
                    context_transform_version=config.context_transform_version,
                ),
                mean,
                std,
            )
            features = jt.array(batch_features, dtype=jt.float32)
            logits = model(features)
            targets = np.zeros((batch_idx.shape[0], logits.shape[1]), dtype=np.float32)
            targets[:, 0] = 1.0
            target_var = jt.array(targets, dtype=jt.float32)
            probs = jt.sigmoid(logits)
            loss = -(
                target_var * jt.log(probs + 1e-8)
                + (1.0 - target_var) * jt.log(1.0 - probs + 1e-8)
            ).mean()
            optimizer.step(loss)
            losses.append(float(loss.item()))

        val_ap, val_mrr = _metrics_from_model(
            model,
            val_features,
            mean,
            std,
            config.batch_size,
            feature_indices,
            context_transform_version=config.context_transform_version,
        )
        val_score = _selected_metric(val_ap, val_mrr, selection_metric)
        if val_score >= best_score:
            best_ap = val_ap
            best_mrr = val_mrr
            best_score = val_score
            best_state = _snapshot_state(model)
            patience_counter = 0
        else:
            patience_counter += 1
        mean_loss = float(np.mean(losses)) if losses else 0.0
        log(
            f"[fusion:{candidate_name}] epoch={epoch} loss={mean_loss:.5f} "
            f"val_ap={val_ap:.5f} val_mrr={val_mrr:.5f} "
            f"best_{selection_metric}={best_score:.5f} patience={patience_counter}",
            enabled=verbose,
        )
        if config.early_stop_patience > 0 and patience_counter >= config.early_stop_patience:
            log(f"[fusion:{candidate_name}] early_stop epoch={epoch}", enabled=verbose)
            break

    _load_state(model, best_state)
    return model, FusionResult(
        best_val_ap=float(best_ap),
        best_val_mrr=float(best_mrr),
        state=best_state,
        mean=mean,
        std=std,
        feature_indices=feature_indices,
        candidate_name=candidate_name,
    )


def fit_fusion_mlp_streaming(
    train_features: np.ndarray,
    val_features: np.ndarray,
    config: FusionConfig,
    rng: np.random.Generator,
    verbose: bool,
    feature_indices: tuple[int, ...] | None = None,
    candidate_name: str = "all",
) -> tuple[FusionMLP, FusionResult]:
    if train_features.size == 0 or val_features.size == 0:
        raise ValueError("fusion reranker requires non-empty train and validation features")

    if feature_indices is None:
        feature_indices = tuple(range(train_features.shape[-1]))
    _set_jittor_seed_from_rng(rng)
    mean, std = _feature_normalizer_streaming(
        train_features,
        feature_indices,
        config.batch_size,
        context_transform_version=config.context_transform_version,
    )
    selection_metric = config.selection_metric.lower()
    if selection_metric not in {"ap", "mrr"}:
        raise ValueError(f"unsupported fusion selection metric: {config.selection_metric}")

    model = FusionMLP(input_dim=mean.shape[0], hidden_dim=config.hidden_dim)
    _initialize_fusion_mlp_from_rng(model, rng)
    optimizer = jt.nn.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_ap, best_mrr = _metrics_from_model_streaming(
        model,
        val_features,
        mean,
        std,
        config.batch_size,
        feature_indices,
        context_transform_version=config.context_transform_version,
    )
    best_score = _selected_metric(best_ap, best_mrr, selection_metric)
    best_state = _snapshot_state(model)
    train_size = train_features.shape[0]
    patience_counter = 0

    epochs = range(1, config.epochs + 1)
    for epoch in track(epochs, description=f"fusion:{candidate_name}", total=config.epochs, enabled=verbose):
        order = rng.permutation(train_size)
        losses: list[float] = []
        for start in range(0, train_size, config.batch_size):
            batch_idx = order[start : start + config.batch_size]
            batch_features = _normalize(
                _prepare_fusion_features(
                    train_features[batch_idx],
                    feature_indices,
                    context_transform_version=config.context_transform_version,
                ),
                mean,
                std,
            )
            features = jt.array(batch_features, dtype=jt.float32)
            logits = model(features)
            targets = np.zeros((batch_idx.shape[0], logits.shape[1]), dtype=np.float32)
            targets[:, 0] = 1.0
            target_var = jt.array(targets, dtype=jt.float32)
            probs = jt.sigmoid(logits)
            loss = -(
                target_var * jt.log(probs + 1e-8)
                + (1.0 - target_var) * jt.log(1.0 - probs + 1e-8)
            ).mean()
            optimizer.step(loss)
            losses.append(float(loss.item()))
            del batch_features, features, logits, targets, target_var, probs, loss
            if start and start % max(config.batch_size * 16, 1) == 0:
                release_memory()

        val_ap, val_mrr = _metrics_from_model_streaming(
            model,
            val_features,
            mean,
            std,
            config.batch_size,
            feature_indices,
            context_transform_version=config.context_transform_version,
        )
        val_score = _selected_metric(val_ap, val_mrr, selection_metric)
        if val_score >= best_score:
            best_ap = val_ap
            best_mrr = val_mrr
            best_score = val_score
            best_state = _snapshot_state(model)
            patience_counter = 0
        else:
            patience_counter += 1
        mean_loss = float(np.mean(losses)) if losses else 0.0
        log(
            f"[fusion:{candidate_name}] epoch={epoch} loss={mean_loss:.5f} "
            f"val_ap={val_ap:.5f} val_mrr={val_mrr:.5f} "
            f"best_{selection_metric}={best_score:.5f} patience={patience_counter}",
            enabled=verbose,
        )
        release_memory()
        if config.early_stop_patience > 0 and patience_counter >= config.early_stop_patience:
            log(f"[fusion:{candidate_name}] early_stop epoch={epoch}", enabled=verbose)
            break

    _load_state(model, best_state)
    return model, FusionResult(
        best_val_ap=float(best_ap),
        best_val_mrr=float(best_mrr),
        state=best_state,
        mean=mean,
        std=std,
        feature_indices=feature_indices,
        candidate_name=candidate_name,
    )


def fit_fusion_mlp_listwise_fixed(
    train_features: np.ndarray,
    val_features: np.ndarray,
    config: FusionConfig,
    rng: np.random.Generator,
    verbose: bool,
    feature_indices: tuple[int, ...] | None = None,
    candidate_name: str = "listwise_fixed",
    train_row_weights: np.ndarray | None = None,
) -> tuple[FusionMLP, FusionResult, tuple[float, ...]]:
    """Train listwise for exactly ``config.epochs`` and evaluate validation once."""
    if train_features.size == 0 or val_features.size == 0:
        raise ValueError("fusion reranker requires non-empty train and validation features")
    if train_features.ndim != 3 or val_features.ndim != 3:
        raise ValueError("fusion reranker features must have shape [queries, candidates, features]")
    if train_features.shape[-1] != val_features.shape[-1]:
        raise ValueError("fusion train and validation feature dimensions must match")
    if config.epochs <= 0:
        raise ValueError("fixed listwise fusion requires at least one epoch")
    if config.batch_size <= 0:
        raise ValueError("fixed listwise fusion requires a positive batch size")

    if feature_indices is None:
        feature_indices = tuple(range(train_features.shape[-1]))
    train_size = int(train_features.shape[0])
    row_weights = _validate_train_row_weights(
        train_row_weights,
        train_size=train_size,
    )
    _set_jittor_seed_from_rng(rng)
    mean, std = _feature_normalizer_streaming(
        train_features,
        feature_indices,
        config.batch_size,
        context_transform_version=config.context_transform_version,
        row_weights=row_weights,
    )
    model = FusionMLP(input_dim=mean.shape[0], hidden_dim=config.hidden_dim)
    _initialize_fusion_mlp_from_rng(model, rng)
    optimizer = jt.nn.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    epoch_losses: list[float] = []

    epochs = range(1, config.epochs + 1)
    for epoch in track(epochs, description=f"fusion:{candidate_name}", total=config.epochs, enabled=verbose):
        order = rng.permutation(train_size)
        losses: list[float] = []
        for start in range(0, train_size, config.batch_size):
            batch_idx = order[start : start + config.batch_size]
            batch_features = _normalize(
                _prepare_fusion_features(
                    train_features[batch_idx],
                    feature_indices,
                    context_transform_version=config.context_transform_version,
                ),
                mean,
                std,
            )
            features = jt.array(batch_features, dtype=jt.float32)
            logits = model(features)
            batch_weights = (
                None
                if row_weights is None
                else jt.array(row_weights[batch_idx], dtype=jt.float32)
            )
            loss = _listwise_positive_loss(logits, batch_weights)
            optimizer.step(loss)
            losses.append(float(loss.item()))
            del batch_features, features, logits, batch_weights, loss
            if start and start % max(config.batch_size * 16, 1) == 0:
                release_memory()

        mean_loss = float(np.mean(losses)) if losses else math.nan
        if not math.isfinite(mean_loss):
            raise FloatingPointError(f"non-finite listwise fusion loss at epoch {epoch}")
        epoch_losses.append(mean_loss)
        log(f"[fusion:{candidate_name}] epoch={epoch} loss={mean_loss:.5f}", enabled=verbose)
        release_memory()

    state = _snapshot_state(model)
    val_ap, val_mrr = _metrics_from_model_streaming(
        model,
        val_features,
        mean,
        std,
        config.batch_size,
        feature_indices,
        context_transform_version=config.context_transform_version,
    )
    log(
        f"[fusion:{candidate_name}] fixed_training_complete val_ap={val_ap:.5f} val_mrr={val_mrr:.5f}",
        enabled=verbose,
    )
    return model, FusionResult(
        best_val_ap=float(val_ap),
        best_val_mrr=float(val_mrr),
        state=state,
        mean=mean,
        std=std,
        feature_indices=feature_indices,
        candidate_name=candidate_name,
    ), tuple(epoch_losses)


def fit_fusion_mlp_listwise_streaming(
    train_features: np.ndarray,
    val_features: np.ndarray,
    config: FusionConfig,
    rng: np.random.Generator,
    verbose: bool,
    feature_indices: tuple[int, ...] | None = None,
    candidate_name: str = "listwise_streaming",
    train_row_weights: np.ndarray | None = None,
) -> tuple[FusionMLP, FusionResult, tuple[dict[str, float | int], ...]]:
    """Train listwise and early-stop on full-candidate validation MRR."""
    if train_features.size == 0 or val_features.size == 0:
        raise ValueError(
            "fusion reranker requires non-empty train and validation features"
        )
    if train_features.ndim != 3 or val_features.ndim != 3:
        raise ValueError(
            "fusion reranker features must have shape [queries, candidates, features]"
        )
    if train_features.shape[-1] != val_features.shape[-1]:
        raise ValueError("fusion train and validation feature dimensions must match")
    if config.epochs <= 0:
        raise ValueError("streaming listwise fusion requires at least one epoch")
    if config.batch_size <= 0:
        raise ValueError("streaming listwise fusion requires a positive batch size")
    selection_metric = config.selection_metric.lower()
    if selection_metric not in {"ap", "mrr"}:
        raise ValueError(
            f"unsupported fusion selection metric: {config.selection_metric}"
        )

    if feature_indices is None:
        feature_indices = tuple(range(train_features.shape[-1]))
    train_size = int(train_features.shape[0])
    row_weights = _validate_train_row_weights(
        train_row_weights,
        train_size=train_size,
    )
    _set_jittor_seed_from_rng(rng)
    mean, std = _feature_normalizer_streaming(
        train_features,
        feature_indices,
        config.batch_size,
        context_transform_version=config.context_transform_version,
        row_weights=row_weights,
    )
    model = FusionMLP(
        input_dim=mean.shape[0],
        hidden_dim=config.hidden_dim,
    )
    _initialize_fusion_mlp_from_rng(model, rng)
    optimizer = jt.nn.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    best_ap, best_mrr = _metrics_from_model_streaming(
        model,
        val_features,
        mean,
        std,
        config.batch_size,
        feature_indices,
        context_transform_version=config.context_transform_version,
    )
    best_score = _selected_metric(best_ap, best_mrr, selection_metric)
    best_state = _snapshot_state(model)
    patience_counter = 0
    history: list[dict[str, float | int]] = []

    epochs = range(1, config.epochs + 1)
    for epoch in track(
        epochs,
        description=f"fusion:{candidate_name}",
        total=config.epochs,
        enabled=verbose,
    ):
        order = rng.permutation(train_size)
        losses: list[float] = []
        for start in range(0, train_size, config.batch_size):
            batch_idx = order[start : start + config.batch_size]
            selected = _prepare_fusion_features(
                train_features[batch_idx],
                feature_indices,
                context_transform_version=config.context_transform_version,
            )
            batch_features = _normalize(selected, mean, std)
            features = jt.array(batch_features, dtype=jt.float32)
            logits = model(features)
            batch_weights = (
                None
                if row_weights is None
                else jt.array(row_weights[batch_idx], dtype=jt.float32)
            )
            loss = _listwise_positive_loss(logits, batch_weights)
            optimizer.step(loss)
            losses.append(float(loss.item()))
            del selected, batch_features, features, logits, batch_weights, loss
            if start and start % max(config.batch_size * 16, 1) == 0:
                release_memory()

        mean_loss = float(np.mean(losses)) if losses else math.nan
        if not math.isfinite(mean_loss):
            raise FloatingPointError(
                f"non-finite listwise fusion loss at epoch {epoch}"
            )
        val_ap, val_mrr = _metrics_from_model_streaming(
            model,
            val_features,
            mean,
            std,
            config.batch_size,
            feature_indices,
            context_transform_version=config.context_transform_version,
        )
        val_score = _selected_metric(val_ap, val_mrr, selection_metric)
        if val_score >= best_score:
            best_ap = val_ap
            best_mrr = val_mrr
            best_score = val_score
            best_state = _snapshot_state(model)
            patience_counter = 0
        else:
            patience_counter += 1
        history.append(
            {
                "epoch": int(epoch),
                "loss": mean_loss,
                "val_ap": float(val_ap),
                "val_mrr": float(val_mrr),
                "best_score": float(best_score),
                "patience": int(patience_counter),
            }
        )
        log(
            f"[fusion:{candidate_name}] epoch={epoch} loss={mean_loss:.5f} "
            f"val_ap={val_ap:.5f} val_mrr={val_mrr:.5f} "
            f"best_{selection_metric}={best_score:.5f} "
            f"patience={patience_counter}",
            enabled=verbose,
        )
        release_memory()
        if (
            config.early_stop_patience > 0
            and patience_counter >= config.early_stop_patience
        ):
            log(
                f"[fusion:{candidate_name}] early_stop epoch={epoch}",
                enabled=verbose,
            )
            break

    _load_state(model, best_state)
    return model, FusionResult(
        best_val_ap=float(best_ap),
        best_val_mrr=float(best_mrr),
        state=best_state,
        mean=mean,
        std=std,
        feature_indices=feature_indices,
        candidate_name=candidate_name,
    ), tuple(history)


def _validate_train_row_weights(
    train_row_weights: np.ndarray | None,
    *,
    train_size: int,
) -> np.ndarray | None:
    if train_row_weights is None:
        return None
    weights = np.asarray(train_row_weights, dtype=np.float32)
    if weights.ndim != 1 or weights.shape[0] != train_size:
        raise ValueError(
            "listwise fusion train_row_weights must contain one weight per "
            "training query"
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("listwise fusion train_row_weights must be finite")
    if np.any(weights <= 0.0):
        raise ValueError("listwise fusion train_row_weights must be positive")
    return weights


def predict_logits(
    model: FusionMLP,
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    prepared = align_fusion_input_features(
        features,
        expected_dim=int(np.asarray(mean).shape[0]),
    )
    normalized = _normalize(prepared, mean, std)
    with jt.no_grad():
        logits = model(jt.array(normalized, dtype=jt.float32))
        return np.asarray(logits.numpy(), dtype=np.float32)


def build_fusion_from_state(input_dim: int, hidden_dim: int, state: dict[str, np.ndarray]) -> FusionMLP:
    first_layer = state.get("linear1.weight")
    if first_layer is not None:
        first_layer_shape = np.asarray(first_layer).shape
        if len(first_layer_shape) != 2 or first_layer_shape[1] <= 0:
            raise ValueError(
                "fusion state linear1.weight must have shape "
                "[hidden_dim, input_dim]"
            )
        input_dim = int(first_layer_shape[1])
    model = FusionMLP(input_dim=input_dim, hidden_dim=hidden_dim)
    _load_state(model, state)
    return model


def _feature_normalizer(
    features: np.ndarray,
    feature_indices: tuple[int, ...],
    *,
    context_transform_version: int = 0,
    batch_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if context_transform_version != 0:
        return _feature_normalizer_streaming(
            features,
            feature_indices,
            batch_size=max(int(batch_size or 512), 1),
            context_transform_version=context_transform_version,
        )
    selected = _prepare_fusion_features(
        features,
        feature_indices,
        context_transform_version=context_transform_version,
    )
    flat = selected.reshape((-1, selected.shape[-1]))
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def _feature_normalizer_streaming(
    features: np.ndarray,
    feature_indices: tuple[int, ...],
    batch_size: int,
    *,
    context_transform_version: int = 0,
    row_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    weights = _validate_train_row_weights(
        row_weights,
        train_size=int(features.shape[0]),
    )
    total_weight = 0.0
    feature_sum: np.ndarray | None = None
    feature_sq_sum: np.ndarray | None = None
    step = max(int(batch_size), 1)
    for start in range(0, features.shape[0], step):
        end = min(start + step, features.shape[0])
        selected = _prepare_fusion_features(
            features[start:end],
            feature_indices,
            context_transform_version=context_transform_version,
        ).astype(np.float64, copy=False)
        flat = selected.reshape((-1, selected.shape[-1]))
        if weights is None:
            batch_sum = flat.sum(axis=0)
            batch_sq_sum = (flat * flat).sum(axis=0)
            batch_weight = float(flat.shape[0])
        else:
            batch_weights = weights[start:end].astype(
                np.float64,
                copy=False,
            )
            batch_sum = np.einsum(
                "qcf,q->f",
                selected,
                batch_weights,
                optimize=True,
            )
            batch_sq_sum = np.einsum(
                "qcf,qcf,q->f",
                selected,
                selected,
                batch_weights,
                optimize=True,
            )
            batch_weight = float(
                batch_weights.sum(dtype=np.float64) * selected.shape[1]
            )
        if feature_sum is None:
            feature_sum = batch_sum
            feature_sq_sum = batch_sq_sum
        else:
            feature_sum += batch_sum
            feature_sq_sum += batch_sq_sum
        total_weight += batch_weight
        del selected, flat, batch_sum, batch_sq_sum
    if (
        total_weight <= 0.0
        or feature_sum is None
        or feature_sq_sum is None
    ):
        raise ValueError("fusion reranker requires non-empty training features")
    mean64 = feature_sum / total_weight
    variance64 = np.maximum(
        feature_sq_sum / total_weight - mean64 * mean64,
        0.0,
    )
    mean = mean64.astype(np.float32)
    std = np.sqrt(variance64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def _normalize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((features - mean) / std).astype(np.float32, copy=False)


def _select_features(features: np.ndarray, feature_indices: tuple[int, ...]) -> np.ndarray:
    if len(feature_indices) == features.shape[-1] and feature_indices == tuple(range(features.shape[-1])):
        return features
    start = feature_indices[0]
    stop = feature_indices[-1] + 1
    if feature_indices == tuple(range(start, stop)):
        return features[..., start:stop]
    return features[..., feature_indices]


def _prepare_fusion_features(
    features: np.ndarray,
    feature_indices: tuple[int, ...],
    *,
    context_transform_version: int,
) -> np.ndarray:
    selected = _select_features(features, feature_indices)
    if context_transform_version == 0:
        return selected
    from .setwise import setwise_context_features  # noqa: PLC0415

    return setwise_context_features(
        selected,
        transform_version=context_transform_version,
    )


def align_fusion_input_features(
    features: np.ndarray,
    *,
    expected_dim: int,
) -> np.ndarray:
    """Match raw selected features to a stored FusionMLP input contract."""

    values = np.asarray(features)
    if values.ndim != 3 or values.shape[-1] <= 0:
        raise ValueError(
            "fusion features must have shape [queries, candidates, features]"
        )
    expected_dim = int(expected_dim)
    if expected_dim <= 0:
        raise ValueError("fusion expected feature dimension must be positive")
    if values.shape[-1] == expected_dim:
        return values

    multiplier_to_version = {3: 1, 5: 2}
    if expected_dim % values.shape[-1] == 0:
        multiplier = expected_dim // values.shape[-1]
        transform_version = multiplier_to_version.get(multiplier)
        if transform_version is not None:
            from .setwise import setwise_context_features  # noqa: PLC0415

            return setwise_context_features(
                values,
                transform_version=transform_version,
            )
    raise ValueError(
        "fusion feature dimension does not match its stored normalizer: "
        f"features={values.shape[-1]} expected={expected_dim}"
    )


def _metrics_from_model(
    model: FusionMLP,
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    feature_indices: tuple[int, ...],
    *,
    context_transform_version: int = 0,
) -> tuple[float, float]:
    scores = np.empty(features.shape[:2], dtype=np.float32)
    with jt.no_grad():
        for start in range(0, features.shape[0], max(int(batch_size), 1)):
            end = min(start + max(int(batch_size), 1), features.shape[0])
            normalized = _normalize(
                _prepare_fusion_features(
                    features[start:end],
                    feature_indices,
                    context_transform_version=context_transform_version,
                ),
                mean,
                std,
            )
            batch_scores = model(jt.array(normalized, dtype=jt.float32))
            scores[start:end] = np.asarray(batch_scores.numpy(), dtype=np.float32)
    return _ap_from_scores(scores), _mrr_from_scores(scores)


def _metrics_from_model_streaming(
    model: FusionMLP,
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    feature_indices: tuple[int, ...],
    *,
    context_transform_version: int = 0,
) -> tuple[float, float]:
    flat_scores = np.empty(features.shape[0] * features.shape[1], dtype=np.float32)
    reciprocal_rank_sum = 0.0
    with jt.no_grad():
        for start in range(0, features.shape[0], max(int(batch_size), 1)):
            end = min(start + max(int(batch_size), 1), features.shape[0])
            normalized = _normalize(
                _prepare_fusion_features(
                    features[start:end],
                    feature_indices,
                    context_transform_version=context_transform_version,
                ),
                mean,
                std,
            )
            batch_scores = model(jt.array(normalized, dtype=jt.float32))
            score_array = np.asarray(batch_scores.numpy(), dtype=np.float32)
            flat_scores[start * features.shape[1] : end * features.shape[1]] = score_array.ravel()
            if score_array.shape[1] <= 1:
                reciprocal_rank_sum += float(score_array.shape[0])
            else:
                ranks = 1 + (score_array[:, 1:] > score_array[:, 0:1]).sum(axis=1)
                reciprocal_rank_sum += float(np.sum(1.0 / ranks))
            del normalized, batch_scores, score_array
    return _ap_from_flat_scores(flat_scores, features.shape[1]), float(reciprocal_rank_sum / max(features.shape[0], 1))


def _ap_from_flat_scores(scores: np.ndarray, candidate_count: int) -> float:
    labels = np.zeros(scores.shape[0], dtype=np.int8)
    labels[::candidate_count] = 1
    return float(average_precision_score(labels, scores))


def _ap_from_scores(scores: np.ndarray) -> float:
    labels = np.zeros(scores.shape, dtype=np.int8)
    labels[:, 0] = 1
    return float(average_precision_score(labels.ravel(), scores.ravel()))


def _mrr_from_scores(scores: np.ndarray) -> float:
    positive_scores = scores[:, 0:1]
    ranks = 1 + (scores[:, 1:] > positive_scores).sum(axis=1)
    return float(np.mean(1.0 / ranks))


def _selected_metric(ap: float, mrr: float, metric: str) -> float:
    if metric == "ap":
        return ap
    if metric == "mrr":
        return mrr
    raise ValueError(f"unsupported fusion selection metric: {metric}")


def _set_jittor_seed_from_rng(rng: np.random.Generator) -> None:
    jt.sync_all()
    seed = _seed_from_rng_state(rng, salt="jittor")
    jt.set_seed(seed % (2**31 - 1))


def _initialize_fusion_mlp_from_rng(model: FusionMLP, rng: np.random.Generator) -> None:
    init_rng = np.random.default_rng(_seed_from_rng_state(rng, salt="fusion-init"))
    state: dict[str, np.ndarray] = {}
    weight_shapes: dict[str, tuple[int, ...]] = {}
    for key, value in model.state_dict().items():
        shape = tuple(int(dim) for dim in value.shape)
        if key.endswith(".weight") and len(shape) >= 2:
            fan_in = int(shape[1] * np.prod(shape[2:], dtype=np.int64))
            bound = math.sqrt(1.0 / max(fan_in, 1))
            state[key] = init_rng.uniform(-bound, bound, size=shape).astype(np.float32)
            weight_shapes[key] = shape
        elif key.endswith(".bias"):
            weight_key = f"{key.removesuffix('.bias')}.weight"
            weight_shape = weight_shapes.get(weight_key)
            fan_in = int(weight_shape[1] * np.prod(weight_shape[2:], dtype=np.int64)) if weight_shape else 1
            bound = math.sqrt(1.0 / max(fan_in, 1))
            state[key] = init_rng.uniform(-bound, bound, size=shape).astype(np.float32)
        else:
            state[key] = np.asarray(value.numpy(), dtype=np.float32).copy()
    _load_state(model, state)


def _seed_from_rng_state(rng: np.random.Generator, *, salt: str) -> int:
    state_bytes = f"{salt}:{rng.bit_generator.state!r}".encode()
    return int.from_bytes(hashlib.blake2b(state_bytes, digest_size=4).digest(), "little")


def _snapshot_state(model: FusionMLP) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(value.numpy(), dtype=np.float32).copy()
        for key, value in model.state_dict().items()
    }


def _load_state(model: FusionMLP, state: dict[str, np.ndarray]) -> None:
    model.load_state_dict({key: jt.array(value, dtype=jt.float32) for key, value in state.items()})
