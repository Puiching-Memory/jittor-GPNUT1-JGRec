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
    selection_metric: str = "ap"
    early_stop_patience: int = 10


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
    input_dim = len(feature_indices)
    mean, std = _feature_normalizer(train_features, feature_indices)
    selection_metric = config.selection_metric.lower()
    if selection_metric not in {"ap", "mrr"}:
        raise ValueError(f"unsupported fusion selection metric: {config.selection_metric}")

    model = FusionMLP(input_dim=input_dim, hidden_dim=config.hidden_dim)
    _initialize_fusion_mlp_from_rng(model, rng)
    optimizer = jt.nn.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_ap, best_mrr = _metrics_from_model(model, val_features, mean, std, config.batch_size, feature_indices)
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
            batch_features = _normalize(_select_features(train_features[batch_idx], feature_indices), mean, std)
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

        val_ap, val_mrr = _metrics_from_model(model, val_features, mean, std, config.batch_size, feature_indices)
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
    input_dim = len(feature_indices)
    mean, std = _feature_normalizer_streaming(train_features, feature_indices, config.batch_size)
    selection_metric = config.selection_metric.lower()
    if selection_metric not in {"ap", "mrr"}:
        raise ValueError(f"unsupported fusion selection metric: {config.selection_metric}")

    model = FusionMLP(input_dim=input_dim, hidden_dim=config.hidden_dim)
    _initialize_fusion_mlp_from_rng(model, rng)
    optimizer = jt.nn.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_ap, best_mrr = _metrics_from_model_streaming(model, val_features, mean, std, config.batch_size, feature_indices)
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
            batch_features = _normalize(_select_features(train_features[batch_idx], feature_indices), mean, std)
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

        val_ap, val_mrr = _metrics_from_model_streaming(model, val_features, mean, std, config.batch_size, feature_indices)
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


def predict_logits(model: FusionMLP, features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    normalized = _normalize(features, mean, std)
    with jt.no_grad():
        logits = model(jt.array(normalized, dtype=jt.float32))
        return np.asarray(logits.numpy(), dtype=np.float32)


def build_fusion_from_state(input_dim: int, hidden_dim: int, state: dict[str, np.ndarray]) -> FusionMLP:
    model = FusionMLP(input_dim=input_dim, hidden_dim=hidden_dim)
    _load_state(model, state)
    return model


def _feature_normalizer(features: np.ndarray, feature_indices: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    selected = _select_features(features, feature_indices)
    flat = selected.reshape((-1, selected.shape[-1]))
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def _feature_normalizer_streaming(
    features: np.ndarray,
    feature_indices: tuple[int, ...],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    total = 0
    feature_sum: np.ndarray | None = None
    feature_sq_sum: np.ndarray | None = None
    step = max(int(batch_size), 1)
    for start in range(0, features.shape[0], step):
        end = min(start + step, features.shape[0])
        selected = _select_features(features[start:end], feature_indices).astype(np.float64, copy=False)
        flat = selected.reshape((-1, selected.shape[-1]))
        batch_sum = flat.sum(axis=0)
        batch_sq_sum = (flat * flat).sum(axis=0)
        if feature_sum is None:
            feature_sum = batch_sum
            feature_sq_sum = batch_sq_sum
        else:
            feature_sum += batch_sum
            feature_sq_sum += batch_sq_sum
        total += flat.shape[0]
        del selected, flat, batch_sum, batch_sq_sum
    if total <= 0 or feature_sum is None or feature_sq_sum is None:
        raise ValueError("fusion reranker requires non-empty training features")
    mean64 = feature_sum / total
    variance64 = np.maximum(feature_sq_sum / total - mean64 * mean64, 0.0)
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


def _metrics_from_model(
    model: FusionMLP,
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    feature_indices: tuple[int, ...],
) -> tuple[float, float]:
    scores = np.empty(features.shape[:2], dtype=np.float32)
    with jt.no_grad():
        for start in range(0, features.shape[0], max(int(batch_size), 1)):
            end = min(start + max(int(batch_size), 1), features.shape[0])
            normalized = _normalize(_select_features(features[start:end], feature_indices), mean, std)
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
) -> tuple[float, float]:
    flat_scores = np.empty(features.shape[0] * features.shape[1], dtype=np.float32)
    reciprocal_rank_sum = 0.0
    with jt.no_grad():
        for start in range(0, features.shape[0], max(int(batch_size), 1)):
            end = min(start + max(int(batch_size), 1), features.shape[0])
            normalized = _normalize(_select_features(features[start:end], feature_indices), mean, std)
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
    state_bytes = f"{salt}:{rng.bit_generator.state!r}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(state_bytes, digest_size=4).digest(), "little")


def _snapshot_state(model: FusionMLP) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(value.numpy(), dtype=np.float32).copy()
        for key, value in model.state_dict().items()
    }


def _load_state(model: FusionMLP, state: dict[str, np.ndarray]) -> None:
    model.load_state_dict({key: jt.array(value, dtype=jt.float32) for key, value in state.items()})
