from __future__ import annotations

from dataclasses import dataclass

import jittor as jt
import numpy as np

from .logging import log, track


@dataclass(frozen=True)
class FusionConfig:
    epochs: int = 5
    batch_size: int = 512
    lr: float = 0.001
    weight_decay: float = 0.0
    hidden_dim: int = 64


@dataclass(frozen=True)
class FusionResult:
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

    input_dim = train_features.shape[-1]
    mean, std = _feature_normalizer(train_features)
    train_x = _normalize(train_features, mean, std)
    val_x = _normalize(val_features, mean, std)

    model = FusionMLP(input_dim=input_dim, hidden_dim=config.hidden_dim)
    optimizer = jt.nn.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_mrr = _mrr_from_model(model, val_x)
    best_state = _snapshot_state(model)
    train_size = train_x.shape[0]

    epochs = range(1, config.epochs + 1)
    for epoch in track(epochs, description=f"fusion:{candidate_name}", total=config.epochs, enabled=verbose):
        order = rng.permutation(train_size)
        losses: list[float] = []
        for start in range(0, train_size, config.batch_size):
            batch_idx = order[start : start + config.batch_size]
            features = jt.array(train_x[batch_idx], dtype=jt.float32)
            logits = model(features)
            shifted = logits - logits.max(dim=1, keepdims=True)
            log_probs = shifted - jt.log(jt.exp(shifted).sum(dim=1, keepdims=True))
            loss = -log_probs[:, 0].mean()
            optimizer.step(loss)
            losses.append(float(loss.item()))

        val_mrr = _mrr_from_model(model, val_x)
        if val_mrr >= best_mrr:
            best_mrr = val_mrr
            best_state = _snapshot_state(model)
        mean_loss = float(np.mean(losses)) if losses else 0.0
        log(
            f"[fusion:{candidate_name}] epoch={epoch} loss={mean_loss:.5f} "
            f"val_mrr={val_mrr:.5f} best={best_mrr:.5f}",
            enabled=verbose,
        )

    _load_state(model, best_state)
    if feature_indices is None:
        feature_indices = tuple(range(input_dim))
    return model, FusionResult(
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


def _feature_normalizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = features.reshape((-1, features.shape[-1]))
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def _normalize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((features - mean) / std).astype(np.float32, copy=False)


def _mrr_from_model(model: FusionMLP, features: np.ndarray) -> float:
    with jt.no_grad():
        scores = np.asarray(model(jt.array(features, dtype=jt.float32)).numpy(), dtype=np.float32)
    positive_scores = scores[:, 0:1]
    ranks = 1 + (scores[:, 1:] > positive_scores).sum(axis=1)
    return float(np.mean(1.0 / ranks))


def _snapshot_state(model: FusionMLP) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(value.numpy(), dtype=np.float32).copy()
        for key, value in model.state_dict().items()
    }


def _load_state(model: FusionMLP, state: dict[str, np.ndarray]) -> None:
    model.load_state_dict({key: jt.array(value, dtype=jt.float32) for key, value in state.items()})
