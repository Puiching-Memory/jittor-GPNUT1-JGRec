from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score

from jgrec.logging import log


@dataclass(frozen=True)
class LGBMFusionResult:
    best_val_ap: float
    best_val_mrr: float
    model_text: str
    feature_indices: tuple[int, ...]
    candidate_name: str
    mlp_weight: float = 0.5
    blend_mode: str = "probability"
    mlp_temperature: float = 1.0
    lgbm_temperature: float = 1.0
    rrf_k: float = 60.0


def fit_fusion_lgbm(
    train_features: np.ndarray,
    val_features: np.ndarray,
    selection_metric: str = "mrr",
    verbose: bool = True,
    feature_indices: tuple[int, ...] | None = None,
    candidate_name: str = "all",
) -> LGBMFusionResult:
    import lightgbm as lgb  # noqa: PLC0415

    selection_metric = selection_metric.strip().lower()
    if selection_metric not in {"ap", "mrr"}:
        raise ValueError(f"Unsupported LightGBM selection metric: {selection_metric!r}")

    if feature_indices is None:
        feature_indices = tuple(range(train_features.shape[-1]))

    train_X, train_y, train_group = _flatten_for_ranking(train_features, feature_indices)
    val_X, val_y, val_group = _flatten_for_ranking(val_features, feature_indices)

    train_ds = lgb.Dataset(train_X, label=train_y, group=train_group, free_raw_data=True)
    val_ds = lgb.Dataset(val_X, label=val_y, group=val_group, reference=train_ds, free_raw_data=True)

    params = {
        "objective": "lambdarank",
        "metric": "None" if selection_metric == "mrr" else "map",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    train_kwargs = {}
    if selection_metric == "mrr":
        train_kwargs["feval"] = _full_candidate_mrr_evaluator(val_features.shape[1])

    booster = lgb.train(
        params,
        train_ds,
        num_boost_round=500,
        valid_sets=[val_ds],
        callbacks=[
            lgb.early_stopping(50, first_metric_only=True, verbose=verbose),
            lgb.log_evaluation(10 if verbose else 0),
        ],
        **train_kwargs,
    )
    del train_ds, train_X, train_y

    n_candidates = val_features.shape[1]
    val_scores = booster.predict(val_X, num_iteration=booster.best_iteration)
    del val_ds, val_X, val_y
    val_ap, val_mrr = _eval_ranking(val_scores, n_candidates)
    name = f"lgbm_{candidate_name}"
    log(
        f"[fusion-lgbm:{name}] early_stop_metric={selection_metric} "
        f"best_iteration={booster.best_iteration} val_ap={val_ap:.5f} val_mrr={val_mrr:.5f}",
        enabled=verbose,
    )

    return LGBMFusionResult(
        best_val_ap=val_ap,
        best_val_mrr=val_mrr,
        model_text=booster.model_to_string(),
        feature_indices=feature_indices,
        candidate_name=name,
    )


def predict_logits_lgbm(model_text: str, features: np.ndarray) -> np.ndarray:
    import lightgbm as lgb  # noqa: PLC0415

    booster = lgb.Booster(model_str=model_text)
    shape = features.shape[:2]
    flat = np.ascontiguousarray(features.reshape(-1, features.shape[-1]), dtype=np.float32)
    scores = booster.predict(flat)
    return scores.reshape(shape).astype(np.float32)


def _flatten_for_ranking(
    features: np.ndarray,
    feature_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    n_events, n_candidates = features.shape[0], features.shape[1]
    selected = _select_columns(features, feature_indices)
    X = np.ascontiguousarray(selected.reshape(-1, len(feature_indices)), dtype=np.float32)
    y = np.zeros(n_events * n_candidates, dtype=np.float32)
    y[::n_candidates] = 1.0
    group = [n_candidates] * n_events
    return X, y, group


def _select_columns(features: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    n_features = features.shape[-1]
    if indices == tuple(range(n_features)):
        return features
    start, stop = indices[0], indices[-1] + 1
    if indices == tuple(range(start, stop)):
        return features[..., start:stop]
    return features[..., indices]


def _eval_ranking(scores: np.ndarray, n_candidates: int) -> tuple[float, float]:
    labels = np.zeros(scores.shape[0], dtype=np.int8)
    labels[::n_candidates] = 1
    ap = float(average_precision_score(labels, scores))
    return ap, _full_candidate_mrr(scores, n_candidates)


def _full_candidate_mrr(scores: np.ndarray, candidate_count: int) -> float:
    scores_2d = np.asarray(scores).reshape(-1, candidate_count)
    ranks = 1 + (scores_2d[:, 1:] > scores_2d[:, 0:1]).sum(axis=1)
    return float(np.mean(1.0 / ranks))


def _full_candidate_mrr_evaluator(candidate_count: int):
    def evaluate(scores: np.ndarray, _dataset) -> tuple[str, float, bool]:
        return "full_candidate_mrr", _full_candidate_mrr(scores, candidate_count), True

    return evaluate
