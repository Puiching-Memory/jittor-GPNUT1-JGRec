from __future__ import annotations

from dataclasses import replace
from typing import Any

from jgrec.rankers.hybrid.config import (
    TrainingConfig,
    graph_window_edge_parameters,
)
from jgrec.rankers.hybrid.fusion import FusionResult
from jgrec.rankers.hybrid.fusion_lgbm import LGBMFusionResult

WINNING_GNN_EPOCHS = 50
WINNING_GNN_MAX_TRAIN_EDGES = 40_000
WINNING_GNN_SHORT_EDGE_WEIGHTING = "none"
SETWISE_CONTEXT_MULTIPLIER = 3


def install_gnn_short_setwise_fusion(
    source_state: dict[str, Any],
    *,
    setwise_result: FusionResult,
    hidden_dim: int,
    setwise_weight: float,
) -> dict[str, Any]:
    """Install the matched fusion head without rebuilding any encoder tower."""

    config = source_state.get("config")
    if not isinstance(config, TrainingConfig):
        raise TypeError("checkpoint Dataset2 state has no TrainingConfig")
    graph_config = config.graph_config()
    short_weighting, _ = graph_window_edge_parameters(
        graph_config,
        "gnn_short",
    )
    if graph_config.epochs != WINNING_GNN_EPOCHS:
        raise ValueError(
            "GNN short checkpoint integration requires the winning 50 epochs"
        )
    if graph_config.max_train_edges != WINNING_GNN_MAX_TRAIN_EDGES:
        raise ValueError(
            "GNN short checkpoint integration requires the winning "
            "40,000 max train edges"
        )
    if short_weighting != WINNING_GNN_SHORT_EDGE_WEIGHTING:
        raise ValueError(
            "GNN short checkpoint integration requires the winning "
            "unweighted short window"
        )

    feature_names = tuple(source_state.get("feature_names", ()))
    if feature_names.count("gnn_short") != 1:
        raise ValueError("checkpoint feature schema must contain one gnn_short")
    context_feature_count = len(feature_names) * SETWISE_CONTEXT_MULTIPLIER
    if setwise_result.mean.shape != (context_feature_count,):
        raise ValueError(
            "Setwise mean must match the "
            f"{context_feature_count}-feature context schema"
        )
    if setwise_result.std.shape != (context_feature_count,):
        raise ValueError(
            "Setwise std must match the "
            f"{context_feature_count}-feature context schema"
        )
    if not setwise_result.feature_indices:
        raise ValueError("Setwise feature_indices must not be empty")
    if min(setwise_result.feature_indices) < 0 or max(
        setwise_result.feature_indices
    ) >= context_feature_count:
        raise ValueError("Setwise feature_indices are outside the context schema")
    if hidden_dim <= 0:
        raise ValueError("Setwise hidden_dim must be positive")
    if not 0.0 <= setwise_weight <= 1.0:
        raise ValueError("Setwise blend weight must be within [0, 1]")
    if source_state.get("setwise_fusion_state") is not None:
        raise ValueError("source checkpoint already has a Setwise fusion state")
    if source_state.get("setwise_fusion_result") is not None:
        raise ValueError("source checkpoint already has a Setwise fusion result")

    lgbm_result = source_state.get("lgbm_result")
    if not isinstance(lgbm_result, LGBMFusionResult):
        raise TypeError("checkpoint Dataset2 state has no LightGBM fusion result")

    candidate_state = dict(source_state)
    candidate_state["setwise_fusion_state"] = setwise_result.state
    candidate_state["setwise_fusion_result"] = setwise_result
    candidate_state["setwise_hidden_dim"] = int(hidden_dim)
    candidate_state["lgbm_result"] = replace(
        lgbm_result,
        mlp_weight=float(setwise_weight),
    )
    return candidate_state
