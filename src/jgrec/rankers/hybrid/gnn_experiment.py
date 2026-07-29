from __future__ import annotations

from dataclasses import replace

from .config import GraphTowerConfig

GNN_CAPACITY_VARIANTS = {
    "short_none": ("gnn_short", "none"),
    "short_repeat": ("gnn_short", "repeat"),
    "short_time_decay": ("gnn_short", "time_decay"),
    "recent_none": ("gnn_recent", "none"),
    "recent_time_decay": ("gnn_recent", "time_decay"),
}


def resolve_gnn_capacity_experiment(
    graph_config: GraphTowerConfig,
    *,
    variant_names: tuple[str, ...],
    epochs: int,
    max_train_edges: int,
) -> tuple[GraphTowerConfig, dict[str, tuple[str, str]]]:
    variants = {
        name: GNN_CAPACITY_VARIANTS[name]
        for name in variant_names
    }
    return (
        replace(
            graph_config,
            epochs=epochs,
            max_train_edges=max_train_edges,
        ),
        variants,
    )
