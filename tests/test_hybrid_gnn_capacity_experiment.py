from __future__ import annotations

from jgrec.rankers.hybrid.config import GraphTowerConfig
from jgrec.rankers.hybrid.gnn_experiment import (
    resolve_gnn_capacity_experiment,
)


def test_resolve_gnn_capacity_experiment_changes_only_requested_capacity() -> None:
    baseline = GraphTowerConfig(
        epochs=50,
        max_train_edges=40_000,
        embedding_dim=128,
        layers=2,
    )

    config, variants = resolve_gnn_capacity_experiment(
        baseline,
        variant_names=("short_none",),
        epochs=50,
        max_train_edges=200_000,
    )

    assert config == GraphTowerConfig(
        epochs=50,
        max_train_edges=200_000,
        embedding_dim=128,
        layers=2,
    )
    assert baseline.max_train_edges == 40_000
    assert variants == {"short_none": ("gnn_short", "none")}
