from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CRAFTBaselineConfig:
    val_ratio: float = 0.15
    epochs: int = 100
    batch_size: int = 200
    lr: float = 0.0001
    early_stop_patience: int = 10
    num_neighbors: int = 30
    hidden_size: int = 64
    n_layers: int = 2
    n_heads: int = 2
    dropout: float = 0.1

