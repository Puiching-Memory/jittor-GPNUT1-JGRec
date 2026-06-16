from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalGraphTrainingConfig:
    val_ratio: float = 0.15
    max_train_events: int = 20_000
    max_val_events: int = 5_000
    num_negatives: int = 99
    max_fit_events: int = 0
    epochs: int = 8
    train_batch_size: int = 256
    lr: float = 0.001
    weight_decay: float = 0.0
    selection_metric: str = "ap"
    early_stop_patience: int = 10
    seed: int = 42
    verbose: bool = True
    history_len: int = 64
    candidate_history_len: int = 32
    hidden_size: int = 128
    layers: int = 3
    heads: int = 4
    dropout: float = 0.15
    training_candidates: str = "test_like"
    validation_candidates: str = "test_like"
    candidate_recent_feature_group: str = "recency_rank"
    # Transductive signal over the given test candidate lists (inputs, not
    # labels); matches the hybrid champion behavior. Toggle off to ablate.
    candidate_include_test_frequency: bool = True
    refit_full: bool = True
