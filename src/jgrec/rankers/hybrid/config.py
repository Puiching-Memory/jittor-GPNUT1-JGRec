from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .fusion import FusionConfig

GRAPH_WINDOW_NAMES = ("gnn_full", "gnn_recent", "gnn_short")
SEQUENCE_FEATURE_NAMES = ("gru_score",)
TWO_TOWER_FEATURE_NAMES = ("two_tower_dot", "two_tower_cosine")


@dataclass(frozen=True)
class StructureTowerConfig:
    enabled: bool = True
    cooccur_enabled: bool = True
    transition_enabled: bool = True
    cooccur_history_limit: int = 128
    future_only_transition_cooccur: bool = False


@dataclass(frozen=True)
class CandidatePriorConfig:
    enabled: bool = True


@dataclass(frozen=True)
class GraphTowerConfig:
    enabled: bool = True
    model_name: str = "xsimgcl"
    edge_weighting: str = "none"
    time_decay_ratio: float = 0.05
    embedding_dim: int = 128
    layers: int = 2
    epochs: int = 3
    batch_size: int = 8192
    max_graph_edges: int = 0
    max_train_edges: int = 200_000
    lr: float = 1e-3
    weight_decay: float = 0.0
    reg_weight: float = 1e-5
    cl_rate: float = 1e-4
    temperature: float = 0.05
    eps: float = 0.1


@dataclass(frozen=True)
class SequenceTowerConfig:
    enabled: bool = True
    epochs: int = 3
    batch_size: int = 512
    score_batch_size: int = 1024
    max_samples: int = 50_000
    max_seq_len: int = 64
    hidden_size: int = 128
    layers: int = 2
    heads: int = 4
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 0.0


@dataclass(frozen=True)
class TwoTowerConfig:
    enabled: bool = True
    embedding_dim: int = 64
    hidden_dim: int = 64
    epochs: int = 3
    batch_size: int = 512
    score_batch_size: int = 2048
    max_samples: int = 50_000
    lr: float = 1e-3
    weight_decay: float = 0.0
    num_negatives: int = 31
    hard_negative_ratio: float = 0.5
    popular_negative_ratio: float = 0.25
    negative_sampling_workers: int = 0


@dataclass(frozen=True)
class TrainingConfig:
    val_ratio: float = 0.15
    context_ratio: float = 0.75
    max_train_events: int = 20_000
    max_val_events: int = 5_000
    supervised_feature_batch_size: int = 4096
    supervised_feature_memmap: bool = False
    num_negatives: int = 31
    max_fit_events: int = 0
    epochs: int = 5
    train_batch_size: int = 512
    lr: float = 0.001
    weight_decay: float = 0.0
    selection_metric: str = "ap"
    early_stop_patience: int = 10
    seed: int = 42
    verbose: bool = True
    encoder_state_cache_enabled: bool = True
    auto_strategy_enabled: bool = True
    candidate_prior_enabled: bool = True
    test_candidate_negative_ratio: float = 0.0
    dataset_train_path: Path | None = None
    dataset_test_path: Path | None = None
    auto_mode: str = "manual"
    profile_holdout_pair_hit_rate: float = 0.0
    profile_candidate_unseen_dst_rate: float = 0.0
    structure_enabled: bool = True
    structure_cooccur_enabled: bool = True
    structure_transition_enabled: bool = True
    structure_cooccur_history_limit: int = 128
    structure_future_only_transition_cooccur: bool = False
    gnn_enabled: bool = True
    gnn_model: str = "xsimgcl"
    gnn_edge_weighting: str = "none"
    gnn_time_decay_ratio: float = 0.05
    gnn_embedding_dim: int = 128
    gnn_layers: int = 2
    gnn_epochs: int = 3
    gnn_batch_size: int = 2048
    gnn_max_graph_edges: int = 0
    gnn_max_train_edges: int = 40_000
    gnn_lr: float = 0.001
    gnn_reg_weight: float = 1e-5
    gnn_cl_rate: float = 1e-4
    seq_enabled: bool = True
    seq_epochs: int = 3
    seq_batch_size: int = 512
    seq_score_batch_size: int = 1024
    seq_max_samples: int = 50_000
    seq_max_len: int = 64
    seq_hidden_size: int = 128
    seq_layers: int = 2
    seq_heads: int = 4
    seq_dropout: float = 0.2
    two_tower_enabled: bool = True
    two_tower_embedding_dim: int = 64
    two_tower_hidden_dim: int = 64
    two_tower_epochs: int = 3
    two_tower_batch_size: int = 512
    two_tower_score_batch_size: int = 2048
    two_tower_max_samples: int = 50_000
    fusion_hidden_dim: int = 64
    hard_negative_ratio: float = 0.5
    popular_negative_ratio: float = 0.25
    negative_sampling_workers: int = 0

    def candidate_prior_config(self) -> CandidatePriorConfig:
        return CandidatePriorConfig(enabled=self.candidate_prior_enabled)

    def structure_config(self) -> StructureTowerConfig:
        return StructureTowerConfig(
            enabled=self.structure_enabled,
            cooccur_enabled=self.structure_cooccur_enabled,
            transition_enabled=self.structure_transition_enabled,
            cooccur_history_limit=self.structure_cooccur_history_limit,
            future_only_transition_cooccur=self.structure_future_only_transition_cooccur,
        )

    def graph_config(self) -> GraphTowerConfig:
        return GraphTowerConfig(
            enabled=self.gnn_enabled,
            model_name=self.gnn_model,
            edge_weighting=self.gnn_edge_weighting,
            time_decay_ratio=self.gnn_time_decay_ratio,
            embedding_dim=self.gnn_embedding_dim,
            layers=self.gnn_layers,
            epochs=self.gnn_epochs,
            batch_size=self.gnn_batch_size,
            max_graph_edges=self.gnn_max_graph_edges,
            max_train_edges=self.gnn_max_train_edges,
            lr=self.gnn_lr,
            weight_decay=self.weight_decay,
            reg_weight=self.gnn_reg_weight,
            cl_rate=self.gnn_cl_rate,
        )

    def sequence_config(self) -> SequenceTowerConfig:
        return SequenceTowerConfig(
            enabled=self.seq_enabled,
            epochs=self.seq_epochs,
            batch_size=self.seq_batch_size,
            score_batch_size=self.seq_score_batch_size,
            max_samples=self.seq_max_samples,
            max_seq_len=self.seq_max_len,
            hidden_size=self.seq_hidden_size,
            layers=self.seq_layers,
            heads=self.seq_heads,
            dropout=self.seq_dropout,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def two_tower_config(self) -> TwoTowerConfig:
        return TwoTowerConfig(
            enabled=self.two_tower_enabled,
            embedding_dim=self.two_tower_embedding_dim,
            hidden_dim=self.two_tower_hidden_dim,
            epochs=self.two_tower_epochs,
            batch_size=self.two_tower_batch_size,
            score_batch_size=self.two_tower_score_batch_size,
            max_samples=self.two_tower_max_samples,
            lr=self.lr,
            weight_decay=self.weight_decay,
            num_negatives=self.num_negatives,
            hard_negative_ratio=self.hard_negative_ratio,
            popular_negative_ratio=self.popular_negative_ratio,
            negative_sampling_workers=self.negative_sampling_workers,
        )

    def fusion_config(self) -> FusionConfig:
        from .fusion import FusionConfig

        return FusionConfig(
            epochs=self.epochs,
            batch_size=self.train_batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            hidden_dim=self.fusion_hidden_dim,
            selection_metric=self.selection_metric,
            early_stop_patience=self.early_stop_patience,
        )
