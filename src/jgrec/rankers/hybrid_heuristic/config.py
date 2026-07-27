from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .fusion import FusionConfig

GRAPH_WINDOW_NAMES = ("gnn_full", "gnn_recent", "gnn_short")
SEQUENCE_FEATURE_NAMES = ("gru_dot", "gru_cosine", "gru_decay_dot")
TWO_TOWER_FEATURE_NAMES = ("two_tower_dot", "two_tower_cosine")
HEURISTIC_FEATURE_NAMES = (
    # 四象限启发式：local=src 出边记忆, global=dst 入边记忆, freq/recency
    "heur_lr_freq",
    "heur_lr_recency",
    "heur_gr_freq",
    "heur_gr_recency",
    "heur_lp_log",
    "heur_gp_log",
    "heur_combined",
    # 时间窗共同邻居（短/中/长窗 + 衰减加权）
    "heur_cn_short",
    "heur_cn_medium",
    "heur_cn_long",
    "heur_cn_decay",
    # 方向化时间衰减共现 + 2跳路径分
    "heur_cooccur_fwd",
    "heur_cooccur_bwd",
    "heur_hop2_score",
)
HEURISTIC_FEATURE_DIM = len(HEURISTIC_FEATURE_NAMES)
TARGET_WINDOW_FRACTION_LABELS = ("001", "005", "020", "100")
TARGET_WINDOW_FEATURE_NAMES = tuple(
    f"{name}_w{label}"
    for label in TARGET_WINDOW_FRACTION_LABELS
    for name in (
        "target_pop",
        "target_pop_share",
        "target_recency",
        "target_pop_rank",
    )
)
SOURCE_PROFILE_FEATURE_NAMES = (
    "source_profile_cooccur_sum",
    "source_profile_cooccur_max",
    "source_profile_cosine_sum",
    "source_profile_cosine_max",
    "source_profile_recent_cosine_sum",
    "source_profile_recent_cosine_max",
    "source_profile_item2vec_dot",
    "source_profile_item2vec_cosine",
    "source_profile_recent_item2vec_dot",
    "source_profile_recent_item2vec_cosine",
    # 候选感知重排序特征（方向 B，OTTO/KDD 冠军思路）
    "source_profile_posdecay_cooccur",  # 位置衰减加权共现和（越近权重越高）
    "source_profile_last_position_inv",  # 候选在 src 序列中最近出现位置倒数
    "source_profile_item2vec_sim_max",  # 候选与历史各项 item2vec 相似度 max
)
DEFAULT_PREDICTION_CACHE_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class HeuristicTowerConfig:
    enabled: bool = True
    window_short_ratio: float = 0.05
    window_medium_ratio: float = 0.20
    window_long_ratio: float = 0.50
    cooccur_time_decay: bool = True
    hop2_enabled: bool = True
    vectorize_quadrant: bool = True
    cache_max_bytes: int = DEFAULT_PREDICTION_CACHE_BYTES // 2


@dataclass(frozen=True)
class StructureTowerConfig:
    enabled: bool = True
    cooccur_enabled: bool = True
    transition_enabled: bool = True
    cooccur_history_limit: int = 128
    future_only_transition_cooccur: bool = False
    bridge_overlap_threshold: float = 0.50
    bridge_min_role_degree: int = 2
    predict_neighbor_limit: int = 0
    cache_max_bytes: int = DEFAULT_PREDICTION_CACHE_BYTES // 2


@dataclass(frozen=True)
class CandidatePriorConfig:
    enabled: bool = True
    # Test-candidate frequency is a transductive signal over the GIVEN candidate
    # lists (inputs, not labels) and was part of the 1.1983 champion baseline.
    include_test_frequency: bool = True


@dataclass(frozen=True)
class TargetWindowConfig:
    enabled: bool = True
    window_fractions: tuple[float, ...] = (0.01, 0.05, 0.20, 1.00)


@dataclass(frozen=True)
class GraphTowerConfig:
    enabled: bool = True
    model_name: str = "xsimgcl"
    edge_weighting: str = "none"
    time_decay_ratio: float = 0.05
    embedding_dim: int = 128
    layers: int = 2
    epochs: int = 50
    early_stop_patience: int = 3
    early_stop_val_ratio: float = 0.1
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
    epochs: int = 50
    early_stop_patience: int = 3
    early_stop_val_ratio: float = 0.1
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
    epochs: int = 50
    early_stop_patience: int = 3
    early_stop_val_ratio: float = 0.1
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
class SourceProfileConfig:
    enabled: bool = True
    deterministic_enabled: bool = True
    item2vec_enabled: bool = True
    embedding_dim: int = 64
    epochs: int = 50
    early_stop_patience: int = 3
    early_stop_val_ratio: float = 0.1
    batch_size: int = 2048
    score_batch_size: int = 8192
    max_samples: int = 100_000
    window_size: int = 16
    recent_k: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    predict_history_limit: int = 0
    cache_max_bytes: int = DEFAULT_PREDICTION_CACHE_BYTES // 2


@dataclass(frozen=True)
class TrainingConfig:
    val_ratio: float = 0.15
    context_ratio: float = 0.75
    max_train_events: int = 20_000
    max_val_events: int = 5_000
    supervised_feature_batch_size: int = 4096
    supervised_feature_memmap: bool = True
    num_negatives: int = 31
    max_fit_events: int = 0
    epochs: int = 15
    train_batch_size: int = 512
    lr: float = 0.001
    weight_decay: float = 0.0
    selection_metric: str = "ap"
    early_stop_patience: int = 3
    seed: int = 42
    verbose: bool = True
    encoder_state_cache_enabled: bool = True
    auto_strategy_enabled: bool = True
    candidate_prior_enabled: bool = True
    # Transductive signal over the given test candidate lists; part of the
    # 1.1983 champion baseline. Disabling it (the reverted "leakage fix") was the
    # main online regression.
    candidate_prior_include_test_frequency: bool = True
    target_window_enabled: bool = True
    target_window_fractions: tuple[float, ...] = (0.01, 0.05, 0.20, 1.00)
    test_candidate_negative_ratio: float = 0.0
    dataset_train_path: Path | None = None
    dataset_test_path: Path | None = None
    auto_mode: str = "manual"
    profile_holdout_pair_hit_rate: float = 0.0
    profile_candidate_unseen_dst_rate: float = 0.0
    structure_enabled: bool = True
    heuristic_enabled: bool = True
    heuristic_cooccur_time_decay: bool = True
    heuristic_hop2_enabled: bool = True
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
    gnn_epochs: int = 10
    gnn_early_stop_patience: int = 3
    gnn_early_stop_val_ratio: float = 0.1
    gnn_batch_size: int = 2048
    gnn_max_graph_edges: int = 0
    gnn_max_train_edges: int = 40_000
    gnn_lr: float = 0.001
    gnn_reg_weight: float = 1e-5
    gnn_cl_rate: float = 1e-4
    seq_enabled: bool = True
    seq_epochs: int = 10
    seq_early_stop_patience: int = 3
    seq_early_stop_val_ratio: float = 0.1
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
    two_tower_epochs: int = 10
    two_tower_early_stop_patience: int = 3
    two_tower_early_stop_val_ratio: float = 0.1
    two_tower_batch_size: int = 512
    two_tower_score_batch_size: int = 2048
    two_tower_max_samples: int = 50_000
    source_profile_enabled: bool = True
    source_profile_deterministic_enabled: bool = True
    source_profile_item2vec_enabled: bool = True
    source_profile_embedding_dim: int = 64
    source_profile_epochs: int = 10
    source_profile_early_stop_patience: int = 3
    source_profile_early_stop_val_ratio: float = 0.1
    source_profile_batch_size: int = 2048
    source_profile_score_batch_size: int = 8192
    source_profile_max_samples: int = 100_000
    source_profile_window_size: int = 16
    source_profile_recent_k: int = 32
    source_profile_predict_history_limit: int = 0
    structure_predict_neighbor_limit: int = 0
    prediction_cache_max_bytes: int = DEFAULT_PREDICTION_CACHE_BYTES
    fusion_hidden_dim: int = 64
    fusion_mode: str = "mlp"
    hard_negative_ratio: float = 0.5
    popular_negative_ratio: float = 0.25
    negative_sampling_workers: int = 0

    def candidate_prior_config(self) -> CandidatePriorConfig:
        return CandidatePriorConfig(
            enabled=self.candidate_prior_enabled,
            include_test_frequency=self.candidate_prior_include_test_frequency,
        )

    def target_window_config(self) -> TargetWindowConfig:
        return TargetWindowConfig(
            enabled=self.target_window_enabled,
            window_fractions=self.target_window_fractions,
        )

    def structure_config(self) -> StructureTowerConfig:
        structure_cache_bytes = max(int(self.prediction_cache_max_bytes), 0) // 2
        return StructureTowerConfig(
            enabled=self.structure_enabled,
            cooccur_enabled=self.structure_cooccur_enabled,
            transition_enabled=self.structure_transition_enabled,
            cooccur_history_limit=self.structure_cooccur_history_limit,
            future_only_transition_cooccur=self.structure_future_only_transition_cooccur,
            predict_neighbor_limit=self.structure_predict_neighbor_limit,
            cache_max_bytes=structure_cache_bytes,
        )

    def heuristic_config(self) -> HeuristicTowerConfig:
        heuristic_cache_bytes = max(int(self.prediction_cache_max_bytes), 0) // 2
        return HeuristicTowerConfig(
            enabled=self.heuristic_enabled,
            cooccur_time_decay=self.heuristic_cooccur_time_decay,
            hop2_enabled=self.heuristic_hop2_enabled,
            cache_max_bytes=heuristic_cache_bytes,
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
            early_stop_patience=self.gnn_early_stop_patience,
            early_stop_val_ratio=self.gnn_early_stop_val_ratio,
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
            early_stop_patience=self.seq_early_stop_patience,
            early_stop_val_ratio=self.seq_early_stop_val_ratio,
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
            early_stop_patience=self.two_tower_early_stop_patience,
            early_stop_val_ratio=self.two_tower_early_stop_val_ratio,
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

    def source_profile_config(self) -> SourceProfileConfig:
        total_cache_bytes = max(int(self.prediction_cache_max_bytes), 0)
        return SourceProfileConfig(
            enabled=self.source_profile_enabled,
            deterministic_enabled=self.source_profile_deterministic_enabled,
            item2vec_enabled=self.source_profile_item2vec_enabled,
            embedding_dim=self.source_profile_embedding_dim,
            epochs=self.source_profile_epochs,
            early_stop_patience=self.source_profile_early_stop_patience,
            early_stop_val_ratio=self.source_profile_early_stop_val_ratio,
            batch_size=self.source_profile_batch_size,
            score_batch_size=self.source_profile_score_batch_size,
            max_samples=self.source_profile_max_samples,
            window_size=self.source_profile_window_size,
            recent_k=self.source_profile_recent_k,
            lr=self.lr,
            weight_decay=self.weight_decay,
            predict_history_limit=self.source_profile_predict_history_limit,
            cache_max_bytes=total_cache_bytes - total_cache_bytes // 2,
        )

    def fusion_config(self) -> FusionConfig:
        from .fusion import FusionConfig  # noqa: PLC0415

        return FusionConfig(
            epochs=self.epochs,
            batch_size=self.train_batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            hidden_dim=self.fusion_hidden_dim,
            selection_metric=self.selection_metric,
            early_stop_patience=self.early_stop_patience,
        )
