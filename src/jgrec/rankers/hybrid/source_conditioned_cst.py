from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import jittor as jt

from .candidate_set_transformer import (
    CandidateSetTransformerConfig,
    _CandidateTransformerBlock,
    candidate_relative_features,
)
from .sequence_model import TIME_DELTA_BUCKETS

ABCDVariant = Literal["A", "B", "C", "D"]


@dataclass(frozen=True)
class SourceConditionedCSTConfig:
    input_dim: int
    num_items: int
    variant: ABCDVariant
    model_dim: int = 64
    heads: int = 4
    candidate_layers: int = 2
    source_layers: int = 1
    source_max_length: int = 64
    dropout: float = 0.05
    feedforward_multiplier: int = 2
    relative_context: str = "mean_max"

    def __post_init__(self) -> None:
        if self.variant not in {"A", "B", "C", "D"}:
            raise ValueError("source-conditioned CST variant must be A/B/C/D")
        if self.input_dim <= 0 or self.num_items <= 0:
            raise ValueError("input_dim and num_items must be positive")
        if self.model_dim <= 0 or self.heads <= 0:
            raise ValueError("model_dim and heads must be positive")
        if self.model_dim % self.heads != 0:
            raise ValueError("model_dim must be divisible by heads")
        if self.candidate_layers <= 0 or self.source_layers <= 0:
            raise ValueError("transformer layer counts must be positive")
        if self.source_max_length <= 0:
            raise ValueError("source_max_length must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.feedforward_multiplier <= 0:
            raise ValueError("feedforward_multiplier must be positive")
        if self.relative_context not in {"none", "mean_max"}:
            raise ValueError("relative_context must be none or mean_max")

    @property
    def use_candidate_ids(self) -> bool:
        return self.variant in {"B", "C", "D"}

    @property
    def use_source_sequence(self) -> bool:
        return self.variant in {"C", "D"}

    @property
    def use_candidate_self_attention(self) -> bool:
        return self.variant in {"A", "B", "D"}


def abcd_model_config(
    variant: str,
    *,
    input_dim: int,
    num_items: int,
    model_dim: int = 64,
    heads: int = 4,
    candidate_layers: int = 2,
    source_layers: int = 1,
    source_max_length: int = 64,
    dropout: float = 0.05,
    feedforward_multiplier: int = 2,
    relative_context: str = "mean_max",
) -> SourceConditionedCSTConfig:
    return SourceConditionedCSTConfig(
        input_dim=input_dim,
        num_items=num_items,
        variant=variant,  # type: ignore[arg-type]
        model_dim=model_dim,
        heads=heads,
        candidate_layers=candidate_layers,
        source_layers=source_layers,
        source_max_length=source_max_length,
        dropout=dropout,
        feedforward_multiplier=feedforward_multiplier,
        relative_context=relative_context,
    )


class SourceConditionedCandidateSetTransformer(jt.nn.Module):
    """Candidate scorer with optional shared-ID and source-history context."""

    def __init__(self, config: SourceConditionedCSTConfig) -> None:
        super().__init__()
        self.config = config
        projection_input_dim = config.input_dim * (
            3 if config.relative_context == "mean_max" else 1
        )
        self.input_projection = jt.nn.Linear(
            projection_input_dim,
            config.model_dim,
        )
        block_config = _candidate_block_config(config)
        self.blocks = jt.nn.ModuleList(
            [
                _CandidateTransformerBlock(block_config)
                for _ in range(config.candidate_layers)
            ]
            if config.use_candidate_self_attention
            else []
        )
        self.output_norm = jt.nn.LayerNorm(config.model_dim)
        self.score_head = jt.nn.Linear(config.model_dim, 1)

        if config.use_candidate_ids:
            self.item_embedding = jt.nn.Embedding(
                config.num_items + 1,
                config.model_dim,
            )
            self.candidate_id_scale = jt.nn.Parameter(
                jt.array([0.1], dtype=jt.float32)
            )
        if config.use_source_sequence:
            self.source_time_embedding = jt.nn.Embedding(
                TIME_DELTA_BUCKETS + 1,
                config.model_dim,
            )
            self.source_position_embedding = jt.nn.Embedding(
                config.source_max_length,
                config.model_dim,
            )
            self.source_input_norm = jt.nn.LayerNorm(config.model_dim)
            self.source_blocks = jt.nn.ModuleList(
                [
                    _CandidateTransformerBlock(block_config)
                    for _ in range(config.source_layers)
                ]
            )
            self.source_output_norm = jt.nn.LayerNorm(config.model_dim)
            self.source_cross_attention = _SourceCrossAttentionBlock(config)
            self.source_scale = jt.nn.Parameter(
                jt.array([0.1], dtype=jt.float32)
            )

    @property
    def candidate_item_embedding(self) -> jt.nn.Embedding:
        if not self.config.use_candidate_ids:
            raise AttributeError("candidate item embedding is disabled")
        return self.item_embedding

    @property
    def source_item_embedding(self) -> jt.nn.Embedding:
        if not self.config.use_source_sequence:
            raise AttributeError("source item embedding is disabled")
        return self.item_embedding

    def execute(
        self,
        features: jt.Var,
        candidate_ids: jt.Var | None = None,
        source_items: jt.Var | None = None,
        source_time_buckets: jt.Var | None = None,
        source_lengths: jt.Var | None = None,
        candidate_mask: jt.Var | None = None,
    ) -> jt.Var:
        self._validate_inputs(
            features,
            candidate_ids,
            source_items,
            source_time_buckets,
            source_lengths,
            candidate_mask,
        )
        contextual_features = candidate_relative_features(
            features,
            mode=self.config.relative_context,
        )
        hidden = self.input_projection(contextual_features)
        if self.config.use_candidate_ids:
            hidden = hidden + (
                self.candidate_id_scale
                * self.item_embedding(candidate_ids)
            )
        if self.config.use_source_sequence:
            source_hidden, source_mask = self._encode_source(
                source_items,
                source_time_buckets,
                source_lengths,
            )
            source_context = self.source_cross_attention(
                hidden,
                source_hidden,
                source_mask,
            )
            hidden = hidden + self.source_scale * source_context
        for block in self.blocks:
            hidden = block(hidden, candidate_mask)
        hidden = self.output_norm(hidden)
        scores = self.score_head(hidden).reshape(features.shape[:2])
        if candidate_mask is not None:
            scores = jt.where(
                candidate_mask,
                scores,
                jt.full_like(scores, -1e9),
            )
        return scores

    def _encode_source(
        self,
        source_items: jt.Var,
        source_time_buckets: jt.Var,
        source_lengths: jt.Var,
    ) -> tuple[jt.Var, jt.Var]:
        source_length = int(source_items.shape[1])
        positions = jt.arange(source_length).reshape((1, -1))
        source_mask = positions < source_lengths.unsqueeze(1)
        position_ids = jt.arange(source_length).reshape((1, -1))
        hidden = (
            self.item_embedding(source_items)
            + self.source_time_embedding(source_time_buckets)
            + self.source_position_embedding(position_ids)
        )
        hidden = self.source_input_norm(hidden)
        for block in self.source_blocks:
            hidden = block(hidden, source_mask)
        return self.source_output_norm(hidden), source_mask

    def _validate_inputs(
        self,
        features: jt.Var,
        candidate_ids: jt.Var | None,
        source_items: jt.Var | None,
        source_time_buckets: jt.Var | None,
        source_lengths: jt.Var | None,
        candidate_mask: jt.Var | None,
    ) -> None:
        if (
            len(features.shape) != 3
            or int(features.shape[-1]) != self.config.input_dim
        ):
            raise ValueError(
                "source-conditioned features must be [B, C, input_dim]"
            )
        if candidate_mask is not None and (
            candidate_mask.shape != features.shape[:2]
        ):
            raise ValueError("candidate mask must match [B, C]")
        if self.config.use_candidate_ids and (
            candidate_ids is None
            or candidate_ids.shape != features.shape[:2]
        ):
            raise ValueError("candidate IDs must match [B, C]")
        if not self.config.use_source_sequence:
            return
        if (
            source_items is None
            or source_time_buckets is None
            or source_lengths is None
        ):
            raise ValueError("source sequence tensors are required")
        if (
            len(source_items.shape) != 2
            or source_items.shape[0] != features.shape[0]
            or source_items.shape[1] != self.config.source_max_length
            or source_time_buckets.shape != source_items.shape
            or source_lengths.shape != (features.shape[0],)
        ):
            raise ValueError("source sequence tensor shapes differ")


class _SourceCrossAttentionBlock(jt.nn.Module):
    def __init__(self, config: SourceConditionedCSTConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.model_dim // config.heads
        self.model_dim = config.model_dim
        self.scale = math.sqrt(self.head_dim)
        self.query_norm = jt.nn.LayerNorm(config.model_dim)
        self.memory_norm = jt.nn.LayerNorm(config.model_dim)
        self.query = jt.nn.Linear(config.model_dim, config.model_dim)
        self.key = jt.nn.Linear(config.model_dim, config.model_dim)
        self.value = jt.nn.Linear(config.model_dim, config.model_dim)
        self.output = jt.nn.Linear(config.model_dim, config.model_dim)
        self.attention_dropout = jt.nn.Dropout(config.dropout)
        self.output_dropout = jt.nn.Dropout(config.dropout)
        inner_dim = config.model_dim * config.feedforward_multiplier
        self.feedforward_norm = jt.nn.LayerNorm(config.model_dim)
        self.feedforward = jt.nn.Sequential(
            jt.nn.Linear(config.model_dim, inner_dim),
            jt.nn.ReLU(),
            jt.nn.Dropout(config.dropout),
            jt.nn.Linear(inner_dim, config.model_dim),
            jt.nn.Dropout(config.dropout),
        )

    def execute(
        self,
        candidate_hidden: jt.Var,
        source_hidden: jt.Var,
        source_mask: jt.Var,
    ) -> jt.Var:
        batch_size, candidate_count, _ = candidate_hidden.shape
        query = self._split_heads(
            self.query(self.query_norm(candidate_hidden))
        )
        memory = self.memory_norm(source_hidden)
        key = self._split_heads(self.key(memory)).permute(0, 1, 3, 2)
        value = self._split_heads(self.value(memory))
        attention_scores = jt.matmul(query, key) / self.scale
        key_mask = source_mask.float().unsqueeze(1).unsqueeze(1)
        attention_scores = attention_scores + (1.0 - key_mask) * -1e9
        attention = jt.nn.softmax(attention_scores, dim=-1)
        attended = jt.matmul(self.attention_dropout(attention), value)
        attended = attended.permute(0, 2, 1, 3).reshape(
            (batch_size, candidate_count, self.model_dim)
        )
        attended = self.output_dropout(self.output(attended))
        has_history = (source_mask.sum(dim=1) > 0).float().reshape(
            (batch_size, 1, 1)
        )
        hidden = attended * has_history
        return hidden + self.feedforward(self.feedforward_norm(hidden))

    def _split_heads(self, values: jt.Var) -> jt.Var:
        return values.reshape(
            (
                values.shape[0],
                values.shape[1],
                self.heads,
                self.head_dim,
            )
        ).permute(0, 2, 1, 3)


def _candidate_block_config(
    config: SourceConditionedCSTConfig,
) -> CandidateSetTransformerConfig:
    return CandidateSetTransformerConfig(
        input_dim=config.input_dim,
        model_dim=config.model_dim,
        heads=config.heads,
        layers=max(config.candidate_layers, 1),
        dropout=config.dropout,
        feedforward_multiplier=config.feedforward_multiplier,
        relative_context=config.relative_context,
    )
