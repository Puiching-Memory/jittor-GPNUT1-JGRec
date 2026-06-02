from __future__ import annotations

import math
from dataclasses import dataclass

import jittor as jt
from jittor import nn
from jittor_geometric.nn.models.craft import CrossAttention


@dataclass(frozen=True)
class TemporalGraphModelConfig:
    num_nodes: int
    history_len: int
    candidate_history_len: int
    hidden_size: int = 128
    layers: int = 3
    heads: int = 4
    dropout: float = 0.15
    time_span: int = 1


class EndToEndTemporalGraphModel(jt.nn.Module):
    """Candidate-set temporal graph ranker trained by one listwise objective."""

    def __init__(self, config: TemporalGraphModelConfig) -> None:
        super().__init__()
        if config.hidden_size % config.heads != 0:
            raise ValueError("hidden_size must be divisible by heads")
        self.config = config
        self.hidden_size = config.hidden_size
        self.time_norm = max(math.log1p(max(config.time_span, 1)), 1.0)

        self.node_embedding = nn.Embedding(config.num_nodes, config.hidden_size)
        self.role_embedding = nn.Embedding(6, config.hidden_size)
        self.time_projection = nn.Linear(1, config.hidden_size)
        self.memory_gate = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.memory_candidate = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.query_projection = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.stats_projection = nn.Linear(3, config.hidden_size)
        self.cross_attention = CrossAttention(
            n_layers=config.layers,
            n_heads=config.heads,
            hidden_size=config.hidden_size,
            inner_size=config.hidden_size * 4,
            hidden_dropout_prob=config.dropout,
            attn_dropout_prob=config.dropout,
            hidden_act="gelu",
            layer_norm_eps=1e-12,
        )
        self.scorer = nn.Sequential(
            nn.Linear(config.hidden_size * 5, config.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1),
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

    def execute(
        self,
        src_ids: jt.Var,
        candidate_ids: jt.Var,
        cur_times: jt.Var,
        src_neighbor_ids: jt.Var,
        src_neighbor_times: jt.Var,
        candidate_neighbor_ids: jt.Var,
        candidate_neighbor_times: jt.Var,
    ) -> jt.Var:
        batch_size, candidate_count = candidate_ids.shape
        src_tokens, src_mask = self._history_tokens(
            node_ids=src_neighbor_ids,
            neighbor_times=src_neighbor_times,
            cur_times=cur_times,
            role_id=1,
        )
        src_base = self.node_embedding(src_ids)
        src_hist = _masked_mean(src_tokens, src_mask, dim=1)
        src_state = self._memory_update(src_base, src_hist)

        flat_candidate_ids = candidate_ids.reshape((-1,))
        flat_times = cur_times.unsqueeze(1).expand(batch_size, candidate_count).reshape((-1,))
        flat_candidate_neighbors = candidate_neighbor_ids.reshape((batch_size * candidate_count, -1))
        flat_candidate_neighbor_times = candidate_neighbor_times.reshape((batch_size * candidate_count, -1))
        candidate_tokens, candidate_mask = self._history_tokens(
            node_ids=flat_candidate_neighbors,
            neighbor_times=flat_candidate_neighbor_times,
            cur_times=flat_times,
            role_id=2,
        )
        candidate_base = self.node_embedding(flat_candidate_ids)
        candidate_hist = _masked_mean(candidate_tokens, candidate_mask, dim=1)
        candidate_state = self._memory_update(candidate_base, candidate_hist).reshape(
            (batch_size, candidate_count, self.hidden_size)
        )
        candidate_tokens = candidate_tokens.reshape(
            (batch_size, candidate_count, self.config.candidate_history_len, self.hidden_size)
        )
        candidate_mask = candidate_mask.reshape((batch_size, candidate_count, self.config.candidate_history_len))

        src_state_expanded = src_state.unsqueeze(1).expand(batch_size, candidate_count, self.hidden_size)
        query = self.query_projection(
            jt.concat(
                [
                    src_state_expanded,
                    candidate_state,
                    src_state_expanded * candidate_state,
                ],
                dim=-1,
            ).reshape((batch_size * candidate_count, -1))
        ).reshape((batch_size * candidate_count, 1, self.hidden_size))

        key, key_mask = self._pair_keys(
            src_state=src_state,
            candidate_state=candidate_state,
            src_tokens=src_tokens,
            src_mask=src_mask,
            candidate_tokens=candidate_tokens,
            candidate_mask=candidate_mask,
            candidate_count=candidate_count,
        )
        attention_mask = (1.0 - key_mask.float().unsqueeze(1).unsqueeze(1)) * -10000.0
        attended = self.cross_attention(
            query,
            attention_mask,
            key,
            output_all_encoded_layers=False,
        )[-1].reshape((batch_size, candidate_count, self.hidden_size))

        stats = self._pair_stats(
            src_neighbor_ids=src_neighbor_ids,
            candidate_ids=candidate_ids,
            src_mask=src_mask,
            candidate_mask=candidate_mask,
        )
        stats_state = self.stats_projection(stats.reshape((batch_size * candidate_count, -1))).reshape(
            (batch_size, candidate_count, self.hidden_size)
        )
        scorer_input = jt.concat(
            [
                attended,
                src_state_expanded,
                candidate_state,
                src_state_expanded * candidate_state,
                stats_state,
            ],
            dim=-1,
        )
        logits = self.scorer(scorer_input.reshape((batch_size * candidate_count, -1))).reshape(
            (batch_size, candidate_count)
        )
        return logits

    def _history_tokens(
        self,
        node_ids: jt.Var,
        neighbor_times: jt.Var,
        cur_times: jt.Var,
        role_id: int,
    ) -> tuple[jt.Var, jt.Var]:
        mask = node_ids != 0
        node_emb = self.node_embedding(node_ids)
        delta = cur_times.unsqueeze(-1) - neighbor_times
        delta = jt.maximum(delta.float(), jt.zeros_like(delta).float())
        time_emb = self.time_projection((jt.log(delta + 1.0) / self.time_norm).reshape((-1, 1))).reshape(node_emb.shape)
        role = self.role_embedding(jt.array([role_id], dtype=jt.int32)).reshape((1, 1, self.hidden_size))
        tokens = self.layer_norm(node_emb + time_emb + role)
        tokens = self.dropout(tokens)
        return tokens, mask

    def _memory_update(self, base: jt.Var, history: jt.Var) -> jt.Var:
        x = jt.concat([base, history], dim=-1)
        gate = jt.sigmoid(self.memory_gate(x))
        candidate = jt.tanh(self.memory_candidate(x))
        return self.layer_norm(base + gate * candidate)

    def _pair_keys(
        self,
        src_state: jt.Var,
        candidate_state: jt.Var,
        src_tokens: jt.Var,
        src_mask: jt.Var,
        candidate_tokens: jt.Var,
        candidate_mask: jt.Var,
        candidate_count: int,
    ) -> tuple[jt.Var, jt.Var]:
        batch_size = src_state.shape[0]
        src_self = src_state.unsqueeze(1).expand(batch_size, candidate_count, self.hidden_size).reshape(
            (batch_size * candidate_count, 1, self.hidden_size)
        )
        dst_self = candidate_state.reshape((batch_size * candidate_count, 1, self.hidden_size))
        src_hist = src_tokens.unsqueeze(1).expand(
            batch_size,
            candidate_count,
            self.config.history_len,
            self.hidden_size,
        ).reshape((batch_size * candidate_count, self.config.history_len, self.hidden_size))
        dst_hist = candidate_tokens.reshape(
            (batch_size * candidate_count, self.config.candidate_history_len, self.hidden_size)
        )
        key = jt.concat([src_self, dst_self, src_hist, dst_hist], dim=1)

        self_mask = jt.ones((batch_size * candidate_count, 2), dtype=jt.float32)
        src_hist_mask = src_mask.unsqueeze(1).expand(batch_size, candidate_count, self.config.history_len).reshape(
            (batch_size * candidate_count, self.config.history_len)
        )
        dst_hist_mask = candidate_mask.reshape((batch_size * candidate_count, self.config.candidate_history_len))
        key_mask = jt.concat([self_mask, src_hist_mask.float(), dst_hist_mask.float()], dim=1) > 0
        return key, key_mask

    def _pair_stats(
        self,
        src_neighbor_ids: jt.Var,
        candidate_ids: jt.Var,
        src_mask: jt.Var,
        candidate_mask: jt.Var,
    ) -> jt.Var:
        batch_size, candidate_count = candidate_ids.shape
        repeat_hits = (
            (src_neighbor_ids.unsqueeze(1) == candidate_ids.unsqueeze(-1))
            * src_mask.unsqueeze(1)
            * (candidate_ids.unsqueeze(-1) != 0)
        ).float()
        repeat_count = repeat_hits.sum(dim=-1, keepdims=True) / max(self.config.history_len, 1)
        src_len = src_mask.float().sum(dim=1, keepdims=True) / max(self.config.history_len, 1)
        src_len = src_len.unsqueeze(1).expand(batch_size, candidate_count, 1)
        candidate_len = candidate_mask.float().sum(dim=2, keepdims=True) / max(self.config.candidate_history_len, 1)
        return jt.concat([repeat_count, src_len, candidate_len], dim=-1)


def _masked_mean(values: jt.Var, mask: jt.Var, dim: int) -> jt.Var:
    mask_f = mask.float().unsqueeze(-1)
    numerator = (values * mask_f).sum(dim=dim)
    denominator = mask_f.sum(dim=dim) + 1e-6
    return numerator / denominator
