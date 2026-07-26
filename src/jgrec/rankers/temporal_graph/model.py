from __future__ import annotations

import math
from dataclasses import dataclass, field

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
    candidate_feature_dim: int = 6


@dataclass
class DiagnosisTrace:
    """Intermediate states captured during a diagnostic forward pass."""

    # memory gate diagnostics
    src_gate_values: jt.Var = None          # [batch, hidden] sigmoid gate for src
    candidate_gate_values: jt.Var = None    # [batch*cand, hidden] sigmoid gate for candidates

    # attention diagnostics (manually recomputed)
    attention_weights: jt.Var = None        # [batch*cand, heads, 1, key_len] softmax weights
    attention_key_mask: jt.Var = None       # [batch*cand, key_len] bool mask

    # scorer input signal strengths (L2 norms per signal block)
    signal_norms: dict = field(default_factory=dict)
    # keys: "attended", "src_state", "candidate_state", "interaction", "stats_state"
    # values: [batch, candidate_count] L2 norm per sample

    # time encoding diagnostics
    time_deltas_src: jt.Var = None          # [batch, history_len] raw time deltas
    time_encodings_src: jt.Var = None       # [batch, history_len, hidden] time embeddings
    time_deltas_candidate: jt.Var = None    # [batch*cand, candidate_history_len]
    time_encodings_candidate: jt.Var = None # [batch*cand, candidate_history_len, hidden]

    # raw stats before projection
    pair_stats_raw: jt.Var = None           # [batch, candidate_count, 7+cand_feat_dim]

    # logits
    logits: jt.Var = None                   # [batch, candidate_count]


# Time projection is kept as Linear(1, hidden) because experiments show
# that more complex encodings (sinusoidal, bucket, MLP) don't improve AP/MRR
# on this task, suggesting fine-grained time info is not the bottleneck.


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
        # Initialize gate bias to -2 so initial gate ≈ sigmoid(-2) ≈ 0.12
        # This makes the model initially trust base embedding over history
        jt.init.constant_(self.memory_gate.bias, -2.0)
        self.query_projection = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.stats_projection = nn.Linear(7 + config.candidate_feature_dim, config.hidden_size)
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
        self.scorer_input_norm = nn.ModuleList([nn.LayerNorm(config.hidden_size, eps=1e-12) for _ in range(5)])
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
        candidate_features: jt.Var,
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
            cur_times=cur_times,
            src_neighbor_ids=src_neighbor_ids,
            src_neighbor_times=src_neighbor_times,
            candidate_ids=candidate_ids,
            candidate_neighbor_times=candidate_neighbor_times,
            src_mask=src_mask,
            candidate_mask=candidate_mask,
        )
        if self.config.candidate_feature_dim > 0:
            stats = jt.concat([stats, candidate_features.float()], dim=-1)
        stats_state = self.stats_projection(stats.reshape((batch_size * candidate_count, -1))).reshape(
            (batch_size, candidate_count, self.hidden_size)
        )
        interaction = src_state_expanded * candidate_state

        # Normalize each signal block before concatenation to equalize magnitudes
        signals = [attended, src_state_expanded, candidate_state, interaction, stats_state]
        normed_signals = [norm(sig) for norm, sig in zip(self.scorer_input_norm, signals, strict=True)]

        scorer_input = jt.concat(normed_signals, dim=-1)
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
        # Store gate values for regularization (accumulate across src + candidate)
        if not hasattr(self, '_gate_buffer'):
            self._gate_buffer = []
        self._gate_buffer.append(gate)
        return self.layer_norm(base + gate * candidate)

    def gate_regularization_loss(self, lam: float = 0.1) -> jt.Var:
        """Penalize gate values near 0.5 to encourage 0/1 decisions.

        loss = lambda * mean(gate * (1 - gate))
        gate*(1-gate) is maximized at 0.5 and minimized at 0 or 1.
        """
        if not hasattr(self, '_gate_buffer') or not self._gate_buffer:
            return jt.array(0.0)
        total = jt.array(0.0)
        for gate in self._gate_buffer:
            total = total + (gate * (1.0 - gate)).mean()
        avg = total / len(self._gate_buffer)
        return lam * avg

    def clear_gate_buffer(self) -> None:
        """Clear accumulated gate values. Call before each forward pass."""
        self._gate_buffer = []

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
        cur_times: jt.Var,
        src_neighbor_ids: jt.Var,
        src_neighbor_times: jt.Var,
        candidate_ids: jt.Var,
        candidate_neighbor_times: jt.Var,
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
        position_weights = (
            (jt.arange(self.config.history_len).float() + 1.0).reshape((1, 1, self.config.history_len))
            / max(self.config.history_len, 1)
        )
        repeat_recent_position = (repeat_hits * position_weights).max(dim=-1, keepdims=True)
        src_len = src_mask.float().sum(dim=1, keepdims=True) / max(self.config.history_len, 1)
        src_len = src_len.unsqueeze(1).expand(batch_size, candidate_count, 1)
        candidate_len = candidate_mask.float().sum(dim=2, keepdims=True) / max(self.config.candidate_history_len, 1)

        src_last_time = (src_neighbor_times * src_mask).max(dim=1, keepdims=True)
        src_delta = jt.maximum(cur_times.reshape((-1, 1)) - src_last_time, jt.zeros_like(src_last_time))
        src_recency = _log_recency(src_delta.float(), self.time_norm)
        src_recency = src_recency.unsqueeze(1).expand(batch_size, candidate_count, 1) * (src_len > 0).float()

        candidate_last_time = (candidate_neighbor_times * candidate_mask).max(dim=2, keepdims=True)
        candidate_delta = jt.maximum(
            cur_times.reshape((batch_size, 1, 1)) - candidate_last_time,
            jt.zeros_like(candidate_last_time),
        )
        candidate_recency = _log_recency(candidate_delta.float(), self.time_norm) * (candidate_len > 0).float()

        pair_last_time = (src_neighbor_times.unsqueeze(1) * repeat_hits).max(dim=2, keepdims=True)
        pair_delta = jt.maximum(
            cur_times.reshape((batch_size, 1, 1)) - pair_last_time,
            jt.zeros_like(pair_last_time),
        )
        pair_recency = _log_recency(pair_delta.float(), self.time_norm) * (repeat_count > 0).float()

        return jt.concat(
            [
                repeat_count,
                repeat_recent_position,
                src_len,
                candidate_len,
                src_recency,
                candidate_recency,
                pair_recency,
            ],
            dim=-1,
        )

    def diagnose_forward(
        self,
        src_ids: jt.Var,
        candidate_ids: jt.Var,
        cur_times: jt.Var,
        src_neighbor_ids: jt.Var,
        src_neighbor_times: jt.Var,
        candidate_neighbor_ids: jt.Var,
        candidate_neighbor_times: jt.Var,
        candidate_features: jt.Var,
    ) -> DiagnosisTrace:
        """Run a forward pass while capturing intermediate diagnostic states.

        Returns a DiagnosisTrace with gate values, attention weights,
        signal strengths, time encodings, and raw stats.
        """
        trace = DiagnosisTrace()
        batch_size, candidate_count = candidate_ids.shape

        # --- src history ---
        src_tokens, src_mask, src_time_deltas, src_time_embs = self._history_tokens_diagnose(
            node_ids=src_neighbor_ids,
            neighbor_times=src_neighbor_times,
            cur_times=cur_times,
            role_id=1,
        )
        trace.time_deltas_src = src_time_deltas
        trace.time_encodings_src = src_time_embs

        src_base = self.node_embedding(src_ids)
        src_hist = _masked_mean(src_tokens, src_mask, dim=1)
        src_state, src_gate = self._memory_update_diagnose(src_base, src_hist)
        trace.src_gate_values = src_gate

        # --- candidate history ---
        flat_candidate_ids = candidate_ids.reshape((-1,))
        flat_times = cur_times.unsqueeze(1).expand(batch_size, candidate_count).reshape((-1,))
        flat_candidate_neighbors = candidate_neighbor_ids.reshape((batch_size * candidate_count, -1))
        flat_candidate_neighbor_times = candidate_neighbor_times.reshape((batch_size * candidate_count, -1))
        candidate_tokens, candidate_mask, cand_time_deltas, cand_time_embs = self._history_tokens_diagnose(
            node_ids=flat_candidate_neighbors,
            neighbor_times=flat_candidate_neighbor_times,
            cur_times=flat_times,
            role_id=2,
        )
        trace.time_deltas_candidate = cand_time_deltas
        trace.time_encodings_candidate = cand_time_embs

        candidate_base = self.node_embedding(flat_candidate_ids)
        candidate_hist = _masked_mean(candidate_tokens, candidate_mask, dim=1)
        candidate_state_flat, cand_gate = self._memory_update_diagnose(candidate_base, candidate_hist)
        trace.candidate_gate_values = cand_gate
        candidate_state = candidate_state_flat.reshape(
            (batch_size, candidate_count, self.hidden_size)
        )
        candidate_tokens = candidate_tokens.reshape(
            (batch_size, candidate_count, self.config.candidate_history_len, self.hidden_size)
        )
        candidate_mask = candidate_mask.reshape((batch_size, candidate_count, self.config.candidate_history_len))

        # --- query / key / attention ---
        src_state_expanded = src_state.unsqueeze(1).expand(batch_size, candidate_count, self.hidden_size)
        query = self.query_projection(
            jt.concat(
                [src_state_expanded, candidate_state, src_state_expanded * candidate_state],
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
        trace.attention_key_mask = key_mask

        attention_mask = (1.0 - key_mask.float().unsqueeze(1).unsqueeze(1)) * -10000.0
        trace.attention_weights = self._extract_attention_weights(query, key, attention_mask)

        attended = self.cross_attention(
            query,
            attention_mask,
            key,
            output_all_encoded_layers=False,
        )[-1].reshape((batch_size, candidate_count, self.hidden_size))

        # --- stats ---
        stats = self._pair_stats(
            cur_times=cur_times,
            src_neighbor_ids=src_neighbor_ids,
            src_neighbor_times=src_neighbor_times,
            candidate_ids=candidate_ids,
            candidate_neighbor_times=candidate_neighbor_times,
            src_mask=src_mask,
            candidate_mask=candidate_mask,
        )
        if self.config.candidate_feature_dim > 0:
            stats = jt.concat([stats, candidate_features.float()], dim=-1)
        trace.pair_stats_raw = stats

        stats_state = self.stats_projection(stats.reshape((batch_size * candidate_count, -1))).reshape(
            (batch_size, candidate_count, self.hidden_size)
        )

        # --- scorer input signal norms ---
        interaction = src_state_expanded * candidate_state
        trace.signal_norms = {
            "attended": jt.norm(attended, p=2, dim=-1),
            "src_state": jt.norm(src_state_expanded, p=2, dim=-1),
            "candidate_state": jt.norm(candidate_state, p=2, dim=-1),
            "interaction": jt.norm(interaction, p=2, dim=-1),
            "stats_state": jt.norm(stats_state, p=2, dim=-1),
        }

        # Normalize each signal block before concatenation to equalize magnitudes
        signals = [attended, src_state_expanded, candidate_state, interaction, stats_state]
        normed_signals = [norm(sig) for norm, sig in zip(self.scorer_input_norm, signals, strict=True)]

        scorer_input = jt.concat(normed_signals, dim=-1)
        logits = self.scorer(scorer_input.reshape((batch_size * candidate_count, -1))).reshape(
            (batch_size, candidate_count)
        )
        trace.logits = logits
        return trace

    def _history_tokens_diagnose(
        self,
        node_ids: jt.Var,
        neighbor_times: jt.Var,
        cur_times: jt.Var,
        role_id: int,
    ) -> tuple[jt.Var, jt.Var, jt.Var, jt.Var]:
        """Like _history_tokens but also returns raw time deltas and time embeddings."""
        mask = node_ids != 0
        node_emb = self.node_embedding(node_ids)
        delta = cur_times.unsqueeze(-1) - neighbor_times
        delta = jt.maximum(delta.float(), jt.zeros_like(delta).float())
        time_input = (jt.log(delta + 1.0) / self.time_norm).reshape((-1, 1))
        time_emb = self.time_projection(time_input).reshape(node_emb.shape)
        role = self.role_embedding(jt.array([role_id], dtype=jt.int32)).reshape((1, 1, self.hidden_size))
        tokens = self.layer_norm(node_emb + time_emb + role)
        tokens = self.dropout(tokens)
        return tokens, mask, delta, time_emb

    def _memory_update_diagnose(self, base: jt.Var, history: jt.Var) -> tuple[jt.Var, jt.Var]:
        """Like _memory_update but also returns the gate values."""
        x = jt.concat([base, history], dim=-1)
        gate = jt.sigmoid(self.memory_gate(x))
        candidate = jt.tanh(self.memory_candidate(x))
        return self.layer_norm(base + gate * candidate), gate

    def _extract_attention_weights(
        self,
        query: jt.Var,
        key: jt.Var,
        attention_mask: jt.Var,
    ) -> jt.Var:
        """Manually recompute attention weights from the first CrossAttention layer.

        Returns: [batch*cand, heads, 1, key_len]
        """
        first_layer = self.cross_attention.layer[0]
        attn_module = first_layer.multi_head_attention
        n_heads = attn_module.num_attention_heads
        head_size = attn_module.attention_head_size
        sqrt_head = attn_module.sqrt_attention_head_size

        q = attn_module.query(query)
        k = attn_module.key(key)

        q_shape = (*q.shape[:-1], n_heads, head_size)
        q = q.view(*q_shape)
        k_shape = (*k.shape[:-1], n_heads, head_size)
        k = k.view(*k_shape)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)

        scores = jt.matmul(q, k) / sqrt_head
        scores = scores + attention_mask
        probs = nn.Softmax(dim=-1)(scores)
        return probs


def _masked_mean(values: jt.Var, mask: jt.Var, dim: int) -> jt.Var:
    mask_f = mask.float().unsqueeze(-1)
    numerator = (values * mask_f).sum(dim=dim)
    denominator = mask_f.sum(dim=dim) + 1e-6
    return numerator / denominator


def _log_recency(delta: jt.Var, time_norm: float) -> jt.Var:
    return jt.exp(-jt.log(delta + 1.0) / time_norm)
