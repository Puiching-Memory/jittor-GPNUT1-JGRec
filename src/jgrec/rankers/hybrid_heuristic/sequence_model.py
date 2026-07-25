from __future__ import annotations

import math

import jittor as jt
import numpy as np

TIME_DELTA_BUCKETS = 32
_BUCKET_BOUNDARIES = np.array(
    [0, 1, 2, 5, 10, 30, 60, 120, 300, 600, 1800, 3600, 7200, 14400,
     28800, 43200, 86400, 172800, 345600, 604800, 1209600, 2592000,
     5184000, 7776000, 15552000, 31104000, 63072000, 94608000,
     157680000, 315360000, 630720000],
    dtype=np.int64,
)


def time_delta_bucket(deltas: np.ndarray) -> np.ndarray:
    return np.searchsorted(_BUCKET_BOUNDARIES, deltas, side="right").astype(np.int32)


def compute_time_deltas(times: np.ndarray) -> np.ndarray:
    if len(times) <= 1:
        return np.zeros(len(times), dtype=np.int64)
    deltas = np.empty(len(times), dtype=np.int64)
    deltas[0] = 0
    deltas[1:] = np.diff(times)
    return np.maximum(deltas, 0)


def extract_last_hidden(all_hidden: jt.Var, lengths: np.ndarray) -> jt.Var:
    last_index = jt.array(np.maximum(lengths - 1, 0), dtype=jt.int32)
    positions = jt.arange(all_hidden.shape[1]).reshape((1, -1))
    last_mask = (positions == last_index.unsqueeze(1)).float32().unsqueeze(-1)
    return (all_hidden * last_mask).sum(dim=1)


def decay_weighted_scores(
    all_hidden: np.ndarray,
    item_emb: np.ndarray,
    lengths: np.ndarray,
    decay_weights: np.ndarray,
    score_scale: float,
) -> np.ndarray:
    batch, _, _ = all_hidden.shape
    _, cand_count, _ = item_emb.shape
    result = np.zeros((batch, cand_count), dtype=np.float32)
    for i in range(batch):
        length = int(lengths[i])
        if length <= 0:
            continue
        h = all_hidden[i, :length, :]
        w = decay_weights[i, :length]
        w_norm = w / (w.sum() + 1e-8)
        weighted_h = (h * w_norm[:, None]).sum(axis=0)
        result[i] = item_emb[i] @ weighted_h / score_scale
    return result


class GRUSequenceModel(jt.nn.Module):
    def __init__(self, num_items: int, hidden_size: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.hidden_size = max(int(hidden_size), 1)
        self.layers = max(int(layers), 1)
        self.dropout = max(float(dropout), 0.0)
        self.score_scale = math.sqrt(self.hidden_size)
        self.item_embedding = jt.nn.Embedding(int(num_items) + 1, self.hidden_size)
        self.time_delta_embedding = jt.nn.Embedding(TIME_DELTA_BUCKETS + 1, self.hidden_size)
        input_size = self.hidden_size * 2
        self.cells = jt.nn.ModuleList([
            _GRUCell(input_size if i == 0 else self.hidden_size, self.hidden_size)
            for i in range(self.layers)
        ])

    def item_vectors(self, item_ids: jt.Var) -> jt.Var:
        return self.item_embedding(item_ids)

    def sequence_vectors(self, seqs: jt.Var, time_deltas: jt.Var, lengths: jt.Var) -> jt.Var:
        all_hidden = self.all_hidden_states(seqs, time_deltas, lengths)
        return extract_last_hidden(all_hidden, np.asarray(lengths.numpy(), dtype=np.int32))

    def all_hidden_states(self, seqs: jt.Var, time_deltas: jt.Var, lengths: jt.Var) -> jt.Var:
        item_emb = self.item_embedding(seqs)
        td_emb = self.time_delta_embedding(time_deltas)
        x = jt.concat([item_emb, td_emb], dim=-1)
        for layer_idx, cell in enumerate(self.cells):
            x = _run_gru_layer(cell, x, lengths, self.hidden_size)
            if self.dropout > 0.0 and layer_idx + 1 < self.layers and self.is_training():
                x = jt.nn.dropout(x, p=self.dropout)
        return x


class _GRUCell(jt.nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_gate = jt.nn.Linear(input_size, hidden_size * 3)
        self.hidden_gate = jt.nn.Linear(hidden_size, hidden_size * 3)

    def execute(self, x: jt.Var, hidden: jt.Var) -> jt.Var:
        input_gates = self.input_gate(x)
        hidden_gates = self.hidden_gate(hidden)
        ir, iu, in_ = _split3(input_gates, self.hidden_size)
        hr, hu, hn = _split3(hidden_gates, self.hidden_size)
        reset = jt.sigmoid(ir + hr)
        update = jt.sigmoid(iu + hu)
        candidate = jt.tanh(in_ + reset * hn)
        return (1.0 - update) * candidate + update * hidden


def _split3(gates: jt.Var, h: int) -> tuple[jt.Var, jt.Var, jt.Var]:
    return gates[:, :h], gates[:, h : h * 2], gates[:, h * 2 :]


def _run_gru_layer(cell: _GRUCell, x: jt.Var, lengths: jt.Var, hidden_size: int) -> jt.Var:
    batch_size = x.shape[0]
    hidden = jt.zeros((batch_size, hidden_size), dtype=jt.float32)
    outputs = []
    for step in range(x.shape[1]):
        updated = cell(x[:, step, :], hidden)
        mask = (lengths > step).float32().unsqueeze(-1)
        hidden = updated * mask + hidden * (1.0 - mask)
        outputs.append(hidden.unsqueeze(1))
    return jt.concat(outputs, dim=1)
