from __future__ import annotations

import math
from collections import defaultdict, deque

import jittor as jt
import numpy as np

from jgrec.core.types import InteractionTable, TestQuery, TestQueryArray
from jgrec.idmap import NodeIdMap
from jgrec.logging import log, track

from .config import SEQUENCE_FEATURE_NAMES, SequenceTowerConfig


class SequenceTower:
    def __init__(self, id_map: NodeIdMap, config: SequenceTowerConfig) -> None:
        self.id_map = id_map
        self.config = config
        self.model: _GRUSequenceModel | None = None
        self.src_sequences: dict[int, tuple[int, ...]] = {}
        self.seen_items: np.ndarray | None = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return SEQUENCE_FEATURE_NAMES

    def fit(self, interactions: InteractionTable, rng: np.random.Generator, verbose: bool = True) -> None:
        self.src_sequences, self.seen_items = _final_sequences(interactions, self.id_map, self.config.max_seq_len)
        if not self.config.enabled or self.config.epochs < 1:
            return
        if self.id_map.num_dst < 2:
            return

        samples = _build_sequence_samples(interactions, self.id_map, self.config, rng)
        if samples is None:
            return

        seqs, lengths, pos_items, neg_items = samples
        self.model = _GRUSequenceModel(
            num_items=self.id_map.num_dst,
            hidden_size=self.config.hidden_size,
            layers=self.config.layers,
            dropout=self.config.dropout,
        )
        optimizer = jt.nn.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        train_size = seqs.shape[0]

        epochs = range(1, self.config.epochs + 1)
        for epoch in track(epochs, description="gru-seq", total=self.config.epochs, enabled=verbose):
            order = rng.permutation(train_size)
            losses: list[float] = []
            for start in range(0, train_size, self.config.batch_size):
                batch_idx = order[start : start + self.config.batch_size]
                seq_output = self.model.sequence_vectors(
                    jt.array(seqs[batch_idx], dtype=jt.int32),
                    jt.array(lengths[batch_idx], dtype=jt.int32),
                )
                pos_emb = self.model.item_vectors(jt.array(pos_items[batch_idx], dtype=jt.int32))
                neg_emb = self.model.item_vectors(jt.array(neg_items[batch_idx], dtype=jt.int32))
                pos_scores = (seq_output * pos_emb).sum(dim=-1) / self.model.score_scale
                neg_scores = (seq_output * neg_emb).sum(dim=-1) / self.model.score_scale
                loss = -jt.log(jt.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
                optimizer.step(loss)
                losses.append(float(loss.item()))

            mean_loss = float(np.mean(losses)) if losses else 0.0
            log(f"[gru-seq] epoch={epoch} loss={mean_loss:.5f}", enabled=verbose)

    def scores_for_queries(self, queries: TestQueryArray | list[TestQuery]) -> np.ndarray:
        if not queries:
            return np.empty((0, 0, len(SEQUENCE_FEATURE_NAMES)), dtype=np.float32)
        return self.scores_for_query_array(TestQueryArray.from_queries(queries))

    def scores_for_query_array(self, queries: TestQueryArray) -> np.ndarray:
        if not queries:
            return np.empty((0, 0, len(SEQUENCE_FEATURE_NAMES)), dtype=np.float32)

        candidate_count = queries.candidate_count
        scores = np.zeros((len(queries), candidate_count, len(SEQUENCE_FEATURE_NAMES)), dtype=np.float32)
        if self.model is None:
            return scores

        seqs = np.zeros((len(queries), self.config.max_seq_len), dtype=np.int32)
        lengths = np.ones(len(queries), dtype=np.int32)
        candidate_ids = np.zeros((len(queries), candidate_count), dtype=np.int32)
        candidate_valid = np.zeros((len(queries), candidate_count), dtype=bool)
        active = np.zeros(len(queries), dtype=bool)
        src_ids = self.id_map.src_ids(queries.src)
        dst_ids_by_row = self.id_map.dst_ids(queries.candidates)

        for row_idx, src_id in enumerate(src_ids):
            if src_id >= 0:
                history = self.src_sequences.get(int(src_id), ())
                if history:
                    length = min(len(history), self.config.max_seq_len)
                    seqs[row_idx, :length] = history[-length:]
                    lengths[row_idx] = length
                    active[row_idx] = True

            dst_ids = dst_ids_by_row[row_idx]
            valid = dst_ids >= 0
            if self.seen_items is not None:
                valid = valid & self.seen_items[dst_ids.clip(min=0) + 1]
            candidate_valid[row_idx, valid] = True
            candidate_ids[row_idx, valid] = dst_ids[valid] + 1

        if not np.any(active):
            return scores

        with jt.no_grad():
            score_batch_size = max(int(self.config.score_batch_size), 1)
            for start in range(0, len(queries), score_batch_size):
                end = min(start + score_batch_size, len(queries))
                seq_output = self.model.sequence_vectors(
                    jt.array(seqs[start:end], dtype=jt.int32),
                    jt.array(lengths[start:end], dtype=jt.int32),
                )
                item_emb = self.model.item_vectors(jt.array(candidate_ids[start:end], dtype=jt.int32))
                batch_scores = (seq_output.unsqueeze(1) * item_emb).sum(dim=-1) / self.model.score_scale
                scores[start:end, :, 0] = np.asarray(batch_scores.numpy(), dtype=np.float32)
        scores[~active, :, 0] = 0.0
        scores[:, :, 0][~candidate_valid] = 0.0
        return scores


class _GRUSequenceModel(jt.nn.Module):
    def __init__(self, num_items: int, hidden_size: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.hidden_size = max(int(hidden_size), 1)
        self.layers = max(int(layers), 1)
        self.dropout = max(float(dropout), 0.0)
        self.score_scale = math.sqrt(self.hidden_size)
        self.item_embedding = jt.nn.Embedding(int(num_items) + 1, self.hidden_size)
        self.cells = jt.nn.ModuleList([_GRUCell(self.hidden_size, self.hidden_size) for _ in range(self.layers)])

    def item_vectors(self, item_ids: jt.Var) -> jt.Var:
        return self.item_embedding(item_ids)

    def sequence_vectors(self, seqs: jt.Var, lengths: jt.Var) -> jt.Var:
        x = self.item_embedding(seqs)
        for layer_idx, cell in enumerate(self.cells):
            x = self._run_layer(cell, x, lengths)
            if self.dropout > 0.0 and layer_idx + 1 < self.layers and self.is_training():
                x = jt.nn.dropout(x, p=self.dropout)
        last_index = jt.maximum(lengths - 1, 0)
        positions = jt.arange(seqs.shape[1]).reshape((1, -1))
        last_mask = (positions == last_index.unsqueeze(1)).float32().unsqueeze(-1)
        return (x * last_mask).sum(dim=1)

    def _run_layer(self, cell: _GRUCell, x: jt.Var, lengths: jt.Var) -> jt.Var:
        batch_size = x.shape[0]
        hidden = jt.zeros((batch_size, self.hidden_size), dtype=jt.float32)
        outputs = []
        for step in range(x.shape[1]):
            updated = cell(x[:, step, :], hidden)
            mask = (lengths > step).float32().unsqueeze(-1)
            hidden = updated * mask + hidden * (1.0 - mask)
            outputs.append(hidden.unsqueeze(1))
        return jt.concat(outputs, dim=1)


class _GRUCell(jt.nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_gate = jt.nn.Linear(input_size, hidden_size * 3)
        self.hidden_gate = jt.nn.Linear(hidden_size, hidden_size * 3)

    def execute(self, x: jt.Var, hidden: jt.Var) -> jt.Var:
        input_gates = self.input_gate(x)
        hidden_gates = self.hidden_gate(hidden)
        input_reset, input_update, input_new = _split_gates(input_gates, self.hidden_size)
        hidden_reset, hidden_update, hidden_new = _split_gates(hidden_gates, self.hidden_size)
        reset = jt.sigmoid(input_reset + hidden_reset)
        update = jt.sigmoid(input_update + hidden_update)
        candidate = jt.tanh(input_new + reset * hidden_new)
        return (1.0 - update) * candidate + update * hidden


def _split_gates(gates: jt.Var, hidden_size: int) -> tuple[jt.Var, jt.Var, jt.Var]:
    return (
        gates[:, :hidden_size],
        gates[:, hidden_size : hidden_size * 2],
        gates[:, hidden_size * 2 :],
    )


def _final_sequences(
    interactions: InteractionTable,
    id_map: NodeIdMap,
    max_seq_len: int,
) -> tuple[dict[int, tuple[int, ...]], np.ndarray]:
    histories: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=max_seq_len))
    seen_items = np.zeros(id_map.num_dst + 1, dtype=bool)
    for src, dst in zip(interactions.src, interactions.dst, strict=True):
        src_id = id_map.src_id(int(src))
        dst_id = id_map.dst_id(int(dst))
        if src_id < 0 or dst_id < 0:
            continue
        item_id = dst_id + 1
        histories[src_id].append(item_id)
        seen_items[item_id] = True
    return {src_id: tuple(values) for src_id, values in histories.items()}, seen_items


def _build_sequence_samples(
    interactions: InteractionTable,
    id_map: NodeIdMap,
    config: SequenceTowerConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    histories: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=config.max_seq_len))
    seqs: list[np.ndarray] = []
    lengths: list[int] = []
    pos_items: list[int] = []
    neg_items: list[int] = []
    seen = 0

    for src, dst in zip(interactions.src, interactions.dst, strict=True):
        src_id = id_map.src_id(int(src))
        dst_id = id_map.dst_id(int(dst))
        if src_id < 0 or dst_id < 0:
            continue

        history = histories[src_id]
        if history:
            seen += 1
            slot = len(seqs)
            if config.max_samples > 0 and slot >= config.max_samples:
                replace = int(rng.integers(0, seen))
                if replace >= config.max_samples:
                    history.append(dst_id + 1)
                    continue
                slot = replace

            seq = np.zeros(config.max_seq_len, dtype=np.int32)
            hist_values = tuple(history)
            length = min(len(hist_values), config.max_seq_len)
            seq[:length] = hist_values[-length:]
            pos = dst_id + 1
            neg = int(rng.integers(1, id_map.num_dst + 1))
            if neg == pos:
                neg = 1 + (neg % id_map.num_dst)

            if slot == len(seqs):
                seqs.append(seq)
                lengths.append(length)
                pos_items.append(pos)
                neg_items.append(neg)
            else:
                seqs[slot] = seq
                lengths[slot] = length
                pos_items[slot] = pos
                neg_items[slot] = neg

        history.append(dst_id + 1)

    if not seqs:
        return None
    return (
        np.asarray(seqs, dtype=np.int32),
        np.asarray(lengths, dtype=np.int32),
        np.asarray(pos_items, dtype=np.int32),
        np.asarray(neg_items, dtype=np.int32),
    )
