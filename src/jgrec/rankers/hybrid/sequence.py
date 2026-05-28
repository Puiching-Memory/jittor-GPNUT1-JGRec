from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import jittor as jt
import numpy as np
from jittor_geometric.nn.models.sasrec import SASRec

from jgrec.core.types import Interaction, TestQuery
from jgrec.idmap import NodeIdMap
from jgrec.logging import log, track


SEQUENCE_FEATURE_NAMES = ("sasrec_score",)


@dataclass(frozen=True)
class SequenceTowerConfig:
    enabled: bool = True
    epochs: int = 3
    batch_size: int = 512
    max_samples: int = 50_000
    max_seq_len: int = 64
    hidden_size: int = 128
    layers: int = 2
    heads: int = 4
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 0.0


class SequenceTower:
    def __init__(self, id_map: NodeIdMap, config: SequenceTowerConfig) -> None:
        self.id_map = id_map
        self.config = config
        self.model: SASRec | None = None
        self.src_sequences: dict[int, tuple[int, ...]] = {}
        self.seen_items: np.ndarray | None = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return SEQUENCE_FEATURE_NAMES

    def fit(self, interactions: list[Interaction], rng: np.random.Generator, verbose: bool = True) -> None:
        self.src_sequences, self.seen_items = _final_sequences(interactions, self.id_map, self.config.max_seq_len)
        if not self.config.enabled or self.config.epochs < 1:
            return
        if self.id_map.num_dst < 2:
            return

        samples = _build_sequence_samples(interactions, self.id_map, self.config, rng)
        if samples is None:
            return

        seqs, lengths, pos_items, neg_items = samples
        self.model = SASRec(
            n_layers=self.config.layers,
            n_heads=self.config.heads,
            hidden_size=self.config.hidden_size,
            inner_size=self.config.hidden_size * 4,
            hidden_dropout_prob=self.config.dropout,
            attn_dropout_prob=self.config.dropout,
            hidden_act="gelu",
            layer_norm_eps=1e-12,
            initializer_range=0.02,
            n_items=self.id_map.num_dst,
            max_seq_length=self.config.max_seq_len,
        )
        optimizer = jt.nn.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        train_size = seqs.shape[0]

        epochs = range(1, self.config.epochs + 1)
        for epoch in track(epochs, description="sasrec", total=self.config.epochs, enabled=verbose):
            order = rng.permutation(train_size)
            losses: list[float] = []
            for start in range(0, train_size, self.config.batch_size):
                batch_idx = order[start : start + self.config.batch_size]
                seq_output = self.model.forward(
                    jt.array(seqs[batch_idx], dtype=jt.int32),
                    jt.array(lengths[batch_idx], dtype=jt.int32),
                )
                pos_emb = self.model.item_embedding(jt.array(pos_items[batch_idx], dtype=jt.int32))
                neg_emb = self.model.item_embedding(jt.array(neg_items[batch_idx], dtype=jt.int32))
                pos_scores = (seq_output * pos_emb).sum(dim=-1)
                neg_scores = (seq_output * neg_emb).sum(dim=-1)
                loss = -jt.log(jt.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
                optimizer.step(loss)
                losses.append(float(loss.item()))

            mean_loss = float(np.mean(losses)) if losses else 0.0
            log(f"[sasrec] epoch={epoch} loss={mean_loss:.5f}", enabled=verbose)

    def scores_for_queries(self, queries: list[TestQuery]) -> np.ndarray:
        if not queries:
            return np.empty((0, 0, len(SEQUENCE_FEATURE_NAMES)), dtype=np.float32)

        candidate_count = len(queries[0].candidates)
        scores = np.zeros((len(queries), candidate_count, len(SEQUENCE_FEATURE_NAMES)), dtype=np.float32)
        if self.model is None:
            return scores

        seqs = np.zeros((len(queries), self.config.max_seq_len), dtype=np.int32)
        lengths = np.ones(len(queries), dtype=np.int32)
        candidate_ids = np.zeros((len(queries), candidate_count), dtype=np.int32)
        candidate_valid = np.zeros((len(queries), candidate_count), dtype=bool)
        active = np.zeros(len(queries), dtype=bool)

        for row_idx, query in enumerate(queries):
            src_id = self.id_map.src_id(query.src)
            if src_id >= 0:
                history = self.src_sequences.get(src_id, ())
                if history:
                    length = min(len(history), self.config.max_seq_len)
                    seqs[row_idx, :length] = history[-length:]
                    lengths[row_idx] = length
                    active[row_idx] = True

            dst_ids = self.id_map.dst_ids(query.candidates)
            valid = dst_ids >= 0
            if self.seen_items is not None:
                valid = valid & self.seen_items[dst_ids.clip(min=0) + 1]
            candidate_valid[row_idx, valid] = True
            candidate_ids[row_idx, valid] = dst_ids[valid] + 1

        if not np.any(active):
            return scores

        with jt.no_grad():
            seq_output = self.model.forward(jt.array(seqs, dtype=jt.int32), jt.array(lengths, dtype=jt.int32))
            item_emb = self.model.item_embedding(jt.array(candidate_ids, dtype=jt.int32))
            batch_scores = (seq_output.unsqueeze(1) * item_emb).sum(dim=-1)
            scores[:, :, 0] = np.asarray(batch_scores.numpy(), dtype=np.float32)
        scores[~active, :, 0] = 0.0
        scores[:, :, 0][~candidate_valid] = 0.0
        return scores


def _final_sequences(
    interactions: list[Interaction],
    id_map: NodeIdMap,
    max_seq_len: int,
) -> tuple[dict[int, tuple[int, ...]], np.ndarray]:
    histories: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=max_seq_len))
    seen_items = np.zeros(id_map.num_dst + 1, dtype=bool)
    for item in interactions:
        src_id = id_map.src_id(item.src)
        dst_id = id_map.dst_id(item.dst)
        if src_id < 0 or dst_id < 0:
            continue
        item_id = dst_id + 1
        histories[src_id].append(item_id)
        seen_items[item_id] = True
    return {src_id: tuple(values) for src_id, values in histories.items()}, seen_items


def _build_sequence_samples(
    interactions: list[Interaction],
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

    for item in interactions:
        src_id = id_map.src_id(item.src)
        dst_id = id_map.dst_id(item.dst)
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
