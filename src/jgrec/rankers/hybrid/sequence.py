from __future__ import annotations

from collections import defaultdict, deque

import jittor as jt
import numpy as np

from jgrec.contest_checkpoint import get_model_state, set_model_state
from jgrec.core.types import InteractionTable, TestQuery, TestQueryArray
from jgrec.idmap import NodeIdMap
from jgrec.logging import log, track
from jgrec.rankers.common.early_stop import LossEarlyStopper
from jgrec.rankers.common.optimization import (
    set_tower_optimizer_learning_rate,
)

from .config import SEQUENCE_FEATURE_NAMES, SequenceTowerConfig
from .sequence_model import (
    GRUSequenceModel,
    compute_time_deltas,
    decay_weighted_scores,
    extract_last_hidden,
    time_delta_bucket,
)


class SequenceTower:
    def __init__(self, id_map: NodeIdMap, config: SequenceTowerConfig) -> None:
        self.id_map = id_map
        self.config = config
        self.model: GRUSequenceModel | None = None
        self.src_sequences: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.seen_items: np.ndarray | None = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return SEQUENCE_FEATURE_NAMES

    def snapshot(self) -> dict:
        return {
            "src_sequences": {
                src_id: (items.copy(), times.copy())
                for src_id, (items, times) in self.src_sequences.items()
            },
            "seen_items": None if self.seen_items is None else self.seen_items.copy(),
            "model_state": get_model_state(self.model) if self.model is not None else None,
        }

    def hydrate(self, snapshot: dict) -> None:
        self.src_sequences = {
            int(src_id): (
                np.asarray(items, dtype=np.int32).copy(),
                np.asarray(times, dtype=np.int64).copy(),
            )
            for src_id, (items, times) in snapshot["src_sequences"].items()
        }
        seen_items = snapshot.get("seen_items")
        self.seen_items = None if seen_items is None else np.asarray(seen_items, dtype=bool).copy()
        model_state = snapshot.get("model_state")
        if model_state is None:
            self.model = None
            return
        self.model = GRUSequenceModel(
            num_items=self.id_map.num_dst,
            hidden_size=self.config.hidden_size,
            layers=self.config.layers,
            dropout=self.config.dropout,
        )
        set_model_state(self.model, model_state)

    def fit(self, interactions: InteractionTable, rng: np.random.Generator, verbose: bool = True) -> None:
        self.src_sequences, self.seen_items = _final_sequences(interactions, self.id_map, self.config.max_seq_len)
        if not self.config.enabled or self.config.epochs < 1:
            return
        if self.id_map.num_dst < 2:
            return

        samples = _build_samples(interactions, self.id_map, self.config, rng)
        if samples is None:
            return

        seqs, time_deltas, lengths, pos_items, neg_items = samples
        self.model = GRUSequenceModel(
            num_items=self.id_map.num_dst,
            hidden_size=self.config.hidden_size,
            layers=self.config.layers,
            dropout=self.config.dropout,
        )
        optimizer = jt.nn.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        full_size = seqs.shape[0]
        val_size = max(0, int(full_size * self.config.early_stop_val_ratio))
        val_perm = rng.permutation(full_size)
        val_idx, train_idx = val_perm[:val_size], val_perm[val_size:]

        stopper = LossEarlyStopper(patience=self.config.early_stop_patience)
        for epoch in track(range(1, self.config.epochs + 1), description="gru-seq", total=self.config.epochs, enabled=verbose):
            learning_rate = set_tower_optimizer_learning_rate(
                optimizer,
                initial_lr=self.config.lr,
                epoch=epoch,
                total_epochs=self.config.epochs,
                schedule=getattr(self.config, "lr_schedule", "constant"),
                min_lr_ratio=getattr(self.config, "min_lr_ratio", 0.0),
            )
            order = rng.permutation(train_idx.shape[0])
            losses: list[float] = []
            for start in range(0, train_idx.shape[0], self.config.batch_size):
                bi = train_idx[order[start : start + self.config.batch_size]]
                loss = _bpr_loss(self.model, seqs, time_deltas, lengths, pos_items, neg_items, bi)
                optimizer.step(loss)
                losses.append(float(loss.item()))

            mean_loss = float(np.mean(losses)) if losses else 0.0
            val_loss = _eval_loss(self.model, seqs, time_deltas, lengths, pos_items, neg_items, val_idx, self.config.batch_size) if val_size > 0 else 0.0
            log(
                f"[gru-seq] epoch={epoch} lr={learning_rate:.8g} "
                f"loss={mean_loss:.5f} val_loss={val_loss:.5f}",
                enabled=verbose,
            )
            if stopper.update(epoch, val_loss if val_size > 0 else mean_loss, self.model):
                log(f"[gru-seq] early_stop epoch={epoch} best={stopper.best_epoch} val={stopper.best_loss:.5f}", enabled=verbose)
                break
        stopper.restore_best(self.model)

    def scores_for_queries(self, queries: TestQueryArray | list[TestQuery]) -> np.ndarray:
        if not queries:
            return np.empty((0, 0, len(SEQUENCE_FEATURE_NAMES)), dtype=np.float32)
        return self.scores_for_query_array(TestQueryArray.from_queries(queries))

    def scores_for_query_array(self, queries: TestQueryArray) -> np.ndarray:
        n_feat = len(SEQUENCE_FEATURE_NAMES)
        if not queries:
            return np.empty((0, 0, n_feat), dtype=np.float32)

        n_q, n_c = len(queries), queries.candidate_count
        scores = np.zeros((n_q, n_c, n_feat), dtype=np.float32)
        if self.model is None:
            return scores

        seqs, td, lengths, decay_w, cand_ids, cand_valid, active = self._prepare_predict(queries)
        if not np.any(active):
            return scores

        with jt.no_grad():
            bs = max(int(self.config.score_batch_size), 1)
            for start in range(0, n_q, bs):
                end = min(start + bs, n_q)
                sl = slice(start, end)
                _score_batch(self.model, seqs[sl], td[sl], lengths[sl], decay_w[sl], cand_ids[sl], scores[sl])

        for fi in range(n_feat):
            scores[~active, :, fi] = 0.0
            scores[:, :, fi][~cand_valid] = 0.0
        return scores

    def _prepare_predict(self, queries: TestQueryArray):
        n_q, n_c = len(queries), queries.candidate_count
        seqs = np.zeros((n_q, self.config.max_seq_len), dtype=np.int32)
        td = np.zeros((n_q, self.config.max_seq_len), dtype=np.int32)
        lengths = np.ones(n_q, dtype=np.int32)
        decay_w = np.zeros((n_q, self.config.max_seq_len), dtype=np.float32)
        cand_ids = np.zeros((n_q, n_c), dtype=np.int32)
        cand_valid = np.zeros((n_q, n_c), dtype=bool)
        active = np.zeros(n_q, dtype=bool)
        src_ids = self.id_map.src_ids(queries.src)
        dst_ids_by_row = self.id_map.dst_ids(queries.candidates)

        for row, src_id in enumerate(src_ids):
            if src_id >= 0:
                entry = self.src_sequences.get(int(src_id))
                if entry is not None:
                    items, times = entry
                    cutoff = int(np.searchsorted(times, int(queries.time[row]), side="left"))
                    if cutoff > 0:
                        length = min(cutoff, self.config.max_seq_len)
                        seqs[row, :length] = items[cutoff - length : cutoff]
                        lengths[row] = length
                        active[row] = True
                        vis_t = times[cutoff - length : cutoff]
                        td[row, :length] = time_delta_bucket(compute_time_deltas(vis_t))
                        diffs = int(queries.time[row]) - vis_t.astype(np.int64)
                        half_life = max(np.median(diffs), 1)
                        decay_w[row, :length] = np.exp(-0.693 * diffs / half_life)

            dst_ids = dst_ids_by_row[row]
            valid = dst_ids >= 0
            if self.seen_items is not None:
                valid = valid & self.seen_items[dst_ids.clip(min=0) + 1]
            cand_valid[row, valid] = True
            cand_ids[row, valid] = dst_ids[valid] + 1

        return seqs, td, lengths, decay_w, cand_ids, cand_valid, active


def _bpr_loss(model, seqs, td, lengths, pos, neg, idx) -> jt.Var:
    seq_out = model.sequence_vectors(
        jt.array(seqs[idx], dtype=jt.int32),
        jt.array(td[idx], dtype=jt.int32),
        jt.array(lengths[idx], dtype=jt.int32),
    )
    pos_emb = model.item_vectors(jt.array(pos[idx], dtype=jt.int32))
    neg_emb = model.item_vectors(jt.array(neg[idx], dtype=jt.int32))
    pos_s = (seq_out * pos_emb).sum(dim=-1) / model.score_scale
    neg_s = (seq_out * neg_emb).sum(dim=-1) / model.score_scale
    return -jt.log(jt.sigmoid(pos_s - neg_s) + 1e-8).mean()


def _eval_loss(model, seqs, td, lengths, pos, neg, val_idx, batch_size) -> float:
    if val_idx.shape[0] == 0:
        return 0.0
    with jt.no_grad():
        losses = []
        for s in range(0, val_idx.shape[0], batch_size):
            bi = val_idx[s : s + batch_size]
            losses.append(float(_bpr_loss(model, seqs, td, lengths, pos, neg, bi).item()))
        return float(np.mean(losses)) if losses else 0.0


def _score_batch(model, seqs, td, lengths, decay_w, cand_ids, out):
    all_h = model.all_hidden_states(
        jt.array(seqs, dtype=jt.int32),
        jt.array(td, dtype=jt.int32),
        jt.array(lengths, dtype=jt.int32),
    )
    last_h = extract_last_hidden(all_h, lengths)
    item_emb = model.item_vectors(jt.array(cand_ids, dtype=jt.int32))

    dot = (last_h.unsqueeze(1) * item_emb).sum(dim=-1) / model.score_scale
    out[:, :, 0] = np.asarray(dot.numpy(), dtype=np.float32)

    l_norm = jt.norm(last_h, dim=-1, keepdim=True).clamp(min_v=1e-8)
    i_norm = jt.norm(item_emb, dim=-1, keepdim=True).clamp(min_v=1e-8)
    cos = ((last_h.unsqueeze(1) / l_norm.unsqueeze(1)) * (item_emb / i_norm)).sum(dim=-1)
    out[:, :, 1] = np.asarray(cos.numpy(), dtype=np.float32)

    out[:, :, 2] = decay_weighted_scores(
        np.asarray(all_h.numpy(), dtype=np.float32),
        np.asarray(item_emb.numpy(), dtype=np.float32),
        lengths, decay_w, model.score_scale,
    )


def _final_sequences(
    interactions: InteractionTable, id_map: NodeIdMap, max_seq_len: int,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], np.ndarray]:
    item_hist: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=max_seq_len))
    time_hist: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=max_seq_len))
    seen = np.zeros(id_map.num_dst + 1, dtype=bool)
    for src, dst, t in zip(interactions.src, interactions.dst, interactions.time, strict=True):
        sid, did = id_map.src_id(int(src)), id_map.dst_id(int(dst))
        if sid < 0 or did < 0:
            continue
        iid = did + 1
        item_hist[sid].append(iid)
        time_hist[sid].append(int(t))
        seen[iid] = True
    seqs = {
        sid: (
            np.fromiter(items, dtype=np.int32, count=len(items)),
            np.fromiter(time_hist[sid], dtype=np.int64, count=len(items)),
        )
        for sid, items in item_hist.items()
    }
    return seqs, seen


def _build_samples(
    interactions: InteractionTable, id_map: NodeIdMap, config: SequenceTowerConfig, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    histories: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=config.max_seq_len))
    t_hist: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=config.max_seq_len))
    seqs, tds, lengths, pos_items, neg_items = [], [], [], [], []
    seen = 0

    for src, dst, t in zip(interactions.src, interactions.dst, interactions.time, strict=True):
        sid, did = id_map.src_id(int(src)), id_map.dst_id(int(dst))
        if sid < 0 or did < 0:
            continue
        history = histories[sid]
        if history:
            seen += 1
            slot = len(seqs)
            if config.max_samples > 0 and slot >= config.max_samples:
                replace = int(rng.integers(0, seen))
                if replace >= config.max_samples:
                    history.append(did + 1)
                    t_hist[sid].append(int(t))
                    continue
                slot = replace

            hv = tuple(history)
            length = min(len(hv), config.max_seq_len)
            seq = np.zeros(config.max_seq_len, dtype=np.int32)
            seq[:length] = hv[-length:]

            tv = tuple(t_hist[sid])
            td_arr = compute_time_deltas(np.array(tv[-length:], dtype=np.int64))
            td_buck = np.zeros(config.max_seq_len, dtype=np.int32)
            td_buck[:length] = time_delta_bucket(td_arr)

            pos = did + 1
            neg = int(rng.integers(1, id_map.num_dst + 1))
            if neg == pos:
                neg = 1 + (neg % id_map.num_dst)

            if slot == len(seqs):
                seqs.append(seq)
                tds.append(td_buck)
                lengths.append(length)
                pos_items.append(pos)
                neg_items.append(neg)
            else:
                seqs[slot], tds[slot] = seq, td_buck
                lengths[slot], pos_items[slot], neg_items[slot] = length, pos, neg

        history.append(did + 1)
        t_hist[sid].append(int(t))

    if not seqs:
        return None
    return (
        np.asarray(seqs, dtype=np.int32),
        np.asarray(tds, dtype=np.int32),
        np.asarray(lengths, dtype=np.int32),
        np.asarray(pos_items, dtype=np.int32),
        np.asarray(neg_items, dtype=np.int32),
    )
