from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jittor as jt
import numpy as np
from sklearn.metrics import average_precision_score

from jgrec.core.types import Interaction, TestQuery
from jgrec.logging import log, track

from .index import TemporalNodeMap
from .model import EndToEndTemporalGraphModel


@dataclass(frozen=True)
class TemporalTrainingBatch:
    src_ids: np.ndarray
    times: np.ndarray
    candidates: np.ndarray
    src_neighbor_ids: np.ndarray
    src_neighbor_times: np.ndarray
    candidate_neighbor_ids: np.ndarray
    candidate_neighbor_times: np.ndarray


@dataclass(frozen=True)
class TemporalTrainingResult:
    best_val_ap: float
    best_val_mrr: float
    best_epoch: int
    state: dict[str, np.ndarray]


def train_listwise(
    model: EndToEndTemporalGraphModel,
    train_events: list[Interaction],
    val_events: list[Interaction],
    node_map: TemporalNodeMap,
    neighbor_sampler: Any,
    dst_pool: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    num_negatives: int,
    lr: float,
    weight_decay: float,
    early_stop_patience: int,
    max_train_events: int,
    max_val_events: int,
    selection_metric: str,
    rng: np.random.Generator,
    verbose: bool,
) -> TemporalTrainingResult:
    if not train_events:
        raise ValueError("temporal graph ranker requires non-empty train events")
    train_events = _sample_events(train_events, max_train_events, rng)
    val_events = _sample_events(val_events, max_val_events, rng)
    optimizer = jt.nn.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_ap, best_mrr = evaluate_listwise(
        model=model,
        events=val_events,
        node_map=node_map,
        neighbor_sampler=neighbor_sampler,
        dst_pool=dst_pool,
        batch_size=batch_size,
        num_negatives=num_negatives,
        rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
    )
    best_score = _select_metric(best_ap, best_mrr, selection_metric)
    best_state = snapshot_state(model)
    best_epoch = 0
    patience_counter = 0

    for epoch in track(range(1, epochs + 1), description="temporal-graph", total=epochs, enabled=verbose):
        model.train()
        losses: list[float] = []
        for batch_events in _event_batches(train_events, batch_size):
            batch = build_training_batch(
                events=batch_events,
                node_map=node_map,
                neighbor_sampler=neighbor_sampler,
                dst_pool=dst_pool,
                num_negatives=num_negatives,
                rng=rng,
                history_len=model.config.history_len,
                candidate_history_len=model.config.candidate_history_len,
            )
            logits = model(*_batch_to_jittor(batch))
            loss = _candidate_softmax_loss(logits)
            optimizer.step(loss)
            jt.sync_all()
            losses.append(float(loss.item()))

        val_ap, val_mrr = evaluate_listwise(
            model=model,
            events=val_events,
            node_map=node_map,
            neighbor_sampler=neighbor_sampler,
            dst_pool=dst_pool,
            batch_size=batch_size,
            num_negatives=num_negatives,
            rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
        )
        val_score = _select_metric(val_ap, val_mrr, selection_metric)
        if val_score >= best_score:
            best_ap = val_ap
            best_mrr = val_mrr
            best_score = val_score
            best_epoch = epoch
            best_state = snapshot_state(model)
            patience_counter = 0
        else:
            patience_counter += 1
        log(
            f"[temporal-graph] epoch={epoch} loss={float(np.mean(losses)) if losses else 0.0:.5f} "
            f"val_ap={val_ap:.5f} val_mrr={val_mrr:.5f} "
            f"best_{selection_metric}={best_score:.5f} patience={patience_counter}",
            enabled=verbose,
        )
        if early_stop_patience > 0 and patience_counter >= early_stop_patience:
            log(f"[temporal-graph] early_stop epoch={epoch}", enabled=verbose)
            break

    load_state(model, best_state)
    return TemporalTrainingResult(
        best_val_ap=float(best_ap),
        best_val_mrr=float(best_mrr),
        best_epoch=max(best_epoch, 1),
        state=best_state,
    )


def fit_full_epochs(
    model: EndToEndTemporalGraphModel,
    events: list[Interaction],
    node_map: TemporalNodeMap,
    neighbor_sampler: Any,
    dst_pool: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    num_negatives: int,
    lr: float,
    weight_decay: float,
    max_train_events: int,
    rng: np.random.Generator,
    verbose: bool,
) -> None:
    train_events = _sample_events(events, max_train_events, rng)
    optimizer = jt.nn.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in track(range(1, epochs + 1), description="temporal-graph:full", total=epochs, enabled=verbose):
        model.train()
        losses: list[float] = []
        for batch_events in _event_batches(train_events, batch_size):
            batch = build_training_batch(
                events=batch_events,
                node_map=node_map,
                neighbor_sampler=neighbor_sampler,
                dst_pool=dst_pool,
                num_negatives=num_negatives,
                rng=rng,
                history_len=model.config.history_len,
                candidate_history_len=model.config.candidate_history_len,
            )
            logits = model(*_batch_to_jittor(batch))
            loss = _candidate_softmax_loss(logits)
            optimizer.step(loss)
            jt.sync_all()
            losses.append(float(loss.item()))
        log(
            f"[temporal-graph:full] epoch={epoch} loss={float(np.mean(losses)) if losses else 0.0:.5f}",
            enabled=verbose,
        )


def evaluate_listwise(
    model: EndToEndTemporalGraphModel,
    events: list[Interaction],
    node_map: TemporalNodeMap,
    neighbor_sampler: Any,
    dst_pool: np.ndarray,
    *,
    batch_size: int,
    num_negatives: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if not events:
        return 0.0, 0.0
    model.eval()
    score_batches: list[np.ndarray] = []
    for batch_events in _event_batches(events, batch_size):
        batch = build_training_batch(
            events=batch_events,
            node_map=node_map,
            neighbor_sampler=neighbor_sampler,
            dst_pool=dst_pool,
            num_negatives=num_negatives,
            rng=rng,
            history_len=model.config.history_len,
            candidate_history_len=model.config.candidate_history_len,
        )
        with jt.no_grad():
            logits = model(*_batch_to_jittor(batch))
        score_batches.append(np.asarray(logits.numpy(), dtype=np.float32))
    scores = np.concatenate(score_batches, axis=0)
    return _ap_from_scores(scores), _mrr_from_scores(scores)


def build_training_batch(
    events: list[Interaction],
    node_map: TemporalNodeMap,
    neighbor_sampler: Any,
    dst_pool: np.ndarray,
    num_negatives: int,
    rng: np.random.Generator,
    history_len: int,
    candidate_history_len: int,
) -> TemporalTrainingBatch:
    src_ids = np.asarray([node_map.src_id(item.src) for item in events], dtype=np.int32)
    times = np.asarray([item.time for item in events], dtype=np.int32)
    positives = np.asarray([node_map.dst_id(item.dst) for item in events], dtype=np.int32)
    candidates = _sample_candidate_ids(positives, dst_pool, num_negatives, rng)
    return build_prediction_batch(
        src_ids=src_ids,
        times=times,
        candidates=candidates,
        neighbor_sampler=neighbor_sampler,
        history_len=history_len,
        candidate_history_len=candidate_history_len,
    )


def build_prediction_batch(
    src_ids: np.ndarray,
    times: np.ndarray,
    candidates: np.ndarray,
    neighbor_sampler: Any,
    history_len: int,
    candidate_history_len: int,
) -> TemporalTrainingBatch:
    src_neighbor_ids, _, src_neighbor_times = neighbor_sampler.get_historical_neighbors_left(
        node_ids=src_ids,
        node_interact_times=times,
        num_neighbors=history_len,
    )
    flat_candidates = candidates.reshape(-1)
    flat_times = np.broadcast_to(times[:, np.newaxis], candidates.shape).reshape(-1)
    candidate_neighbor_ids, _, candidate_neighbor_times = neighbor_sampler.get_historical_neighbors_left(
        node_ids=flat_candidates,
        node_interact_times=flat_times,
        num_neighbors=candidate_history_len,
    )
    return TemporalTrainingBatch(
        src_ids=src_ids.astype(np.int32, copy=False),
        times=times.astype(np.int32, copy=False),
        candidates=candidates.astype(np.int32, copy=False),
        src_neighbor_ids=np.asarray(src_neighbor_ids, dtype=np.int32),
        src_neighbor_times=np.asarray(src_neighbor_times, dtype=np.int32),
        candidate_neighbor_ids=np.asarray(candidate_neighbor_ids, dtype=np.int32).reshape(
            candidates.shape[0],
            candidates.shape[1],
            candidate_history_len,
        ),
        candidate_neighbor_times=np.asarray(candidate_neighbor_times, dtype=np.int32).reshape(
            candidates.shape[0],
            candidates.shape[1],
            candidate_history_len,
        ),
    )


def queries_to_prediction_batch(
    queries: list[TestQuery],
    node_map: TemporalNodeMap,
    neighbor_sampler: Any,
    history_len: int,
    candidate_history_len: int,
) -> TemporalTrainingBatch:
    src_ids = np.asarray([node_map.src_id(query.src) for query in queries], dtype=np.int32)
    times = np.asarray([query.time for query in queries], dtype=np.int32)
    candidates = np.asarray([node_map.dst_ids(query.candidates) for query in queries], dtype=np.int32)
    return build_prediction_batch(
        src_ids=src_ids,
        times=times,
        candidates=candidates,
        neighbor_sampler=neighbor_sampler,
        history_len=history_len,
        candidate_history_len=candidate_history_len,
    )


def predict_logits(model: EndToEndTemporalGraphModel, batch: TemporalTrainingBatch) -> np.ndarray:
    model.eval()
    with jt.no_grad():
        logits = model(*_batch_to_jittor(batch))
    return np.asarray(logits.numpy(), dtype=np.float32)


def snapshot_state(model: EndToEndTemporalGraphModel) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(value.numpy(), dtype=np.float32).copy()
        for key, value in model.state_dict().items()
    }


def load_state(model: EndToEndTemporalGraphModel, state: dict[str, np.ndarray]) -> None:
    model.load_state_dict({key: jt.array(value, dtype=jt.float32) for key, value in state.items()})


def _batch_to_jittor(batch: TemporalTrainingBatch) -> tuple[jt.Var, ...]:
    return (
        jt.array(batch.src_ids, dtype=jt.int32),
        jt.array(batch.candidates, dtype=jt.int32),
        jt.array(batch.times, dtype=jt.int32),
        jt.array(batch.src_neighbor_ids, dtype=jt.int32),
        jt.array(batch.src_neighbor_times, dtype=jt.int32),
        jt.array(batch.candidate_neighbor_ids, dtype=jt.int32),
        jt.array(batch.candidate_neighbor_times, dtype=jt.int32),
    )


def _sample_candidate_ids(
    positives: np.ndarray,
    dst_pool: np.ndarray,
    num_negatives: int,
    rng: np.random.Generator,
) -> np.ndarray:
    candidate_count = num_negatives + 1
    candidates = np.empty((positives.shape[0], candidate_count), dtype=np.int32)
    candidates[:, 0] = positives
    replace = dst_pool.shape[0] < max(num_negatives * 2, 1)
    for row_idx, positive in enumerate(positives):
        used = {int(positive)}
        negatives: list[int] = []
        attempts = 0
        while len(negatives) < num_negatives and attempts < 25:
            attempts += 1
            draw_size = max((num_negatives - len(negatives)) * 2, 16)
            sampled = rng.choice(dst_pool, size=draw_size, replace=replace)
            for value in sampled:
                item = int(value)
                if item in used:
                    continue
                used.add(item)
                negatives.append(item)
                if len(negatives) >= num_negatives:
                    break
        if len(negatives) < num_negatives:
            for value in dst_pool:
                item = int(value)
                if item in used:
                    continue
                used.add(item)
                negatives.append(item)
                if len(negatives) >= num_negatives:
                    break
        if len(negatives) < num_negatives:
            negatives.extend([int(positive)] * (num_negatives - len(negatives)))
        candidates[row_idx, 1:] = np.asarray(negatives[:num_negatives], dtype=np.int32)
    return candidates


def _candidate_softmax_loss(logits: jt.Var) -> jt.Var:
    shifted = logits - logits.max(dim=1, keepdims=True)
    log_probs = shifted - jt.log(jt.exp(shifted).sum(dim=1, keepdims=True))
    return -log_probs[:, 0].mean()


def _event_batches(events: list[Interaction], batch_size: int):
    for start in range(0, len(events), batch_size):
        yield events[start : start + batch_size]


def _sample_events(events: list[Interaction], max_events: int, rng: np.random.Generator) -> list[Interaction]:
    if max_events <= 0 or len(events) <= max_events:
        return list(events)
    indices = np.sort(rng.choice(len(events), size=max_events, replace=False))
    return [events[int(index)] for index in indices]


def _ap_from_scores(scores: np.ndarray) -> float:
    labels = np.zeros(scores.shape, dtype=np.int8)
    labels[:, 0] = 1
    return float(average_precision_score(labels.ravel(), scores.ravel()))


def _mrr_from_scores(scores: np.ndarray) -> float:
    positive_scores = scores[:, 0:1]
    ranks = 1 + (scores[:, 1:] > positive_scores).sum(axis=1)
    return float(np.mean(1.0 / ranks))


def _select_metric(ap: float, mrr: float, metric: str) -> float:
    normalized = metric.lower()
    if normalized == "ap":
        return ap
    if normalized == "mrr":
        return mrr
    raise ValueError(f"unsupported selection metric: {metric}")
