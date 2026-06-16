from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jittor as jt
import numpy as np
from sklearn.metrics import average_precision_score

from jgrec.core.types import InteractionTable, TestQueryArray
from jgrec.logging import log, track

from .index import TemporalNodeMap
from .model import EndToEndTemporalGraphModel

CANDIDATE_RECENT_WINDOW_LABELS = ("005", "020")
CANDIDATE_RECENT_WINDOW_FRACTIONS = (0.05, 0.20)
CANDIDATE_BASE_FEATURE_DIM = 6
CANDIDATE_FEATURES_PER_RECENT_WINDOW = 4
CANDIDATE_PRIOR_FEATURE_NAMES = (
    "candidate_train_seen",
    "candidate_test_freq",
    "candidate_unseen_test_freq",
    "candidate_test_freq_row_rank",
    "candidate_train_freq",
    "candidate_train_freq_row_rank",
    *(
        f"candidate_train_recent_{name}_w{label}"
        for label in CANDIDATE_RECENT_WINDOW_LABELS
        for name in (
            "pop",
            "share",
            "recency",
            "rank",
        )
    ),
)
CANDIDATE_PRIOR_FEATURE_DIM = len(CANDIDATE_PRIOR_FEATURE_NAMES)


@dataclass(frozen=True)
class TemporalTrainingBatch:
    src_ids: np.ndarray
    times: np.ndarray
    candidates: np.ndarray
    src_neighbor_ids: np.ndarray
    src_neighbor_times: np.ndarray
    candidate_neighbor_ids: np.ndarray
    candidate_neighbor_times: np.ndarray
    candidate_features: np.ndarray


@dataclass(frozen=True)
class TemporalTrainingResult:
    best_val_ap: float
    best_val_mrr: float
    best_epoch: int
    state: dict[str, np.ndarray]


@dataclass(frozen=True)
class TestCandidateIndex:
    by_src: dict[int, tuple[np.ndarray, ...]]
    global_candidates: np.ndarray

    @classmethod
    def from_queries(cls, queries: TestQueryArray, node_map: TemporalNodeMap) -> TestCandidateIndex:
        by_src_lists: dict[int, list[np.ndarray]] = {}
        global_chunks: list[np.ndarray] = []
        chunk_size = 4096
        for start in range(0, len(queries), chunk_size):
            stop = min(start + chunk_size, len(queries))
            candidate_rows = node_map.dst_ids(queries.candidates[start:stop])
            for src, candidate_ids in zip(queries.src[start:stop], candidate_rows, strict=True):
                candidate_ids = candidate_ids[candidate_ids > 0]
                if candidate_ids.size == 0:
                    continue
                compact = candidate_ids.astype(np.int32, copy=False)
                by_src_lists.setdefault(int(src), []).append(compact)
                global_chunks.append(compact)
        by_src = {src: tuple(rows) for src, rows in by_src_lists.items()}
        global_candidates = np.concatenate(global_chunks) if global_chunks else np.empty(0, dtype=np.int32)
        return cls(by_src=by_src, global_candidates=global_candidates.astype(np.int32, copy=False))


@dataclass(frozen=True)
class CandidatePriorIndex:
    train_dst_ids: frozenset[int]
    train_dst_counts: dict[int, int]
    train_dst_total: int
    test_candidate_counts: dict[int, int]
    test_candidate_total: int
    recent_train_features: dict[int, tuple[float, ...]]
    recent_feature_mask: tuple[bool, ...]
    include_test_frequency: bool = False

    @classmethod
    def from_test_candidates(
        cls,
        candidate_index: TestCandidateIndex,
        train_dst_ids: np.ndarray,
        train_times: np.ndarray | None = None,
        recent_feature_group: str = "none",
        include_test_frequency: bool = False,
    ) -> CandidatePriorIndex:
        if candidate_index.global_candidates.size:
            candidate_values = candidate_index.global_candidates
        else:
            rows = [row for source_rows in candidate_index.by_src.values() for row in source_rows]
            candidate_values = np.concatenate(rows) if rows else np.empty(0, dtype=np.int32)
        if candidate_values.size:
            values, counts = np.unique(candidate_values[candidate_values > 0], return_counts=True)
            test_counts = {
                int(value): int(count)
                for value, count in zip(values, counts, strict=True)
            }
            total = int(counts.sum())
        else:
            test_counts = {}
            total = 0
        raw_train_values = np.asarray(train_dst_ids, dtype=np.int32)
        train_mask = raw_train_values > 0
        train_values = raw_train_values[train_mask]
        aligned_train_times = None
        if train_times is not None:
            raw_train_times = np.asarray(train_times, dtype=np.int64)
            if raw_train_times.shape[0] != raw_train_values.shape[0]:
                raise ValueError("train_times must align with train_dst_ids")
            aligned_train_times = raw_train_times[train_mask]
        if train_values.size:
            train_unique, train_counts_array = np.unique(train_values, return_counts=True)
            train_counts = {
                int(value): int(count)
                for value, count in zip(train_unique, train_counts_array, strict=True)
            }
            train_total = int(train_counts_array.sum())
        else:
            train_counts = {}
            train_total = 0
        return cls(
            train_dst_ids=frozenset(train_counts),
            train_dst_counts=train_counts,
            train_dst_total=train_total,
            test_candidate_counts=test_counts,
            test_candidate_total=total,
            recent_train_features=_recent_train_feature_map(train_values, aligned_train_times),
            recent_feature_mask=_recent_feature_mask(recent_feature_group),
            include_test_frequency=include_test_frequency,
        )


def train_listwise(
    model: EndToEndTemporalGraphModel,
    train_events: InteractionTable,
    val_events: InteractionTable,
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
    train_candidate_index: TestCandidateIndex | None,
    validation_candidate_index: TestCandidateIndex | None,
    candidate_prior_index: CandidatePriorIndex | None,
    rng: np.random.Generator,
    verbose: bool,
) -> TemporalTrainingResult:
    if len(train_events) == 0:
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
        candidate_index=validation_candidate_index,
        candidate_prior_index=candidate_prior_index,
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
                candidate_index=train_candidate_index,
                candidate_prior_index=candidate_prior_index,
            )
            model.clear_gate_buffer()
            logits = model(*_batch_to_jittor(batch))
            loss = _candidate_softmax_loss(logits)
            loss = loss + model.gate_regularization_loss(lam=0.05)
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
            candidate_index=validation_candidate_index,
            candidate_prior_index=candidate_prior_index,
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
    events: InteractionTable,
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
    train_candidate_index: TestCandidateIndex | None,
    candidate_prior_index: CandidatePriorIndex | None,
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
                candidate_index=train_candidate_index,
                candidate_prior_index=candidate_prior_index,
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
    events: InteractionTable,
    node_map: TemporalNodeMap,
    neighbor_sampler: Any,
    dst_pool: np.ndarray,
    *,
    batch_size: int,
    num_negatives: int,
    candidate_index: TestCandidateIndex | None = None,
    candidate_prior_index: CandidatePriorIndex | None = None,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if len(events) == 0:
        return 0.0, 0.0
    model.eval()
    score_batches: list[np.ndarray] = []
    for batch_events in _event_batches(events, batch_size):
        batch = build_evaluation_batch(
            events=batch_events,
            node_map=node_map,
            neighbor_sampler=neighbor_sampler,
            dst_pool=dst_pool,
            num_negatives=num_negatives,
            candidate_index=candidate_index,
            candidate_prior_index=candidate_prior_index,
            rng=rng,
            history_len=model.config.history_len,
            candidate_history_len=model.config.candidate_history_len,
        )
        with jt.no_grad():
            logits = model(*_batch_to_jittor(batch))
        score_batches.append(np.asarray(logits.numpy(), dtype=np.float32))
    scores = np.concatenate(score_batches, axis=0)
    return _ap_from_scores(scores), _mrr_from_scores(scores)


def build_evaluation_batch(
    events: InteractionTable,
    node_map: TemporalNodeMap,
    neighbor_sampler: Any,
    dst_pool: np.ndarray,
    num_negatives: int,
    candidate_index: TestCandidateIndex | None,
    candidate_prior_index: CandidatePriorIndex | None,
    rng: np.random.Generator,
    history_len: int,
    candidate_history_len: int,
) -> TemporalTrainingBatch:
    if candidate_index is None:
        return build_training_batch(
            events=events,
            node_map=node_map,
            neighbor_sampler=neighbor_sampler,
            dst_pool=dst_pool,
            num_negatives=num_negatives,
            rng=rng,
            history_len=history_len,
            candidate_history_len=candidate_history_len,
            candidate_prior_index=candidate_prior_index,
        )

    src_ids = node_map.src_ids(events.src)
    times = events.time.astype(np.int32, copy=False)
    positives = node_map.dst_ids(events.dst)
    src_neighbor_ids, _, src_neighbor_times = neighbor_sampler.get_historical_neighbors_left(
        node_ids=src_ids,
        node_interact_times=times,
        num_neighbors=history_len,
    )
    candidates = _sample_test_like_candidate_ids(
        events=events,
        positives=positives,
        candidate_index=candidate_index,
        dst_pool=dst_pool,
        num_negatives=num_negatives,
        rng=rng,
    )
    return build_prediction_batch(
        src_ids=src_ids,
        times=times,
        candidates=candidates,
        neighbor_sampler=neighbor_sampler,
        history_len=history_len,
        candidate_history_len=candidate_history_len,
        src_neighbor_ids=np.asarray(src_neighbor_ids, dtype=np.int32),
        src_neighbor_times=np.asarray(src_neighbor_times, dtype=np.int32),
        candidate_prior_index=candidate_prior_index,
    )


def build_training_batch(
    events: InteractionTable,
    node_map: TemporalNodeMap,
    neighbor_sampler: Any,
    dst_pool: np.ndarray,
    num_negatives: int,
    rng: np.random.Generator,
    history_len: int,
    candidate_history_len: int,
    candidate_index: TestCandidateIndex | None = None,
    candidate_prior_index: CandidatePriorIndex | None = None,
) -> TemporalTrainingBatch:
    src_ids = node_map.src_ids(events.src)
    times = events.time.astype(np.int32, copy=False)
    positives = node_map.dst_ids(events.dst)
    src_neighbor_ids, _, src_neighbor_times = neighbor_sampler.get_historical_neighbors_left(
        node_ids=src_ids,
        node_interact_times=times,
        num_neighbors=history_len,
    )
    if candidate_index is None:
        candidates = _sample_candidate_ids(
            positives=positives,
            dst_pool=dst_pool,
            num_negatives=num_negatives,
            rng=rng,
            forbidden=src_neighbor_ids,
        )
    else:
        candidates = _sample_test_like_candidate_ids(
            events=events,
            positives=positives,
            candidate_index=candidate_index,
            dst_pool=dst_pool,
            num_negatives=num_negatives,
            rng=rng,
        )
    return build_prediction_batch(
        src_ids=src_ids,
        times=times,
        candidates=candidates,
        neighbor_sampler=neighbor_sampler,
        history_len=history_len,
        candidate_history_len=candidate_history_len,
        src_neighbor_ids=np.asarray(src_neighbor_ids, dtype=np.int32),
        src_neighbor_times=np.asarray(src_neighbor_times, dtype=np.int32),
        candidate_prior_index=candidate_prior_index,
    )


def build_prediction_batch(
    src_ids: np.ndarray,
    times: np.ndarray,
    candidates: np.ndarray,
    neighbor_sampler: Any,
    history_len: int,
    candidate_history_len: int,
    src_neighbor_ids: np.ndarray | None = None,
    src_neighbor_times: np.ndarray | None = None,
    candidate_prior_index: CandidatePriorIndex | None = None,
) -> TemporalTrainingBatch:
    if src_neighbor_ids is None or src_neighbor_times is None:
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
        candidate_features=build_candidate_prior_features(
            candidates,
            candidate_prior_index,
        ),
    )


def queries_to_prediction_batch(
    queries: TestQueryArray,
    node_map: TemporalNodeMap,
    neighbor_sampler: Any,
    history_len: int,
    candidate_history_len: int,
    candidate_prior_index: CandidatePriorIndex | None = None,
) -> TemporalTrainingBatch:
    src_ids = node_map.src_ids(queries.src)
    times = queries.time.astype(np.int32, copy=False)
    candidates = node_map.dst_ids(queries.candidates)
    return build_prediction_batch(
        src_ids=src_ids,
        times=times,
        candidates=candidates,
        neighbor_sampler=neighbor_sampler,
        history_len=history_len,
        candidate_history_len=candidate_history_len,
        candidate_prior_index=candidate_prior_index,
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
        _int32_var(batch.src_ids),
        _int32_var(batch.candidates),
        _int32_var(batch.times),
        _int32_var(batch.src_neighbor_ids),
        _int32_var(batch.src_neighbor_times),
        _int32_var(batch.candidate_neighbor_ids),
        _int32_var(batch.candidate_neighbor_times),
        _float32_var(batch.candidate_features),
    )


def _int32_var(array: np.ndarray) -> jt.Var:
    return jt.Var(array.astype(np.int32, copy=False))


def _float32_var(array: np.ndarray) -> jt.Var:
    return jt.Var(array.astype(np.float32, copy=False))


def build_candidate_prior_features(
    candidates: np.ndarray,
    candidate_prior_index: CandidatePriorIndex | None,
) -> np.ndarray:
    features = np.zeros((*candidates.shape, CANDIDATE_PRIOR_FEATURE_DIM), dtype=np.float32)
    if candidate_prior_index is None:
        return features

    flat_candidates = candidates.reshape(-1)
    train_seen = np.fromiter(
        (int(candidate) in candidate_prior_index.train_dst_ids for candidate in flat_candidates),
        dtype=bool,
        count=flat_candidates.size,
    ).reshape(candidates.shape)
    train_freq = np.fromiter(
        (
            candidate_prior_index.train_dst_counts.get(int(candidate), 0)
            / max(float(candidate_prior_index.train_dst_total), 1.0)
            for candidate in flat_candidates
        ),
        dtype=np.float32,
        count=flat_candidates.size,
    ).reshape(candidates.shape)
    valid = candidates > 0

    features[:, :, 0] = train_seen.astype(np.float32, copy=False)
    # Columns 1-3 carry test-set candidate frequency, which leaks test-side label
    # information. Keep them zero unless explicitly opted in; the train-only
    # frequency prior (columns 4-5) provides the non-leaky popularity signal.
    if candidate_prior_index.include_test_frequency:
        test_freq = np.fromiter(
            (
                candidate_prior_index.test_candidate_counts.get(int(candidate), 0)
                / max(float(candidate_prior_index.test_candidate_total), 1.0)
                for candidate in flat_candidates
            ),
            dtype=np.float32,
            count=flat_candidates.size,
        ).reshape(candidates.shape)
        features[:, :, 1] = test_freq
        features[:, :, 2] = np.where(train_seen, 0.0, test_freq).astype(np.float32, copy=False)
        features[:, :, 3] = _row_rank_features(test_freq)
    features[:, :, 4] = train_freq
    features[:, :, 5] = _row_rank_features(train_freq)
    _fill_recent_train_features(features, candidates, candidate_prior_index)
    features[~valid] = 0.0
    return features


def _fill_recent_train_features(
    features: np.ndarray,
    candidates: np.ndarray,
    candidate_prior_index: CandidatePriorIndex,
) -> None:
    if not candidate_prior_index.recent_train_features:
        return
    recent = np.zeros((candidates.shape[0], candidates.shape[1], CANDIDATE_PRIOR_FEATURE_DIM - CANDIDATE_BASE_FEATURE_DIM), dtype=np.float32)
    flat_candidates = candidates.reshape(-1)
    flat_recent = recent.reshape((flat_candidates.shape[0], -1))
    for row_idx, candidate in enumerate(flat_candidates):
        values = candidate_prior_index.recent_train_features.get(int(candidate))
        if values is None:
            continue
        flat_recent[row_idx] = np.asarray(values, dtype=np.float32)

    for window_idx, _ in enumerate(CANDIDATE_RECENT_WINDOW_LABELS):
        local_start = window_idx * CANDIDATE_FEATURES_PER_RECENT_WINDOW
        pop = recent[:, :, local_start]
        recent[:, :, local_start + 3] = _row_rank_features(pop)
    if candidate_prior_index.recent_feature_mask:
        mask = np.asarray(candidate_prior_index.recent_feature_mask, dtype=np.float32).reshape((1, 1, -1))
        recent *= mask
    features[:, :, CANDIDATE_BASE_FEATURE_DIM:] = recent


def _recent_feature_mask(group: str) -> tuple[bool, ...]:
    normalized = group.lower()
    feature_names = tuple(
        name
        for _label in CANDIDATE_RECENT_WINDOW_LABELS
        for name in ("pop", "share", "recency", "rank")
    )
    if normalized == "none":
        enabled: set[str] = set()
    elif normalized == "recency_rank":
        enabled = {"recency", "rank"}
    else:
        raise ValueError(f"unsupported candidate recent feature group: {group}")
    return tuple(name in enabled for name in feature_names)


def _recent_train_feature_map(
    train_dst_ids: np.ndarray,
    train_times: np.ndarray | None,
) -> dict[int, tuple[float, ...]]:
    if train_times is None or train_dst_ids.size == 0:
        return {}
    if train_dst_ids.shape[0] != train_times.shape[0]:
        raise ValueError("train_times must align with train_dst_ids")
    min_time = int(train_times.min())
    max_time = int(train_times.max())
    graph_span = max(max_time - min_time, 1)
    feature_values: dict[int, list[float]] = {int(dst): [] for dst in np.unique(train_dst_ids)}
    for fraction in CANDIDATE_RECENT_WINDOW_FRACTIONS:
        width = max(math.ceil(float(fraction) * graph_span), 1)
        start_time = max_time + 1 - width
        in_window = train_times >= start_time
        window_dst_ids = train_dst_ids[in_window]
        window_times = train_times[in_window]
        window_total = int(window_dst_ids.shape[0])
        denominator = math.log1p(max(window_total, 1))
        if window_total:
            values, counts = np.unique(window_dst_ids, return_counts=True)
            last_times = {
                int(dst): int(window_times[window_dst_ids == dst].max())
                for dst in values
            }
            count_map = {
                int(dst): int(count)
                for dst, count in zip(values, counts, strict=True)
            }
        else:
            count_map = {}
            last_times = {}
        for dst in feature_values:
            count = count_map.get(dst, 0)
            if count > 0:
                pop = math.log1p(count) / denominator
                share = count / max(float(window_total), 1.0)
                recency = math.exp(-max((max_time + 1) - last_times[dst], 0) / max(float(width), 1.0))
            else:
                pop = 0.0
                share = 0.0
                recency = 0.0
            feature_values[dst].extend([pop, share, recency, 0.0])
    return {
        dst: tuple(float(value) for value in values)
        for dst, values in feature_values.items()
    }


def _sample_candidate_ids(
    positives: np.ndarray,
    dst_pool: np.ndarray,
    num_negatives: int,
    rng: np.random.Generator,
    forbidden: np.ndarray | None = None,
) -> np.ndarray:
    candidate_count = num_negatives + 1
    candidates = np.empty((positives.shape[0], candidate_count), dtype=np.int32)
    candidates[:, 0] = positives
    if dst_pool.size == 0:
        candidates[:, 1:] = positives[:, np.newaxis]
        return candidates
    replace = dst_pool.shape[0] < max(num_negatives * 2, 1)
    for row_idx, positive in enumerate(positives):
        used = {int(positive)}
        if forbidden is not None:
            used.update(int(value) for value in forbidden[row_idx] if int(value) != 0)
        negatives: list[int] = []
        attempts = 0
        while len(negatives) < num_negatives and attempts < 25:
            attempts += 1
            draw_size = max((num_negatives - len(negatives)) * 2, 16)
            draw_replace = replace or dst_pool.shape[0] < draw_size
            sampled = rng.choice(dst_pool, size=draw_size, replace=draw_replace)
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


def _sample_test_like_candidate_ids(
    events: InteractionTable,
    positives: np.ndarray,
    candidate_index: TestCandidateIndex,
    dst_pool: np.ndarray,
    num_negatives: int,
    rng: np.random.Generator,
) -> np.ndarray:
    candidate_count = num_negatives + 1
    candidates = np.empty((positives.shape[0], candidate_count), dtype=np.int32)
    candidates[:, 0] = positives
    global_pool = candidate_index.global_candidates if candidate_index.global_candidates.size else dst_pool
    fallback_pool = dst_pool if dst_pool.size else global_pool

    for row_idx, positive in enumerate(positives):
        used = {int(positive), 0}
        negatives: list[int] = []
        raw_src = int(events.src[row_idx])
        source_rows = candidate_index.by_src.get(raw_src)
        if source_rows:
            row = source_rows[int(rng.integers(0, len(source_rows)))]
            filtered = row[(row != int(positive)) & (row != 0)]
            if filtered.size >= num_negatives:
                first_values = filtered[:num_negatives]
                if np.unique(first_values).size == num_negatives:
                    candidates[row_idx, 1:] = first_values.astype(np.int32, copy=False)
                    continue
            for value in row:
                item = int(value)
                if item in used:
                    continue
                used.add(item)
                negatives.append(item)
                if len(negatives) >= num_negatives:
                    break

        attempts = 0
        while len(negatives) < num_negatives and attempts < 100 and global_pool.size:
            attempts += 1
            need = num_negatives - len(negatives)
            draw_size = max(need * 3, 128)
            sampled = rng.choice(global_pool, size=draw_size, replace=global_pool.size < draw_size)
            for value in sampled:
                item = int(value)
                if item in used:
                    continue
                used.add(item)
                negatives.append(item)
                if len(negatives) >= num_negatives:
                    break

        if len(negatives) < num_negatives:
            for value in fallback_pool:
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


def _event_batches(events: InteractionTable, batch_size: int):
    for start in range(0, len(events), batch_size):
        yield events[start : start + batch_size]


def _sample_events(events: InteractionTable, max_events: int, rng: np.random.Generator) -> InteractionTable:
    if max_events <= 0 or len(events) <= max_events:
        return events
    indices = np.sort(rng.choice(len(events), size=max_events, replace=False))
    return events.take(indices)


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


def _row_rank_features(values: np.ndarray) -> np.ndarray:
    # Tied candidates share an averaged rank so the feature cannot encode column
    # position. Without this, stable-sort tie-breaking would award the positive
    # (always column 0 during training) the best rank among ties -> label leak.
    if values.size == 0:
        return np.empty(values.shape, dtype=np.float32)
    ranks = _average_descending_ranks(values)
    return (1.0 / ranks).astype(np.float32, copy=False)


def _average_descending_ranks(values: np.ndarray) -> np.ndarray:
    """Fractional (average) 1-based ranks in descending order, ties averaged."""
    n_rows, n_cols = values.shape
    order = np.argsort(-values, axis=1, kind="mergesort")
    sorted_values = np.take_along_axis(values, order, axis=1)
    positions = np.broadcast_to(np.arange(1, n_cols + 1, dtype=np.float64), (n_rows, n_cols))
    is_group_start = np.ones((n_rows, n_cols), dtype=bool)
    is_group_start[:, 1:] = sorted_values[:, 1:] != sorted_values[:, :-1]
    group_id = np.cumsum(is_group_start, axis=1) - 1
    flat_group = (np.arange(n_rows)[:, None] * n_cols + group_id).ravel()
    minlength = n_rows * n_cols
    sums = np.bincount(flat_group, weights=positions.ravel(), minlength=minlength)
    counts = np.bincount(flat_group, minlength=minlength)
    avg_rank = sums / np.maximum(counts, 1)
    sorted_ranks = avg_rank[flat_group].reshape(n_rows, n_cols)
    ranks = np.empty((n_rows, n_cols), dtype=np.float64)
    np.put_along_axis(ranks, order, sorted_ranks, axis=1)
    return ranks
