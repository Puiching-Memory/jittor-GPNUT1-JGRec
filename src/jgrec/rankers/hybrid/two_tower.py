from __future__ import annotations

import math
from dataclasses import dataclass

import jittor as jt
import numpy as np

from jgrec.core.memory import release_memory
from jgrec.core.types import Interaction, TestQuery
from jgrec.idmap import NodeIdMap
from jgrec.logging import log, track
from jgrec.rankers.common.temporal_index import TemporalInteractionIndex

from .config import TWO_TOWER_FEATURE_NAMES, TwoTowerConfig
from .sampling import (
    DenseCandidatePool,
    NegativeSamplingContext,
    NegativeSamplingJob,
    build_candidate_pool,
    sample_mixed_negatives_batch_seeded,
)

COUNT_BUCKETS = 16
RECENCY_BUCKETS = 16
TIME_BUCKETS = 24


@dataclass(frozen=True)
class _TowerBatch:
    src_ids: np.ndarray
    src_activity_buckets: np.ndarray
    src_recency_buckets: np.ndarray
    src_time_buckets: np.ndarray
    dst_ids: np.ndarray
    dst_popularity_buckets: np.ndarray
    dst_recency_buckets: np.ndarray
    dst_time_buckets: np.ndarray


@dataclass(frozen=True)
class _TowerTrainingContext:
    id_map: NodeIdMap
    index: TemporalInteractionIndex
    negative_context: NegativeSamplingContext
    dst_pool: np.ndarray
    candidate_pool: DenseCandidatePool | set[int]
    min_time: int
    graph_span: int

    @classmethod
    def from_interactions(
        cls,
        interactions: list[Interaction],
        id_map: NodeIdMap,
        index: TemporalInteractionIndex,
    ) -> _TowerTrainingContext:
        dst_pool = np.asarray(id_map.dst_values, dtype=np.int64)
        return cls(
            id_map=id_map,
            index=index,
            negative_context=NegativeSamplingContext(index=index, dst_values=id_map.dst_values),
            dst_pool=dst_pool,
            candidate_pool=build_candidate_pool(dst_pool),
            min_time=interactions[0].time,
            graph_span=max(interactions[-1].time - interactions[0].time, 1),
        )


class TwoTower:
    def __init__(self, id_map: NodeIdMap, config: TwoTowerConfig) -> None:
        self.id_map = id_map
        self.config = config
        self.model: _TwoTowerModel | None = None
        self.index = TemporalInteractionIndex()
        self.min_time = 0
        self.max_time = 0
        self.graph_span = 1

    @property
    def feature_names(self) -> tuple[str, ...]:
        return TWO_TOWER_FEATURE_NAMES

    def fit(
        self,
        interactions: list[Interaction],
        rng: np.random.Generator,
        verbose: bool = True,
        shared_index: TemporalInteractionIndex | None = None,
    ) -> None:
        if not interactions:
            raise ValueError("training interactions are empty")

        interactions = _ensure_time_order(interactions)
        needs_structural_negatives = self.config.enabled and self.config.num_negatives > 0 and self.config.hard_negative_ratio > 0
        owns_index = not _can_reuse_index(shared_index, needs_structural_negatives)
        if owns_index:
            self.index = TemporalInteractionIndex()
            self.index.fit(
                interactions,
                build_transitions=needs_structural_negatives,
                build_cooccurs=needs_structural_negatives,
            )
        else:
            self.index = shared_index
        self.min_time = interactions[0].time
        self.max_time = interactions[-1].time
        self.graph_span = max(self.max_time - self.min_time, 1)

        if not self.config.enabled or self.config.epochs < 1 or self.config.num_negatives < 1:
            return
        if self.id_map.num_src == 0 or self.id_map.num_dst < 2:
            return

        sampled_events = _sample_events(interactions, self.config.max_samples, rng)
        if not sampled_events:
            return

        self.model = _TwoTowerModel(
            num_src=self.id_map.num_src,
            num_dst=self.id_map.num_dst,
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
        )
        optimizer = jt.nn.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        train_size = len(sampled_events)
        negative_seeds = rng.integers(0, 2**32 - 1, size=train_size, dtype=np.uint32)
        training_context = _TowerTrainingContext.from_interactions(
            interactions=interactions,
            id_map=self.id_map,
            index=self.index,
        )

        epochs = range(1, self.config.epochs + 1)
        for epoch in track(epochs, description="two-tower", total=self.config.epochs, enabled=verbose):
            order = rng.permutation(train_size)
            losses: list[float] = []
            for start in range(0, train_size, self.config.batch_size):
                batch_idx = order[start : start + self.config.batch_size]
                samples = _build_training_batch_for_events(
                    events=[sampled_events[int(index)] for index in batch_idx],
                    negative_seeds=negative_seeds[batch_idx],
                    training_context=training_context,
                    config=self.config,
                )
                logits = self.model(
                    jt.array(samples.src_ids, dtype=jt.int32),
                    jt.array(samples.src_activity_buckets, dtype=jt.int32),
                    jt.array(samples.src_recency_buckets, dtype=jt.int32),
                    jt.array(samples.src_time_buckets, dtype=jt.int32),
                    jt.array(samples.dst_ids, dtype=jt.int32),
                    jt.array(samples.dst_popularity_buckets, dtype=jt.int32),
                    jt.array(samples.dst_recency_buckets, dtype=jt.int32),
                    jt.array(samples.dst_time_buckets, dtype=jt.int32),
                )
                targets = np.zeros((samples.src_ids.shape[0], logits.shape[1]), dtype=np.float32)
                targets[:, 0] = 1.0
                target_var = jt.array(targets, dtype=jt.float32)
                probs = jt.sigmoid(logits)
                loss = -(
                    target_var * jt.log(probs + 1e-8)
                    + (1.0 - target_var) * jt.log(1.0 - probs + 1e-8)
                ).mean()
                optimizer.step(loss)
                losses.append(float(loss.item()))
                del samples
                if start and start % max(self.config.batch_size * 16, 1) == 0:
                    release_memory()

            mean_loss = float(np.mean(losses)) if losses else 0.0
            log(f"[two-tower] epoch={epoch} loss={mean_loss:.5f}", enabled=verbose)
            release_memory()

        if owns_index:
            self.index.transition_times = {}
            self.index.transitions_by_left = {}
            self.index.cooccur_times = {}
            self.index.cooccurs_by_left = {}

    def scores_for_queries(self, queries: list[TestQuery]) -> np.ndarray:
        if not queries:
            return np.empty((0, 0, len(TWO_TOWER_FEATURE_NAMES)), dtype=np.float32)

        candidate_count = len(queries[0].candidates)
        scores = np.zeros((len(queries), candidate_count, len(TWO_TOWER_FEATURE_NAMES)), dtype=np.float32)
        if self.model is None:
            return scores

        batch = self._batch_for_queries(queries)
        valid_src = batch.src_ids < self.id_map.num_src
        valid_dst = batch.dst_ids < self.id_map.num_dst
        with jt.no_grad():
            score_batch_size = max(int(self.config.score_batch_size), 1)
            for start in range(0, len(queries), score_batch_size):
                end = min(start + score_batch_size, len(queries))
                src_vec = self.model.source_vectors(
                    jt.array(batch.src_ids[start:end], dtype=jt.int32),
                    jt.array(batch.src_activity_buckets[start:end], dtype=jt.int32),
                    jt.array(batch.src_recency_buckets[start:end], dtype=jt.int32),
                    jt.array(batch.src_time_buckets[start:end], dtype=jt.int32),
                )
                dst_vec = self.model.destination_vectors(
                    jt.array(batch.dst_ids[start:end], dtype=jt.int32),
                    jt.array(batch.dst_popularity_buckets[start:end], dtype=jt.int32),
                    jt.array(batch.dst_recency_buckets[start:end], dtype=jt.int32),
                    jt.array(batch.dst_time_buckets[start:end], dtype=jt.int32),
                )
                raw_dot = (src_vec.unsqueeze(1) * dst_vec).sum(dim=-1)
                scaled_dot = raw_dot / math.sqrt(max(self.config.embedding_dim, 1))
                src_norm = jt.sqrt((src_vec * src_vec).sum(dim=-1, keepdims=True) + 1e-8)
                dst_norm = jt.sqrt((dst_vec * dst_vec).sum(dim=-1) + 1e-8)
                cosine = raw_dot / (src_norm * dst_norm)
                scores[start:end, :, 0] = np.asarray(scaled_dot.numpy(), dtype=np.float32)
                scores[start:end, :, 1] = np.asarray(cosine.numpy(), dtype=np.float32)

        scores[:, :, 0][~valid_src, :] = 0.0
        scores[:, :, 1][~valid_src, :] = 0.0
        scores[:, :, 0][~valid_dst] = 0.0
        scores[:, :, 1][~valid_dst] = 0.0
        return scores

    def _batch_for_queries(self, queries: list[TestQuery]) -> _TowerBatch:
        candidate_count = len(queries[0].candidates)
        src_ids = np.empty(len(queries), dtype=np.int32)
        src_activity = np.empty(len(queries), dtype=np.int32)
        src_recency = np.empty(len(queries), dtype=np.int32)
        src_time = np.empty(len(queries), dtype=np.int32)
        dst_ids = np.empty((len(queries), candidate_count), dtype=np.int32)
        dst_popularity = np.empty((len(queries), candidate_count), dtype=np.int32)
        dst_recency = np.empty((len(queries), candidate_count), dtype=np.int32)
        dst_time = np.empty((len(queries), candidate_count), dtype=np.int32)

        for row_idx, query in enumerate(queries):
            if len(query.candidates) != candidate_count:
                raise ValueError("all queries in a batch must have the same candidate count")
            source_view = self.index.source_view(query.src, query.time)
            src_ids[row_idx] = _src_model_id(self.id_map, query.src)
            src_activity[row_idx] = _count_bucket(source_view.cutoff)
            src_recency[row_idx] = _history_recency_bucket(source_view.visible_times, query.time, self.graph_span)
            time_bucket = _time_bucket(query.time, self.min_time, self.graph_span)
            src_time[row_idx] = time_bucket
            dst_time[row_idx, :] = time_bucket

            for col_idx, dst in enumerate(query.candidates):
                dst_int = int(dst)
                destination_view = self.index.destination_view(dst_int, query.time)
                dst_ids[row_idx, col_idx] = _dst_model_id(self.id_map, dst_int)
                dst_popularity[row_idx, col_idx] = _count_bucket(destination_view.cutoff)
                dst_recency[row_idx, col_idx] = _history_recency_bucket(
                    destination_view.visible_times,
                    query.time,
                    self.graph_span,
                )

        return _TowerBatch(
            src_ids=src_ids,
            src_activity_buckets=src_activity,
            src_recency_buckets=src_recency,
            src_time_buckets=src_time,
            dst_ids=dst_ids,
            dst_popularity_buckets=dst_popularity,
            dst_recency_buckets=dst_recency,
            dst_time_buckets=dst_time,
        )


def _can_reuse_index(index: TemporalInteractionIndex | None, needs_structural_negatives: bool) -> bool:
    if index is None or index.total_edges <= 0:
        return False
    if not needs_structural_negatives:
        return True
    return bool(index.built_transitions and index.built_cooccurs)


class _TwoTowerModel(jt.nn.Module):
    def __init__(self, num_src: int, num_dst: int, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding_dim = max(int(embedding_dim), 1)
        hidden_dim = max(int(hidden_dim), 1)
        input_dim = self.embedding_dim * 4
        self.src_id_embedding = jt.nn.Embedding(int(num_src) + 1, self.embedding_dim)
        self.dst_id_embedding = jt.nn.Embedding(int(num_dst) + 1, self.embedding_dim)
        self.src_activity_embedding = jt.nn.Embedding(COUNT_BUCKETS + 1, self.embedding_dim)
        self.src_recency_embedding = jt.nn.Embedding(RECENCY_BUCKETS + 1, self.embedding_dim)
        self.src_time_embedding = jt.nn.Embedding(TIME_BUCKETS, self.embedding_dim)
        self.dst_popularity_embedding = jt.nn.Embedding(COUNT_BUCKETS + 1, self.embedding_dim)
        self.dst_recency_embedding = jt.nn.Embedding(RECENCY_BUCKETS + 1, self.embedding_dim)
        self.dst_time_embedding = jt.nn.Embedding(TIME_BUCKETS, self.embedding_dim)
        self.src_linear1 = jt.nn.Linear(input_dim, hidden_dim)
        self.src_linear2 = jt.nn.Linear(hidden_dim, self.embedding_dim)
        self.dst_linear1 = jt.nn.Linear(input_dim, hidden_dim)
        self.dst_linear2 = jt.nn.Linear(hidden_dim, self.embedding_dim)

    def execute(
        self,
        src_ids: jt.Var,
        src_activity_buckets: jt.Var,
        src_recency_buckets: jt.Var,
        src_time_buckets: jt.Var,
        dst_ids: jt.Var,
        dst_popularity_buckets: jt.Var,
        dst_recency_buckets: jt.Var,
        dst_time_buckets: jt.Var,
    ) -> jt.Var:
        src_vec = self.source_vectors(src_ids, src_activity_buckets, src_recency_buckets, src_time_buckets)
        dst_vec = self.destination_vectors(dst_ids, dst_popularity_buckets, dst_recency_buckets, dst_time_buckets)
        return (src_vec.unsqueeze(1) * dst_vec).sum(dim=-1) / math.sqrt(self.embedding_dim)

    def source_vectors(
        self,
        src_ids: jt.Var,
        activity_buckets: jt.Var,
        recency_buckets: jt.Var,
        time_buckets: jt.Var,
    ) -> jt.Var:
        x = jt.concat(
            [
                self.src_id_embedding(src_ids),
                self.src_activity_embedding(activity_buckets),
                self.src_recency_embedding(recency_buckets),
                self.src_time_embedding(time_buckets),
            ],
            dim=-1,
        )
        return self.src_linear2(jt.nn.relu(self.src_linear1(x)))

    def destination_vectors(
        self,
        dst_ids: jt.Var,
        popularity_buckets: jt.Var,
        recency_buckets: jt.Var,
        time_buckets: jt.Var,
    ) -> jt.Var:
        x = jt.concat(
            [
                self.dst_id_embedding(dst_ids),
                self.dst_popularity_embedding(popularity_buckets),
                self.dst_recency_embedding(recency_buckets),
                self.dst_time_embedding(time_buckets),
            ],
            dim=-1,
        )
        original_shape = x.shape[:-1]
        x = x.reshape((-1, x.shape[-1]))
        x = self.dst_linear2(jt.nn.relu(self.dst_linear1(x)))
        return x.reshape((*original_shape, self.embedding_dim))


def _build_training_batch_for_events(
    events: list[Interaction],
    negative_seeds: np.ndarray,
    training_context: _TowerTrainingContext,
    config: TwoTowerConfig,
) -> _TowerBatch:
    if not events:
        raise ValueError("two-tower training batch is empty")
    candidate_count = config.num_negatives + 1
    jobs = [
        NegativeSamplingJob(src=event.src, positive_dst=event.dst, query_time=event.time)
        for event in events
    ]
    negatives_by_event = sample_mixed_negatives_batch_seeded(
        jobs=jobs,
        seeds=negative_seeds,
        context=training_context.negative_context,
        dst_pool=training_context.dst_pool,
        num_negatives=config.num_negatives,
        hard_negative_ratio=config.hard_negative_ratio,
        popular_negative_ratio=config.popular_negative_ratio,
        candidate_pool=training_context.candidate_pool,
        workers=config.negative_sampling_workers,
    )

    src_ids = np.empty(len(events), dtype=np.int32)
    src_activity = np.empty(len(events), dtype=np.int32)
    src_recency = np.empty(len(events), dtype=np.int32)
    src_time = np.empty(len(events), dtype=np.int32)
    dst_ids = np.empty((len(events), candidate_count), dtype=np.int32)
    dst_popularity = np.empty((len(events), candidate_count), dtype=np.int32)
    dst_recency = np.empty((len(events), candidate_count), dtype=np.int32)
    dst_time = np.empty((len(events), candidate_count), dtype=np.int32)

    for row_idx, (event, negatives) in enumerate(zip(events, negatives_by_event)):
        source_view = training_context.index.source_view(event.src, event.time)
        src_ids[row_idx] = _src_model_id(training_context.id_map, event.src)
        src_activity[row_idx] = _count_bucket(source_view.cutoff)
        src_recency[row_idx] = _history_recency_bucket(source_view.visible_times, event.time, training_context.graph_span)
        time_bucket = _time_bucket(event.time, training_context.min_time, training_context.graph_span)
        src_time[row_idx] = time_bucket
        dst_time[row_idx, :] = time_bucket

        for col_idx, dst in enumerate((event.dst, *negatives)):
            dst_int = int(dst)
            destination_view = training_context.index.destination_view(dst_int, event.time)
            dst_ids[row_idx, col_idx] = _dst_model_id(training_context.id_map, dst_int)
            dst_popularity[row_idx, col_idx] = _count_bucket(destination_view.cutoff)
            dst_recency[row_idx, col_idx] = _history_recency_bucket(
                destination_view.visible_times,
                event.time,
                training_context.graph_span,
            )

    return _TowerBatch(
        src_ids=src_ids,
        src_activity_buckets=src_activity,
        src_recency_buckets=src_recency,
        src_time_buckets=src_time,
        dst_ids=dst_ids,
        dst_popularity_buckets=dst_popularity,
        dst_recency_buckets=dst_recency,
        dst_time_buckets=dst_time,
    )


def _sample_events(
    events: list[Interaction],
    max_events: int,
    rng: np.random.Generator,
) -> list[Interaction]:
    if max_events <= 0 or len(events) <= max_events:
        return list(events)
    indices = np.sort(rng.choice(len(events), size=max_events, replace=False))
    return [events[int(index)] for index in indices]


def _src_model_id(id_map: NodeIdMap, src: int) -> int:
    mapped = id_map.src_id(src)
    return mapped if mapped >= 0 else id_map.num_src


def _dst_model_id(id_map: NodeIdMap, dst: int) -> int:
    mapped = id_map.dst_id(dst)
    return mapped if mapped >= 0 else id_map.num_dst


def _count_bucket(count: int) -> int:
    if count <= 0:
        return 0
    return min(int(math.log2(count)) + 1, COUNT_BUCKETS)


def _history_recency_bucket(times: np.ndarray, query_time: int, graph_span: int) -> int:
    if times.size == 0:
        return 0
    last_time = int(times[-1])
    score = math.exp(-max(int(query_time) - last_time, 0) / max(graph_span, 1))
    return min(int(score * (RECENCY_BUCKETS - 1)) + 1, RECENCY_BUCKETS)


def _time_bucket(query_time: int, min_time: int, graph_span: int) -> int:
    position = (int(query_time) - int(min_time)) / max(graph_span, 1)
    position = min(max(position, 0.0), 1.0)
    return min(int(position * TIME_BUCKETS), TIME_BUCKETS - 1)


def _ensure_time_order(interactions: list[Interaction]) -> list[Interaction]:
    if all(left.time <= right.time for left, right in zip(interactions, interactions[1:])):
        return interactions
    return sorted(interactions, key=lambda item: item.time)
