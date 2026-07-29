from __future__ import annotations

import math
from dataclasses import dataclass

import jittor as jt
import numpy as np

from jgrec.contest_checkpoint import get_model_state, set_model_state
from jgrec.core.memory import release_memory
from jgrec.core.types import InteractionTable, TestQuery, TestQueryArray
from jgrec.idmap import NodeIdMap
from jgrec.logging import log, track
from jgrec.rankers.common.early_stop import LossEarlyStopper
from jgrec.rankers.common.optimization import (
    set_tower_optimizer_learning_rate,
)
from jgrec.rankers.common.temporal_index import TemporalInteractionIndex

from .auto_strategy import DatasetProfile, test_candidate_arrays
from .config import TWO_TOWER_FEATURE_NAMES, TwoTowerConfig
from .in_batch_negatives import _in_batch_positive_mask
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


def _listwise_positive_loss(logits: jt.Var) -> jt.Var:
    if len(logits.shape) != 2 or logits.shape[1] <= 0:
        raise ValueError("listwise logits must have shape [queries, candidates]")
    return -jt.nn.log_softmax(logits, dim=1)[:, 0].mean()


def _multi_positive_in_batch_loss(
    logits: jt.Var,
    positive_dst_ids: np.ndarray,
    *,
    temperature: float,
) -> jt.Var:
    if len(logits.shape) != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("in-batch logits must be a square matrix")
    destination_ids = np.asarray(positive_dst_ids)
    if destination_ids.shape != (int(logits.shape[0]),):
        raise ValueError("positive destination IDs must align with logits")
    temperature_value = float(temperature)
    if temperature_value <= 0.0:
        raise ValueError("in-batch temperature must be positive")

    positive_mask = jt.array(
        _in_batch_positive_mask(destination_ids).astype(np.float32),
        dtype=jt.float32,
    )
    log_probs = jt.nn.log_softmax(
        logits / temperature_value,
        dim=1,
    )
    positive_probability = (jt.exp(log_probs) * positive_mask).sum(dim=1)
    return -jt.log(positive_probability + 1e-8).mean()


def _full_candidate_mrr(scores: np.ndarray) -> float:
    values = np.asarray(scores)
    if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] <= 0:
        raise ValueError("candidate scores must have shape [queries, candidates]")
    ranks = 1 + (values[:, 1:] > values[:, 0:1]).sum(axis=1)
    return float(np.mean(1.0 / ranks))


def _early_stop_signal(
    metric: str,
    *,
    val_loss: float,
    val_mrr: float,
) -> float:
    normalized = str(metric).lower()
    if normalized == "loss":
        return float(val_loss)
    if normalized == "mrr":
        return -float(val_mrr)
    raise ValueError(f"unsupported two-tower early-stop metric: {metric}")


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


def _take_tower_batch(batch: _TowerBatch, indices: np.ndarray) -> _TowerBatch:
    return _TowerBatch(
        src_ids=batch.src_ids[indices],
        src_activity_buckets=batch.src_activity_buckets[indices],
        src_recency_buckets=batch.src_recency_buckets[indices],
        src_time_buckets=batch.src_time_buckets[indices],
        dst_ids=batch.dst_ids[indices],
        dst_popularity_buckets=batch.dst_popularity_buckets[indices],
        dst_recency_buckets=batch.dst_recency_buckets[indices],
        dst_time_buckets=batch.dst_time_buckets[indices],
    )


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
        interactions: InteractionTable,
        id_map: NodeIdMap,
        index: TemporalInteractionIndex,
        dataset_profile: DatasetProfile | None = None,
    ) -> _TowerTrainingContext:
        dst_pool = np.asarray(id_map.dst_values, dtype=np.int64)
        test_values, test_weights = test_candidate_arrays(dataset_profile)
        return cls(
            id_map=id_map,
            index=index,
            negative_context=NegativeSamplingContext(
                index=index,
                dst_values=id_map.dst_values,
                test_candidate_values=test_values,
                test_candidate_weights=test_weights,
            ),
            dst_pool=dst_pool,
            candidate_pool=build_candidate_pool(dst_pool, test_values),
            min_time=int(interactions.time[0]),
            graph_span=max(int(interactions.time[-1]) - int(interactions.time[0]), 1),
        )


class TwoTower:
    def __init__(
        self,
        id_map: NodeIdMap,
        config: TwoTowerConfig,
        dataset_profile: DatasetProfile | None = None,
    ) -> None:
        self.id_map = id_map
        self.config = config
        self.dataset_profile = dataset_profile
        self.model: _TwoTowerModel | None = None
        self.index = TemporalInteractionIndex()
        self.min_time = 0
        self.max_time = 0
        self.graph_span = 1

    @property
    def feature_names(self) -> tuple[str, ...]:
        return TWO_TOWER_FEATURE_NAMES

    def snapshot(self) -> dict:
        return {
            "model_state": get_model_state(self.model) if self.model is not None else None,
            "index": self.index.shallow_copy(),
            "min_time": self.min_time,
            "max_time": self.max_time,
            "graph_span": self.graph_span,
        }

    def hydrate(self, snapshot: dict) -> None:
        self.index = snapshot["index"].shallow_copy()
        self.min_time = int(snapshot["min_time"])
        self.max_time = int(snapshot["max_time"])
        self.graph_span = int(snapshot["graph_span"])
        model_state = snapshot.get("model_state")
        if model_state is None:
            self.model = None
            return
        self.model = _TwoTowerModel(
            num_src=self.id_map.num_src,
            num_dst=self.id_map.num_dst,
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
        )
        set_model_state(self.model, model_state)

    def fit(
        self,
        interactions: InteractionTable,
        rng: np.random.Generator,
        verbose: bool = True,
        shared_index: TemporalInteractionIndex | None = None,
    ) -> None:
        if len(interactions) == 0:
            raise ValueError("training interactions are empty")

        interactions = interactions.sort_by_time()
        test_candidate_ratio = float(
            getattr(self.config, "test_candidate_negative_ratio", 0.0)
        )
        needs_structural_negatives = (
            self.config.enabled
            and self.config.num_negatives > 0
            and self.config.hard_negative_ratio > 0
            and test_candidate_ratio < 1.0
        )
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
        self.min_time = int(interactions.time[0])
        self.max_time = int(interactions.time[-1])
        self.graph_span = max(self.max_time - self.min_time, 1)

        if not self.config.enabled or self.config.epochs < 1 or self.config.num_negatives < 1:
            return
        if self.id_map.num_src == 0 or self.id_map.num_dst < 2:
            return

        sampled_events = _sample_events(interactions, self.config.max_samples, rng)
        if len(sampled_events) == 0:
            return

        self.model = _TwoTowerModel(
            num_src=self.id_map.num_src,
            num_dst=self.id_map.num_dst,
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
        )
        optimizer = jt.nn.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        full_size = len(sampled_events)
        negative_seeds = rng.integers(0, 2**32 - 1, size=full_size, dtype=np.uint32)
        training_context = _TowerTrainingContext.from_interactions(
            interactions=interactions,
            id_map=self.id_map,
            index=self.index,
            dataset_profile=self.dataset_profile,
        )
        log(
            f"[two-tower] prepare_groups rows={full_size} "
            f"candidates={self.config.num_negatives + 1}",
            enabled=verbose,
        )
        training_samples = _build_training_batch_for_events(
            events=sampled_events,
            negative_seeds=negative_seeds,
            training_context=training_context,
            config=self.config,
        )
        log(
            f"[two-tower] groups_ready rows={full_size} "
            f"candidates={self.config.num_negatives + 1}",
            enabled=verbose,
        )
        val_size = max(0, int(full_size * self.config.early_stop_val_ratio))
        val_perm = rng.permutation(full_size)
        val_idx = val_perm[:val_size]
        train_idx = val_perm[val_size:]
        train_size = train_idx.shape[0]

        epochs = range(1, self.config.epochs + 1)
        stopper = LossEarlyStopper(patience=self.config.early_stop_patience)
        for epoch in track(epochs, description="two-tower", total=self.config.epochs, enabled=verbose):
            learning_rate = set_tower_optimizer_learning_rate(
                optimizer,
                initial_lr=self.config.lr,
                epoch=epoch,
                total_epochs=self.config.epochs,
                schedule=getattr(self.config, "lr_schedule", "constant"),
                min_lr_ratio=getattr(self.config, "min_lr_ratio", 0.0),
            )
            order = rng.permutation(train_size)
            losses: list[float] = []
            for start in range(0, train_size, self.config.batch_size):
                batch_idx = train_idx[order[start : start + self.config.batch_size]]
                loss = self._two_tower_loss(
                    training_samples,
                    batch_idx,
                    train=True,
                )
                optimizer.step(loss)
                losses.append(float(loss.item()))
                if start and start % max(self.config.batch_size * 16, 1) == 0:
                    release_memory()

            mean_loss = float(np.mean(losses)) if losses else 0.0
            if val_size > 0:
                val_loss, val_mrr = self._two_tower_val_metrics(
                    training_samples,
                    val_idx,
                )
            else:
                val_loss, val_mrr = 0.0, 0.0
            log(
                f"[two-tower] epoch={epoch} lr={learning_rate:.8g} "
                f"loss={mean_loss:.5f} "
                f"val_loss={val_loss:.5f} val_mrr={val_mrr:.5f}",
                enabled=verbose,
            )
            release_memory()
            stop_signal = (
                _early_stop_signal(
                    getattr(self.config, "early_stop_metric", "loss"),
                    val_loss=val_loss,
                    val_mrr=val_mrr,
                )
                if val_size > 0
                else mean_loss
            )
            if stopper.update(epoch, stop_signal, self.model):
                metric = str(
                    getattr(self.config, "early_stop_metric", "loss")
                ).lower()
                best_value = (
                    -float(stopper.best_loss)
                    if metric == "mrr"
                    else float(stopper.best_loss)
                )
                log(
                    f"[two-tower] early_stop epoch={epoch} "
                    f"best_epoch={stopper.best_epoch} "
                    f"best_{metric}={best_value:.5f}",
                    enabled=verbose,
                )
                break
        stopper.restore_best(self.model)

        del training_samples, train_idx, val_idx, val_perm
        if owns_index:
            self.index.transition_times = {}
            self.index.transitions_by_left = {}
            self.index.cooccur_times = {}
            self.index.cooccurs_by_left = {}

    def scores_for_queries(self, queries: TestQueryArray | list[TestQuery]) -> np.ndarray:
        if not queries:
            return np.empty((0, 0, len(TWO_TOWER_FEATURE_NAMES)), dtype=np.float32)
        return self.scores_for_query_array(TestQueryArray.from_queries(queries))

    def _two_tower_loss(
        self,
        training_samples: _TowerBatch,
        batch_idx: np.ndarray,
        *,
        train: bool,
    ) -> jt.Var:
        logits = self._two_tower_logits(
            training_samples,
            batch_idx,
        )
        objective = str(getattr(self.config, "objective", "bce")).lower()
        if objective == "listwise":
            base_loss = _listwise_positive_loss(logits)
        elif objective == "bce":
            targets = np.zeros(logits.shape, dtype=np.float32)
            targets[:, 0] = 1.0
            target_var = jt.array(targets, dtype=jt.float32)
            probs = jt.sigmoid(logits)
            base_loss = -(
                target_var * jt.log(probs + 1e-8)
                + (1.0 - target_var) * jt.log(1.0 - probs + 1e-8)
            ).mean()
        else:
            raise ValueError(f"unsupported two-tower objective: {objective}")

        in_batch_weight = float(
            getattr(self.config, "in_batch_negative_weight", 1.0)
        )
        if (
            not train
            or not bool(
                getattr(self.config, "in_batch_negatives", False)
            )
            or in_batch_weight <= 0.0
            or batch_idx.shape[0] <= 1
        ):
            return base_loss
        return base_loss + in_batch_weight * self._two_tower_in_batch_loss(
            training_samples,
            batch_idx,
        )

    def _two_tower_in_batch_loss(
        self,
        training_samples: _TowerBatch,
        batch_idx: np.ndarray,
    ) -> jt.Var:
        samples = _take_tower_batch(training_samples, batch_idx)
        source_vectors = self.model.source_vectors(
            jt.array(samples.src_ids, dtype=jt.int32),
            jt.array(samples.src_activity_buckets, dtype=jt.int32),
            jt.array(samples.src_recency_buckets, dtype=jt.int32),
            jt.array(samples.src_time_buckets, dtype=jt.int32),
        )
        positive_dst_ids = samples.dst_ids[:, 0]
        neutral_context = np.zeros_like(positive_dst_ids)
        destination_vectors = self.model.destination_vectors(
            jt.array(positive_dst_ids, dtype=jt.int32),
            jt.array(neutral_context, dtype=jt.int32),
            jt.array(neutral_context, dtype=jt.int32),
            jt.array(neutral_context, dtype=jt.int32),
        )
        logits = (
            source_vectors.unsqueeze(1)
            * destination_vectors.unsqueeze(0)
        ).sum(dim=-1) / math.sqrt(max(self.config.embedding_dim, 1))
        del samples
        return _multi_positive_in_batch_loss(
            logits,
            positive_dst_ids,
            temperature=float(
                getattr(self.config, "in_batch_temperature", 1.0)
            ),
        )

    def _two_tower_logits(
        self,
        training_samples: _TowerBatch,
        batch_idx: np.ndarray,
    ) -> jt.Var:
        samples = _take_tower_batch(training_samples, batch_idx)
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
        del samples
        return logits

    def _two_tower_val_metrics(
        self,
        training_samples: _TowerBatch,
        val_idx: np.ndarray,
    ) -> tuple[float, float]:
        if val_idx.shape[0] == 0:
            return 0.0, 0.0
        with jt.no_grad():
            weighted_loss = 0.0
            reciprocal_rank_sum = 0.0
            rows = 0
            for start in range(0, val_idx.shape[0], self.config.batch_size):
                batch_idx = val_idx[start : start + self.config.batch_size]
                logits = self._two_tower_logits(
                    training_samples,
                    batch_idx,
                )
                objective = str(
                    getattr(self.config, "objective", "bce")
                ).lower()
                if objective == "listwise":
                    loss = _listwise_positive_loss(logits)
                elif objective == "bce":
                    targets = np.zeros(logits.shape, dtype=np.float32)
                    targets[:, 0] = 1.0
                    target_var = jt.array(targets, dtype=jt.float32)
                    probs = jt.sigmoid(logits)
                    loss = -(
                        target_var * jt.log(probs + 1e-8)
                        + (1.0 - target_var)
                        * jt.log(1.0 - probs + 1e-8)
                    ).mean()
                else:
                    raise ValueError(
                        f"unsupported two-tower objective: {objective}"
                    )
                score_array = np.asarray(logits.numpy(), dtype=np.float32)
                batch_rows = int(score_array.shape[0])
                weighted_loss += float(loss.item()) * batch_rows
                reciprocal_rank_sum += (
                    _full_candidate_mrr(score_array) * batch_rows
                )
                rows += batch_rows
            if rows <= 0:
                return 0.0, 0.0
            return weighted_loss / rows, reciprocal_rank_sum / rows

    def scores_for_query_array(self, queries: TestQueryArray) -> np.ndarray:
        if not queries:
            return np.empty((0, 0, len(TWO_TOWER_FEATURE_NAMES)), dtype=np.float32)

        scores = np.zeros((len(queries), queries.candidate_count, len(TWO_TOWER_FEATURE_NAMES)), dtype=np.float32)
        if self.model is None:
            return scores

        batch = self._batch_for_query_array(queries)
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
        return self._batch_for_query_array(TestQueryArray.from_queries(queries))

    def _batch_for_query_array(self, queries: TestQueryArray) -> _TowerBatch:
        candidate_count = queries.candidate_count
        src_ids = np.empty(len(queries), dtype=np.int32)
        src_activity = np.empty(len(queries), dtype=np.int32)
        src_recency = np.empty(len(queries), dtype=np.int32)
        src_time = np.empty(len(queries), dtype=np.int32)
        dst_ids = np.empty((len(queries), candidate_count), dtype=np.int32)
        dst_popularity = np.empty((len(queries), candidate_count), dtype=np.int32)
        dst_recency = np.empty((len(queries), candidate_count), dtype=np.int32)
        dst_time = np.empty((len(queries), candidate_count), dtype=np.int32)

        for row_idx in range(len(queries)):
            src = int(queries.src[row_idx])
            query_time = int(queries.time[row_idx])
            source_view = self.index.source_view(src, query_time)
            src_ids[row_idx] = _src_model_id(self.id_map, src)
            src_activity[row_idx] = _count_bucket(source_view.cutoff)
            src_recency[row_idx] = _history_recency_bucket(source_view.visible_times, query_time, self.graph_span)
            time_bucket = _time_bucket(query_time, self.min_time, self.graph_span)
            src_time[row_idx] = time_bucket
            dst_time[row_idx, :] = time_bucket

            for col_idx, dst in enumerate(queries.candidates[row_idx]):
                dst_int = int(dst)
                destination_view = self.index.destination_view(dst_int, query_time)
                dst_ids[row_idx, col_idx] = _dst_model_id(self.id_map, dst_int)
                dst_popularity[row_idx, col_idx] = _count_bucket(destination_view.cutoff)
                dst_recency[row_idx, col_idx] = _history_recency_bucket(
                    destination_view.visible_times,
                    query_time,
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
    events: InteractionTable,
    negative_seeds: np.ndarray,
    training_context: _TowerTrainingContext,
    config: TwoTowerConfig,
) -> _TowerBatch:
    if len(events) == 0:
        raise ValueError("two-tower training batch is empty")
    candidate_count = config.num_negatives + 1
    jobs = [
        NegativeSamplingJob(src=int(src), positive_dst=int(dst), query_time=int(time))
        for src, dst, time in zip(events.src, events.dst, events.time, strict=True)
    ]
    negatives_by_event = sample_mixed_negatives_batch_seeded(
        jobs=jobs,
        seeds=negative_seeds,
        context=training_context.negative_context,
        dst_pool=training_context.dst_pool,
        num_negatives=config.num_negatives,
        hard_negative_ratio=config.hard_negative_ratio,
        popular_negative_ratio=config.popular_negative_ratio,
        test_candidate_negative_ratio=float(
            getattr(config, "test_candidate_negative_ratio", 0.0)
        ),
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

    for row_idx, (event_src, event_dst, event_time, negatives) in enumerate(
        zip(events.src, events.dst, events.time, negatives_by_event, strict=True)
    ):
        src_int = int(event_src)
        dst_int = int(event_dst)
        time_int = int(event_time)
        source_view = training_context.index.source_view(src_int, time_int)
        src_ids[row_idx] = _src_model_id(training_context.id_map, src_int)
        src_activity[row_idx] = _count_bucket(source_view.cutoff)
        src_recency[row_idx] = _history_recency_bucket(source_view.visible_times, time_int, training_context.graph_span)
        time_bucket = _time_bucket(time_int, training_context.min_time, training_context.graph_span)
        src_time[row_idx] = time_bucket
        dst_time[row_idx, :] = time_bucket

        for col_idx, candidate_dst in enumerate((dst_int, *negatives)):
            candidate_dst_int = int(candidate_dst)
            destination_view = training_context.index.destination_view(candidate_dst_int, time_int)
            dst_ids[row_idx, col_idx] = _dst_model_id(training_context.id_map, candidate_dst_int)
            dst_popularity[row_idx, col_idx] = _count_bucket(destination_view.cutoff)
            dst_recency[row_idx, col_idx] = _history_recency_bucket(
                destination_view.visible_times,
                time_int,
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
    events: InteractionTable,
    max_events: int,
    rng: np.random.Generator,
) -> InteractionTable:
    if max_events <= 0 or len(events) <= max_events:
        return events
    indices = np.sort(rng.choice(len(events), size=max_events, replace=False))
    return events.take(indices)


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
