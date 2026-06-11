from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from jgrec.core.memory import release_memory
from jgrec.core.types import InteractionTable, TestQuery, TestQueryArray
from jgrec.idmap import NodeIdMap
from jgrec.logging import log, track
from jgrec.rankers.common.temporal_index import TemporalInteractionIndex

from .config import SOURCE_PROFILE_FEATURE_NAMES, SourceProfileConfig

SOURCE_PROFILE_FEATURE_DIM = len(SOURCE_PROFILE_FEATURE_NAMES)
DETERMINISTIC_FEATURE_DIM = 6
ITEM2VEC_FEATURE_DIM = 4
EPSILON = 1e-8


def _jt():
    import jittor as jt  # noqa: PLC0415

    return jt


class SourceProfileTower:
    def __init__(self, id_map: NodeIdMap, config: SourceProfileConfig) -> None:
        self.id_map = id_map
        self.config = config
        self.index = TemporalInteractionIndex()
        self.item_pair_counts: dict[int, dict[int, int]] = {}
        self.item_degrees: dict[int, int] = {}
        self.embeddings: np.ndarray | None = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return SOURCE_PROFILE_FEATURE_NAMES

    def fit(
        self,
        interactions: InteractionTable,
        rng: np.random.Generator,
        verbose: bool = True,
        shared_index: TemporalInteractionIndex | None = None,
        deterministic_ready: bool = False,
    ) -> None:
        if len(interactions) == 0:
            raise ValueError("training interactions are empty")
        interactions = interactions.sort_by_time()
        if shared_index is not None and shared_index.total_edges > 0:
            self.index = shared_index
        else:
            self.index = TemporalInteractionIndex()
            self.index.fit(
                interactions,
                build_transitions=False,
                build_cooccurs=False,
            )
        if self.config.deterministic_enabled and not deterministic_ready:
            self.fit_deterministic(interactions)
        elif not self.config.deterministic_enabled:
            self.item_pair_counts = {}
            self.item_degrees = {}
        if self.config.item2vec_enabled and self.config.epochs > 0 and self.id_map.num_dst >= 2:
            self._fit_item2vec(interactions, rng=rng, verbose=verbose)
        else:
            self.embeddings = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "item_pair_counts": {int(left): dict(counts) for left, counts in self.item_pair_counts.items()},
            "item_degrees": dict(self.item_degrees),
        }

    def hydrate(self, snapshot: dict[str, Any]) -> None:
        self.item_pair_counts = {
            int(left): {int(right): int(count) for right, count in counts.items()}
            for left, counts in snapshot["item_pair_counts"].items()
        }
        self.item_degrees = {int(dst): int(count) for dst, count in snapshot["item_degrees"].items()}

    def fit_deterministic(self, interactions: InteractionTable) -> None:
        self._fit_deterministic(interactions)

    def scores_for_queries(self, queries: TestQueryArray | list[TestQuery]) -> np.ndarray:
        if not queries:
            return np.empty((0, 0, SOURCE_PROFILE_FEATURE_DIM), dtype=np.float32)
        if isinstance(queries, TestQueryArray):
            return self.scores_for_query_array(queries)
        return self.scores_for_query_array(TestQueryArray.from_queries(queries))

    def scores_for_query_array(self, queries: TestQueryArray) -> np.ndarray:
        if not queries:
            return np.empty((0, 0, SOURCE_PROFILE_FEATURE_DIM), dtype=np.float32)
        scores = np.zeros((len(queries), queries.candidate_count, SOURCE_PROFILE_FEATURE_DIM), dtype=np.float32)
        score_batch_size = max(int(self.config.score_batch_size), 1)
        for start in range(0, len(queries), score_batch_size):
            end = min(start + score_batch_size, len(queries))
            for row_idx in range(start, end):
                src = int(queries.src[row_idx])
                query_time = int(queries.time[row_idx])
                history = self.index.source_view(src, query_time).visible_dsts
                if history.size == 0:
                    continue
                candidates = queries.candidates[row_idx].astype(np.int64, copy=False)
                if self.config.deterministic_enabled:
                    self._fill_deterministic_features(history, candidates, scores[row_idx])
                if self.embeddings is not None:
                    self._fill_item2vec_features(history, candidates, scores[row_idx])
        return scores

    def _fit_deterministic(self, interactions: InteractionTable) -> None:
        pair_counts: dict[int, dict[int, int]] = defaultdict(dict)
        degrees: dict[int, int] = {}
        window_size = max(int(self.config.window_size), 1)
        grouped: dict[int, list[int]] = defaultdict(list)
        for src, dst in zip(interactions.src, interactions.dst, strict=True):
            grouped[int(src)].append(int(dst))

        for dsts in grouped.values():
            for dst in dsts:
                degrees[int(dst)] = degrees.get(int(dst), 0) + 1
            for idx, left in enumerate(dsts):
                left_int = int(left)
                start = max(0, idx - window_size)
                for right in dsts[start:idx]:
                    right_int = int(right)
                    if left_int == right_int:
                        continue
                    pair_counts[left_int][right_int] = pair_counts[left_int].get(right_int, 0) + 1
                    pair_counts[right_int][left_int] = pair_counts[right_int].get(left_int, 0) + 1
        self.item_pair_counts = {left: dict(counts) for left, counts in pair_counts.items()}
        self.item_degrees = degrees

    def _fit_item2vec(self, interactions: InteractionTable, rng: np.random.Generator, verbose: bool) -> None:
        samples = _item2vec_samples(interactions, self.id_map, self.config, rng)
        if samples is None:
            self.embeddings = None
            return

        centers, positives, negatives = samples
        jt = _jt()
        model = _Item2VecModel(num_items=self.id_map.num_dst, embedding_dim=self.config.embedding_dim)
        optimizer = jt.nn.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        train_size = int(centers.shape[0])
        for epoch in track(range(1, self.config.epochs + 1), description="source-profile", total=self.config.epochs, enabled=verbose):
            order = rng.permutation(train_size)
            losses: list[float] = []
            for start in range(0, train_size, max(int(self.config.batch_size), 1)):
                batch_idx = order[start : start + int(self.config.batch_size)]
                center = jt.array(centers[batch_idx], dtype=jt.int32)
                pos = jt.array(positives[batch_idx], dtype=jt.int32)
                neg = jt.array(negatives[batch_idx], dtype=jt.int32)
                loss = model(center, pos, neg)
                optimizer.step(loss)
                losses.append(float(loss.item()))
            log(f"[source-profile] epoch={epoch} loss={float(np.mean(losses)) if losses else 0.0:.5f}", enabled=verbose)
            release_memory()

        with jt.no_grad():
            self.embeddings = np.asarray(model.item_embedding.weight.numpy(), dtype=np.float32)

    def _fill_deterministic_features(
        self,
        history: np.ndarray,
        candidates: np.ndarray,
        output: np.ndarray,
    ) -> None:
        recent_k = max(int(self.config.recent_k), 1)
        recent_history = history[-recent_k:]
        for col_idx, candidate in enumerate(candidates):
            candidate_int = int(candidate)
            total = 0.0
            max_value = 0.0
            cosine_total = 0.0
            cosine_max = 0.0
            recent_cosine_total = 0.0
            recent_cosine_max = 0.0
            candidate_degree = self.item_degrees.get(candidate_int, 0)
            for seen in history:
                cooccur = self._cooccur_count(int(seen), candidate_int)
                if cooccur <= 0:
                    continue
                value = math.log1p(cooccur)
                total += value
                max_value = max(max_value, value)
                cosine = self._cosine_from_count(cooccur, int(seen), candidate_int, candidate_degree)
                cosine_total += cosine
                cosine_max = max(cosine_max, cosine)
            for seen in recent_history:
                cooccur = self._cooccur_count(int(seen), candidate_int)
                if cooccur <= 0:
                    continue
                cosine = self._cosine_from_count(cooccur, int(seen), candidate_int, candidate_degree)
                recent_cosine_total += cosine
                recent_cosine_max = max(recent_cosine_max, cosine)
            output[col_idx, 0] = np.float32(total)
            output[col_idx, 1] = np.float32(max_value)
            output[col_idx, 2] = np.float32(cosine_total)
            output[col_idx, 3] = np.float32(cosine_max)
            output[col_idx, 4] = np.float32(recent_cosine_total)
            output[col_idx, 5] = np.float32(recent_cosine_max)

    def _fill_item2vec_features(self, history: np.ndarray, candidates: np.ndarray, output: np.ndarray) -> None:
        if self.embeddings is None:
            return
        history_ids = self.id_map.dst_ids(history)
        history_ids = history_ids[history_ids >= 0]
        if history_ids.size == 0:
            return
        recent_k = max(int(self.config.recent_k), 1)
        recent_ids = history_ids[-recent_k:]
        full_profile = _mean_embedding(self.embeddings, history_ids)
        recent_profile = _mean_embedding(self.embeddings, recent_ids)
        dst_ids = self.id_map.dst_ids(candidates)
        for col_idx, dst_id in enumerate(dst_ids):
            if dst_id < 0:
                continue
            if int(candidates[col_idx]) not in self.index.dst_times:
                continue
            item_vec = self.embeddings[int(dst_id)]
            output[col_idx, 6] = np.float32(float(np.dot(full_profile, item_vec)) / math.sqrt(self.embeddings.shape[1]))
            output[col_idx, 7] = np.float32(_cosine(full_profile, item_vec))
            output[col_idx, 8] = np.float32(float(np.dot(recent_profile, item_vec)) / math.sqrt(self.embeddings.shape[1]))
            output[col_idx, 9] = np.float32(_cosine(recent_profile, item_vec))

    def _cooccur_count(self, left: int, right: int) -> int:
        if left == right:
            return 0
        return self.item_pair_counts.get(int(left), {}).get(int(right), 0)

    def _cosine_from_count(self, cooccur: int, left: int, right: int, right_degree: int) -> float:
        left_degree = self.item_degrees.get(int(left), 0)
        denominator = math.sqrt(max(left_degree, 1) * max(right_degree, 1))
        return float(cooccur) / max(denominator, EPSILON)


class _Item2VecModel:
    def __new__(cls, num_items: int, embedding_dim: int):
        jt = _jt()

        class Item2VecModel(jt.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.item_embedding = jt.nn.Embedding(int(num_items), int(embedding_dim))

            def execute(self, center_ids, pos_ids, neg_ids):
                center = self.item_embedding(center_ids)
                pos = self.item_embedding(pos_ids)
                neg = self.item_embedding(neg_ids)
                pos_scores = (center * pos).sum(dim=-1)
                neg_scores = (center * neg).sum(dim=-1)
                return -jt.log(jt.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

        return Item2VecModel()


def _item2vec_samples(
    interactions: InteractionTable,
    id_map: NodeIdMap,
    config: SourceProfileConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    window_size = max(int(config.window_size), 1)
    max_samples = max(int(config.max_samples), 0)
    centers: list[int] = []
    positives: list[int] = []
    seen = 0
    histories: dict[int, list[int]] = defaultdict(list)
    for src, dst in zip(interactions.src, interactions.dst, strict=True):
        dst_id = id_map.dst_id(int(dst))
        if dst_id < 0:
            continue
        history = histories[int(src)]
        for previous in history[-window_size:]:
            if previous == dst_id:
                continue
            seen += 2
            _append_or_reservoir(centers, positives, previous, dst_id, seen - 1, max_samples, rng)
            _append_or_reservoir(centers, positives, dst_id, previous, seen, max_samples, rng)
        history.append(dst_id)
    if not centers:
        return None
    centers_array = np.asarray(centers, dtype=np.int32)
    positives_array = np.asarray(positives, dtype=np.int32)
    negatives = rng.integers(0, id_map.num_dst, size=centers_array.shape[0], dtype=np.int32)
    same = negatives == positives_array
    if np.any(same):
        negatives[same] = (negatives[same] + 1) % id_map.num_dst
    return centers_array, positives_array, negatives


def _append_or_reservoir(
    centers: list[int],
    positives: list[int],
    center: int,
    positive: int,
    seen: int,
    max_samples: int,
    rng: np.random.Generator,
) -> None:
    if max_samples <= 0 or len(centers) < max_samples:
        centers.append(int(center))
        positives.append(int(positive))
        return
    replace = int(rng.integers(0, seen))
    if replace < max_samples:
        centers[replace] = int(center)
        positives[replace] = int(positive)


def _mean_embedding(embeddings: np.ndarray, ids: np.ndarray) -> np.ndarray:
    if ids.size == 0:
        return np.zeros(embeddings.shape[1], dtype=np.float32)
    return np.asarray(embeddings[ids].mean(axis=0), dtype=np.float32)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= EPSILON:
        return 0.0
    return float(np.dot(left, right) / denominator)
