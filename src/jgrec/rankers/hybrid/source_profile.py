from __future__ import annotations

import math
from collections import OrderedDict, defaultdict
from typing import Any

import numpy as np

from jgrec.core.memory import release_memory
from jgrec.core.types import InteractionTable, TestQuery, TestQueryArray
from jgrec.idmap import NodeIdMap
from jgrec.logging import log, track
from jgrec.rankers.common.sparse_counts import SparseCountMap
from jgrec.rankers.common.temporal_index import TemporalInteractionIndex

from .config import SOURCE_PROFILE_FEATURE_NAMES, SourceProfileConfig

SOURCE_PROFILE_FEATURE_DIM = len(SOURCE_PROFILE_FEATURE_NAMES)
DETERMINISTIC_FEATURE_DIM = 6
ITEM2VEC_FEATURE_DIM = 4
EPSILON = 1e-8
SOURCE_PROFILE_CACHE_LIMIT = 2048
SOURCE_PROFILE_CACHE_MIN_HISTORY = 32

DeterministicSummary = tuple[
    dict[int, tuple[float, float, float, float]],
    dict[int, tuple[float, float]],
]


def _jt():
    import jittor as jt  # noqa: PLC0415

    return jt


class SourceProfileTower:
    def __init__(self, id_map: NodeIdMap, config: SourceProfileConfig) -> None:
        self.id_map = id_map
        self.config = config
        self.index = TemporalInteractionIndex()
        self.item_pair_counts: dict[int, dict[int, int]] = {}
        self.item_pair_counts_sparse: SparseCountMap = SparseCountMap.empty()
        self.item_degrees: dict[int, int] = {}
        self.embeddings: np.ndarray | None = None
        self._deterministic_cache: OrderedDict[tuple[int, int], DeterministicSummary] = OrderedDict()
        self._embedding_profile_cache: OrderedDict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = OrderedDict()

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
        self._clear_score_caches()

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
        self.item_pair_counts_sparse = SparseCountMap.from_nested_dict(self.item_pair_counts)
        self.item_degrees = {int(dst): int(count) for dst, count in snapshot["item_degrees"].items()}
        self._clear_score_caches()

    def fit_deterministic(self, interactions: InteractionTable) -> None:
        self._fit_deterministic(interactions)
        self._clear_score_caches()

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
        hist_limit = self.config.predict_history_limit
        for start in range(0, len(queries), score_batch_size):
            end = min(start + score_batch_size, len(queries))
            for row_idx in range(start, end):
                src = int(queries.src[row_idx])
                query_time = int(queries.time[row_idx])
                source_view = self.index.source_view(src, query_time)
                history = source_view.visible_dsts
                if history.size == 0:
                    continue
                if hist_limit > 0 and history.size > hist_limit:
                    history = history[-hist_limit:]
                cache_key = (src, int(source_view.cutoff))
                use_cache = history.size >= SOURCE_PROFILE_CACHE_MIN_HISTORY
                candidates = queries.candidates[row_idx].astype(np.int64, copy=False)
                if self.config.deterministic_enabled:
                    self._fill_deterministic_features(history, candidates, scores[row_idx], cache_key, use_cache)
                if self.embeddings is not None:
                    self._fill_item2vec_features(history, candidates, scores[row_idx], cache_key, use_cache)
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
        self.item_pair_counts_sparse = SparseCountMap.from_nested_dict(self.item_pair_counts)
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
        cache_key: tuple[int, int] | None = None,
        use_cache: bool = False,
    ) -> None:
        if use_cache and cache_key is not None:
            self._fill_deterministic_features_from_summary(cache_key, history, candidates, output)
            return

        recent_k = max(int(self.config.recent_k), 1)
        sparse = self.item_pair_counts_sparse
        for col_idx, candidate in enumerate(candidates):
            candidate_int = int(candidate)
            candidate_degree = max(self.item_degrees.get(candidate_int, 0), 1)
            counts = sparse.batch_get_counts(history.astype(np.int32, copy=False), candidate_int)
            mask = (counts > 0) & (history != candidate_int)
            if not np.any(mask):
                continue
            valid_counts = counts[mask].astype(np.float32)
            valid_history = history[mask]
            values = np.log1p(valid_counts)
            seen_degrees = np.array([max(self.item_degrees.get(int(s), 0), 1) for s in valid_history], dtype=np.float32)
            cosines = valid_counts / np.sqrt(seen_degrees * candidate_degree)
            output[col_idx, 0] = values.sum()
            output[col_idx, 1] = values.max()
            output[col_idx, 2] = cosines.sum()
            output[col_idx, 3] = cosines.max()
            recent_mask = mask[-recent_k:]
            if np.any(recent_mask):
                recent_counts = counts[-recent_k:][recent_mask].astype(np.float32)
                recent_seen = history[-recent_k:][recent_mask]
                recent_degrees = np.array([max(self.item_degrees.get(int(s), 0), 1) for s in recent_seen], dtype=np.float32)
                recent_cosines = recent_counts / np.sqrt(recent_degrees * candidate_degree)
                output[col_idx, 4] = recent_cosines.sum()
                output[col_idx, 5] = recent_cosines.max()

    def _fill_deterministic_features_from_summary(
        self,
        cache_key: tuple[int, int],
        history: np.ndarray,
        candidates: np.ndarray,
        output: np.ndarray,
    ) -> None:
        full_scores, recent_scores = self._deterministic_summary(cache_key, history)
        for col_idx, candidate in enumerate(candidates):
            candidate_int = int(candidate)
            full = full_scores.get(candidate_int)
            if full is not None:
                output[col_idx, 0] = np.float32(full[0])
                output[col_idx, 1] = np.float32(full[1])
                output[col_idx, 2] = np.float32(full[2])
                output[col_idx, 3] = np.float32(full[3])
            recent = recent_scores.get(candidate_int)
            if recent is not None:
                output[col_idx, 4] = np.float32(recent[0])
                output[col_idx, 5] = np.float32(recent[1])

    def _deterministic_summary(self, cache_key: tuple[int, int], history: np.ndarray) -> DeterministicSummary:
        cached = self._deterministic_cache.get(cache_key)
        if cached is not None:
            self._deterministic_cache.move_to_end(cache_key)
            return cached

        recent_k = max(int(self.config.recent_k), 1)
        recent_start = max(history.size - recent_k, 0)
        full_scores: dict[int, tuple[float, float, float, float]] = {}
        recent_scores: dict[int, tuple[float, float]] = {}
        sparse = self.item_pair_counts_sparse
        for history_idx, seen in enumerate(history):
            seen_int = int(seen)
            row = sparse.get_row(seen_int)
            if row is None:
                continue
            cols, cooccurs = row
            mask = (cols != seen_int) & (cooccurs > 0)
            if not np.any(mask):
                continue
            cols, cooccurs = cols[mask], cooccurs[mask]
            values = np.log1p(cooccurs.astype(np.float32))
            seen_degree = max(self.item_degrees.get(seen_int, 0), 1)
            cand_degrees = np.array([max(self.item_degrees.get(int(c), 0), 1) for c in cols], dtype=np.float32)
            cosines = cooccurs.astype(np.float32) / np.sqrt(seen_degree * cand_degrees)
            is_recent = history_idx >= recent_start
            for i in range(len(cols)):
                cand = int(cols[i])
                val = float(values[i])
                cos = float(cosines[i])
                t, mx, ct, cm = full_scores.get(cand, (0.0, 0.0, 0.0, 0.0))
                full_scores[cand] = (t + val, max(mx, val), ct + cos, max(cm, cos))
                if is_recent:
                    rt, rm = recent_scores.get(cand, (0.0, 0.0))
                    recent_scores[cand] = (rt + cos, max(rm, cos))
        summary = (full_scores, recent_scores)
        self._cache_put(self._deterministic_cache, cache_key, summary)
        return summary

    def _fill_item2vec_features(
        self,
        history: np.ndarray,
        candidates: np.ndarray,
        output: np.ndarray,
        cache_key: tuple[int, int] | None = None,
        use_cache: bool = False,
    ) -> None:
        if self.embeddings is None:
            return
        full_profile, recent_profile = self._embedding_profiles(history, cache_key, use_cache)
        dst_ids = self.id_map.dst_ids(candidates)
        valid = dst_ids >= 0
        if np.any(valid):
            valid &= np.asarray([int(candidate) in self.index.dst_times for candidate in candidates], dtype=bool)
        if not np.any(valid):
            return

        valid_rows = np.flatnonzero(valid)
        item_vectors = self.embeddings[dst_ids[valid_rows]]
        scale = math.sqrt(self.embeddings.shape[1])
        output[valid_rows, 6] = np.asarray(item_vectors @ full_profile / scale, dtype=np.float32)
        output[valid_rows, 7] = _cosine_many(item_vectors, full_profile)
        output[valid_rows, 8] = np.asarray(item_vectors @ recent_profile / scale, dtype=np.float32)
        output[valid_rows, 9] = _cosine_many(item_vectors, recent_profile)

    def _embedding_profiles(
        self,
        history: np.ndarray,
        cache_key: tuple[int, int] | None,
        use_cache: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.embeddings is None:
            empty = np.empty(0, dtype=np.float32)
            return empty, empty
        if use_cache and cache_key is not None:
            cached = self._embedding_profile_cache.get(cache_key)
            if cached is not None:
                self._embedding_profile_cache.move_to_end(cache_key)
                return cached

        history_ids = self.id_map.dst_ids(history)
        history_ids = history_ids[history_ids >= 0]
        if history_ids.size == 0:
            profiles = (
                np.zeros(self.embeddings.shape[1], dtype=np.float32),
                np.zeros(self.embeddings.shape[1], dtype=np.float32),
            )
        else:
            recent_k = max(int(self.config.recent_k), 1)
            recent_ids = history_ids[-recent_k:]
            profiles = (
                _mean_embedding(self.embeddings, history_ids),
                _mean_embedding(self.embeddings, recent_ids),
            )
        if use_cache and cache_key is not None:
            self._cache_put(self._embedding_profile_cache, cache_key, profiles)
        return profiles

    def _cooccur_count(self, left: int, right: int) -> int:
        if left == right:
            return 0
        return self.item_pair_counts.get(int(left), {}).get(int(right), 0)

    def _cosine_from_count(self, cooccur: int, left: int, right: int, right_degree: int) -> float:
        left_degree = self.item_degrees.get(int(left), 0)
        denominator = math.sqrt(max(left_degree, 1) * max(right_degree, 1))
        return float(cooccur) / max(denominator, EPSILON)

    def _clear_score_caches(self) -> None:
        self._deterministic_cache.clear()
        self._embedding_profile_cache.clear()

    @staticmethod
    def _cache_put(cache: OrderedDict, key, value) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > SOURCE_PROFILE_CACHE_LIMIT:
            cache.popitem(last=False)


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


def _cosine_many(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    right_norm = float(np.linalg.norm(right))
    if right_norm <= EPSILON:
        return np.zeros(left.shape[0], dtype=np.float32)
    left_norm = np.linalg.norm(left, axis=1)
    denominator = left_norm * right_norm
    result = np.zeros(left.shape[0], dtype=np.float32)
    valid = denominator > EPSILON
    if np.any(valid):
        result[valid] = np.asarray(left[valid] @ right / denominator[valid], dtype=np.float32)
    return result
