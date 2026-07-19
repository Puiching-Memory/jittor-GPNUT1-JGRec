from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from jgrec.core.memory import release_memory
from jgrec.core.types import InteractionTable, TestQuery, TestQueryArray
from jgrec.idmap import NodeIdMap
from jgrec.logging import log, track
from jgrec.rankers.common.byte_budget_lru import ByteBudgetLRU
from jgrec.rankers.common.early_stop import LossEarlyStopper
from jgrec.rankers.common.sparse_counts import SparseCountMap
from jgrec.rankers.common.temporal_index import TemporalInteractionIndex

from .config import SOURCE_PROFILE_FEATURE_NAMES, SourceProfileConfig

SOURCE_PROFILE_FEATURE_DIM = len(SOURCE_PROFILE_FEATURE_NAMES)
DETERMINISTIC_FEATURE_DIM = 6
ITEM2VEC_FEATURE_DIM = 4
EPSILON = 1e-8
SOURCE_PROFILE_CACHE_LIMIT = 256
SOURCE_PROFILE_CACHE_MIN_HISTORY = 32


@dataclass(frozen=True)
class _CompactDeterministicSummary:
    full_candidate_ids: np.ndarray
    full_values: np.ndarray
    recent_candidate_ids: np.ndarray
    recent_values: np.ndarray

    @classmethod
    def from_dicts(
        cls,
        full_scores: dict[int, tuple[float, float, float, float]],
        recent_scores: dict[int, tuple[float, float]],
    ) -> _CompactDeterministicSummary:
        full_ids = np.asarray(sorted(full_scores), dtype=np.int64)
        recent_ids = np.asarray(sorted(recent_scores), dtype=np.int64)
        full_values = np.asarray([full_scores[int(key)] for key in full_ids], dtype=np.float64).reshape((-1, 4))
        recent_values = np.asarray([recent_scores[int(key)] for key in recent_ids], dtype=np.float64).reshape((-1, 2))
        return cls(full_ids, full_values, recent_ids, recent_values)

    @property
    def nbytes(self) -> int:
        return sum(
            array.nbytes
            for array in (self.full_candidate_ids, self.full_values, self.recent_candidate_ids, self.recent_values)
        )

    def fill_candidates(self, candidates: np.ndarray, output: np.ndarray) -> None:
        _fill_matching_rows(self.full_candidate_ids, self.full_values, candidates, output[:, :4])
        _fill_matching_rows(self.recent_candidate_ids, self.recent_values, candidates, output[:, 4:6])


def _fill_matching_rows(
    keys: np.ndarray,
    values: np.ndarray,
    candidates: np.ndarray,
    output: np.ndarray,
) -> None:
    if keys.size == 0 or candidates.size == 0:
        return
    positions = np.searchsorted(keys, candidates)
    in_bounds = positions < len(keys)
    rows = np.flatnonzero(in_bounds)
    rows = rows[keys[positions[rows]] == candidates[rows]]
    output[rows] = values[positions[rows]]


def _jt():
    import jittor as jt  # noqa: PLC0415

    return jt


class SourceProfileTower:
    def __init__(self, id_map: NodeIdMap, config: SourceProfileConfig) -> None:
        self.id_map = id_map
        self.config = config
        self.index = TemporalInteractionIndex()
        self.item_pair_counts_sparse: SparseCountMap = SparseCountMap.empty()
        self._candidate_ids = np.asarray(self.id_map.dst_values, dtype=np.int64)
        self._item_degree_values = np.zeros(self._candidate_ids.size, dtype=np.int32)
        self._seen_candidate_mask = np.zeros(self._candidate_ids.size, dtype=bool)
        self.embeddings: np.ndarray | None = None
        total_cache_bytes = max(int(self.config.cache_max_bytes), 0)
        embedding_cache_bytes = total_cache_bytes // 8
        deterministic_cache_bytes = total_cache_bytes - embedding_cache_bytes
        self._deterministic_cache: ByteBudgetLRU[tuple[int, int], _CompactDeterministicSummary] = ByteBudgetLRU(
            max_bytes=deterministic_cache_bytes,
            max_entries=SOURCE_PROFILE_CACHE_LIMIT,
        )
        self._embedding_profile_cache: ByteBudgetLRU[tuple[int, int], tuple[np.ndarray, np.ndarray]] = ByteBudgetLRU(
            max_bytes=embedding_cache_bytes,
            max_entries=SOURCE_PROFILE_CACHE_LIMIT,
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return SOURCE_PROFILE_FEATURE_NAMES

    @property
    def cache_bytes(self) -> int:
        return self._deterministic_cache.current_bytes + self._embedding_profile_cache.current_bytes

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
        self._refresh_seen_candidates()
        if self.config.deterministic_enabled and not deterministic_ready:
            self.fit_deterministic(interactions)
        elif not self.config.deterministic_enabled:
            self._item_degree_values.fill(0)
        if self.config.item2vec_enabled and self.config.epochs > 0 and self.id_map.num_dst >= 2:
            self._fit_item2vec(interactions, rng=rng, verbose=verbose)
        else:
            self.embeddings = None
        self._clear_score_caches()

    def snapshot(self) -> dict[str, Any]:
        item_degrees = {
            int(candidate): int(degree)
            for candidate, degree in zip(self._candidate_ids, self._item_degree_values, strict=True)
            if degree > 0
        }
        return {
            "index": self.index.shallow_copy(),
            "item_pair_counts": self.item_pair_counts_sparse.snapshot(),
            "item_degrees": item_degrees,
            "embeddings": None if self.embeddings is None else self.embeddings.copy(),
        }

    def hydrate(self, snapshot: dict[str, Any]) -> None:
        index = snapshot.get("index")
        self.index = TemporalInteractionIndex() if index is None else index.shallow_copy()
        self._refresh_seen_candidates()
        self.item_pair_counts_sparse = SparseCountMap.from_snapshot(snapshot["item_pair_counts"])
        self._set_item_degrees(snapshot["item_degrees"])
        embeddings = snapshot.get("embeddings")
        self.embeddings = None if embeddings is None else np.asarray(embeddings, dtype=np.float32).copy()
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
        nested = {left: dict(counts) for left, counts in pair_counts.items()}
        self.item_pair_counts_sparse = SparseCountMap.from_nested_dict(nested)
        self._set_item_degrees(degrees)

    def _fit_item2vec(self, interactions: InteractionTable, rng: np.random.Generator, verbose: bool) -> None:
        samples = _item2vec_samples(interactions, self.id_map, self.config, rng)
        if samples is None:
            self.embeddings = None
            return

        centers, positives, negatives = samples
        jt = _jt()
        model = _Item2VecModel(num_items=self.id_map.num_dst, embedding_dim=self.config.embedding_dim)
        optimizer = jt.nn.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        full_size = int(centers.shape[0])
        val_size = max(0, int(full_size * self.config.early_stop_val_ratio))
        val_perm = rng.permutation(full_size)
        val_idx = val_perm[:val_size]
        train_idx = val_perm[val_size:]
        train_size = train_idx.shape[0]
        stopper = LossEarlyStopper(patience=self.config.early_stop_patience)
        for epoch in track(
            range(1, self.config.epochs + 1), description="source-profile", total=self.config.epochs, enabled=verbose
        ):
            order = rng.permutation(train_size)
            losses: list[float] = []
            for start in range(0, train_size, max(int(self.config.batch_size), 1)):
                batch_idx = train_idx[order[start : start + int(self.config.batch_size)]]
                loss = self._item2vec_loss(model, centers, positives, negatives, batch_idx)
                optimizer.step(loss)
                losses.append(float(loss.item()))
            mean_loss = float(np.mean(losses)) if losses else 0.0
            val_loss = self._item2vec_val_loss(model, centers, positives, negatives, val_idx) if val_size > 0 else 0.0
            log(f"[source-profile] epoch={epoch} loss={mean_loss:.5f} val_loss={val_loss:.5f}", enabled=verbose)
            release_memory()
            stop_signal = val_loss if val_size > 0 else mean_loss
            if stopper.update(epoch, stop_signal, model):
                log(
                    f"[source-profile] early_stop epoch={epoch} best_epoch={stopper.best_epoch} best_val={stopper.best_loss:.5f}",
                    enabled=verbose,
                )
                break
        stopper.restore_best(model)

        with jt.no_grad():
            self.embeddings = np.asarray(model.item_embedding.weight.numpy(), dtype=np.float32)

    @staticmethod
    def _item2vec_loss(model, centers, positives, negatives, batch_idx):
        jt = _jt()
        center = jt.array(centers[batch_idx], dtype=jt.int32)
        pos = jt.array(positives[batch_idx], dtype=jt.int32)
        neg = jt.array(negatives[batch_idx], dtype=jt.int32)
        return model(center, pos, neg)

    def _item2vec_val_loss(self, model, centers, positives, negatives, val_idx) -> float:
        if val_idx.shape[0] == 0:
            return 0.0
        jt = _jt()
        with jt.no_grad():
            losses: list[float] = []
            for start in range(0, val_idx.shape[0], max(int(self.config.batch_size), 1)):
                batch_idx = val_idx[start : start + int(self.config.batch_size)]
                loss = self._item2vec_loss(model, centers, positives, negatives, batch_idx)
                losses.append(float(loss.item()))
            return float(np.mean(losses)) if losses else 0.0

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
        history_int32 = history.astype(np.int32, copy=False)
        history_degrees = self._degrees_for(history_int32)
        candidate_degrees = self._degrees_for(candidates)
        for col_idx, candidate in enumerate(candidates):
            candidate_int = int(candidate)
            candidate_degree = candidate_degrees[col_idx]
            counts = sparse.batch_get_counts(history_int32, candidate_int)
            mask = (counts > 0) & (history != candidate_int)
            if not np.any(mask):
                continue
            valid_counts = counts[mask].astype(np.float32)
            values = np.log1p(valid_counts)
            seen_degrees = history_degrees[mask]
            cosines = valid_counts / np.sqrt(seen_degrees * candidate_degree)
            output[col_idx, 0] = values.sum()
            output[col_idx, 1] = values.max()
            output[col_idx, 2] = cosines.sum()
            output[col_idx, 3] = cosines.max()
            recent_mask = mask[-recent_k:]
            if np.any(recent_mask):
                recent_counts = counts[-recent_k:][recent_mask].astype(np.float32)
                recent_degrees = history_degrees[-recent_k:][recent_mask]
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
        summary = self._deterministic_summary(cache_key, history)
        summary.fill_candidates(candidates, output)

    def _deterministic_summary(
        self,
        cache_key: tuple[int, int],
        history: np.ndarray,
    ) -> _CompactDeterministicSummary:
        cached = self._deterministic_cache.get(cache_key)
        if cached is not None:
            return cached

        summary = self._build_deterministic_summary(history)
        self._deterministic_cache.put(cache_key, summary, size_bytes=summary.nbytes)
        return summary

    def _build_deterministic_summary(self, history: np.ndarray) -> _CompactDeterministicSummary:
        recent_k = max(int(self.config.recent_k), 1)
        recent_start = max(history.size - recent_k, 0)
        candidate_ids = self._candidate_ids
        if candidate_ids.size == 0:
            return _CompactDeterministicSummary(
                full_candidate_ids=np.empty(0, dtype=np.int64),
                full_values=np.empty((0, 4), dtype=np.float64),
                recent_candidate_ids=np.empty(0, dtype=np.int64),
                recent_values=np.empty((0, 2), dtype=np.float64),
            )

        candidate_degrees = np.maximum(self._item_degree_values, 1).astype(np.float32)
        full_values = np.zeros((candidate_ids.size, 4), dtype=np.float64)
        recent_values = np.zeros((candidate_ids.size, 2), dtype=np.float64)
        full_touched = np.zeros(candidate_ids.size, dtype=bool)
        recent_touched = np.zeros(candidate_ids.size, dtype=bool)
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
            positions = np.searchsorted(candidate_ids, cols)
            valid = positions < candidate_ids.size
            valid_rows = np.flatnonzero(valid)
            valid[valid_rows] &= candidate_ids[positions[valid_rows]] == cols[valid_rows]
            if not np.all(valid):
                positions = positions[valid]
                cooccurs = cooccurs[valid]
            cooccurs_float = cooccurs.astype(np.float32)
            values = np.log1p(cooccurs_float)
            seen_degree = self._degree_for(seen_int)
            cosines = cooccurs_float / np.sqrt(seen_degree * candidate_degrees[positions])
            np.add.at(full_values[:, 0], positions, values)
            np.maximum.at(full_values[:, 1], positions, values)
            np.add.at(full_values[:, 2], positions, cosines)
            np.maximum.at(full_values[:, 3], positions, cosines)
            full_touched[positions] = True
            if history_idx >= recent_start:
                np.add.at(recent_values[:, 0], positions, cosines)
                np.maximum.at(recent_values[:, 1], positions, cosines)
                recent_touched[positions] = True

        return _CompactDeterministicSummary(
            full_candidate_ids=candidate_ids[full_touched],
            full_values=full_values[full_touched],
            recent_candidate_ids=candidate_ids[recent_touched],
            recent_values=recent_values[recent_touched],
        )

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
        dst_ids, valid = self._candidate_positions(candidates)
        valid_rows = np.flatnonzero(valid)
        valid_rows = valid_rows[self._seen_candidate_mask[dst_ids[valid_rows]]]
        if valid_rows.size == 0:
            return

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
                return cached

        history_ids, valid = self._candidate_positions(history)
        history_ids = history_ids[valid]
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
            profile_bytes = profiles[0].nbytes + profiles[1].nbytes
            self._embedding_profile_cache.put(cache_key, profiles, size_bytes=profile_bytes)
        return profiles

    def _cosine_from_count(self, cooccur: int, left: int, right: int, right_degree: int) -> float:
        left_degree = self._degree_for(left)
        denominator = math.sqrt(max(left_degree, 1) * max(right_degree, 1))
        return float(cooccur) / max(denominator, EPSILON)

    def _set_item_degrees(self, item_degrees: dict[int, int]) -> None:
        self._item_degree_values.fill(0)
        if not item_degrees or self._candidate_ids.size == 0:
            return
        raw_ids = np.fromiter(item_degrees, dtype=np.int64, count=len(item_degrees))
        degrees = np.fromiter(item_degrees.values(), dtype=np.int32, count=len(item_degrees))
        positions = np.searchsorted(self._candidate_ids, raw_ids)
        valid = positions < self._candidate_ids.size
        rows = np.flatnonzero(valid)
        rows = rows[self._candidate_ids[positions[rows]] == raw_ids[rows]]
        self._item_degree_values[positions[rows]] = degrees[rows]

    def _refresh_seen_candidates(self) -> None:
        self._seen_candidate_mask.fill(False)
        if not self.index.dst_times or self._candidate_ids.size == 0:
            return
        raw_ids = np.fromiter(self.index.dst_times, dtype=np.int64, count=len(self.index.dst_times))
        positions, valid = self._candidate_positions(raw_ids)
        self._seen_candidate_mask[positions[valid]] = True

    def _candidate_positions(self, raw_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw_ids = np.asarray(raw_ids)
        positions = np.searchsorted(self._candidate_ids, raw_ids)
        valid = positions < self._candidate_ids.size
        rows = np.flatnonzero(valid)
        valid[rows] &= self._candidate_ids[positions[rows]] == raw_ids[rows]
        return positions, valid

    def _degrees_for(self, raw_ids: np.ndarray) -> np.ndarray:
        raw_ids = np.asarray(raw_ids)
        degrees = np.ones(raw_ids.shape, dtype=np.float32)
        if raw_ids.size == 0 or self._candidate_ids.size == 0:
            return degrees
        positions, valid = self._candidate_positions(raw_ids)
        rows = np.flatnonzero(valid)
        degrees[rows] = np.maximum(self._item_degree_values[positions[rows]], 1)
        return degrees

    def _degree_for(self, raw_id: int) -> int:
        position = int(np.searchsorted(self._candidate_ids, raw_id))
        if position >= self._candidate_ids.size or self._candidate_ids[position] != raw_id:
            return 1
        return max(int(self._item_degree_values[position]), 1)

    def _clear_score_caches(self) -> None:
        self._deterministic_cache.clear()
        self._embedding_profile_cache.clear()


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
