from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

import jittor as jt
import numpy as np

from .data import Interaction, TestQuery


@dataclass
class SrcHistory:
    total: int = 0
    last_time: int = 0
    dst_counts: Counter[int] = field(default_factory=Counter)
    recent_dsts: deque[int] = field(default_factory=lambda: deque(maxlen=32))


class HeuristicJittorRanker:
    """Lightweight temporal graph candidate ranker.

    This MVP keeps the modeling deterministic and scalable: historical graph
    statistics are computed on CPU, then candidate features are fused through a
    small Jittor tensor expression. It satisfies the contest's Jittor framework
    requirement while leaving a clear upgrade path to trainable Jittor modules.
    """

    def __init__(self, recent_window: int = 32) -> None:
        self.recent_window = recent_window
        self.src_histories: dict[int, SrcHistory] = {}
        self.dst_counts: Counter[int] = Counter()
        self.dst_recent_time: dict[int, int] = {}
        self.pair_recent_time: dict[tuple[int, int], int] = {}
        self.pair_counts: Counter[tuple[int, int]] = Counter()
        self.total_edges = 0
        self.min_time = 0
        self.max_time = 0
        self._weights = jt.array([3.25, 1.35, 0.90, 0.55, 0.45, 0.20], dtype=jt.float32)

    def fit(self, interactions: list[Interaction]) -> None:
        if not interactions:
            raise ValueError("training interactions are empty")

        interactions.sort(key=lambda item: item.time)
        self.total_edges = len(interactions)
        self.min_time = interactions[0].time
        self.max_time = interactions[-1].time

        histories: dict[int, SrcHistory] = defaultdict(lambda: SrcHistory(recent_dsts=deque(maxlen=self.recent_window)))
        for item in interactions:
            history = histories[item.src]
            history.total += 1
            history.last_time = item.time
            history.dst_counts[item.dst] += 1
            history.recent_dsts.append(item.dst)

            self.dst_counts[item.dst] += 1
            self.dst_recent_time[item.dst] = item.time
            pair = (item.src, item.dst)
            self.pair_counts[pair] += 1
            self.pair_recent_time[pair] = item.time

        self.src_histories = dict(histories)

    def predict(self, query: TestQuery) -> np.ndarray:
        return self.predict_batch([query])[0]

    def predict_batch(self, queries: list[TestQuery]) -> np.ndarray:
        if not queries:
            return np.empty((0, 100), dtype=np.float64)

        features = np.stack([self._features(query) for query in queries], axis=0)
        with jt.no_grad():
            logits = (jt.array(features, dtype=jt.float32) * self._weights).sum(dim=2)
            logits = logits - logits.max(dim=1, keepdims=True)
            probs = jt.nn.softmax(logits, dim=1)
            return np.asarray(probs.numpy(), dtype=np.float64)

    def _features(self, query: TestQuery) -> np.ndarray:
        history = self.src_histories.get(query.src)
        src_total = history.total if history is not None else 0
        graph_span = max(self.max_time - self.min_time, 1)

        rows: list[list[float]] = []
        recent_rank: dict[int, int] = {}
        if history is not None:
            for rank, dst in enumerate(reversed(history.recent_dsts), start=1):
                recent_rank.setdefault(dst, rank)

        for dst in query.candidates:
            pair = (query.src, dst)
            pair_count = self.pair_counts.get(pair, 0)
            pair_last_time = self.pair_recent_time.get(pair)
            dst_count = self.dst_counts.get(dst, 0)
            dst_last_time = self.dst_recent_time.get(dst)

            repeat_rate = pair_count / max(src_total, 1)
            pair_strength = math.log1p(pair_count)
            dst_popularity = math.log1p(dst_count) / math.log1p(max(self.total_edges, 1))

            if pair_last_time is None:
                pair_recency = 0.0
            else:
                pair_recency = math.exp(-max(query.time - pair_last_time, 0) / graph_span)

            if dst_last_time is None:
                dst_recency = 0.0
            else:
                dst_recency = math.exp(-max(query.time - dst_last_time, 0) / graph_span)

            rank = recent_rank.get(dst)
            recent_hit = 0.0 if rank is None else 1.0 / rank

            rows.append(
                [
                    pair_strength,
                    repeat_rate,
                    pair_recency,
                    dst_popularity,
                    dst_recency,
                    recent_hit,
                ]
            )

        return np.asarray(rows, dtype=np.float32)
