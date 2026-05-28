from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from jgrec.core.types import FitContext, Interaction, TestQuery, TrainingReport
from jgrec.rankers.common import FusionConfig, FusionMLP, FusionResult, fit_fusion_mlp, predict_logits

from .features import THIRD_PARTY_FEATURE_NAMES, ThirdPartyFeatureExtractor
from .indexes import TemporalIndexConfig, TemporalIndexes


@dataclass(frozen=True)
class ThirdPartyRankerConfig:
    val_ratio: float = 0.15
    context_ratio: float = 0.75
    max_train_events: int = 20_000
    max_val_events: int = 5_000
    num_negatives: int = 63
    epochs: int = 8
    batch_size: int = 512
    lr: float = 0.001
    weight_decay: float = 0.0
    hidden_dim: int = 96
    selection_metric: str = "ap"
    early_stop_patience: int = 10
    seed: int = 42
    verbose: bool = True
    recent_window: int = 64
    cooccur_recent_k: int = 16
    cooccur_max_sources: int = 500_000

    def index_config(self) -> TemporalIndexConfig:
        return TemporalIndexConfig(
            recent_window=self.recent_window,
            cooccur_recent_k=self.cooccur_recent_k,
            cooccur_max_sources=self.cooccur_max_sources,
        )

    def fusion_config(self) -> FusionConfig:
        return FusionConfig(
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            hidden_dim=self.hidden_dim,
            selection_metric=self.selection_metric,
            early_stop_patience=self.early_stop_patience,
        )


class ThirdPartyRanker:
    name = "third_party"

    def __init__(self, config: ThirdPartyRankerConfig | None = None) -> None:
        self.config = config or ThirdPartyRankerConfig()
        self.indexes: TemporalIndexes | None = None
        self.extractor: ThirdPartyFeatureExtractor | None = None
        self.fusion: FusionMLP | None = None
        self.fusion_result: FusionResult | None = None

    def fit(self, interactions: list[Interaction], context: FitContext) -> TrainingReport:
        if not interactions:
            raise ValueError("training interactions are empty")
        config = replace(self.config, seed=context.seed, verbose=context.verbose)
        interactions = sorted(interactions, key=lambda item: item.time)
        rng = np.random.default_rng(config.seed)

        n_events = len(interactions)
        val_size = max(1, int(n_events * config.val_ratio))
        train_end = max(2, n_events - val_size)
        context_end = max(1, min(train_end - 1, int(train_end * config.context_ratio)))
        context_events = interactions[:context_end]
        train_events = _sample_events(interactions[context_end:train_end], config.max_train_events, rng)
        val_context_events = interactions[:train_end]
        val_events = _sample_events(interactions[train_end:], config.max_val_events, rng)
        if not train_events or not val_events:
            raise ValueError(
                "invalid temporal split for third_party ranker: "
                f"context={len(context_events)}, train={len(train_events)}, val={len(val_events)}"
            )

        dst_pool = np.asarray(sorted({item.dst for item in interactions}), dtype=np.int64)
        train_extractor = _fit_extractor(context_events, config)
        train_queries = _build_supervised_queries(train_events, train_extractor.indexes, dst_pool, config.num_negatives, rng)
        train_features = train_extractor.features_for_queries(train_queries)

        val_extractor = _fit_extractor(val_context_events, config)
        val_queries = _build_supervised_queries(val_events, val_extractor.indexes, dst_pool, config.num_negatives, rng)
        val_features = val_extractor.features_for_queries(val_queries)

        self.fusion, self.fusion_result = fit_fusion_mlp(
            train_features=train_features,
            val_features=val_features,
            config=config.fusion_config(),
            rng=rng,
            verbose=config.verbose,
            feature_indices=tuple(range(len(THIRD_PARTY_FEATURE_NAMES))),
            candidate_name="third_party_stats",
        )
        self.extractor = _fit_extractor(interactions, config)
        self.indexes = self.extractor.indexes
        return TrainingReport(
            train_events=len(train_queries),
            val_events=len(val_queries),
            best_val_ap=self.fusion_result.best_val_ap,
            best_val_mrr=self.fusion_result.best_val_mrr,
            feature_names=THIRD_PARTY_FEATURE_NAMES,
            selected_fusion=self.fusion_result.candidate_name,
            model_name=self.name,
        )

    def predict_batch(self, queries: list[TestQuery]) -> np.ndarray:
        if not queries:
            return np.empty((0, 100), dtype=np.float64)
        if self.extractor is None or self.fusion is None or self.fusion_result is None:
            raise RuntimeError("ranker is not fitted")
        features = self.extractor.features_for_queries(queries)
        logits = predict_logits(self.fusion, features, self.fusion_result.mean, self.fusion_result.std)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return probs.astype(np.float64, copy=False)


def _fit_extractor(interactions: list[Interaction], config: ThirdPartyRankerConfig) -> ThirdPartyFeatureExtractor:
    indexes = TemporalIndexes(config.index_config())
    indexes.fit(interactions)
    return ThirdPartyFeatureExtractor(indexes)


def _sample_events(events: list[Interaction], max_events: int, rng: np.random.Generator) -> list[Interaction]:
    if max_events <= 0 or len(events) <= max_events:
        return list(events)
    indices = np.sort(rng.choice(len(events), size=max_events, replace=False))
    return [events[int(index)] for index in indices]


def _build_supervised_queries(
    positives: list[Interaction],
    indexes: TemporalIndexes,
    dst_pool: np.ndarray,
    num_negatives: int,
    rng: np.random.Generator,
) -> list[TestQuery]:
    popular = np.asarray(
        [
            dst
            for dst, _ in sorted(
                indexes.destinations.items(),
                key=lambda pair: pair[1].total,
                reverse=True,
            )[: max(num_negatives * 20, 100)]
        ],
        dtype=np.int64,
    )
    queries: list[TestQuery] = []
    for event in positives:
        negatives = _sample_negatives(event.src, event.dst, indexes, dst_pool, popular, num_negatives, rng)
        queries.append(TestQuery(src=event.src, time=event.time, candidates=(event.dst, *negatives)))
    return queries


def _sample_negatives(
    src: int,
    positive_dst: int,
    indexes: TemporalIndexes,
    dst_pool: np.ndarray,
    popular_pool: np.ndarray,
    num_negatives: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    used = {positive_dst}
    negatives: list[int] = []
    source = indexes.sources.get(src)
    if source is not None:
        hard_candidates: list[int] = []
        for recent_dst in reversed(source.recent_dsts):
            hard_candidates.extend(indexes.cooccur.get(recent_dst, {}).keys())
            hard_candidates.extend(indexes.transition.get(recent_dst, {}).keys())
        rng.shuffle(hard_candidates)
        for value in hard_candidates:
            dst = int(value)
            if dst in used:
                continue
            used.add(dst)
            negatives.append(dst)
            if len(negatives) >= max(1, num_negatives // 4):
                break
        used.update(source.dst_counts)

    target_popular = max(1, num_negatives // 3)
    if popular_pool.size:
        for value in rng.choice(popular_pool, size=min(popular_pool.size, target_popular * 3), replace=False):
            dst = int(value)
            if dst in used:
                continue
            used.add(dst)
            negatives.append(dst)
            if len(negatives) >= target_popular + max(1, num_negatives // 4):
                break

    attempts = 0
    while len(negatives) < num_negatives and attempts < 20:
        attempts += 1
        draw_size = max((num_negatives - len(negatives)) * 3, 16)
        sampled = rng.choice(dst_pool, size=draw_size, replace=len(dst_pool) < draw_size)
        for value in sampled:
            dst = int(value)
            if dst in used:
                continue
            used.add(dst)
            negatives.append(dst)
            if len(negatives) >= num_negatives:
                break

    if len(negatives) < num_negatives:
        for value in dst_pool:
            dst = int(value)
            if dst in used:
                continue
            used.add(dst)
            negatives.append(dst)
            if len(negatives) >= num_negatives:
                break
    if len(negatives) < num_negatives:
        negatives.extend([positive_dst] * (num_negatives - len(negatives)))
    return tuple(negatives[:num_negatives])

