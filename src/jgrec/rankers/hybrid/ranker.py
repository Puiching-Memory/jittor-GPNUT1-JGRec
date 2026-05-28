from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from jgrec.core.types import FitContext, Interaction, TestQuery, TrainingReport
from jgrec.idmap import NodeIdMap
from jgrec.logging import log

from .fusion import FusionConfig, FusionMLP, FusionResult, fit_fusion_mlp, predict_logits
from .gnn import GRAPH_WINDOW_NAMES, GraphTower, GraphTowerConfig
from .sequence import SEQUENCE_FEATURE_NAMES, SequenceTower, SequenceTowerConfig
from .stats import STAT_FEATURE_NAMES, TemporalStats


@dataclass(frozen=True)
class TrainingConfig:
    val_ratio: float = 0.15
    context_ratio: float = 0.75
    max_train_events: int = 20_000
    max_val_events: int = 5_000
    num_negatives: int = 31
    max_fit_events: int = 0
    epochs: int = 5
    train_batch_size: int = 512
    lr: float = 0.001
    weight_decay: float = 0.0
    selection_metric: str = "ap"
    early_stop_patience: int = 10
    seed: int = 42
    verbose: bool = True
    gnn_enabled: bool = True
    gnn_model: str = "xsimgcl"
    gnn_embedding_dim: int = 128
    gnn_layers: int = 2
    gnn_epochs: int = 3
    gnn_batch_size: int = 2048
    gnn_max_graph_edges: int = 0
    gnn_max_train_edges: int = 40_000
    gnn_lr: float = 0.001
    gnn_reg_weight: float = 1e-5
    gnn_cl_rate: float = 1e-4
    seq_enabled: bool = True
    seq_epochs: int = 3
    seq_batch_size: int = 512
    seq_max_samples: int = 50_000
    seq_max_len: int = 64
    seq_hidden_size: int = 128
    seq_layers: int = 2
    seq_heads: int = 4
    seq_dropout: float = 0.2
    fusion_hidden_dim: int = 64

    def graph_config(self) -> GraphTowerConfig:
        return GraphTowerConfig(
            enabled=self.gnn_enabled,
            model_name=self.gnn_model,
            embedding_dim=self.gnn_embedding_dim,
            layers=self.gnn_layers,
            epochs=self.gnn_epochs,
            batch_size=self.gnn_batch_size,
            max_graph_edges=self.gnn_max_graph_edges,
            max_train_edges=self.gnn_max_train_edges,
            lr=self.gnn_lr,
            weight_decay=self.weight_decay,
            reg_weight=self.gnn_reg_weight,
            cl_rate=self.gnn_cl_rate,
        )

    def sequence_config(self) -> SequenceTowerConfig:
        return SequenceTowerConfig(
            enabled=self.seq_enabled,
            epochs=self.seq_epochs,
            batch_size=self.seq_batch_size,
            max_samples=self.seq_max_samples,
            max_seq_len=self.seq_max_len,
            hidden_size=self.seq_hidden_size,
            layers=self.seq_layers,
            heads=self.seq_heads,
            dropout=self.seq_dropout,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def fusion_config(self) -> FusionConfig:
        return FusionConfig(
            epochs=self.epochs,
            batch_size=self.train_batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            hidden_dim=self.fusion_hidden_dim,
            selection_metric=self.selection_metric,
            early_stop_patience=self.early_stop_patience,
        )


class HybridFeatureEncoder:
    def __init__(
        self,
        id_map: NodeIdMap,
        recent_window: int,
        graph_config: GraphTowerConfig,
        sequence_config: SequenceTowerConfig,
    ) -> None:
        self.id_map = id_map
        self.stats = TemporalStats(recent_window=recent_window)
        self.graph = GraphTower(id_map=id_map, config=graph_config)
        self.sequence = SequenceTower(id_map=id_map, config=sequence_config)
        self.feature_names = STAT_FEATURE_NAMES + GRAPH_WINDOW_NAMES + SEQUENCE_FEATURE_NAMES

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def fit(self, interactions: list[Interaction], rng: np.random.Generator, verbose: bool) -> None:
        self.stats.fit(interactions)
        self.graph.fit(interactions, rng=rng, verbose=verbose)
        self.sequence.fit(interactions, rng=rng, verbose=verbose)

    def features_for_queries(self, queries: list[TestQuery]) -> np.ndarray:
        if not queries:
            return np.empty((0, 0, self.feature_dim), dtype=np.float32)
        stat_features = self.stats.features_for_queries(queries)
        graph_features = self.graph.scores_for_queries(queries)
        sequence_features = self.sequence.scores_for_queries(queries)
        return np.concatenate([stat_features, graph_features, sequence_features], axis=2).astype(np.float32, copy=False)


class TemporalHybridRanker:
    """Aggressive GNN/sequence/statistics hybrid candidate reranker."""

    def __init__(self, recent_window: int = 32) -> None:
        self.recent_window = recent_window
        self.id_map: NodeIdMap | None = None
        self.encoder: HybridFeatureEncoder | None = None
        self.fusion: FusionMLP | None = None
        self.fusion_result: FusionResult | None = None
        self.training_report: TrainingReport | None = None
        self.feature_names: tuple[str, ...] = ()
        self._fusion_hidden_dim = 64

    def fit(self, interactions: list[Interaction], training_config: TrainingConfig) -> TrainingReport:
        if not interactions:
            raise ValueError("training interactions are empty")

        interactions.sort(key=lambda item: item.time)
        if training_config.max_fit_events > 0 and len(interactions) > training_config.max_fit_events:
            interactions = interactions[-training_config.max_fit_events :]
        self.id_map = NodeIdMap.from_interactions(interactions)
        self.feature_names = STAT_FEATURE_NAMES + GRAPH_WINDOW_NAMES + SEQUENCE_FEATURE_NAMES
        self._fusion_hidden_dim = training_config.fusion_hidden_dim

        fusion, fusion_result, report = self._learn_fusion(interactions, training_config)
        self.fusion = fusion
        self.fusion_result = fusion_result

        rng = np.random.default_rng(training_config.seed + 10_000)
        final_config = _config_for_selected_features(training_config, fusion_result.feature_indices)
        self.encoder = self._fit_encoder(interactions, final_config, rng, verbose=training_config.verbose)
        self.training_report = report
        return report

    def predict(self, query: TestQuery) -> np.ndarray:
        return self.predict_batch([query])[0]

    def predict_batch(self, queries: list[TestQuery]) -> np.ndarray:
        if not queries:
            return np.empty((0, 100), dtype=np.float64)
        if self.encoder is None or self.fusion is None or self.fusion_result is None:
            raise RuntimeError("ranker is not fitted")

        features = self.encoder.features_for_queries(queries)
        if self.fusion_result.feature_indices:
            features = features[:, :, self.fusion_result.feature_indices]
        logits = predict_logits(self.fusion, features, self.fusion_result.mean, self.fusion_result.std)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return probs.astype(np.float64, copy=False)

    def _learn_fusion(
        self,
        interactions: list[Interaction],
        config: TrainingConfig,
    ) -> tuple[FusionMLP, FusionResult, TrainingReport]:
        n_events = len(interactions)
        if n_events < 100 or config.num_negatives < 1 or config.epochs < 1:
            raise ValueError(
                "not enough training signal for hybrid reranker: "
                f"events={n_events}, num_negatives={config.num_negatives}, epochs={config.epochs}"
            )

        rng = np.random.default_rng(config.seed)
        val_size = max(1, int(n_events * config.val_ratio))
        train_end = max(2, n_events - val_size)
        context_end = max(1, min(train_end - 1, int(train_end * config.context_ratio)))

        context_events = interactions[:context_end]
        train_events = interactions[context_end:train_end]
        val_context_events = interactions[:train_end]
        val_events = interactions[train_end:]
        if not train_events or not val_events:
            raise ValueError(
                "invalid temporal split for hybrid reranker: "
                f"context={len(context_events)}, train={len(train_events)}, val={len(val_events)}"
            )

        train_events = _sample_events(train_events, config.max_train_events, rng)
        val_events = _sample_events(val_events, config.max_val_events, rng)
        dst_pool = np.asarray(sorted({item.dst for item in interactions}), dtype=np.int64)

        train_encoder = self._fit_encoder(context_events, config, rng, verbose=config.verbose)
        train_queries = _build_supervised_queries(train_events, train_encoder, dst_pool, config.num_negatives, rng)
        train_features = train_encoder.features_for_queries(train_queries)

        val_encoder = self._fit_encoder(val_context_events, config, rng, verbose=config.verbose)
        val_queries = _build_supervised_queries(val_events, val_encoder, dst_pool, config.num_negatives, rng)
        val_features = val_encoder.features_for_queries(val_queries)

        fusion, result = self._fit_best_fusion(
            train_features=train_features,
            val_features=val_features,
            config=config,
            rng=rng,
            verbose=config.verbose,
        )
        report = TrainingReport(
            train_events=len(train_queries),
            val_events=len(val_queries),
            best_val_ap=result.best_val_ap,
            best_val_mrr=result.best_val_mrr,
            feature_names=tuple(self.feature_names[idx] for idx in result.feature_indices),
            selected_fusion=result.candidate_name,
            model_name="hybrid",
        )
        return fusion, result, report

    def _fit_best_fusion(
        self,
        train_features: np.ndarray,
        val_features: np.ndarray,
        config: TrainingConfig,
        rng: np.random.Generator,
        verbose: bool,
    ) -> tuple[FusionMLP, FusionResult]:
        masks = _feature_masks(train_features.shape[-1])
        best_model: FusionMLP | None = None
        best_result: FusionResult | None = None
        for name, indices in masks:
            candidate_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
            model, result = fit_fusion_mlp(
                train_features=train_features[:, :, indices],
                val_features=val_features[:, :, indices],
                config=config.fusion_config(),
                rng=candidate_rng,
                verbose=verbose,
                feature_indices=indices,
                candidate_name=name,
            )
            selected_score = _selected_report_metric(result, config.selection_metric)
            log(
                f"[fusion-select] candidate={name} "
                f"val_ap={result.best_val_ap:.5f} val_mrr={result.best_val_mrr:.5f}",
                enabled=verbose,
            )
            if best_result is None or selected_score >= _selected_report_metric(best_result, config.selection_metric):
                best_model = model
                best_result = result

        if best_model is None or best_result is None:
            raise RuntimeError("no fusion candidate was trained")
        log(
            f"[fusion-select] chosen={best_result.candidate_name} "
            f"best_ap={best_result.best_val_ap:.5f} best_mrr={best_result.best_val_mrr:.5f}",
            enabled=verbose,
        )
        return best_model, best_result

    def _fit_encoder(
        self,
        interactions: list[Interaction],
        config: TrainingConfig,
        rng: np.random.Generator,
        verbose: bool,
    ) -> HybridFeatureEncoder:
        if self.id_map is None:
            raise RuntimeError("id map is not initialized")
        encoder = HybridFeatureEncoder(
            id_map=self.id_map,
            recent_window=self.recent_window,
            graph_config=config.graph_config(),
            sequence_config=config.sequence_config(),
        )
        encoder.fit(interactions, rng=rng, verbose=verbose)
        return encoder


def _sample_events(
    events: list[Interaction],
    max_events: int,
    rng: np.random.Generator,
) -> list[Interaction]:
    if max_events <= 0 or len(events) <= max_events:
        return list(events)
    indices = np.sort(rng.choice(len(events), size=max_events, replace=False))
    return [events[int(index)] for index in indices]


def _feature_masks(feature_count: int) -> list[tuple[str, tuple[int, ...]]]:
    stats_end = len(STAT_FEATURE_NAMES)
    graph_end = stats_end + len(GRAPH_WINDOW_NAMES)
    masks = [("stats", tuple(range(min(stats_end, feature_count))))]
    if feature_count > stats_end:
        masks.append(("stats_gnn", tuple(range(min(graph_end, feature_count)))))
    if feature_count > graph_end:
        masks.append(("stats_gnn_seq", tuple(range(feature_count))))

    unique: list[tuple[str, tuple[int, ...]]] = []
    seen: set[tuple[int, ...]] = set()
    for name, indices in masks:
        if not indices or indices in seen:
            continue
        seen.add(indices)
        unique.append((name, indices))
    return unique


def _selected_report_metric(result: FusionResult, metric: str) -> float:
    normalized = metric.lower()
    if normalized == "ap":
        return result.best_val_ap
    if normalized == "mrr":
        return result.best_val_mrr
    raise ValueError(f"unsupported fusion selection metric: {metric}")


def _config_for_selected_features(config: TrainingConfig, feature_indices: tuple[int, ...]) -> TrainingConfig:
    stats_end = len(STAT_FEATURE_NAMES)
    graph_end = stats_end + len(GRAPH_WINDOW_NAMES)
    needs_graph = any(stats_end <= idx < graph_end for idx in feature_indices)
    needs_sequence = any(idx >= graph_end for idx in feature_indices)
    return replace(config, gnn_enabled=config.gnn_enabled and needs_graph, seq_enabled=config.seq_enabled and needs_sequence)


def _build_supervised_queries(
    positives: list[Interaction],
    encoder: HybridFeatureEncoder,
    dst_pool: np.ndarray,
    num_negatives: int,
    rng: np.random.Generator,
) -> list[TestQuery]:
    queries: list[TestQuery] = []
    for event in positives:
        negatives = _sample_negatives(event.src, event.dst, encoder, dst_pool, num_negatives, rng)
        candidates = (event.dst, *negatives)
        queries.append(TestQuery(src=event.src, time=event.time, candidates=candidates))
    return queries


def _sample_negatives(
    src: int,
    positive_dst: int,
    encoder: HybridFeatureEncoder,
    dst_pool: np.ndarray,
    num_negatives: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    used = {positive_dst}
    history = encoder.stats.src_histories.get(src)
    if history is not None:
        used.update(history.dst_counts)

    negatives: list[int] = []
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
    return tuple(negatives)


class HybridRankerAdapter:
    name = "hybrid"

    def __init__(self, config: TrainingConfig | None = None, recent_window: int = 32) -> None:
        self.config = config or TrainingConfig()
        self.recent_window = recent_window
        self.impl = TemporalHybridRanker(recent_window=recent_window)

    def fit(self, interactions: list[Interaction], context: FitContext) -> TrainingReport:
        config = replace(
            self.config,
            seed=context.seed,
            verbose=context.verbose,
        )
        return self.impl.fit(interactions, training_config=config)

    def predict_batch(self, queries: list[TestQuery]) -> np.ndarray:
        return self.impl.predict_batch(queries)
