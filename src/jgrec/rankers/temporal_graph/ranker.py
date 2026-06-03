from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from jgrec.core.io import read_test_queries
from jgrec.core.types import (
    INTERACTION_DST,
    INTERACTION_TIME,
    FitContext,
    InteractionArray,
    TestQueryArray,
    TrainingReport,
)
from jgrec.logging import log

from .index import TemporalNodeMap, safe_neighbor_sampler, temporal_data_from_interactions, temporal_loader_api
from .model import EndToEndTemporalGraphModel, TemporalGraphModelConfig
from .trainer import (
    TestCandidateIndex,
    fit_full_epochs,
    predict_logits,
    queries_to_prediction_batch,
    train_listwise,
)


@dataclass(frozen=True)
class TemporalGraphTrainingConfig:
    val_ratio: float = 0.15
    max_train_events: int = 20_000
    max_val_events: int = 5_000
    num_negatives: int = 99
    max_fit_events: int = 0
    epochs: int = 8
    train_batch_size: int = 256
    lr: float = 0.001
    weight_decay: float = 0.0
    selection_metric: str = "ap"
    early_stop_patience: int = 10
    seed: int = 42
    verbose: bool = True
    history_len: int = 64
    candidate_history_len: int = 32
    hidden_size: int = 128
    layers: int = 3
    heads: int = 4
    dropout: float = 0.15
    validation_candidates: str = "random"
    refit_full: bool = True


class TemporalGraphRanker:
    """End-to-end temporal graph candidate ranker."""

    def __init__(self) -> None:
        self.node_map: TemporalNodeMap | None = None
        self.model: EndToEndTemporalGraphModel | None = None
        self.neighbor_sampler = None
        self.config: TemporalGraphTrainingConfig | None = None
        self.training_report: TrainingReport | None = None

    def fit(
        self,
        interactions: InteractionArray,
        training_config: TemporalGraphTrainingConfig,
        context: FitContext,
    ) -> TrainingReport:
        if len(interactions) == 0:
            raise ValueError("training interactions are empty")
        interactions = interactions[np.argsort(interactions[:, INTERACTION_TIME], kind="stable")]
        if training_config.max_fit_events > 0 and len(interactions) > training_config.max_fit_events:
            interactions = interactions[-training_config.max_fit_events :]
        if len(interactions) < 4:
            raise ValueError("temporal graph ranker needs at least four interactions")

        self.config = training_config
        self.node_map = TemporalNodeMap.from_interactions_and_test(interactions, context.dataset.test_path)
        full_data = temporal_data_from_interactions(interactions, self.node_map)
        _, get_neighbor_sampler = temporal_loader_api()
        self.neighbor_sampler = safe_neighbor_sampler(get_neighbor_sampler(full_data, "recent", seed=training_config.seed))
        dst_pool = np.unique(self.node_map.dst_ids(interactions[:, INTERACTION_DST])).astype(np.int32, copy=False)
        dst_pool = dst_pool[dst_pool > 0]

        n_events = len(interactions)
        val_size = max(1, int(n_events * training_config.val_ratio))
        train_end = max(1, n_events - val_size)
        if train_end >= n_events:
            train_end = n_events - 1
        train_events = interactions[:train_end]
        val_events = interactions[train_end:]
        time_span = max(int(interactions[-1, INTERACTION_TIME]) - int(interactions[0, INTERACTION_TIME]), 1)

        rng = np.random.default_rng(training_config.seed)
        self.model = self._build_model(time_span)
        validation_candidate_index = self._validation_candidate_index(context, training_config)
        result = train_listwise(
            model=self.model,
            train_events=train_events,
            val_events=val_events,
            node_map=self.node_map,
            neighbor_sampler=self.neighbor_sampler,
            dst_pool=dst_pool,
            epochs=training_config.epochs,
            batch_size=training_config.train_batch_size,
            num_negatives=training_config.num_negatives,
            lr=training_config.lr,
            weight_decay=training_config.weight_decay,
            early_stop_patience=training_config.early_stop_patience,
            max_train_events=training_config.max_train_events,
            max_val_events=training_config.max_val_events,
            selection_metric=training_config.selection_metric,
            validation_candidate_index=validation_candidate_index,
            rng=rng,
            verbose=training_config.verbose,
        )

        if training_config.refit_full:
            log(
                f"[temporal-graph] refit_full epochs={result.best_epoch} events={len(interactions)}",
                enabled=training_config.verbose,
            )
            self.model = self._build_model(time_span)
            fit_full_epochs(
                model=self.model,
                events=interactions,
                node_map=self.node_map,
                neighbor_sampler=self.neighbor_sampler,
                dst_pool=dst_pool,
                epochs=result.best_epoch,
                batch_size=training_config.train_batch_size,
                num_negatives=training_config.num_negatives,
                lr=training_config.lr,
                weight_decay=training_config.weight_decay,
                max_train_events=training_config.max_train_events,
                rng=np.random.default_rng(training_config.seed + 10_000),
                verbose=training_config.verbose,
            )

        report = TrainingReport(
            train_events=min(len(train_events), training_config.max_train_events or len(train_events)),
            val_events=min(len(val_events), training_config.max_val_events or len(val_events)),
            best_val_ap=result.best_val_ap,
            best_val_mrr=result.best_val_mrr,
            feature_names=("node_embedding", "temporal_memory", "cross_attention", "listwise_softmax"),
            selected_fusion="end_to_end",
            model_name="temporal-graph",
            metrics={
                "best_epoch": float(result.best_epoch),
                "num_nodes": float(self.node_map.num_nodes),
                "num_dst": float(self.node_map.num_dst),
                "validation_test_like": 1.0 if training_config.validation_candidates == "test_like" else 0.0,
            },
        )
        self.training_report = report
        return report

    def predict_batch(self, queries: TestQueryArray) -> np.ndarray:
        if len(queries) == 0:
            return np.empty((0, queries.candidate_count), dtype=np.float64)
        if self.model is None or self.node_map is None or self.neighbor_sampler is None or self.config is None:
            raise RuntimeError("ranker is not fitted")
        batch = queries_to_prediction_batch(
            queries=queries,
            node_map=self.node_map,
            neighbor_sampler=self.neighbor_sampler,
            history_len=self.config.history_len,
            candidate_history_len=self.config.candidate_history_len,
        )
        logits = predict_logits(self.model, batch)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return probs.astype(np.float64, copy=False)

    def _build_model(self, time_span: int) -> EndToEndTemporalGraphModel:
        if self.node_map is None or self.config is None:
            raise RuntimeError("ranker is not initialized")
        return EndToEndTemporalGraphModel(
            TemporalGraphModelConfig(
                num_nodes=self.node_map.num_nodes,
                history_len=self.config.history_len,
                candidate_history_len=self.config.candidate_history_len,
                hidden_size=self.config.hidden_size,
                layers=self.config.layers,
                heads=self.config.heads,
                dropout=self.config.dropout,
                time_span=time_span,
            )
        )

    def _validation_candidate_index(
        self,
        context: FitContext,
        training_config: TemporalGraphTrainingConfig,
    ) -> TestCandidateIndex | None:
        if self.node_map is None:
            raise RuntimeError("ranker is not initialized")
        if training_config.validation_candidates == "random":
            return None
        if training_config.validation_candidates != "test_like":
            raise ValueError(f"unsupported validation candidate protocol: {training_config.validation_candidates}")
        return TestCandidateIndex.from_queries(read_test_queries(context.dataset.test_path), self.node_map)


class TemporalGraphRankerAdapter:
    name = "temporal-graph"

    def __init__(self, config: TemporalGraphTrainingConfig | None = None) -> None:
        self.config = config or TemporalGraphTrainingConfig()
        self.impl = TemporalGraphRanker()

    def fit(self, interactions: InteractionArray, context: FitContext) -> TrainingReport:
        config = replace(
            self.config,
            seed=context.seed,
            verbose=context.verbose,
        )
        return self.impl.fit(interactions, training_config=config, context=context)

    def predict_batch(self, queries: TestQueryArray) -> np.ndarray:
        return self.impl.predict_batch(queries)
