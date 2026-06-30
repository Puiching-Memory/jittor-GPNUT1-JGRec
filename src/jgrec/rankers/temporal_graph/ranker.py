from __future__ import annotations

from dataclasses import replace

import jittor as jt
import numpy as np

from jgrec.checkpoint import get_model_state, load_model_state, save_model_state, set_model_state
from jgrec.core.io import read_test_queries
from jgrec.core.types import (
    FitContext,
    InteractionTable,
    TestQueryArray,
    TrainingReport,
)
from jgrec.logging import log

from .config import TemporalGraphTrainingConfig
from .index import TemporalNodeMap, safe_neighbor_sampler, temporal_data_from_interactions, temporal_loader_api
from .model import EndToEndTemporalGraphModel, TemporalGraphModelConfig
from .trainer import (
    CANDIDATE_PRIOR_FEATURE_DIM,
    CANDIDATE_PRIOR_FEATURE_NAMES,
    CandidatePriorIndex,
    TestCandidateIndex,
    fit_full_epochs,
    predict_logits,
    queries_to_prediction_batch,
    train_listwise,
)


class TemporalGraphRanker:
    """End-to-end temporal graph candidate ranker."""

    def __init__(self) -> None:
        self.node_map: TemporalNodeMap | None = None
        self.model: EndToEndTemporalGraphModel | None = None
        self.neighbor_sampler = None
        self.config: TemporalGraphTrainingConfig | None = None
        self.training_report: TrainingReport | None = None
        self._candidate_prior_index: CandidatePriorIndex | None = None

    def fit(
        self,
        interactions: InteractionTable,
        training_config: TemporalGraphTrainingConfig,
        context: FitContext,
    ) -> TrainingReport:
        if len(interactions) == 0:
            raise ValueError("training interactions are empty")
        _configure_jittor_runtime(training_config.seed)
        interactions = interactions.sort_by_time()
        if training_config.max_fit_events > 0 and len(interactions) > training_config.max_fit_events:
            interactions = interactions.tail(training_config.max_fit_events)
        if len(interactions) < 4:
            raise ValueError("temporal graph ranker needs at least four interactions")

        self.config = training_config
        self.node_map = TemporalNodeMap.from_interactions_and_test(interactions, context.dataset.test_path)
        full_data = temporal_data_from_interactions(interactions, self.node_map)
        _, get_neighbor_sampler = temporal_loader_api()
        self.neighbor_sampler = safe_neighbor_sampler(get_neighbor_sampler(full_data, "recent", seed=training_config.seed))
        dst_pool = np.unique(self.node_map.dst_ids(interactions.dst)).astype(np.int32, copy=False)
        dst_pool = dst_pool[dst_pool > 0]

        n_events = len(interactions)
        val_size = max(1, int(n_events * training_config.val_ratio))
        train_end = max(1, n_events - val_size)
        if train_end >= n_events:
            train_end = n_events - 1
        train_events = interactions[:train_end]
        val_events = interactions[train_end:]
        time_span = max(int(interactions.time[-1]) - int(interactions.time[0]), 1)

        rng = np.random.default_rng(training_config.seed)
        self.model = self._build_model(time_span)
        if context.load_checkpoint_path is not None and context.load_checkpoint_path.exists():
            log(
                f"[temporal-graph] loading checkpoint from {context.load_checkpoint_path}",
                enabled=training_config.verbose,
            )
            self.load_checkpoint(context.load_checkpoint_path)
        test_candidate_index = self._test_candidate_index(context)
        train_candidate_index = self._candidate_index_for_protocol(
            test_candidate_index,
            training_config.training_candidates,
        )
        validation_candidate_index = self._candidate_index_for_protocol(
            test_candidate_index,
            training_config.validation_candidates,
        )
        selection_candidate_prior_index = (
            CandidatePriorIndex.from_test_candidates(
                test_candidate_index,
                self.node_map.dst_ids(train_events.dst),
                train_times=train_events.time,
                recent_feature_group=training_config.candidate_recent_feature_group,
                include_test_frequency=training_config.candidate_include_test_frequency,
            )
            if test_candidate_index is not None
            else None
        )
        self._candidate_prior_index = selection_candidate_prior_index
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
            train_candidate_index=train_candidate_index,
            validation_candidate_index=validation_candidate_index,
            candidate_prior_index=selection_candidate_prior_index,
            rng=rng,
            verbose=training_config.verbose,
        )

        if training_config.refit_full:
            log(
                f"[temporal-graph] refit_full epochs={result.best_epoch} events={len(interactions)}",
                enabled=training_config.verbose,
            )
            full_candidate_prior_index = (
                CandidatePriorIndex.from_test_candidates(
                    test_candidate_index,
                    self.node_map.dst_ids(interactions.dst),
                    train_times=interactions.time,
                    recent_feature_group=training_config.candidate_recent_feature_group,
                    include_test_frequency=training_config.candidate_include_test_frequency,
                )
                if test_candidate_index is not None
                else None
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
                train_candidate_index=train_candidate_index,
                candidate_prior_index=full_candidate_prior_index,
                rng=np.random.default_rng(training_config.seed + 10_000),
                verbose=training_config.verbose,
            )
            self._candidate_prior_index = full_candidate_prior_index

        report = TrainingReport(
            train_events=min(len(train_events), training_config.max_train_events or len(train_events)),
            val_events=min(len(val_events), training_config.max_val_events or len(val_events)),
            best_val_ap=result.best_val_ap,
            best_val_mrr=result.best_val_mrr,
            feature_names=(
                "node_embedding",
                "temporal_memory",
                "cross_attention",
                *CANDIDATE_PRIOR_FEATURE_NAMES,
                "listwise_softmax",
            ),
            selected_fusion="end_to_end",
            model_name="temporal-graph",
            metrics={
                "best_epoch": float(result.best_epoch),
                "num_nodes": float(self.node_map.num_nodes),
                "num_dst": float(self.node_map.num_dst),
                "training_test_like": 1.0 if train_candidate_index is not None else 0.0,
                "validation_test_like": 1.0 if validation_candidate_index is not None else 0.0,
                "use_cuda": 1.0,
            },
        )
        self.training_report = report
        if context.save_checkpoint_path is not None:
            log(
                f"[temporal-graph] saving checkpoint to {context.save_checkpoint_path}",
                enabled=training_config.verbose,
            )
            self.save_checkpoint(context.save_checkpoint_path)
        return report

    def save_checkpoint(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("ranker is not fitted")
        save_model_state(path, get_model_state(self.model))

    def load_checkpoint(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("ranker is not fitted")
        set_model_state(self.model, load_model_state(path))

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
            candidate_prior_index=self._prediction_candidate_prior_index(),
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
                candidate_feature_dim=CANDIDATE_PRIOR_FEATURE_DIM,
            )
        )

    def _test_candidate_index(self, context: FitContext) -> TestCandidateIndex | None:
        if self.node_map is None:
            raise RuntimeError("ranker is not initialized")
        test_path = context.dataset.test_path
        if test_path is None or not test_path.exists():
            return None
        return TestCandidateIndex.from_queries(read_test_queries(test_path), self.node_map)

    def _candidate_index_for_protocol(
        self,
        test_candidate_index: TestCandidateIndex | None,
        protocol: str,
    ) -> TestCandidateIndex | None:
        if protocol == "random":
            return None
        if protocol != "test_like":
            raise ValueError(f"unsupported candidate protocol: {protocol}")
        return test_candidate_index

    def _prediction_candidate_prior_index(self) -> CandidatePriorIndex | None:
        if self.node_map is None or self.config is None:
            raise RuntimeError("ranker is not initialized")
        return self._candidate_prior_index


class TemporalGraphRankerAdapter:
    name = "temporal-graph"

    def __init__(self, config: TemporalGraphTrainingConfig | None = None) -> None:
        self.config = config or TemporalGraphTrainingConfig()
        self.impl = TemporalGraphRanker()

    def fit(self, interactions: InteractionTable, context: FitContext) -> TrainingReport:
        config = replace(
            self.config,
            seed=context.seed,
            verbose=context.verbose,
        )
        return self.impl.fit(interactions, training_config=config, context=context)

    def predict_batch(self, queries: TestQueryArray) -> np.ndarray:
        return self.impl.predict_batch(queries)


def _set_jittor_seed(seed: int) -> None:
    jt.set_global_seed(int(seed))


def _configure_jittor_runtime(seed: int) -> None:
    jt.flags.use_cuda = 1
    _set_jittor_seed(seed)
