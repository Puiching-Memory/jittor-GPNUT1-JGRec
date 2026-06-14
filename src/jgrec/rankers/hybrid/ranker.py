from __future__ import annotations

import os
import tempfile
from dataclasses import replace
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

from jgrec.core.memory import log_event, log_memory, release_memory
from jgrec.core.types import (
    FitContext,
    InteractionTable,
    TestQuery,
    TestQueryArray,
    TrainingReport,
)
from jgrec.idmap import NodeIdMap
from jgrec.logging import log
from jgrec.rankers.common.temporal_index import TemporalInteractionIndex

from .auto_strategy import (
    DatasetProfile,
    choose_auto_strategy,
    profile_dataset,
    profile_dataset_paths,
    test_candidate_arrays,
)
from .candidate_prior import CANDIDATE_PRIOR_FEATURE_NAMES, CandidatePriorTower
from .config import (
    GRAPH_WINDOW_NAMES,
    SEQUENCE_FEATURE_NAMES,
    SOURCE_PROFILE_FEATURE_NAMES,
    TARGET_WINDOW_FEATURE_NAMES,
    TWO_TOWER_FEATURE_NAMES,
    CandidatePriorConfig,
    GraphTowerConfig,
    SequenceTowerConfig,
    SourceProfileConfig,
    StructureTowerConfig,
    TargetWindowConfig,
    TrainingConfig,
    TwoTowerConfig,
)
from .encoder_cache import HybridDeterministicSnapshot, HybridPrefixStateCache, hydrate_deterministic_state
from .sampling import (
    NegativeSamplingContext,
    NegativeSamplingJob,
    build_candidate_pool,
    sample_mixed_negatives,
    sample_mixed_negatives_batch,
)
from .stats import STAT_FEATURE_NAMES, TemporalStats
from .structure import STRUCTURE_FEATURE_NAMES, StructureFeatureTower

if TYPE_CHECKING:
    from .fusion import FusionMLP, FusionResult

_FEATURE_MEMMAP_TEMP_FILES: list[Any] = []
FEATURE_PROFILE_INTERVAL = 10_000
FEATURE_MEMMAP_FLUSH_INTERVAL = 8


class _DisabledGraphTower:
    def fit(self, interactions: InteractionTable, rng: np.random.Generator, verbose: bool = True, **kwargs) -> None:
        return

    def scores_for_queries(self, queries: TestQueryArray | list[TestQuery]) -> np.ndarray:
        return _zero_scores(queries, len(GRAPH_WINDOW_NAMES))

    def scores_for_query_array(self, queries: TestQueryArray) -> np.ndarray:
        return _zero_scores_for_array(queries, len(GRAPH_WINDOW_NAMES))


class _DisabledSequenceTower:
    def fit(self, interactions: InteractionTable, rng: np.random.Generator, verbose: bool = True, **kwargs) -> None:
        return

    def scores_for_queries(self, queries: TestQueryArray | list[TestQuery]) -> np.ndarray:
        return _zero_scores(queries, len(SEQUENCE_FEATURE_NAMES))

    def scores_for_query_array(self, queries: TestQueryArray) -> np.ndarray:
        return _zero_scores_for_array(queries, len(SEQUENCE_FEATURE_NAMES))


class _DisabledTwoTower:
    def fit(self, interactions: InteractionTable, rng: np.random.Generator, verbose: bool = True, **kwargs) -> None:
        return

    def scores_for_queries(self, queries: TestQueryArray | list[TestQuery]) -> np.ndarray:
        return _zero_scores(queries, len(TWO_TOWER_FEATURE_NAMES))

    def scores_for_query_array(self, queries: TestQueryArray) -> np.ndarray:
        return _zero_scores_for_array(queries, len(TWO_TOWER_FEATURE_NAMES))


class _DisabledSourceProfileTower:
    index: Any = None

    def fit(self, interactions: InteractionTable, rng: np.random.Generator, verbose: bool = True, **kwargs) -> None:
        return

    def scores_for_queries(self, queries: TestQueryArray | list[TestQuery]) -> np.ndarray:
        return _zero_scores(queries, len(SOURCE_PROFILE_FEATURE_NAMES))

    def scores_for_query_array(self, queries: TestQueryArray) -> np.ndarray:
        return _zero_scores_for_array(queries, len(SOURCE_PROFILE_FEATURE_NAMES))

    def hydrate(self, snapshot: dict) -> None:
        return


class _DisabledCandidatePriorTower:
    def fit(
        self,
        interactions: InteractionTable,
        test_candidate_counts=None,
    ) -> None:
        return

    def features_for_queries(self, queries: TestQueryArray | list[TestQuery], stat_features: np.ndarray) -> np.ndarray:
        return _zero_scores(queries, len(CANDIDATE_PRIOR_FEATURE_NAMES))

    def features_for_query_array(self, queries: TestQueryArray, stat_features: np.ndarray) -> np.ndarray:
        return _zero_scores_for_array(queries, len(CANDIDATE_PRIOR_FEATURE_NAMES))


class _DisabledTargetWindowTower:
    def fit(self, interactions: InteractionTable) -> None:
        return

    def features_for_queries(self, queries: TestQueryArray | list[TestQuery]) -> np.ndarray:
        return _zero_scores(queries, len(TARGET_WINDOW_FEATURE_NAMES))

    def features_for_query_array(self, queries: TestQueryArray) -> np.ndarray:
        return _zero_scores_for_array(queries, len(TARGET_WINDOW_FEATURE_NAMES))

    def hydrate(self, snapshot: dict) -> None:
        return

    def compact_for_future_queries(self) -> None:
        return


class _DisabledStructureTower:
    index: Any = None

    def fit(self, interactions: InteractionTable, rng: np.random.Generator, verbose: bool = True) -> None:
        return

    def features_for_queries(self, queries: TestQueryArray | list[TestQuery]) -> np.ndarray:
        return _zero_scores(queries, len(STRUCTURE_FEATURE_NAMES))

    def features_for_query_array(self, queries: TestQueryArray) -> np.ndarray:
        return _zero_scores_for_array(queries, len(STRUCTURE_FEATURE_NAMES))

    def compact_for_future_queries(self) -> None:
        return


class HybridFeatureEncoder:
    def __init__(
        self,
        id_map: NodeIdMap,
        recent_window: int,
        graph_config: GraphTowerConfig,
        sequence_config: SequenceTowerConfig,
        candidate_prior_config: CandidatePriorConfig | None = None,
        target_window_config: TargetWindowConfig | None = None,
        dataset_profile: DatasetProfile | None = None,
        two_tower_config: TwoTowerConfig | None = None,
        source_profile_config: SourceProfileConfig | None = None,
        structure_config: StructureTowerConfig | None = None,
    ) -> None:
        self.id_map = id_map
        self.dataset_profile = dataset_profile
        self.stats = TemporalStats(recent_window=recent_window)
        self.candidate_prior = _build_candidate_prior(candidate_prior_config or CandidatePriorConfig(enabled=False))
        self.target_window = _build_target_window(target_window_config or TargetWindowConfig(enabled=False))
        self.structure = _build_structure_tower(structure_config or StructureTowerConfig(enabled=False))
        self.source_profile = _build_source_profile_tower(
            id_map,
            source_profile_config or SourceProfileConfig(enabled=False),
        )
        self.two_tower = _build_two_tower(id_map, two_tower_config or TwoTowerConfig(enabled=False))
        self.graph = _build_graph_tower(id_map, graph_config)
        self.sequence = _build_sequence_tower(id_map, sequence_config)
        self.verbose = False
        self._profile_rows = 0
        self._profile_next_rows = FEATURE_PROFILE_INTERVAL
        self._profile_elapsed = {
            "stats": 0.0,
            "prior": 0.0,
            "target": 0.0,
            "structure": 0.0,
            "profile": 0.0,
            "tower": 0.0,
            "graph": 0.0,
            "sequence": 0.0,
            "concat": 0.0,
        }
        self.feature_names = (
            STAT_FEATURE_NAMES
            + CANDIDATE_PRIOR_FEATURE_NAMES
            + TARGET_WINDOW_FEATURE_NAMES
            + STRUCTURE_FEATURE_NAMES
            + SOURCE_PROFILE_FEATURE_NAMES
            + TWO_TOWER_FEATURE_NAMES
            + GRAPH_WINDOW_NAMES
            + SEQUENCE_FEATURE_NAMES
        )

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def fit(
        self,
        interactions: InteractionTable,
        rng: np.random.Generator,
        verbose: bool,
        deterministic_snapshot: HybridDeterministicSnapshot | None = None,
    ) -> None:
        self.verbose = verbose
        start_total = perf_counter()
        elapsed: dict[str, float] = {}
        if deterministic_snapshot is not None:
            start = perf_counter()
            hydrate_deterministic_state(
                snapshot=deterministic_snapshot,
                stats=self.stats,
                candidate_prior=self.candidate_prior,
                target_window=self.target_window,
                structure=self.structure,
                source_profile=self.source_profile,
            )
            elapsed["deterministic"] = perf_counter() - start
            cache_status = "hit"
        else:
            start = perf_counter()
            self.stats.fit(interactions)
            elapsed["stats"] = perf_counter() - start
            start = perf_counter()
            test_counts = self.dataset_profile.test_candidate_counts if self.dataset_profile is not None else None
            self.candidate_prior.fit(interactions, test_counts)
            elapsed["prior"] = perf_counter() - start
            start = perf_counter()
            self.target_window.fit(interactions)
            elapsed["target"] = perf_counter() - start
            start = perf_counter()
            self.structure.fit(interactions, rng=rng, verbose=verbose)
            elapsed["structure"] = perf_counter() - start
            cache_status = "build"
        two_tower_fit = getattr(self.two_tower, "fit", None)
        start = perf_counter()
        if callable(two_tower_fit):
            two_tower_fit(
                interactions,
                rng=rng,
                verbose=verbose,
                shared_index=getattr(self.structure, "index", None),
            )
        elapsed["tower"] = perf_counter() - start
        start = perf_counter()
        self.graph.fit(interactions, rng=rng, verbose=verbose)
        elapsed["graph"] = perf_counter() - start
        start = perf_counter()
        self.sequence.fit(interactions, rng=rng, verbose=verbose)
        elapsed["sequence"] = perf_counter() - start
        source_profile_fit = getattr(self.source_profile, "fit", None)
        start = perf_counter()
        if callable(source_profile_fit):
            source_profile_fit(
                interactions,
                rng=_copy_rng(rng),
                verbose=verbose,
                shared_index=getattr(self.structure, "index", None),
                deterministic_ready=deterministic_snapshot is not None,
            )
        elapsed["profile"] = perf_counter() - start
        total = perf_counter() - start_total
        pieces = " ".join(f"{name}={seconds:.1f}s" for name, seconds in elapsed.items())
        log_event(f"[encoder-fit] cache={cache_status} {pieces} total={total:.1f}s", enabled=verbose)

    def features_for_queries(self, queries: TestQueryArray | list[TestQuery]) -> np.ndarray:
        if not queries:
            return np.empty((0, 0, self.feature_dim), dtype=np.float32)
        if isinstance(queries, TestQueryArray):
            return self.features_for_query_array(queries)
        return self.features_for_query_array(TestQueryArray.from_queries(queries))

    def features_for_query_array(self, queries: TestQueryArray) -> np.ndarray:
        if not queries:
            return np.empty((0, 0, self.feature_dim), dtype=np.float32)
        detail_profile = self.verbose and self._profile_rows == 0
        rows = len(queries)
        if detail_profile:
            log_event(f"[feature-profile] first_batch rows={rows} stats_start", enabled=True)
        start = perf_counter()
        stat_features = _features_from_tower(self.stats, queries)
        elapsed = perf_counter() - start
        self._profile_elapsed["stats"] += elapsed
        if detail_profile:
            log_event(f"[feature-profile] first_batch rows={rows} stats_done elapsed={elapsed:.1f}s", enabled=True)
            log_event(f"[feature-profile] first_batch rows={rows} prior_start", enabled=True)
        start = perf_counter()
        candidate_prior_features = _candidate_prior_features(self.candidate_prior, queries, stat_features)
        elapsed = perf_counter() - start
        self._profile_elapsed["prior"] += elapsed
        if detail_profile:
            log_event(f"[feature-profile] first_batch rows={rows} prior_done elapsed={elapsed:.1f}s", enabled=True)
            log_event(f"[feature-profile] first_batch rows={rows} target_start", enabled=True)
        start = perf_counter()
        target_window_features = _features_from_tower(self.target_window, queries)
        elapsed = perf_counter() - start
        self._profile_elapsed["target"] += elapsed
        if detail_profile:
            log_event(f"[feature-profile] first_batch rows={rows} target_done elapsed={elapsed:.1f}s", enabled=True)
            log_event(f"[feature-profile] first_batch rows={rows} structure_start", enabled=True)
        start = perf_counter()
        structure_features = _features_from_tower(self.structure, queries)
        elapsed = perf_counter() - start
        self._profile_elapsed["structure"] += elapsed
        if detail_profile:
            log_event(f"[feature-profile] first_batch rows={rows} structure_done elapsed={elapsed:.1f}s", enabled=True)
            log_event(f"[feature-profile] first_batch rows={rows} profile_start", enabled=True)
        start = perf_counter()
        source_profile_features = _scores_from_tower(self.source_profile, queries)
        elapsed = perf_counter() - start
        self._profile_elapsed["profile"] += elapsed
        if detail_profile:
            log_event(f"[feature-profile] first_batch rows={rows} profile_done elapsed={elapsed:.1f}s", enabled=True)
            log_event(f"[feature-profile] first_batch rows={rows} tower_start", enabled=True)
        start = perf_counter()
        two_tower_features = _scores_from_tower(self.two_tower, queries)
        elapsed = perf_counter() - start
        self._profile_elapsed["tower"] += elapsed
        if detail_profile:
            log_event(f"[feature-profile] first_batch rows={rows} tower_done elapsed={elapsed:.1f}s", enabled=True)
            log_event(f"[feature-profile] first_batch rows={rows} graph_start", enabled=True)
        start = perf_counter()
        graph_features = _scores_from_tower(self.graph, queries)
        elapsed = perf_counter() - start
        self._profile_elapsed["graph"] += elapsed
        if detail_profile:
            log_event(f"[feature-profile] first_batch rows={rows} graph_done elapsed={elapsed:.1f}s", enabled=True)
            log_event(f"[feature-profile] first_batch rows={rows} sequence_start", enabled=True)
        start = perf_counter()
        sequence_features = _scores_from_tower(self.sequence, queries)
        elapsed = perf_counter() - start
        self._profile_elapsed["sequence"] += elapsed
        if detail_profile:
            log_event(f"[feature-profile] first_batch rows={rows} sequence_done elapsed={elapsed:.1f}s", enabled=True)
            log_event(f"[feature-profile] first_batch rows={rows} concat_start", enabled=True)
        start = perf_counter()
        features = np.concatenate(
            [
                stat_features,
                candidate_prior_features,
                target_window_features,
                structure_features,
                source_profile_features,
                two_tower_features,
                graph_features,
                sequence_features,
            ],
            axis=2,
        ).astype(np.float32, copy=False)
        elapsed = perf_counter() - start
        self._profile_elapsed["concat"] += elapsed
        if detail_profile:
            log_event(f"[feature-profile] first_batch rows={rows} concat_done elapsed={elapsed:.1f}s", enabled=True)
        self._log_feature_profile(rows)
        return features

    def _log_feature_profile(self, rows: int) -> None:
        self._profile_rows += int(rows)
        if not self.verbose or self._profile_rows < self._profile_next_rows:
            return

        elapsed = " ".join(f"{name}={seconds:.1f}s" for name, seconds in self._profile_elapsed.items())
        log_event(f"[feature-profile] rows={self._profile_rows} {elapsed}", enabled=True)
        while self._profile_rows >= self._profile_next_rows:
            self._profile_next_rows += FEATURE_PROFILE_INTERVAL

    def compact_for_future_queries(self) -> None:
        compact_stats = getattr(self.stats, "compact_for_future_queries", None)
        if callable(compact_stats):
            compact_stats()
        compact_target = getattr(self.target_window, "compact_for_future_queries", None)
        if callable(compact_target):
            compact_target()
        compact_future_structure = getattr(self.structure, "compact_transition_cooccur_for_future_queries", None)
        if callable(compact_future_structure):
            compact_future_structure()
        release_memory()


def _build_candidate_prior(config: CandidatePriorConfig) -> Any:
    if not config.enabled:
        return _DisabledCandidatePriorTower()
    return CandidatePriorTower(config=config)


def _build_target_window(config: TargetWindowConfig) -> Any:
    if not config.enabled:
        return _DisabledTargetWindowTower()
    from .target_window import TargetWindowTower  # noqa: PLC0415

    return TargetWindowTower(config=config)


def _build_structure_tower(config: StructureTowerConfig) -> Any:
    if not config.enabled:
        return _DisabledStructureTower()
    return StructureFeatureTower(config)


def _build_source_profile_tower(id_map: NodeIdMap, config: SourceProfileConfig) -> Any:
    if not config.enabled:
        return _DisabledSourceProfileTower()
    from .source_profile import SourceProfileTower  # noqa: PLC0415

    return SourceProfileTower(id_map=id_map, config=config)


def _build_graph_tower(id_map: NodeIdMap, config: GraphTowerConfig) -> Any:
    if not config.enabled:
        return _DisabledGraphTower()
    from .gnn import GraphTower  # noqa: PLC0415

    return GraphTower(id_map=id_map, config=config)


def _build_sequence_tower(id_map: NodeIdMap, config: SequenceTowerConfig) -> Any:
    if not config.enabled:
        return _DisabledSequenceTower()
    from .sequence import SequenceTower  # noqa: PLC0415

    return SequenceTower(id_map=id_map, config=config)


def _build_two_tower(id_map: NodeIdMap, config: TwoTowerConfig) -> Any:
    if not config.enabled:
        return _DisabledTwoTower()
    from .two_tower import TwoTower  # noqa: PLC0415

    return TwoTower(id_map=id_map, config=config)


def _zero_scores(queries: TestQueryArray | list[TestQuery], feature_count: int) -> np.ndarray:
    if not queries:
        return np.empty((0, 0, feature_count), dtype=np.float32)
    candidate_count = len(queries[0].candidates)
    return np.zeros((len(queries), candidate_count, feature_count), dtype=np.float32)


def _zero_scores_for_array(queries: TestQueryArray, feature_count: int) -> np.ndarray:
    return np.zeros((len(queries), queries.candidate_count, feature_count), dtype=np.float32)


def _features_from_tower(tower: Any, queries: TestQueryArray) -> np.ndarray:
    fast_path = getattr(tower, "features_for_query_array", None)
    if callable(fast_path):
        return fast_path(queries)
    return tower.features_for_queries(list(queries))


def _candidate_prior_features(tower: Any, queries: TestQueryArray, stat_features: np.ndarray) -> np.ndarray:
    fast_path = getattr(tower, "features_for_query_array", None)
    if callable(fast_path):
        return fast_path(queries, stat_features)
    return tower.features_for_queries(list(queries), stat_features)


def _scores_from_tower(tower: Any, queries: TestQueryArray) -> np.ndarray:
    fast_path = getattr(tower, "scores_for_query_array", None)
    if callable(fast_path):
        return fast_path(queries)
    return tower.scores_for_queries(list(queries))


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
        self.dataset_profile: DatasetProfile | None = None

    def fit(self, interactions: InteractionTable, training_config: TrainingConfig) -> TrainingReport:
        if len(interactions) == 0:
            raise ValueError("training interactions are empty")

        interactions = interactions.sort_by_time()
        if training_config.max_fit_events > 0 and len(interactions) > training_config.max_fit_events:
            interactions = interactions.tail(training_config.max_fit_events)
        self.id_map = NodeIdMap.from_interactions(interactions)
        self.feature_names = (
            STAT_FEATURE_NAMES
            + CANDIDATE_PRIOR_FEATURE_NAMES
            + TARGET_WINDOW_FEATURE_NAMES
            + STRUCTURE_FEATURE_NAMES
            + SOURCE_PROFILE_FEATURE_NAMES
            + TWO_TOWER_FEATURE_NAMES
            + GRAPH_WINDOW_NAMES
            + SEQUENCE_FEATURE_NAMES
        )
        self._fusion_hidden_dim = training_config.fusion_hidden_dim

        training_config = self._apply_auto_strategy(interactions, training_config)
        log_memory("hybrid_fit_start", enabled=training_config.verbose)
        log(f"[hybrid-fit] start events={len(interactions)}", enabled=training_config.verbose)
        fusion, fusion_result, report, encoder_cache, cache_config = self._learn_fusion(interactions, training_config)
        self.fusion = fusion
        self.fusion_result = fusion_result

        rng = np.random.default_rng(training_config.seed + 10_000)
        final_config = _config_for_selected_features(training_config, fusion_result.feature_indices)
        final_future_only = self._can_use_future_only_final_encoder()
        if final_future_only:
            final_config = replace(final_config, structure_future_only_transition_cooccur=True)
        cache = self._final_encoder_cache(
            interactions=interactions,
            final_config=final_config,
            existing_cache=encoder_cache,
            existing_config=cache_config,
            verbose=training_config.verbose,
        )
        final_snapshot = cache.snapshot_for_prefix(len(interactions)) if cache is not None else None
        log_event(
            "[hybrid-fit] final_encoder "
            f"prior={final_config.candidate_prior_enabled} profile={final_config.source_profile_enabled} "
            f"target={final_config.target_window_enabled} "
            f"tower={final_config.two_tower_enabled} "
            f"gnn={final_config.gnn_enabled} seq={final_config.seq_enabled} "
            f"future_only_structure={final_config.structure_future_only_transition_cooccur}",
            enabled=training_config.verbose,
        )
        log_memory("final_encoder_start", enabled=training_config.verbose)
        self.encoder = self._fit_encoder(
            interactions,
            final_config,
            rng,
            verbose=training_config.verbose,
            deterministic_snapshot=final_snapshot,
        )
        if cache is not None:
            cache.clear()
        del final_snapshot
        if final_future_only:
            self.encoder.compact_for_future_queries()
        log_memory("final_encoder_done", enabled=training_config.verbose)
        self.training_report = report
        log_event("[hybrid-fit] done", enabled=training_config.verbose)
        return report

    def _can_use_future_only_final_encoder(self) -> bool:
        if self.dataset_profile is None:
            return False
        return (
            self.dataset_profile.test_min_time > 0
            and self.dataset_profile.train_max_time > 0
            and self.dataset_profile.test_min_time > self.dataset_profile.train_max_time
        )

    def _apply_auto_strategy(self, interactions: InteractionTable, config: TrainingConfig) -> TrainingConfig:
        if config.dataset_test_path is None:
            return config

        needs_profile = (
            config.auto_strategy_enabled
            or config.candidate_prior_enabled
            or config.test_candidate_negative_ratio > 0.0
        )
        if not needs_profile:
            return config

        profile = self._dataset_profile(interactions, config)
        self.dataset_profile = profile
        if not config.auto_strategy_enabled:
            log_event(
                "[dataset-profile] "
                f"holdout_pair_hit={profile.holdout_pair_hit_rate:.5f} "
                f"candidate_unseen_dst={profile.candidate_unseen_dst_rate:.5f} "
                f"test_top1pct_share={profile.test_candidate_top1pct_share:.5f}",
                enabled=config.verbose,
            )
            return replace(
                config,
                profile_holdout_pair_hit_rate=profile.holdout_pair_hit_rate,
                profile_candidate_unseen_dst_rate=profile.candidate_unseen_dst_rate,
            )

        strategy = choose_auto_strategy(profile)
        ratio = config.test_candidate_negative_ratio
        if ratio <= 0.0:
            ratio = strategy.test_candidate_negative_ratio
        log_event(
            "[auto-strategy] "
            f"mode={strategy.mode} holdout_pair_hit={profile.holdout_pair_hit_rate:.5f} "
            f"candidate_unseen_dst={profile.candidate_unseen_dst_rate:.5f} "
            f"src_history_p90={profile.src_history_p90:.1f} "
            f"test_top1pct_share={profile.test_candidate_top1pct_share:.5f} "
            f"test_candidate_negative_ratio={ratio:.2f}",
            enabled=config.verbose,
        )
        return replace(
            config,
            auto_mode=strategy.mode,
            profile_holdout_pair_hit_rate=profile.holdout_pair_hit_rate,
            profile_candidate_unseen_dst_rate=profile.candidate_unseen_dst_rate,
            test_candidate_negative_ratio=ratio,
        )

    def _dataset_profile(self, interactions: InteractionTable, config: TrainingConfig) -> DatasetProfile:
        if config.dataset_test_path is None:
            raise RuntimeError("dataset test path is required for profiling")
        if config.dataset_train_path is not None:
            return profile_dataset_paths(config.dataset_train_path, config.dataset_test_path, val_ratio=config.val_ratio)
        return profile_dataset(interactions, config.dataset_test_path, val_ratio=config.val_ratio)

    def predict(self, query: TestQuery) -> np.ndarray:
        return self.predict_batch([query])[0]

    def predict_batch(self, queries: TestQueryArray | list[TestQuery]) -> np.ndarray:
        if not queries:
            return np.empty((0, 100), dtype=np.float64)
        if self.encoder is None or self.fusion is None or self.fusion_result is None:
            raise RuntimeError("ranker is not fitted")

        from .fusion import predict_logits  # noqa: PLC0415

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
        interactions: InteractionTable,
        config: TrainingConfig,
    ) -> tuple[FusionMLP, FusionResult, TrainingReport, HybridPrefixStateCache | None, TrainingConfig]:
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
        if len(train_events) == 0 or len(val_events) == 0:
            raise ValueError(
                "invalid temporal split for hybrid reranker: "
                f"context={len(context_events)}, train={len(train_events)}, val={len(val_events)}"
            )

        train_events = _sample_events(train_events, config.max_train_events, rng)
        val_events = _sample_events(val_events, config.max_val_events, rng)
        dst_pool = np.unique(interactions.dst).astype(np.int64, copy=False)

        log_event(
            "[hybrid-fit] split "
            f"context={len(context_events)} train={len(train_events)} val={len(val_events)} "
            f"dst={len(dst_pool)}",
            enabled=config.verbose,
        )
        log_memory("split_done", enabled=config.verbose)

        supervised_encoder_config = replace(config, structure_future_only_transition_cooccur=True)
        encoder_cache = self._encoder_state_cache(interactions, supervised_encoder_config, config.verbose)
        train_snapshot = encoder_cache.snapshot_for_prefix(context_end) if encoder_cache is not None else None
        train_encoder = self._timed_fit_encoder(
            "train_context_encoder",
            context_events,
            supervised_encoder_config,
            rng,
            config.verbose,
            deterministic_snapshot=train_snapshot,
        )
        if encoder_cache is not None:
            encoder_cache.release_except()
        del train_snapshot
        feature_start = perf_counter()
        log_memory("train_features_start", enabled=config.verbose)
        train_features = _build_supervised_features(
            train_events,
            train_encoder,
            dst_pool,
            config,
            rng,
            label="train_features",
        )
        del train_encoder
        log_event(
            f"[hybrid-fit] train_features shape={train_features.shape} elapsed={perf_counter() - feature_start:.1f}s",
            enabled=config.verbose,
        )
        log_memory("train_features_done", enabled=config.verbose)

        val_snapshot = encoder_cache.snapshot_for_prefix(train_end) if encoder_cache is not None else None
        val_encoder = self._timed_fit_encoder(
            "val_context_encoder",
            val_context_events,
            supervised_encoder_config,
            rng,
            config.verbose,
            deterministic_snapshot=val_snapshot,
        )
        if encoder_cache is not None:
            encoder_cache.release_except()
        del val_snapshot
        feature_start = perf_counter()
        log_memory("val_features_start", enabled=config.verbose)
        val_features = _build_supervised_features(
            val_events,
            val_encoder,
            dst_pool,
            config,
            rng,
            label="val_features",
        )
        del val_encoder
        log_event(
            f"[hybrid-fit] val_features shape={val_features.shape} elapsed={perf_counter() - feature_start:.1f}s",
            enabled=config.verbose,
        )
        log_memory("val_features_done", enabled=config.verbose)

        fusion_start = perf_counter()
        log_memory("fusion_start", enabled=config.verbose)
        fusion, result = self._fit_best_fusion(
            train_features=train_features,
            val_features=val_features,
            config=config,
            rng=rng,
            verbose=config.verbose,
        )
        del train_features
        del val_features
        _release_feature_memmaps()
        release_memory()
        log_event(f"[hybrid-fit] fusion elapsed={perf_counter() - fusion_start:.1f}s", enabled=config.verbose)
        log_memory("fusion_done", enabled=config.verbose)
        report = TrainingReport(
            train_events=len(train_events),
            val_events=len(val_events),
            best_val_ap=result.best_val_ap,
            best_val_mrr=result.best_val_mrr,
            feature_names=tuple(self.feature_names[idx] for idx in result.feature_indices),
            selected_fusion=result.candidate_name,
            model_name="hybrid",
            metrics={
                "auto_mode_code": _auto_mode_code(config.auto_mode),
                "holdout_pair_hit_rate": config.profile_holdout_pair_hit_rate,
                "candidate_unseen_dst_rate": config.profile_candidate_unseen_dst_rate,
                "test_candidate_negative_ratio": config.test_candidate_negative_ratio,
            },
        )
        return fusion, result, report, encoder_cache, supervised_encoder_config

    def _fit_best_fusion(
        self,
        train_features: np.ndarray,
        val_features: np.ndarray,
        config: TrainingConfig,
        rng: np.random.Generator,
        verbose: bool,
    ) -> tuple[FusionMLP, FusionResult]:
        from .fusion import fit_fusion_mlp, fit_fusion_mlp_streaming  # noqa: PLC0415

        masks = _feature_masks(train_features.shape[-1], config=config)
        best_model: FusionMLP | None = None
        best_result: FusionResult | None = None
        for name, indices in masks:
            candidate_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
            fit_func = fit_fusion_mlp_streaming if config.supervised_feature_memmap else fit_fusion_mlp
            log_memory(f"fusion_candidate_start:{name}", enabled=verbose)
            model, result = fit_func(
                train_features=train_features,
                val_features=val_features,
                config=config.fusion_config(),
                rng=candidate_rng,
                verbose=verbose,
                feature_indices=indices,
                candidate_name=name,
            )
            selected_score = _selected_report_metric(result, config.selection_metric)
            log_event(
                f"[fusion-select] candidate={name} "
                f"val_ap={result.best_val_ap:.5f} val_mrr={result.best_val_mrr:.5f}",
                enabled=verbose,
            )
            if best_result is None or selected_score >= _selected_report_metric(best_result, config.selection_metric):
                if best_model is not None:
                    del best_model
                best_model = model
                best_result = result
            else:
                del model
            release_memory()
            log_memory(f"fusion_candidate_done:{name}", enabled=verbose)

        if best_model is None or best_result is None:
            raise RuntimeError("no fusion candidate was trained")
        log_event(
            f"[fusion-select] chosen={best_result.candidate_name} "
            f"best_ap={best_result.best_val_ap:.5f} best_mrr={best_result.best_val_mrr:.5f}",
            enabled=verbose,
        )
        return best_model, best_result

    def _fit_encoder(
        self,
        interactions: InteractionTable,
        config: TrainingConfig,
        rng: np.random.Generator,
        verbose: bool,
        deterministic_snapshot: HybridDeterministicSnapshot | None = None,
    ) -> HybridFeatureEncoder:
        if self.id_map is None:
            raise RuntimeError("id map is not initialized")
        encoder = HybridFeatureEncoder(
            id_map=self.id_map,
            recent_window=self.recent_window,
            candidate_prior_config=config.candidate_prior_config(),
            target_window_config=config.target_window_config(),
            dataset_profile=self.dataset_profile,
            structure_config=config.structure_config(),
            source_profile_config=config.source_profile_config(),
            graph_config=config.graph_config(),
            sequence_config=config.sequence_config(),
            two_tower_config=config.two_tower_config(),
        )
        encoder.fit(interactions, rng=rng, verbose=verbose, deterministic_snapshot=deterministic_snapshot)
        return encoder

    def _timed_fit_encoder(
        self,
        label: str,
        interactions: InteractionTable,
        config: TrainingConfig,
        rng: np.random.Generator,
        verbose: bool,
        deterministic_snapshot: HybridDeterministicSnapshot | None = None,
    ) -> HybridFeatureEncoder:
        start = perf_counter()
        log_event(
            f"[hybrid-fit] {label} start events={len(interactions)} "
            f"prior={config.candidate_prior_enabled} target={config.target_window_enabled} "
            f"profile={config.source_profile_enabled} "
            f"tower={config.two_tower_enabled} "
            f"gnn={config.gnn_enabled} seq={config.seq_enabled} "
            f"future_only_structure={config.structure_future_only_transition_cooccur}",
            enabled=verbose,
        )
        log_memory(f"{label}_start", enabled=verbose)
        encoder = self._fit_encoder(
            interactions,
            config,
            rng,
            verbose=verbose,
            deterministic_snapshot=deterministic_snapshot,
        )
        log_event(f"[hybrid-fit] {label} done elapsed={perf_counter() - start:.1f}s", enabled=verbose)
        log_memory(f"{label}_done", enabled=verbose)
        return encoder

    def _encoder_state_cache(
        self,
        interactions: InteractionTable,
        config: TrainingConfig,
        verbose: bool,
    ) -> HybridPrefixStateCache | None:
        if not config.encoder_state_cache_enabled:
            log_event("[encoder-cache] disabled", enabled=verbose)
            return None
        try:
            test_counts = self.dataset_profile.test_candidate_counts if self.dataset_profile is not None else None
            return HybridPrefixStateCache(
                interactions,
                recent_window=self.recent_window,
                candidate_prior_config=config.candidate_prior_config(),
                target_window_config=config.target_window_config(),
                structure_config=config.structure_config(),
                source_profile_config=config.source_profile_config(),
                test_candidate_counts=test_counts,
                verbose=verbose,
            )
        except Exception as exc:
            log_event(f"[encoder-cache] fallback reason={type(exc).__name__}: {exc}", enabled=verbose)
            return None

    def _final_encoder_cache(
        self,
        *,
        interactions: InteractionTable,
        final_config: TrainingConfig,
        existing_cache: HybridPrefixStateCache | None,
        existing_config: TrainingConfig,
        verbose: bool,
    ) -> HybridPrefixStateCache | None:
        if existing_cache is not None and _can_reuse_encoder_cache(existing_config, final_config):
            log_event("[encoder-cache] reuse prefix builder for final_encoder", enabled=verbose)
            return existing_cache

        if existing_cache is not None:
            existing_cache.clear()
            log_event("[encoder-cache] final_encoder requires fresh deterministic state", enabled=verbose)
        return self._encoder_state_cache(interactions, final_config, verbose)


def _sample_events(
    events: InteractionTable,
    max_events: int,
    rng: np.random.Generator,
) -> InteractionTable:
    if max_events <= 0 or len(events) <= max_events:
        return events
    indices = np.sort(rng.choice(len(events), size=max_events, replace=False))
    return events.take(indices)


def _feature_masks(feature_count: int, config: TrainingConfig | None = None) -> list[tuple[str, tuple[int, ...]]]:
    stats_end = len(STAT_FEATURE_NAMES)
    prior_end = stats_end + len(CANDIDATE_PRIOR_FEATURE_NAMES)
    target_end = prior_end + len(TARGET_WINDOW_FEATURE_NAMES)
    structure_end = target_end + len(STRUCTURE_FEATURE_NAMES)
    profile_end = structure_end + len(SOURCE_PROFILE_FEATURE_NAMES)
    tower_end = profile_end + len(TWO_TOWER_FEATURE_NAMES)
    graph_end = tower_end + len(GRAPH_WINDOW_NAMES)
    ranges = {
        "stats": _feature_range(0, stats_end, feature_count),
        "prior": _feature_range(stats_end, prior_end, feature_count),
        "target": _feature_range(prior_end, target_end, feature_count),
        "structure": _feature_range(target_end, structure_end, feature_count),
        "profile": _feature_range(structure_end, profile_end, feature_count),
        "tower": _feature_range(profile_end, tower_end, feature_count),
        "gnn": _feature_range(tower_end, graph_end, feature_count),
        "seq": _feature_range(graph_end, feature_count, feature_count),
    }
    enabled = {
        "stats": True,
        "prior": config is None or config.candidate_prior_enabled,
        "target": config is None or config.target_window_enabled,
        "structure": config is None or config.structure_enabled,
        "profile": config is None or config.source_profile_enabled,
        "tower": config is None or config.two_tower_enabled,
        "gnn": config is None or config.gnn_enabled,
        "seq": config is None or config.seq_enabled,
    }

    def build_mask(groups: tuple[str, ...]) -> tuple[str, tuple[int, ...]] | None:
        name_parts = ["stats"]
        indices = ranges["stats"]
        for group in groups:
            group_range = ranges[group]
            if not group_range:
                return None
            if not enabled[group]:
                continue
            name_parts.append(group)
            indices += group_range
        if indices == ranges["stats"]:
            return None
        return "_".join(name_parts), indices

    masks: list[tuple[str, tuple[int, ...]]] = [("stats", ranges["stats"])]
    for groups in (
        ("prior",),
        ("prior", "structure"),
        ("prior", "structure", "tower"),
        ("prior", "structure", "tower", "gnn"),
        ("prior", "structure", "tower", "gnn", "seq"),
        ("prior", "target"),
        ("prior", "target", "structure"),
        ("prior", "target", "structure", "profile"),
        ("prior", "target", "structure", "profile", "tower"),
        ("prior", "target", "structure", "profile", "tower", "gnn"),
        ("prior", "target", "structure", "profile", "tower", "gnn", "seq"),
    ):
        mask = build_mask(groups)
        if mask is not None:
            masks.append(mask)

    unique: list[tuple[str, tuple[int, ...]]] = []
    seen: set[tuple[int, ...]] = set()
    for name, indices in masks:
        if not indices or indices in seen:
            continue
        seen.add(indices)
        unique.append((name, indices))
    return unique


def _copy_rng(rng: np.random.Generator) -> np.random.Generator:
    copied = np.random.default_rng()
    copied.bit_generator.state = rng.bit_generator.state
    return copied


def _feature_range(start: int, end: int, feature_count: int) -> tuple[int, ...]:
    start = min(max(start, 0), feature_count)
    end = min(max(end, 0), feature_count)
    if end <= start:
        return ()
    return tuple(range(start, end))


def _selected_report_metric(result: FusionResult, metric: str) -> float:
    normalized = metric.lower()
    if normalized == "ap":
        return result.best_val_ap
    if normalized == "mrr":
        return result.best_val_mrr
    raise ValueError(f"unsupported fusion selection metric: {metric}")


def _auto_mode_code(mode: str) -> float:
    if mode == "repeat_memory":
        return 1.0
    if mode == "new_link_cold":
        return 3.0
    if mode == "balanced":
        return 2.0
    return 0.0


def _config_for_selected_features(config: TrainingConfig, feature_indices: tuple[int, ...]) -> TrainingConfig:
    stats_end = len(STAT_FEATURE_NAMES)
    prior_end = stats_end + len(CANDIDATE_PRIOR_FEATURE_NAMES)
    target_end = prior_end + len(TARGET_WINDOW_FEATURE_NAMES)
    structure_end = target_end + len(STRUCTURE_FEATURE_NAMES)
    profile_end = structure_end + len(SOURCE_PROFILE_FEATURE_NAMES)
    tower_end = profile_end + len(TWO_TOWER_FEATURE_NAMES)
    graph_end = tower_end + len(GRAPH_WINDOW_NAMES)
    needs_prior = any(stats_end <= idx < prior_end for idx in feature_indices)
    needs_target = any(prior_end <= idx < target_end for idx in feature_indices)
    needs_structure = any(target_end <= idx < structure_end for idx in feature_indices)
    needs_profile = any(structure_end <= idx < profile_end for idx in feature_indices)
    needs_tower = any(profile_end <= idx < tower_end for idx in feature_indices)
    needs_graph = any(tower_end <= idx < graph_end for idx in feature_indices)
    needs_sequence = any(idx >= graph_end for idx in feature_indices)
    return replace(
        config,
        candidate_prior_enabled=config.candidate_prior_enabled and needs_prior,
        target_window_enabled=config.target_window_enabled and needs_target,
        structure_enabled=config.structure_enabled and needs_structure,
        source_profile_enabled=config.source_profile_enabled and needs_profile,
        two_tower_enabled=config.two_tower_enabled and needs_tower,
        gnn_enabled=config.gnn_enabled and needs_graph,
        seq_enabled=config.seq_enabled and needs_sequence,
    )


def _can_reuse_encoder_cache(source_config: TrainingConfig, target_config: TrainingConfig) -> bool:
    """Return True when a deterministic prefix cache can hydrate the target encoder exactly."""
    if not source_config.encoder_state_cache_enabled or not target_config.encoder_state_cache_enabled:
        return False
    if (
        target_config.target_window_enabled
        and (
            not source_config.target_window_enabled
            or source_config.target_window_fractions != target_config.target_window_fractions
        )
    ):
        return False
    if (
        target_config.source_profile_enabled
        and target_config.source_profile_deterministic_enabled
        and (
            not source_config.source_profile_enabled
            or not source_config.source_profile_deterministic_enabled
            or source_config.source_profile_window_size != target_config.source_profile_window_size
            or source_config.source_profile_recent_k != target_config.source_profile_recent_k
        )
    ):
        return False
    if not target_config.structure_enabled:
        return True
    if source_config.structure_future_only_transition_cooccur != target_config.structure_future_only_transition_cooccur:
        return False
    if target_config.structure_transition_enabled and not source_config.structure_transition_enabled:
        return False
    if target_config.structure_cooccur_enabled:
        return (
            source_config.structure_cooccur_enabled
            and source_config.structure_cooccur_history_limit == target_config.structure_cooccur_history_limit
        )
    return True


def _build_supervised_queries(
    positives: InteractionTable,
    encoder: HybridFeatureEncoder,
    dst_pool: np.ndarray,
    config: TrainingConfig,
    rng: np.random.Generator,
    candidate_pool=None,
) -> TestQueryArray:
    test_values, test_weights = test_candidate_arrays(encoder.dataset_profile)
    index = getattr(encoder.structure, "index", None)
    if index is None:
        index = TemporalInteractionIndex()
        index.fit(
            positives,
            build_transitions=False,
            build_cooccurs=False,
        )
    context = NegativeSamplingContext(
        index=index,
        dst_values=encoder.id_map.dst_values,
        test_candidate_values=test_values,
        test_candidate_weights=test_weights,
    )
    if candidate_pool is None:
        candidate_pool = build_candidate_pool(dst_pool, test_values)
    jobs = [
        NegativeSamplingJob(src=int(src), positive_dst=int(dst), query_time=int(time))
        for src, dst, time in zip(positives.src, positives.dst, positives.time, strict=True)
    ]
    negatives_by_event = sample_mixed_negatives_batch(
        jobs=jobs,
        context=context,
        dst_pool=dst_pool,
        num_negatives=config.num_negatives,
        rng=rng,
        hard_negative_ratio=config.hard_negative_ratio,
        popular_negative_ratio=config.popular_negative_ratio,
        test_candidate_negative_ratio=config.test_candidate_negative_ratio,
        candidate_pool=candidate_pool,
        workers=config.negative_sampling_workers,
        verbose=config.verbose,
        label="fusion",
    )
    src = np.empty(len(positives), dtype=np.int32)
    time = np.empty(len(positives), dtype=np.int32)
    candidates = np.empty((len(positives), int(config.num_negatives) + 1), dtype=np.int32)
    for row_idx, (event_src, event_dst, event_time, negatives) in enumerate(
        zip(positives.src, positives.dst, positives.time, negatives_by_event, strict=True)
    ):
        src[row_idx] = int(event_src)
        time[row_idx] = int(event_time)
        candidates[row_idx, 0] = int(event_dst)
        candidates[row_idx, 1:] = np.asarray(negatives, dtype=np.int32)
    return TestQueryArray(src=src, time=time, candidates=candidates)


class SupervisedFeatureBuilder:
    def __init__(
        self,
        *,
        encoder: HybridFeatureEncoder,
        dst_pool: np.ndarray,
        config: TrainingConfig,
        label: str = "supervised_features",
    ) -> None:
        self.encoder = encoder
        self.dst_pool = dst_pool
        self.config = config
        self.label = label
        self.test_values, self.test_weights = test_candidate_arrays(encoder.dataset_profile)
        self.index = getattr(encoder.structure, "index", None)
        self.candidate_pool = build_candidate_pool(dst_pool, self.test_values)
        self.sample_elapsed = 0.0
        self.encode_elapsed = 0.0
        self.write_elapsed = 0.0
        self.negative_checksum = 0

    def batch_for_events(self, positives: InteractionTable, rng: np.random.Generator) -> TestQueryArray:
        sample_start = perf_counter()
        jobs = [
            NegativeSamplingJob(src=int(src), positive_dst=int(dst), query_time=int(time))
            for src, dst, time in zip(positives.src, positives.dst, positives.time, strict=True)
        ]
        negatives_by_event = sample_mixed_negatives_batch(
            jobs=jobs,
            context=self._negative_sampling_context(positives),
            dst_pool=self.dst_pool,
            num_negatives=self.config.num_negatives,
            rng=rng,
            hard_negative_ratio=self.config.hard_negative_ratio,
            popular_negative_ratio=self.config.popular_negative_ratio,
            test_candidate_negative_ratio=self.config.test_candidate_negative_ratio,
            candidate_pool=self.candidate_pool,
            workers=self.config.negative_sampling_workers,
            verbose=self.config.verbose,
            label="fusion",
        )
        self.sample_elapsed += perf_counter() - sample_start
        return _supervised_query_array_from_negatives(positives, negatives_by_event, self.config.num_negatives)

    def _negative_sampling_context(self, positives: InteractionTable) -> NegativeSamplingContext:
        index = self.index
        if index is None:
            index = TemporalInteractionIndex()
            index.fit(
                positives,
                build_transitions=False,
                build_cooccurs=False,
            )
        return NegativeSamplingContext(
            index=index,
            dst_values=self.encoder.id_map.dst_values,
            test_candidate_values=self.test_values,
            test_candidate_weights=self.test_weights,
        )

    def features_for_events(
        self,
        positives: InteractionTable,
        rng: np.random.Generator,
        *,
        batch_id: int | None = None,
        total_batches: int | None = None,
        row_start: int = 0,
        row_end: int | None = None,
        total_rows: int | None = None,
    ) -> tuple[np.ndarray, int]:
        row_end = len(positives) if row_end is None else row_end
        total_rows = row_end if total_rows is None else total_rows
        progress = _supervised_progress_label(
            self.label,
            batch_id=batch_id,
            total_batches=total_batches,
            row_start=row_start,
            row_end=row_end,
            total_rows=total_rows,
        )
        if self.config.verbose:
            log_event(f"[hybrid-fit] {progress} sample_start", enabled=True)
        sample_before = self.sample_elapsed
        batch = self.batch_for_events(positives, rng)
        if self.config.verbose:
            log_event(
                f"[hybrid-fit] {progress} sample_done elapsed={self.sample_elapsed - sample_before:.1f}s "
                f"sample_total={self.sample_elapsed:.1f}s",
                enabled=True,
            )
        self.negative_checksum += _candidate_tail_checksum(batch.candidates[:, 1:])
        if self.config.verbose:
            log_event(f"[hybrid-fit] {progress} encode_start", enabled=True)
        encode_start = perf_counter()
        features = self.encoder.features_for_query_array(batch)
        encode_elapsed = perf_counter() - encode_start
        self.encode_elapsed += encode_elapsed
        if self.config.verbose:
            log_event(
                f"[hybrid-fit] {progress} encode_done elapsed={encode_elapsed:.1f}s "
                f"encode_total={self.encode_elapsed:.1f}s",
                enabled=True,
            )
        return features, _candidate_matrix_checksum(batch.candidates)


def _supervised_query_array_from_negatives(
    positives: InteractionTable,
    negatives_by_event: list[tuple[int, ...]],
    num_negatives: int,
) -> TestQueryArray:
    if len(positives) != len(negatives_by_event):
        raise ValueError("positive events and negatives must have the same row count")
    candidate_count = int(num_negatives) + 1
    src = np.empty(len(positives), dtype=np.int32)
    time = np.empty(len(positives), dtype=np.int32)
    candidates = np.empty((len(positives), candidate_count), dtype=np.int32)
    for row_idx, (event_src, event_dst, event_time, negatives) in enumerate(
        zip(positives.src, positives.dst, positives.time, negatives_by_event, strict=True)
    ):
        if len(negatives) != num_negatives:
            raise ValueError("negative row length does not match num_negatives")
        src[row_idx] = int(event_src)
        time[row_idx] = int(event_time)
        candidates[row_idx, 0] = int(event_dst)
        candidates[row_idx, 1:] = np.asarray(negatives, dtype=np.int32)
    return TestQueryArray(src=src, time=time, candidates=candidates)


def _candidate_matrix_checksum(candidates: np.ndarray) -> int:
    if candidates.size == 0:
        return 0
    values = candidates.astype(np.int64, copy=False)
    weights = np.arange(1, values.shape[1] + 1, dtype=np.int64)
    return int((values * weights).sum(dtype=np.int64))


def _candidate_tail_checksum(candidates: np.ndarray) -> int:
    if candidates.size == 0:
        return 0
    values = candidates.astype(np.int64, copy=False)
    weights = np.arange(1, values.shape[1] + 1, dtype=np.int64)
    return int((values * weights).sum(dtype=np.int64))


def _supervised_progress_label(
    label: str,
    *,
    batch_id: int | None,
    total_batches: int | None,
    row_start: int,
    row_end: int,
    total_rows: int,
) -> str:
    batch_text = ""
    if batch_id is not None:
        batch_text = f" batch={batch_id}"
        if total_batches is not None:
            batch_text += f"/{total_batches}"
    return f"{label}{batch_text} rows={row_start}-{row_end}/{total_rows}"


def _build_supervised_features(
    positives: InteractionTable,
    encoder: HybridFeatureEncoder,
    dst_pool: np.ndarray,
    config: TrainingConfig,
    rng: np.random.Generator,
    label: str = "supervised_features",
) -> np.ndarray:
    if len(positives) == 0:
        return np.empty((0, 0, encoder.feature_dim), dtype=np.float32)

    batch_size = max(int(config.supervised_feature_batch_size), 1)
    candidate_count = int(config.num_negatives) + 1
    shape = (len(positives), candidate_count, encoder.feature_dim)
    features = _empty_feature_matrix(shape, config)
    builder = SupervisedFeatureBuilder(encoder=encoder, dst_pool=dst_pool, config=config, label=label)
    start_time = perf_counter()
    candidate_checksum = 0
    total_batches = (len(positives) + batch_size - 1) // batch_size
    for start in range(0, len(positives), batch_size):
        end = min(start + batch_size, len(positives))
        batch_events = positives[start:end]
        batch_id = start // batch_size + 1
        batch_features, batch_candidate_checksum = builder.features_for_events(
            batch_events,
            rng,
            batch_id=batch_id,
            total_batches=total_batches,
            row_start=start,
            row_end=end,
            total_rows=len(positives),
        )
        progress = _supervised_progress_label(
            label,
            batch_id=batch_id,
            total_batches=total_batches,
            row_start=start,
            row_end=end,
            total_rows=len(positives),
        )
        if config.verbose:
            log_event(f"[hybrid-fit] {progress} write_start", enabled=True)
        write_start = perf_counter()
        features[start:end] = batch_features
        write_elapsed = perf_counter() - write_start
        builder.write_elapsed += write_elapsed
        if config.verbose:
            log_event(
                f"[hybrid-fit] {progress} write_done elapsed={write_elapsed:.1f}s "
                f"write_total={builder.write_elapsed:.1f}s",
                enabled=True,
            )
        candidate_checksum += batch_candidate_checksum
        del batch_features
        if isinstance(features, np.memmap) and (start // batch_size + 1) % FEATURE_MEMMAP_FLUSH_INTERVAL == 0:
            features.flush()
        release_memory()
        if config.verbose and (end == len(positives) or end % max(batch_size * 4, 1) == 0):
            log_event(
                f"[hybrid-fit] {label} rows={end}/{len(positives)} "
                f"batch={len(batch_events)} sample={builder.sample_elapsed:.1f}s "
                f"encode={builder.encode_elapsed:.1f}s write={builder.write_elapsed:.1f}s "
                f"elapsed={perf_counter() - start_time:.1f}s",
                enabled=True,
            )
    if isinstance(features, np.memmap):
        features.flush()
    if config.verbose:
        log_event(
            f"[hybrid-fit] {label} done rows={len(positives)} "
            f"sample={builder.sample_elapsed:.1f}s encode={builder.encode_elapsed:.1f}s "
            f"write={builder.write_elapsed:.1f}s elapsed={perf_counter() - start_time:.1f}s "
            f"candidate_checksum={candidate_checksum} negative_checksum={builder.negative_checksum}",
            enabled=True,
        )
    return features


def _empty_feature_matrix(shape: tuple[int, int, int], config: TrainingConfig) -> np.ndarray:
    if not config.supervised_feature_memmap:
        return np.empty(shape, dtype=np.float32)

    fd, path = tempfile.mkstemp(prefix="jgrec-supervised-features-", suffix=".dat")
    os.close(fd)
    matrix = np.memmap(path, mode="w+", dtype=np.float32, shape=shape)
    _FEATURE_MEMMAP_TEMP_FILES.append(path)
    return matrix


def _release_feature_memmaps() -> None:
    while _FEATURE_MEMMAP_TEMP_FILES:
        temp = _FEATURE_MEMMAP_TEMP_FILES.pop()
        try:
            if isinstance(temp, str):
                os.remove(temp)
            else:
                temp.close()
        except OSError:
            pass


def _sample_negatives(
    src: int,
    positive_dst: int,
    query_time: int,
    encoder: HybridFeatureEncoder,
    dst_pool: np.ndarray,
    num_negatives: int,
    rng: np.random.Generator,
    hard_negative_ratio: float = 0.5,
    popular_negative_ratio: float = 0.25,
    test_candidate_negative_ratio: float = 0.0,
    candidate_pool=None,
) -> tuple[int, ...]:
    test_values, test_weights = test_candidate_arrays(encoder.dataset_profile)
    context = NegativeSamplingContext(
        index=encoder.structure.index,
        dst_values=encoder.id_map.dst_values,
        test_candidate_values=test_values,
        test_candidate_weights=test_weights,
    )
    return sample_mixed_negatives(
        src=src,
        positive_dst=positive_dst,
        query_time=query_time,
        context=context,
        dst_pool=dst_pool,
        num_negatives=num_negatives,
        rng=rng,
        hard_negative_ratio=hard_negative_ratio,
        popular_negative_ratio=popular_negative_ratio,
        test_candidate_negative_ratio=test_candidate_negative_ratio,
        candidate_pool=candidate_pool,
    )


class HybridRankerAdapter:
    name = "hybrid"

    def __init__(self, config: TrainingConfig | None = None, recent_window: int = 32) -> None:
        self.config = config or TrainingConfig()
        self.recent_window = recent_window
        self.impl = TemporalHybridRanker(recent_window=recent_window)

    def fit(self, interactions: InteractionTable, context: FitContext) -> TrainingReport:
        import jittor as jt  # noqa: PLC0415

        jt.flags.use_cuda = 1
        config = replace(
            self.config,
            seed=context.seed,
            verbose=context.verbose,
            dataset_train_path=context.dataset.train_path,
            dataset_test_path=context.dataset.test_path,
        )
        return self.impl.fit(interactions, training_config=config)

    def predict_batch(self, queries: TestQueryArray | list[TestQuery]) -> np.ndarray:
        return self.impl.predict_batch(queries)
