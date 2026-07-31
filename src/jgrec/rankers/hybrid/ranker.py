from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, replace
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

from jgrec.contest_checkpoint import get_model_state, set_model_state
from jgrec.core.io import read_test_queries
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
from jgrec.service_normalizer import (
    StreamingFeatureNormalizer,
    normalizer_drift_report,
    replace_result_normalizer,
)

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
from .structure import (
    COOCCUR_TIME_DECAY_FEATURE_NAMES,
    STRUCTURE_FEATURE_NAMES,
    StructureFeatureTower,
)
from .supervised_feature_cache import SupervisedFeatureCache, supervised_feature_cache_key

if TYPE_CHECKING:
    from .candidate_set_transformer import CandidateSetEnsembleCheckpoint
    from .expert_fusion import ExpertBlendCalibration
    from .fusion import FusionMLP, FusionResult
    from .fusion_lgbm import LGBMFusionResult
    from .oof_models import PureJittorOOFStackingCheckpoint
    from .segment_fusion import SegmentGateResult

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

    def tie_break_prior_for_query_array(self, queries: TestQueryArray) -> np.ndarray:
        return np.zeros(queries.candidates.shape, dtype=np.float64)


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
        resolved_structure_config = structure_config or StructureTowerConfig(enabled=False)
        self.structure = _build_structure_tower(resolved_structure_config)
        self._cooccur_time_decay_enabled = bool(
            resolved_structure_config.enabled
            and resolved_structure_config.cooccur_time_decay_enabled
        )
        self.source_profile = _build_source_profile_tower(
            id_map,
            source_profile_config or SourceProfileConfig(enabled=False),
        )
        self.two_tower = _build_two_tower(
            id_map,
            two_tower_config or TwoTowerConfig(enabled=False),
            dataset_profile,
        )
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
            + (
                COOCCUR_TIME_DECAY_FEATURE_NAMES
                if self._cooccur_time_decay_enabled
                else ()
            )
        )

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def snapshot(self) -> dict[str, Any]:
        return {
            "stats": self.stats.snapshot(),
            "candidate_prior": self.candidate_prior.snapshot() if hasattr(self.candidate_prior, "snapshot") else {},
            "target_window": self.target_window.snapshot() if hasattr(self.target_window, "snapshot") else {},
            "structure": self.structure.snapshot() if hasattr(self.structure, "snapshot") else {},
            "source_profile": self.source_profile.snapshot() if hasattr(self.source_profile, "snapshot") else {},
            "two_tower": self.two_tower.snapshot() if hasattr(self.two_tower, "snapshot") else {},
            "graph": self.graph.snapshot() if hasattr(self.graph, "snapshot") else {},
            "sequence": self.sequence.snapshot() if hasattr(self.sequence, "snapshot") else {},
        }

    def hydrate(self, snapshot: dict[str, Any]) -> None:
        self.stats.hydrate(snapshot["stats"])
        for tower_name in (
            "candidate_prior",
            "target_window",
            "structure",
            "source_profile",
            "two_tower",
            "graph",
            "sequence",
        ):
            tower = getattr(self, tower_name)
            tower_snapshot = snapshot.get(tower_name)
            hydrate = getattr(tower, "hydrate", None)
            if tower_snapshot and callable(hydrate):
                hydrate(tower_snapshot)

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
        feature_parts = [
            stat_features,
            candidate_prior_features,
            target_window_features,
            structure_features,
            source_profile_features,
            two_tower_features,
            graph_features,
            sequence_features,
        ]
        if self._cooccur_time_decay_enabled:
            feature_parts.append(self.structure.time_decay_features_for_queries(queries))
        features = np.concatenate(feature_parts, axis=2).astype(np.float32, copy=False)
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

    def clear_batch_caches(self) -> None:
        if hasattr(self.structure, "_full_src_structure_cache"):
            self.structure._full_src_structure_cache.clear()
            self.structure._full_src_neighbor_cache.clear()
            self.structure._full_dst_source_cache.clear()
            self.structure._full_src_cooccur_cache.clear()
        if hasattr(self.source_profile, "_deterministic_cache"):
            self.source_profile._clear_score_caches()

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


def _build_two_tower(
    id_map: NodeIdMap,
    config: TwoTowerConfig,
    dataset_profile: DatasetProfile | None = None,
) -> Any:
    if not config.enabled:
        return _DisabledTwoTower()
    from .two_tower import TwoTower  # noqa: PLC0415

    return TwoTower(
        id_map=id_map,
        config=config,
        dataset_profile=dataset_profile,
    )


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
        self.lgbm_result: LGBMFusionResult | None = None
        self.segment_gate_result: SegmentGateResult | None = None
        self.setwise_fusion: FusionMLP | None = None
        self.setwise_fusion_result: FusionResult | None = None
        self._setwise_hidden_dim = 64
        self.time_ramp_setwise_fusion: FusionMLP | None = None
        self.time_ramp_setwise_result: FusionResult | None = None
        self._time_ramp_setwise_hidden_dim = 64
        self.time_ramp_config: dict[str, float] | None = None
        self.conservative_window_fusions: dict[str, FusionMLP] = {}
        self.conservative_window_results: dict[str, FusionResult] = {}
        self.conservative_window_hidden_dims: dict[str, int] = {}
        self.conservative_window_config: dict[str, float] | None = None
        self.multi_interest_proxy_state: dict[str, np.ndarray] | None = None
        self.candidate_set_ensemble: (
            CandidateSetEnsembleCheckpoint | None
        ) = None
        self.oof_stacking: PureJittorOOFStackingCheckpoint | None = None
        self.cooccur_lift_auxiliary_state = None
        self.cooccur_lift_auxiliary_model = None
        self.training_report: TrainingReport | None = None
        self.feature_names: tuple[str, ...] = ()
        self._fusion_hidden_dim = 64
        self.dataset_profile: DatasetProfile | None = None
        self.config: TrainingConfig | None = None

    def snapshot(self) -> dict[str, Any]:
        if self.config is None or self.id_map is None or self.encoder is None:
            raise RuntimeError("ranker is not fitted")
        if (
            self.oof_stacking is None
            and
            self.candidate_set_ensemble is None
            and (self.fusion is None or self.fusion_result is None)
        ):
            raise RuntimeError("ranker fusion is not fitted")
        oof_stacking_state = None
        if self.oof_stacking is not None:
            from .oof_models import (  # noqa: PLC0415
                snapshot_pure_jittor_oof_stacking,
            )

            oof_stacking_state = snapshot_pure_jittor_oof_stacking(
                self.oof_stacking
            )
        candidate_set_state = None
        if self.candidate_set_ensemble is not None:
            from .candidate_set_transformer import (  # noqa: PLC0415
                snapshot_candidate_set_ensemble,
            )

            candidate_set_state = snapshot_candidate_set_ensemble(
                self.candidate_set_ensemble
            )
        return {
            "config": self.config,
            "id_map": {
                "src_values": self.id_map.src_values,
                "dst_values": self.id_map.dst_values,
            },
            "recent_window": self.recent_window,
            "feature_names": self.feature_names,
            "fusion_hidden_dim": self._fusion_hidden_dim,
            "dataset_profile": self.dataset_profile,
            "encoder": self.encoder.snapshot(),
            "fusion_state": (
                get_model_state(self.fusion)
                if self.fusion is not None
                else None
            ),
            "fusion_result": self.fusion_result,
            "lgbm_result": self.lgbm_result,
            "segment_gate_result": self.segment_gate_result,
            "setwise_fusion_state": (
                get_model_state(self.setwise_fusion)
                if self.setwise_fusion is not None
                else None
            ),
            "setwise_fusion_result": self.setwise_fusion_result,
            "setwise_hidden_dim": self._setwise_hidden_dim,
            "time_ramp_setwise_fusion_state": (
                get_model_state(self.time_ramp_setwise_fusion)
                if self.time_ramp_setwise_fusion is not None
                else None
            ),
            "time_ramp_setwise_result": self.time_ramp_setwise_result,
            "time_ramp_setwise_hidden_dim": (
                self._time_ramp_setwise_hidden_dim
            ),
            "time_ramp_config": self.time_ramp_config,
            "conservative_window_fusion_states": {
                name: get_model_state(model)
                for name, model in self.conservative_window_fusions.items()
            },
            "conservative_window_results": (
                self.conservative_window_results
            ),
            "conservative_window_hidden_dims": (
                self.conservative_window_hidden_dims
            ),
            "conservative_window_config": self.conservative_window_config,
            "multi_interest_proxy_state": self.multi_interest_proxy_state,
            "cooccur_lift_auxiliary_state": (
                self.cooccur_lift_auxiliary_state
            ),
            "oof_stacking_state": oof_stacking_state,
            "candidate_set_ensemble_state": candidate_set_state,
            "training_report": self.training_report,
        }

    def hydrate(self, snapshot: dict[str, Any]) -> None:
        self.config = snapshot["config"]
        self.recent_window = int(snapshot["recent_window"])
        self.feature_names = tuple(snapshot["feature_names"])
        self._fusion_hidden_dim = int(snapshot["fusion_hidden_dim"])
        self.dataset_profile = snapshot.get("dataset_profile")
        id_map_state = snapshot["id_map"]
        src_values = tuple(int(value) for value in id_map_state["src_values"])
        dst_values = tuple(int(value) for value in id_map_state["dst_values"])
        self.id_map = NodeIdMap(
            src_to_id={value: idx for idx, value in enumerate(src_values)},
            dst_to_id={value: idx for idx, value in enumerate(dst_values)},
            src_values=src_values,
            dst_values=dst_values,
        )
        self.encoder = HybridFeatureEncoder(
            id_map=self.id_map,
            recent_window=self.recent_window,
            graph_config=self.config.graph_config(),
            sequence_config=self.config.sequence_config(),
            candidate_prior_config=self.config.candidate_prior_config(),
            target_window_config=self.config.target_window_config(),
            dataset_profile=self.dataset_profile,
            two_tower_config=self.config.two_tower_config(),
            source_profile_config=self.config.source_profile_config(),
            structure_config=self.config.structure_config(),
        )
        self.encoder.hydrate(snapshot["encoder"])
        cooccur_lift_state = snapshot.get(
            "cooccur_lift_auxiliary_state"
        )
        if cooccur_lift_state is None:
            self.cooccur_lift_auxiliary_state = None
            self.cooccur_lift_auxiliary_model = None
        else:
            from .cooccur_lift_checkpoint import (  # noqa: PLC0415
                CooccurLiftAuxiliaryState,
                build_cooccur_lift_auxiliary_model,
            )

            if not isinstance(
                cooccur_lift_state,
                CooccurLiftAuxiliaryState,
            ):
                raise TypeError(
                    "checkpoint cooccur-lift auxiliary state is invalid"
                )
            self.cooccur_lift_auxiliary_state = cooccur_lift_state
            self.cooccur_lift_auxiliary_model = (
                build_cooccur_lift_auxiliary_model(
                    cooccur_lift_state
                )
            )
        oof_stacking_state = snapshot.get("oof_stacking_state")
        if oof_stacking_state is not None:
            if self.cooccur_lift_auxiliary_state is not None:
                raise ValueError(
                    "cooccur-lift auxiliary does not support OOF stacking"
                )
            from .oof_models import (  # noqa: PLC0415
                hydrate_pure_jittor_oof_stacking,
            )

            stacking = hydrate_pure_jittor_oof_stacking(
                oof_stacking_state
            )
            self._validate_oof_stacking_feature_contract(stacking)
            self.oof_stacking = stacking
            self.candidate_set_ensemble = None
            self.fusion = None
            self.fusion_result = None
            self.lgbm_result = None
            self.segment_gate_result = None
            self.setwise_fusion = None
            self.setwise_fusion_result = None
            self.time_ramp_setwise_fusion = None
            self.time_ramp_setwise_result = None
            self.time_ramp_config = None
            self.conservative_window_fusions = {}
            self.conservative_window_results = {}
            self.conservative_window_hidden_dims = {}
            self.conservative_window_config = None
            self.multi_interest_proxy_state = None
            self.training_report = snapshot.get("training_report")
            return
        candidate_set_state = snapshot.get(
            "candidate_set_ensemble_state"
        )
        if candidate_set_state is not None:
            if self.cooccur_lift_auxiliary_state is not None:
                raise ValueError(
                    "cooccur-lift auxiliary does not support candidate-set "
                    "ensembles"
                )
            from .candidate_set_transformer import (  # noqa: PLC0415
                hydrate_candidate_set_ensemble,
            )

            ensemble = hydrate_candidate_set_ensemble(
                candidate_set_state
            )
            self._validate_candidate_set_feature_contract(ensemble)
            self.candidate_set_ensemble = ensemble
            self.oof_stacking = None
            self.fusion = None
            self.fusion_result = None
            self.lgbm_result = None
            self.segment_gate_result = None
            self.setwise_fusion = None
            self.setwise_fusion_result = None
            self.time_ramp_setwise_fusion = None
            self.time_ramp_setwise_result = None
            self.time_ramp_config = None
            self.conservative_window_fusions = {}
            self.conservative_window_results = {}
            self.conservative_window_hidden_dims = {}
            self.conservative_window_config = None
            self.multi_interest_proxy_state = None
            self.training_report = snapshot.get("training_report")
            return

        from .fusion import FusionMLP  # noqa: PLC0415

        self.candidate_set_ensemble = None
        self.oof_stacking = None
        self.fusion_result = snapshot["fusion_result"]
        self.fusion = FusionMLP(
            input_dim=_fusion_result_input_dim(self.fusion_result),
            hidden_dim=self._fusion_hidden_dim,
        )
        set_model_state(self.fusion, snapshot["fusion_state"])
        self.lgbm_result = snapshot.get("lgbm_result")
        self.segment_gate_result = snapshot.get("segment_gate_result")
        self.setwise_fusion_result = snapshot.get("setwise_fusion_result")
        self._setwise_hidden_dim = int(snapshot.get("setwise_hidden_dim", 64))
        proxy_state = snapshot.get("multi_interest_proxy_state")
        self.multi_interest_proxy_state = (
            {
                str(key): np.asarray(value, dtype=np.float32)
                for key, value in proxy_state.items()
            }
            if proxy_state is not None
            else None
        )
        setwise_state = snapshot.get("setwise_fusion_state")
        if self.setwise_fusion_result is None:
            if setwise_state is not None:
                raise ValueError(
                    "checkpoint has Setwise state without a Setwise result"
                )
            self.setwise_fusion = None
        else:
            if setwise_state is None:
                raise ValueError(
                    "checkpoint has a Setwise result without model state"
                )
            self.setwise_fusion = FusionMLP(
                input_dim=_fusion_result_input_dim(
                    self.setwise_fusion_result
                ),
                hidden_dim=self._setwise_hidden_dim,
            )
            set_model_state(self.setwise_fusion, setwise_state)
        self.time_ramp_setwise_result = snapshot.get(
            "time_ramp_setwise_result"
        )
        self._time_ramp_setwise_hidden_dim = int(
            snapshot.get("time_ramp_setwise_hidden_dim", 64)
        )
        self.time_ramp_config = snapshot.get("time_ramp_config")
        time_ramp_state = snapshot.get("time_ramp_setwise_fusion_state")
        if (
            self.time_ramp_setwise_result is None
            and time_ramp_state is None
            and self.time_ramp_config is None
        ):
            self.time_ramp_setwise_fusion = None
        elif (
            self.time_ramp_setwise_result is None
            or time_ramp_state is None
            or self.time_ramp_config is None
        ):
            raise ValueError("checkpoint has an incomplete time-ramp expert")
        else:
            power = float(self.time_ramp_config["power"])
            minimum_time = float(self.time_ramp_config["minimum_time"])
            maximum_time = float(self.time_ramp_config["maximum_time"])
            if (
                not np.isfinite(power)
                or power <= 0.0
                or not np.isfinite(minimum_time)
                or not np.isfinite(maximum_time)
                or maximum_time <= minimum_time
            ):
                raise ValueError("checkpoint has an invalid time-ramp config")
            self.time_ramp_config = {
                "power": power,
                "minimum_time": minimum_time,
                "maximum_time": maximum_time,
            }
            self.time_ramp_setwise_fusion = FusionMLP(
                input_dim=_fusion_result_input_dim(
                    self.time_ramp_setwise_result
                ),
                hidden_dim=self._time_ramp_setwise_hidden_dim,
            )
            set_model_state(
                self.time_ramp_setwise_fusion,
                time_ramp_state,
            )
        conservative_states = snapshot.get(
            "conservative_window_fusion_states",
            {},
        )
        self.conservative_window_results = snapshot.get(
            "conservative_window_results",
            {},
        )
        self.conservative_window_hidden_dims = {
            str(name): int(hidden_dim)
            for name, hidden_dim in snapshot.get(
                "conservative_window_hidden_dims",
                {},
            ).items()
        }
        self.conservative_window_config = snapshot.get(
            "conservative_window_config"
        )
        conservative_keys = set(conservative_states)
        if (
            conservative_keys != set(self.conservative_window_results)
            or conservative_keys != set(self.conservative_window_hidden_dims)
        ):
            raise ValueError(
                "checkpoint has incomplete conservative window experts"
            )
        if self.conservative_window_config is None:
            if conservative_keys:
                raise ValueError(
                    "checkpoint has conservative experts without config"
                )
            self.conservative_window_fusions = {}
        else:
            alpha = float(self.conservative_window_config["alpha"])
            if (
                not np.isfinite(alpha)
                or not 0.0 < alpha <= 1.0
                or not conservative_keys
                or self.setwise_fusion is None
                or self.setwise_fusion_result is None
            ):
                raise ValueError(
                    "checkpoint has an invalid conservative window config"
                )
            self.conservative_window_config = {"alpha": alpha}
            self.conservative_window_fusions = {}
            for name in sorted(conservative_keys):
                result = self.conservative_window_results[name]
                model = FusionMLP(
                    input_dim=_fusion_result_input_dim(result),
                    hidden_dim=self.conservative_window_hidden_dims[name],
                )
                set_model_state(model, conservative_states[name])
                self.conservative_window_fusions[name] = model
        self.training_report = snapshot.get("training_report")

    def install_candidate_set_ensemble(
        self,
        ensemble: CandidateSetEnsembleCheckpoint,
    ) -> None:
        """Replace all legacy trainable rerankers with a pure-Jittor head."""
        if self.config is None or self.id_map is None or self.encoder is None:
            raise RuntimeError(
                "candidate-set ensemble requires a fitted encoder"
            )
        from .candidate_set_transformer import (  # noqa: PLC0415
            snapshot_candidate_set_ensemble,
        )

        self._validate_candidate_set_feature_contract(ensemble)
        snapshot_candidate_set_ensemble(ensemble)
        self.candidate_set_ensemble = ensemble
        self.oof_stacking = None
        self.fusion = None
        self.fusion_result = None
        self.lgbm_result = None
        self.segment_gate_result = None
        self.setwise_fusion = None
        self.setwise_fusion_result = None
        self.time_ramp_setwise_fusion = None
        self.time_ramp_setwise_result = None
        self.time_ramp_config = None
        self.conservative_window_fusions = {}
        self.conservative_window_results = {}
        self.conservative_window_hidden_dims = {}
        self.conservative_window_config = None
        self.multi_interest_proxy_state = None

    def install_pure_jittor_oof_stacking(
        self,
        stacking: PureJittorOOFStackingCheckpoint,
    ) -> None:
        """Replace legacy rerankers with a pure-Jittor OOF stack."""
        if self.config is None or self.id_map is None or self.encoder is None:
            raise RuntimeError(
                "OOF stacking requires a fitted encoder"
            )
        from .oof_models import (  # noqa: PLC0415
            snapshot_pure_jittor_oof_stacking,
        )

        self._validate_oof_stacking_feature_contract(stacking)
        snapshot_pure_jittor_oof_stacking(stacking)
        self.oof_stacking = stacking
        self.candidate_set_ensemble = None
        self.fusion = None
        self.fusion_result = None
        self.lgbm_result = None
        self.segment_gate_result = None
        self.setwise_fusion = None
        self.setwise_fusion_result = None
        self.time_ramp_setwise_fusion = None
        self.time_ramp_setwise_result = None
        self.time_ramp_config = None
        self.conservative_window_fusions = {}
        self.conservative_window_results = {}
        self.conservative_window_hidden_dims = {}
        self.conservative_window_config = None
        self.multi_interest_proxy_state = None

    def _validate_candidate_set_feature_contract(
        self,
        ensemble: CandidateSetEnsembleCheckpoint,
    ) -> None:
        if not ensemble.results:
            raise ValueError("candidate-set ensemble has no experts")
        for result in ensemble.results:
            if result.feature_names != self.feature_names:
                raise ValueError(
                    "candidate-set feature names do not match the encoder"
                )
            if result.model_config.input_dim != len(self.feature_names):
                raise ValueError(
                    "candidate-set input dimension does not match the encoder"
                )

    def _validate_oof_stacking_feature_contract(
        self,
        stacking: PureJittorOOFStackingCheckpoint,
    ) -> None:
        for result in stacking.cst_experts.results:
            if result.feature_names != self.feature_names:
                raise ValueError(
                    "OOF stacking CST features do not match the encoder"
                )
        setwise_result = stacking.setwise_mlp[1]
        if setwise_result.feature_names != self.feature_names:
            raise ValueError(
                "OOF stacking Setwise features do not match the encoder"
            )

    def fit(self, interactions: InteractionTable, training_config: TrainingConfig) -> TrainingReport:
        if len(interactions) == 0:
            raise ValueError("training interactions are empty")

        interactions = interactions.sort_by_time()
        if training_config.max_fit_events > 0 and len(interactions) > training_config.max_fit_events:
            interactions = interactions.tail(training_config.max_fit_events)
        self.id_map = NodeIdMap.from_interactions(interactions)
        self.feature_names = _hybrid_feature_names(training_config)
        self._fusion_hidden_dim = training_config.fusion_hidden_dim

        training_config = self._apply_auto_strategy(interactions, training_config)
        self.config = training_config
        self.encoder = None
        log_memory("hybrid_fit_start", enabled=training_config.verbose)
        log(f"[hybrid-fit] start events={len(interactions)}", enabled=training_config.verbose)
        fusion, fusion_result, lgbm_result, report, encoder_cache, cache_config = self._learn_fusion(
            interactions, training_config,
        )
        self.fusion = fusion
        self.fusion_result = fusion_result
        self.lgbm_result = lgbm_result
        self.segment_gate_result = None
        self.setwise_fusion = None
        self.setwise_fusion_result = None
        self.time_ramp_setwise_fusion = None
        self.time_ramp_setwise_result = None
        self.time_ramp_config = None
        self.conservative_window_fusions = {}
        self.conservative_window_results = {}
        self.conservative_window_hidden_dims = {}
        self.conservative_window_config = None
        self.multi_interest_proxy_state = None
        self.candidate_set_ensemble = None
        self.oof_stacking = None

        final_future_only = self._can_use_future_only_final_encoder()
        if training_config.refit_full:
            rng = np.random.default_rng(training_config.seed + 10_000)
            final_config = _config_for_selected_features(
                training_config,
                fusion_result.feature_indices,
            )
            if final_future_only:
                final_config = replace(
                    final_config,
                    structure_future_only_transition_cooccur=True,
                )
            cache = self._final_encoder_cache(
                interactions=interactions,
                final_config=final_config,
                existing_cache=encoder_cache,
                existing_config=cache_config,
                verbose=training_config.verbose,
            )
            final_snapshot = (
                cache.snapshot_for_prefix(len(interactions))
                if cache is not None
                else None
            )
            log_event(
                "[hybrid-fit] final_encoder refit=full "
                f"prior={final_config.candidate_prior_enabled} "
                f"prior_test_freq={final_config.candidate_prior_include_test_frequency} "
                f"profile={final_config.source_profile_enabled} "
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
        else:
            if self.encoder is None:
                val_size = max(1, int(len(interactions) * training_config.val_ratio))
                train_end = max(2, len(interactions) - val_size)
                serving_config = replace(
                    cache_config,
                    structure_future_only_transition_cooccur=True,
                )
                self.encoder = self._fit_encoder(
                    interactions[:train_end],
                    serving_config,
                    np.random.default_rng(training_config.seed + 20_000),
                    verbose=training_config.verbose,
                )
            if encoder_cache is not None:
                encoder_cache.clear()
            log_event(
                "[hybrid-fit] final_encoder refit=off "
                "serving_state=validation_context",
                enabled=training_config.verbose,
            )
            log_memory("final_encoder_start", enabled=training_config.verbose)
        if final_future_only:
            self.encoder.compact_for_future_queries()
        log_memory("final_encoder_done", enabled=training_config.verbose)
        metrics = dict(report.metrics)
        metrics["refit_full"] = float(training_config.refit_full)
        report = replace(report, metrics=metrics)
        if (
            training_config.service_normalizer_calibration_enabled
            and training_config.dataset_test_path is not None
        ):
            log_event(
                "[hybrid-fit] service_normalizer_start "
                f"path={training_config.dataset_test_path}",
                enabled=training_config.verbose,
            )
            service_queries = read_test_queries(
                training_config.dataset_test_path
            )
            drift = self.recalibrate_service_normalizers(
                service_queries,
                batch_size=(
                    training_config.service_normalizer_calibration_batch_size
                ),
            )
            metrics = dict(report.metrics)
            metrics.update(_service_normalizer_report_metrics(drift))
            report = replace(report, metrics=metrics)
            max_shift = metrics[
                "service_normalizer_max_abs_mean_shift_in_training_std"
            ]
            log_event(
                "[hybrid-fit] service_normalizer_done "
                f"heads={len(drift)} max_standardized_mean_shift={max_shift:.6f}",
                enabled=training_config.verbose,
            )
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

    def recalibrate_service_normalizers(
        self,
        queries: TestQueryArray | list[TestQuery],
        *,
        batch_size: int = 256,
    ) -> dict[str, dict[str, int | float]]:
        """Recompute neural-head normalizers on unlabeled service features."""
        if not queries:
            raise ValueError("service normalizer calibration requires non-empty queries")
        if self.encoder is None:
            raise RuntimeError("service normalizer calibration requires a fitted encoder")
        if batch_size <= 0:
            raise ValueError("service normalizer batch_size must be positive")
        if not isinstance(queries, TestQueryArray):
            queries = TestQueryArray.from_queries(queries)

        raw_results: dict[str, FusionResult] = {}
        if self.fusion_result is not None:
            raw_results["fusion"] = self.fusion_result

        setwise_results: dict[str, FusionResult] = {}
        if self.setwise_fusion_result is not None:
            setwise_results["setwise"] = self.setwise_fusion_result
        if self.time_ramp_setwise_result is not None:
            setwise_results["time_ramp_setwise"] = (
                self.time_ramp_setwise_result
            )
        for name, result in self.conservative_window_results.items():
            setwise_results[f"conservative_window:{name}"] = result
        if not raw_results and not setwise_results:
            raise RuntimeError(
                "service normalizer calibration found no supported neural head"
            )

        accumulators = {
            name: StreamingFeatureNormalizer()
            for name in (*raw_results, *setwise_results)
        }
        for start in range(0, len(queries), batch_size):
            query_batch = queries[start : start + batch_size]
            raw_features = self.encoder.features_for_query_array(query_batch)
            for name, result in raw_results.items():
                accumulators[name].update(
                    _select_result_features(raw_features, result)
                )
            if setwise_results:
                setwise_source_features = raw_features
                if self.multi_interest_proxy_state is not None:
                    if self.id_map is None:
                        raise RuntimeError(
                            "multi-interest calibration requires an id map"
                        )
                    from .multi_interest_proxy import (  # noqa: PLC0415
                        append_multi_interest_features,
                    )

                    setwise_source_features = (
                        append_multi_interest_features(
                            raw_features,
                            query_batch,
                            self.id_map,
                            self.multi_interest_proxy_state,
                        )
                    )
                from .setwise import (  # noqa: PLC0415
                    setwise_context_features,
                )

                setwise_features = setwise_context_features(
                    setwise_source_features
                )
                for name, result in setwise_results.items():
                    accumulators[name].update(
                        _select_result_features(
                            setwise_features,
                            result,
                        )
                    )
            self.encoder.clear_batch_caches()

        calibrated: dict[str, FusionResult] = {}
        report: dict[str, dict[str, int | float]] = {}
        for name, result in {
            **raw_results,
            **setwise_results,
        }.items():
            service = accumulators[name].finalize()
            report[name] = normalizer_drift_report(
                training_mean=result.mean,
                training_std=result.std,
                service=service,
            )
            calibrated[name] = replace_result_normalizer(
                result,
                service,
            )

        if "fusion" in calibrated:
            self.fusion_result = calibrated["fusion"]
        if "setwise" in calibrated:
            self.setwise_fusion_result = calibrated["setwise"]
        if "time_ramp_setwise" in calibrated:
            self.time_ramp_setwise_result = calibrated[
                "time_ramp_setwise"
            ]
        for name in self.conservative_window_results:
            self.conservative_window_results[name] = calibrated[
                f"conservative_window:{name}"
            ]
        return report

    def prediction_order(self, queries: TestQueryArray) -> np.ndarray | None:
        if self.encoder is None or not queries:
            return None
        if not np.all(queries.time > self.encoder.stats.max_time):
            return None
        return np.argsort(queries.src, kind="stable")

    def prediction_tie_break_prior(
        self,
        queries: TestQueryArray | list[TestQuery],
    ) -> np.ndarray:
        if self.encoder is None:
            raise RuntimeError("ranker is not fitted")
        if not isinstance(queries, TestQueryArray):
            queries = TestQueryArray.from_queries(queries)
        return self.encoder.candidate_prior.tie_break_prior_for_query_array(
            queries
        )

    def predict_batch(self, queries: TestQueryArray | list[TestQuery]) -> np.ndarray:
        if not queries:
            return np.empty((0, 100), dtype=np.float64)
        if self.encoder is None:
            raise RuntimeError("ranker is not fitted")

        features = self.encoder.features_for_queries(queries)
        if self.oof_stacking is not None:
            from .oof_models import (  # noqa: PLC0415
                predict_pure_jittor_oof_stacking_scores,
            )

            scores = predict_pure_jittor_oof_stacking_scores(
                self.oof_stacking,
                features,
                batch_size=128,
            )
            return scores.astype(np.float64, copy=False)
        if self.candidate_set_ensemble is not None:
            from .candidate_set_transformer import (  # noqa: PLC0415
                predict_candidate_set_ensemble_probabilities,
            )

            probabilities = (
                predict_candidate_set_ensemble_probabilities(
                    self.candidate_set_ensemble,
                    features,
                    batch_size=128,
                )
            )
            return probabilities.astype(np.float64, copy=False)
        if self.fusion is None or self.fusion_result is None:
            raise RuntimeError("ranker is not fitted")

        from .fusion import predict_logits  # noqa: PLC0415

        if self.multi_interest_proxy_state is not None:
            from .multi_interest_proxy import (  # noqa: PLC0415
                append_multi_interest_features,
            )

            if not isinstance(queries, TestQueryArray):
                queries = TestQueryArray.from_queries(queries)
            setwise_source_features = append_multi_interest_features(
                features,
                queries,
                self.id_map,
                self.multi_interest_proxy_state,
            )
        else:
            setwise_source_features = features
        selected, lgbm_selected = _select_expert_features(
            features,
            mlp_indices=self.fusion_result.feature_indices,
            lgbm_indices=(
                self.lgbm_result.feature_indices
                if self.lgbm_result is not None
                else self.fusion_result.feature_indices
            ),
        )
        setwise_context: np.ndarray | None = None
        champion_setwise_probs: np.ndarray | None = None
        if (
            self.setwise_fusion is not None
            and self.setwise_fusion_result is not None
        ):
            from .setwise import setwise_context_features  # noqa: PLC0415

            setwise_context = setwise_context_features(
                setwise_source_features
            )
            setwise_features = setwise_context
            setwise_indices = self.setwise_fusion_result.feature_indices
            if setwise_indices != tuple(range(setwise_features.shape[-1])):
                setwise_features = setwise_features[..., setwise_indices]
            setwise_logits = predict_logits(
                self.setwise_fusion,
                setwise_features,
                self.setwise_fusion_result.mean,
                self.setwise_fusion_result.std,
            )
            neural_logits = setwise_logits
            probs = _softmax(neural_logits)
            champion_setwise_probs = probs
        else:
            mlp_logits = predict_logits(
                self.fusion,
                selected,
                self.fusion_result.mean,
                self.fusion_result.std,
            )
            neural_logits = mlp_logits
            probs = _softmax(neural_logits)

        if self.lgbm_result is not None:
            from .fusion_lgbm import predict_logits_lgbm  # noqa: PLC0415

            lgbm_logits = predict_logits_lgbm(self.lgbm_result.model_text, lgbm_selected)
            if (
                self.setwise_fusion_result is not None
                or self.segment_gate_result is None
            ):
                mlp_weight: float | np.ndarray = self.lgbm_result.mlp_weight
            else:
                from .segment_fusion import predict_segment_weights  # noqa: PLC0415

                mlp_weight = predict_segment_weights(self.segment_gate_result, features, self.feature_names)
            from .expert_fusion import blend_expert_logits  # noqa: PLC0415

            probs = blend_expert_logits(
                neural_logits,
                lgbm_logits,
                mlp_weight,
                calibration=_expert_blend_calibration(self.lgbm_result),
            )

        if self.conservative_window_config is not None:
            from .conservative_window_blend import (  # noqa: PLC0415
                conservative_window_scores,
            )
            from .setwise import setwise_context_features  # noqa: PLC0415

            if (
                champion_setwise_probs is None
                or not self.conservative_window_fusions
            ):
                raise RuntimeError(
                    "conservative window fusion requires a Setwise champion"
                )
            if setwise_context is None:
                setwise_context = setwise_context_features(
                    setwise_source_features
                )
            window_expert_probs = [champion_setwise_probs]
            for name, model in self.conservative_window_fusions.items():
                result = self.conservative_window_results[name]
                expert_features = setwise_context
                if result.feature_indices != tuple(
                    range(expert_features.shape[-1])
                ):
                    expert_features = expert_features[
                        ..., result.feature_indices
                    ]
                logits = predict_logits(
                    model,
                    expert_features,
                    result.mean,
                    result.std,
                )
                window_expert_probs.append(_softmax(logits))
            window_probs = np.mean(
                np.stack(window_expert_probs, axis=0),
                axis=0,
            )
            if self.lgbm_result is not None:
                from .expert_fusion import blend_expert_logits  # noqa: PLC0415

                window_probs = blend_expert_logits(
                    np.log(np.clip(window_probs, 1e-300, 1.0)),
                    lgbm_logits,
                    mlp_weight,
                    calibration=_expert_blend_calibration(
                        self.lgbm_result
                    ),
                )
            probs = conservative_window_scores(
                probs,
                window_probs,
                alpha=self.conservative_window_config["alpha"],
            )

        if (
            self.time_ramp_setwise_fusion is not None
            and self.time_ramp_setwise_result is not None
            and self.time_ramp_config is not None
        ):
            from .setwise import setwise_context_features  # noqa: PLC0415
            from .time_ramp import (  # noqa: PLC0415
                blend_query_scores,
                time_ramp_weights,
            )

            if setwise_context is None:
                setwise_context = setwise_context_features(
                    setwise_source_features
                )
            time_ramp_features = setwise_context
            time_ramp_indices = (
                self.time_ramp_setwise_result.feature_indices
            )
            if time_ramp_indices != tuple(
                range(time_ramp_features.shape[-1])
            ):
                time_ramp_features = time_ramp_features[
                    ..., time_ramp_indices
                ]
            time_ramp_logits = predict_logits(
                self.time_ramp_setwise_fusion,
                time_ramp_features,
                self.time_ramp_setwise_result.mean,
                self.time_ramp_setwise_result.std,
            )
            time_ramp_probs = _softmax(time_ramp_logits)
            if isinstance(queries, TestQueryArray):
                query_times = queries.time
            else:
                query_times = np.fromiter(
                    (query.time for query in queries),
                    dtype=np.int64,
                    count=len(queries),
                )
            query_weights = time_ramp_weights(
                query_times,
                power=self.time_ramp_config["power"],
                minimum_time=self.time_ramp_config["minimum_time"],
                maximum_time=self.time_ramp_config["maximum_time"],
            )
            probs = blend_query_scores(
                probs,
                time_ramp_probs,
                query_weights,
            )

        if self.cooccur_lift_auxiliary_state is not None:
            from .cooccur_lift_checkpoint import (  # noqa: PLC0415
                predict_cooccur_lift_auxiliary_probabilities,
            )

            if self.cooccur_lift_auxiliary_model is None:
                raise RuntimeError(
                    "cooccur-lift auxiliary model is not hydrated"
                )
            if not isinstance(queries, TestQueryArray):
                queries = TestQueryArray.from_queries(queries)
            auxiliary = predict_cooccur_lift_auxiliary_probabilities(
                self.cooccur_lift_auxiliary_state,
                self.cooccur_lift_auxiliary_model,
                features,
                queries,
            )
            weight = float(self.cooccur_lift_auxiliary_state.weight)
            probs = (1.0 - weight) * probs + weight * auxiliary

        return probs.astype(np.float64, copy=False)

    def _learn_fusion(
        self,
        interactions: InteractionTable,
        config: TrainingConfig,
    ) -> tuple[FusionMLP, FusionResult, TrainingReport, HybridPrefixStateCache | None, TrainingConfig]:
        n_events = len(interactions)
        train_num_negatives = config.resolved_train_num_negatives()
        val_num_negatives = config.resolved_val_num_negatives()
        if n_events < 100 or config.epochs < 1:
            raise ValueError(
                "not enough training signal for hybrid reranker: "
                f"events={n_events}, train_num_negatives={train_num_negatives}, "
                f"val_num_negatives={val_num_negatives}, epochs={config.epochs}"
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
            f"dst={len(dst_pool)} train_negatives={train_num_negatives} "
            f"val_negatives={val_num_negatives}",
            enabled=config.verbose,
        )
        log_memory("split_done", enabled=config.verbose)

        supervised_encoder_config = replace(config, structure_future_only_transition_cooccur=True)
        feature_cache: SupervisedFeatureCache | None = None
        feature_cache_key = ""
        cached_features: tuple[np.ndarray, np.ndarray] | None = None
        cached_fusion_rng_state: dict[str, Any] | None = None
        feature_cache_dir = getattr(config, "supervised_feature_cache_dir", None)
        if feature_cache_dir is not None:
            cache_start = perf_counter()
            feature_cache = SupervisedFeatureCache(feature_cache_dir)
            feature_cache_key = supervised_feature_cache_key(
                interactions,
                supervised_encoder_config,
                recent_window=self.recent_window,
                feature_names=self.feature_names,
                dataset_profile=self.dataset_profile,
            )
            cached_features = feature_cache.load(feature_cache_key)
            if cached_features is not None:
                cached_fusion_rng_state = feature_cache.load_fusion_rng_state(feature_cache_key)
                expected_train_shape = (len(train_events), train_num_negatives + 1, len(self.feature_names))
                expected_val_shape = (len(val_events), val_num_negatives + 1, len(self.feature_names))
                if (
                    cached_fusion_rng_state is None
                    or cached_features[0].shape != expected_train_shape
                    or cached_features[1].shape != expected_val_shape
                ):
                    cached_features = None
            cache_status = "hit" if cached_features is not None else "miss"
            log_event(
                f"[supervised-feature-cache] {cache_status} key={feature_cache_key[:12]} "
                f"path={feature_cache.root} elapsed={perf_counter() - cache_start:.1f}s",
                enabled=config.verbose,
            )

        encoder_cache: HybridPrefixStateCache | None = None
        if cached_features is not None:
            train_features, val_features = cached_features
            if cached_fusion_rng_state is None:
                raise RuntimeError("supervised feature cache hit is missing fusion RNG state")
            rng.bit_generator.state = cached_fusion_rng_state
            log_event(
                f"[hybrid-fit] cached_features train_shape={train_features.shape} val_shape={val_features.shape}",
                enabled=config.verbose,
            )
        else:
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
                replace(config, num_negatives=train_num_negatives),
                rng,
                label="train_features",
            )
            del train_encoder
            release_memory()
            log_event(
                f"[hybrid-fit] train_features shape={train_features.shape} "
                f"elapsed={perf_counter() - feature_start:.1f}s",
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
                replace(config, num_negatives=val_num_negatives),
                rng,
                label="val_features",
            )
            if config.refit_full:
                del val_encoder
            else:
                self.encoder = val_encoder
            log_event(
                f"[hybrid-fit] val_features shape={val_features.shape} "
                f"elapsed={perf_counter() - feature_start:.1f}s",
                enabled=config.verbose,
            )
            log_memory("val_features_done", enabled=config.verbose)
            if feature_cache is not None:
                cache_start = perf_counter()
                try:
                    feature_cache.save(
                        feature_cache_key,
                        train_features,
                        val_features,
                        fusion_rng_state=rng.bit_generator.state,
                    )
                    log_event(
                        f"[supervised-feature-cache] saved key={feature_cache_key[:12]} "
                        f"path={feature_cache.root} elapsed={perf_counter() - cache_start:.1f}s",
                        enabled=config.verbose,
                    )
                except (OSError, ValueError) as exc:
                    log_event(
                        f"[supervised-feature-cache] save_failed key={feature_cache_key[:12]} "
                        f"reason={type(exc).__name__}: {exc}",
                        enabled=config.verbose,
                    )

        fusion_start = perf_counter()
        log_memory("fusion_start", enabled=config.verbose)
        fusion, result = self._fit_best_fusion(
            train_features=train_features,
            val_features=val_features,
            config=config,
            rng=rng,
            verbose=config.verbose,
        )
        lgbm_result = self._fit_lgbm_fusion(
            train_features, val_features, result.feature_indices, result.candidate_name, config,
            mlp_model=fusion, mlp_result=result,
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
                "fusion_context_transform_version": float(
                    config.resolved_fusion_context_transform_version()
                ),
            },
        )
        return fusion, result, lgbm_result, report, encoder_cache, supervised_encoder_config

    def _fit_lgbm_fusion(
        self,
        train_features: np.ndarray,
        val_features: np.ndarray,
        feature_indices: tuple[int, ...],
        candidate_name: str,
        config: TrainingConfig,
        mlp_model: FusionMLP | None = None,
        mlp_result: FusionResult | None = None,
    ) -> LGBMFusionResult | None:
        if config.fusion_mode not in ("lgbm", "ensemble"):
            return None
        from .fusion_lgbm import fit_fusion_lgbm  # noqa: PLC0415

        log_memory("lgbm_fusion_start", enabled=config.verbose)
        result = fit_fusion_lgbm(
            train_features=train_features,
            val_features=val_features,
            selection_metric=config.selection_metric,
            verbose=config.verbose,
            feature_indices=feature_indices,
            candidate_name=candidate_name,
        )
        mlp_weight, calibration = _find_ensemble_weight(
            mlp_model, mlp_result, result, val_features, feature_indices, config,
        )
        from dataclasses import replace as dc_replace  # noqa: PLC0415
        result = dc_replace(
            result,
            mlp_weight=mlp_weight,
            blend_mode=calibration.mode,
            mlp_temperature=calibration.mlp_temperature,
            lgbm_temperature=calibration.lgbm_temperature,
            rrf_k=calibration.rrf_k,
        )
        log_event(
            f"[fusion-lgbm-select] chosen={result.candidate_name} "
            f"val_ap={result.best_val_ap:.5f} val_mrr={result.best_val_mrr:.5f} "
            f"blend={calibration.mode} mlp_weight={mlp_weight:.2f} "
            f"temperatures={calibration.mlp_temperature:.3f}/"
            f"{calibration.lgbm_temperature:.3f} rrf_k={calibration.rrf_k:.1f}",
            enabled=config.verbose,
        )
        log_memory("lgbm_fusion_done", enabled=config.verbose)
        return result

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
            f"prior={config.candidate_prior_enabled} "
            f"prior_test_freq={config.candidate_prior_include_test_frequency} "
            f"target={config.target_window_enabled} "
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


def _hybrid_feature_names(config: TrainingConfig) -> tuple[str, ...]:
    names = (
        STAT_FEATURE_NAMES
        + CANDIDATE_PRIOR_FEATURE_NAMES
        + TARGET_WINDOW_FEATURE_NAMES
        + STRUCTURE_FEATURE_NAMES
        + SOURCE_PROFILE_FEATURE_NAMES
        + TWO_TOWER_FEATURE_NAMES
        + GRAPH_WINDOW_NAMES
        + SEQUENCE_FEATURE_NAMES
    )
    if bool(getattr(config, "structure_cooccur_time_decay_enabled", False)):
        names += COOCCUR_TIME_DECAY_FEATURE_NAMES
    return names


def _service_normalizer_report_metrics(
    report: dict[str, dict[str, int | float]],
) -> dict[str, float]:
    if not report:
        raise ValueError("service normalizer report must be non-empty")
    rows = [float(head["count"]) for head in report.values()]
    standardized_shifts = [
        float(head["max_abs_mean_shift_in_training_std"])
        for head in report.values()
    ]
    absolute_shifts = [
        float(head["max_abs_mean_shift"])
        for head in report.values()
    ]
    max_std_ratios = [
        float(head["max_service_to_training_std_ratio"])
        for head in report.values()
    ]
    min_std_ratios = [
        float(head["min_service_to_training_std_ratio"])
        for head in report.values()
    ]
    return {
        "service_normalizer_head_count": float(len(report)),
        "service_normalizer_candidate_rows": max(rows),
        "service_normalizer_max_abs_mean_shift": max(absolute_shifts),
        "service_normalizer_max_abs_mean_shift_in_training_std": max(
            standardized_shifts
        ),
        "service_normalizer_max_service_to_training_std_ratio": max(
            max_std_ratios
        ),
        "service_normalizer_min_service_to_training_std_ratio": min(
            min_std_ratios
        ),
    }


def _select_result_features(
    features: np.ndarray,
    result: FusionResult,
) -> np.ndarray:
    indices = result.feature_indices
    selected = (
        features
        if indices == tuple(range(features.shape[-1]))
        else features[..., indices]
    )
    from .fusion import align_fusion_input_features  # noqa: PLC0415

    return align_fusion_input_features(
        selected,
        expected_dim=_fusion_result_input_dim(result),
    )


def _fusion_result_input_dim(result: FusionResult) -> int:
    mean = np.asarray(result.mean)
    if mean.ndim != 1 or mean.shape[0] <= 0:
        raise ValueError("fusion result must contain a non-empty mean vector")
    if np.asarray(result.std).shape != mean.shape:
        raise ValueError("fusion result mean and std dimensions must match")
    return int(mean.shape[0])


def _select_expert_features(
    features: np.ndarray,
    *,
    mlp_indices: tuple[int, ...],
    lgbm_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    mlp_features = features[:, :, mlp_indices] if mlp_indices else features
    lgbm_features = features[:, :, lgbm_indices] if lgbm_indices else features
    return mlp_features, lgbm_features


@dataclass(frozen=True)
class _LeaveOneOutFeatureMask:
    group: str
    alias: str
    candidate_name: str
    indices: tuple[int, ...]


@dataclass(frozen=True)
class _FeatureMaskCatalog:
    masks: tuple[tuple[str, tuple[int, ...]], ...]
    enabled_groups: tuple[str, ...]
    full_candidate_name: str
    full_indices: tuple[int, ...]
    leave_one_out: tuple[_LeaveOneOutFeatureMask, ...]


def _feature_masks(
    feature_count: int,
    config: TrainingConfig | None = None,
) -> list[tuple[str, tuple[int, ...]]]:
    catalog = _feature_mask_catalog(feature_count, config=config)
    frozen_candidate = (
        None
        if config is None
        else getattr(
            config,
            "frozen_fusion_feature_candidate",
            None,
        )
    )
    if frozen_candidate is None:
        return list(catalog.masks)

    by_name = dict(catalog.masks)
    selected = by_name.get(frozen_candidate)
    if selected is not None:
        return [(frozen_candidate, selected)]

    leave_one_out_by_alias = {
        entry.alias: entry.indices
        for entry in catalog.leave_one_out
    }
    selected = leave_one_out_by_alias.get(frozen_candidate)
    if selected is not None:
        return [(frozen_candidate, selected)]

    available_names = [name for name, _indices in catalog.masks]
    available_names.extend(
        entry.alias
        for entry in catalog.leave_one_out
        if entry.alias not in by_name
    )
    available = ", ".join(available_names)
    raise ValueError(
        "frozen fusion feature candidate is unavailable: "
        f"{frozen_candidate!r}; available={available}"
    )


def _feature_mask_catalog(
    feature_count: int,
    config: TrainingConfig | None = None,
) -> _FeatureMaskCatalog:
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

    group_order = tuple(ranges)
    enabled_groups = tuple(
        group
        for group in group_order
        if enabled[group] and ranges[group]
    )
    full_indices = tuple(
        index
        for group in enabled_groups
        for index in ranges[group]
    )
    candidate_name_by_indices = {
        indices: name
        for name, indices in unique
    }
    full_candidate_name = candidate_name_by_indices.get(full_indices, "")
    if full_indices and not full_candidate_name:
        full_candidate_name = "full_enabled"
        unique.append((full_candidate_name, full_indices))
        seen.add(full_indices)
        candidate_name_by_indices[full_indices] = full_candidate_name

    leave_one_out: list[_LeaveOneOutFeatureMask] = []
    for group in enabled_groups:
        removed_indices = set(ranges[group])
        indices = tuple(
            index
            for index in full_indices
            if index not in removed_indices
        )
        if not indices:
            continue
        alias = f"loo_without_{group}"
        candidate_name = candidate_name_by_indices.get(indices, "")
        if not candidate_name:
            candidate_name = alias
            unique.append((candidate_name, indices))
            seen.add(indices)
            candidate_name_by_indices[indices] = candidate_name
        leave_one_out.append(
            _LeaveOneOutFeatureMask(
                group=group,
                alias=alias,
                candidate_name=candidate_name,
                indices=indices,
            )
        )

    return _FeatureMaskCatalog(
        masks=tuple(unique),
        enabled_groups=enabled_groups,
        full_candidate_name=full_candidate_name,
        full_indices=full_indices,
        leave_one_out=tuple(leave_one_out),
    )


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


def _find_ensemble_weight(
    mlp_model: FusionMLP | None,
    mlp_result: FusionResult | None,
    lgbm_result: LGBMFusionResult,
    val_features: np.ndarray,
    feature_indices: tuple[int, ...],
    config: TrainingConfig,
) -> tuple[float, ExpertBlendCalibration]:
    from .expert_fusion import ExpertBlendCalibration  # noqa: PLC0415

    mode = str(getattr(config, "expert_blend_mode", "probability")).lower()
    default_calibration = ExpertBlendCalibration(
        mode=mode,
        rrf_k=float(getattr(config, "expert_rrf_k", 60.0)),
    )
    if mlp_model is None or mlp_result is None:
        return 0.0, default_calibration
    from .expert_fusion import (  # noqa: PLC0415
        blend_expert_logits,
        fit_positive_column_temperature,
    )
    from .fusion import predict_logits  # noqa: PLC0415
    from .fusion_lgbm import predict_logits_lgbm  # noqa: PLC0415

    selected = val_features[:, :, feature_indices] if feature_indices else val_features
    mlp_logits = predict_logits(
        mlp_model,
        selected,
        mlp_result.mean,
        mlp_result.std,
    )
    lgbm_logits = predict_logits_lgbm(lgbm_result.model_text, selected)
    calibration = default_calibration
    if mode == "temperature":
        calibration = ExpertBlendCalibration(
            mode=mode,
            mlp_temperature=fit_positive_column_temperature(mlp_logits),
            lgbm_temperature=fit_positive_column_temperature(lgbm_logits),
            rrf_k=default_calibration.rrf_k,
        )

    frozen_weight = getattr(
        config,
        "frozen_ensemble_mlp_weight",
        None,
    )
    if frozen_weight is not None:
        frozen_weight = float(frozen_weight)
        if (
            not np.isfinite(frozen_weight)
            or not 0.0 <= frozen_weight <= 1.0
        ):
            raise ValueError(
                "frozen_ensemble_mlp_weight must be finite and in [0, 1]"
            )
        if config.fusion_mode == "lgbm" and frozen_weight != 0.0:
            raise ValueError(
                "lgbm fusion requires frozen_ensemble_mlp_weight=0"
            )
        log_event(
            f"[ensemble-weight] frozen_w={frozen_weight:.6f} "
            "single_split_scan=false",
            enabled=config.verbose,
        )
        return frozen_weight, calibration

    metric = config.selection_metric.lower()
    best_w, best_score = 0.5, -1.0
    candidate_weights = (
        (0.0,)
        if config.fusion_mode == "lgbm"
        else tuple(value / 10.0 for value in range(11))
    )
    for w in candidate_weights:
        blended = blend_expert_logits(
            mlp_logits,
            lgbm_logits,
            w,
            calibration=calibration,
        )
        score = _mrr_from_probs(blended) if metric == "mrr" else _ap_from_probs(blended)
        if score > best_score:
            best_w, best_score = w, score
    log_event(
        f"[ensemble-weight] blend={mode} best_w={best_w:.1f} "
        f"{metric}={best_score:.5f}",
        enabled=config.verbose,
    )
    return best_w, calibration


def _expert_blend_calibration(
    result: LGBMFusionResult,
) -> ExpertBlendCalibration:
    from .expert_fusion import ExpertBlendCalibration  # noqa: PLC0415

    return ExpertBlendCalibration(
        mode=str(getattr(result, "blend_mode", "probability")),
        mlp_temperature=float(
            getattr(result, "mlp_temperature", 1.0)
        ),
        lgbm_temperature=float(
            getattr(result, "lgbm_temperature", 1.0)
        ),
        rrf_k=float(getattr(result, "rrf_k", 60.0)),
    )


def _mrr_from_probs(probs: np.ndarray) -> float:
    ranks = 1 + (probs[:, 1:] > probs[:, 0:1]).sum(axis=1)
    return float(np.mean(1.0 / ranks))


def _ap_from_probs(probs: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score  # noqa: PLC0415
    labels = np.zeros(probs.shape, dtype=np.int8)
    labels[:, 0] = 1
    return float(average_precision_score(labels.ravel(), probs.ravel()))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_l = np.exp(shifted)
    return exp_l / exp_l.sum(axis=1, keepdims=True)


def _auto_mode_code(mode: str) -> float:
    if mode == "repeat_memory":
        return 1.0
    if mode == "new_link_cold":
        return 3.0
    if mode == "balanced":
        return 2.0
    return 0.0


def _candidate_prior_test_frequency_indices() -> tuple[int, ...]:
    stats_end = len(STAT_FEATURE_NAMES)
    return tuple(
        stats_end + CANDIDATE_PRIOR_FEATURE_NAMES.index(name)
        for name in (
            "candidate_test_freq",
            "candidate_unseen_test_freq",
            "candidate_test_freq_row_rank",
        )
    )


def _selected_features_use_test_frequency(feature_indices: tuple[int, ...]) -> bool:
    test_frequency_indices = _candidate_prior_test_frequency_indices()
    return any(idx in test_frequency_indices for idx in feature_indices)


def _config_for_supervised_encoder(
    config: TrainingConfig,
    *,
    include_test_frequency: bool = True,
) -> TrainingConfig:
    """Fusion supervision needs test.csv candidate frequencies so the MLP learns prior weights.

    This reintroduces mild validation leakage from global test candidate counts, but without
    it fusion selection optimizes on zeroed prior columns and inference collapses to uniform.
    """
    return replace(
        config,
        structure_future_only_transition_cooccur=True,
        candidate_prior_include_test_frequency=include_test_frequency,
    )


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
    selected = replace(
        config,
        candidate_prior_enabled=config.candidate_prior_enabled and needs_prior,
        target_window_enabled=config.target_window_enabled and needs_target,
        structure_enabled=config.structure_enabled and needs_structure,
        source_profile_enabled=config.source_profile_enabled and needs_profile,
        two_tower_enabled=config.two_tower_enabled and needs_tower,
        gnn_enabled=config.gnn_enabled and needs_graph,
        seq_enabled=config.seq_enabled and needs_sequence,
    )
    if selected.candidate_prior_enabled and _selected_features_use_test_frequency(feature_indices):
        selected = replace(selected, candidate_prior_include_test_frequency=True)
    return selected


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
    if bool(getattr(target_config, "structure_cooccur_time_decay_enabled", False)):
        if not bool(getattr(source_config, "structure_cooccur_time_decay_enabled", False)):
            return False
        if float(getattr(source_config, "structure_cooccur_time_decay_ratio", 0.05)) != float(
            getattr(target_config, "structure_cooccur_time_decay_ratio", 0.05)
        ):
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
        encoder.clear_batch_caches()
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

    def prediction_order(self, queries: TestQueryArray) -> np.ndarray | None:
        return self.impl.prediction_order(queries)

    def prediction_tie_break_prior(
        self,
        queries: TestQueryArray | list[TestQuery],
    ) -> np.ndarray:
        return self.impl.prediction_tie_break_prior(queries)

    @property
    def training_report(self) -> TrainingReport | None:
        return self.impl.training_report

    def snapshot(self) -> dict[str, Any]:
        return self.impl.snapshot()

    def hydrate(self, snapshot: dict[str, Any]) -> None:
        self.impl.hydrate(snapshot)
        if self.impl.config is not None:
            self.config = self.impl.config
