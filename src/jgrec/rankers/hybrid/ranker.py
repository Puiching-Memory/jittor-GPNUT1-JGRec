from __future__ import annotations

import tempfile
from dataclasses import replace
from time import perf_counter
from typing import Any

import numpy as np

from jgrec.core.memory import log_event, log_memory, release_memory
from jgrec.core.types import FitContext, Interaction, TestQuery, TrainingReport
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
    TWO_TOWER_FEATURE_NAMES,
    CandidatePriorConfig,
    GraphTowerConfig,
    SequenceTowerConfig,
    StructureTowerConfig,
    TrainingConfig,
    TwoTowerConfig,
)
from .fusion import FusionMLP, FusionResult, fit_fusion_mlp, fit_fusion_mlp_streaming, predict_logits
from .sampling import (
    NegativeSamplingContext,
    NegativeSamplingJob,
    build_candidate_pool,
    sample_mixed_negatives,
    sample_mixed_negatives_batch,
)
from .stats import STAT_FEATURE_NAMES, TemporalStats
from .structure import STRUCTURE_FEATURE_NAMES, StructureFeatureTower

_FEATURE_MEMMAP_TEMP_FILES: list[Any] = []
FEATURE_PROFILE_INTERVAL = 10_000
FEATURE_MEMMAP_FLUSH_INTERVAL = 8


class _DisabledGraphTower:
    def fit(self, interactions: list[Interaction], rng: np.random.Generator, verbose: bool = True, **kwargs) -> None:
        return

    def scores_for_queries(self, queries: list[TestQuery]) -> np.ndarray:
        return _zero_scores(queries, len(GRAPH_WINDOW_NAMES))


class _DisabledSequenceTower:
    def fit(self, interactions: list[Interaction], rng: np.random.Generator, verbose: bool = True, **kwargs) -> None:
        return

    def scores_for_queries(self, queries: list[TestQuery]) -> np.ndarray:
        return _zero_scores(queries, len(SEQUENCE_FEATURE_NAMES))


class _DisabledTwoTower:
    def fit(self, interactions: list[Interaction], rng: np.random.Generator, verbose: bool = True, **kwargs) -> None:
        return

    def scores_for_queries(self, queries: list[TestQuery]) -> np.ndarray:
        return _zero_scores(queries, len(TWO_TOWER_FEATURE_NAMES))


class _DisabledCandidatePriorTower:
    def fit(
        self,
        interactions: list[Interaction],
        test_candidate_counts=None,
    ) -> None:
        return

    def features_for_queries(self, queries: list[TestQuery], stat_features: np.ndarray) -> np.ndarray:
        return _zero_scores(queries, len(CANDIDATE_PRIOR_FEATURE_NAMES))


class _DisabledStructureTower:
    index: Any = None

    def fit(self, interactions: list[Interaction], rng: np.random.Generator, verbose: bool = True) -> None:
        return

    def features_for_queries(self, queries: list[TestQuery]) -> np.ndarray:
        return _zero_scores(queries, len(STRUCTURE_FEATURE_NAMES))

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
        dataset_profile: DatasetProfile | None = None,
        two_tower_config: TwoTowerConfig | None = None,
        structure_config: StructureTowerConfig | None = None,
    ) -> None:
        self.id_map = id_map
        self.dataset_profile = dataset_profile
        self.stats = TemporalStats(recent_window=recent_window)
        self.candidate_prior = _build_candidate_prior(candidate_prior_config or CandidatePriorConfig(enabled=False))
        self.structure = _build_structure_tower(structure_config or StructureTowerConfig(enabled=False))
        self.two_tower = _build_two_tower(id_map, two_tower_config or TwoTowerConfig(enabled=False))
        self.graph = _build_graph_tower(id_map, graph_config)
        self.sequence = _build_sequence_tower(id_map, sequence_config)
        self.verbose = False
        self._profile_rows = 0
        self._profile_next_rows = FEATURE_PROFILE_INTERVAL
        self._profile_elapsed = {
            "stats": 0.0,
            "prior": 0.0,
            "structure": 0.0,
            "tower": 0.0,
            "graph": 0.0,
            "sequence": 0.0,
            "concat": 0.0,
        }
        self.feature_names = (
            STAT_FEATURE_NAMES
            + CANDIDATE_PRIOR_FEATURE_NAMES
            + STRUCTURE_FEATURE_NAMES
            + TWO_TOWER_FEATURE_NAMES
            + GRAPH_WINDOW_NAMES
            + SEQUENCE_FEATURE_NAMES
        )

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def fit(self, interactions: list[Interaction], rng: np.random.Generator, verbose: bool) -> None:
        self.verbose = verbose
        self.stats.fit(interactions)
        test_counts = self.dataset_profile.test_candidate_counts if self.dataset_profile is not None else None
        self.candidate_prior.fit(interactions, test_counts)
        self.structure.fit(interactions, rng=rng, verbose=verbose)
        two_tower_fit = getattr(self.two_tower, "fit", None)
        if callable(two_tower_fit):
            two_tower_fit(
                interactions,
                rng=rng,
                verbose=verbose,
                shared_index=getattr(self.structure, "index", None),
            )
        self.graph.fit(interactions, rng=rng, verbose=verbose)
        self.sequence.fit(interactions, rng=rng, verbose=verbose)

    def features_for_queries(self, queries: list[TestQuery]) -> np.ndarray:
        if not queries:
            return np.empty((0, 0, self.feature_dim), dtype=np.float32)
        start = perf_counter()
        stat_features = self.stats.features_for_queries(queries)
        self._profile_elapsed["stats"] += perf_counter() - start
        start = perf_counter()
        candidate_prior_features = self.candidate_prior.features_for_queries(queries, stat_features)
        self._profile_elapsed["prior"] += perf_counter() - start
        start = perf_counter()
        structure_features = self.structure.features_for_queries(queries)
        self._profile_elapsed["structure"] += perf_counter() - start
        start = perf_counter()
        two_tower_features = self.two_tower.scores_for_queries(queries)
        self._profile_elapsed["tower"] += perf_counter() - start
        start = perf_counter()
        graph_features = self.graph.scores_for_queries(queries)
        self._profile_elapsed["graph"] += perf_counter() - start
        start = perf_counter()
        sequence_features = self.sequence.scores_for_queries(queries)
        self._profile_elapsed["sequence"] += perf_counter() - start
        start = perf_counter()
        features = np.concatenate(
            [
                stat_features,
                candidate_prior_features,
                structure_features,
                two_tower_features,
                graph_features,
                sequence_features,
            ],
            axis=2,
        ).astype(np.float32, copy=False)
        self._profile_elapsed["concat"] += perf_counter() - start
        self._log_feature_profile(len(queries))
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
        compact_future_structure = getattr(self.structure, "compact_transition_cooccur_for_future_queries", None)
        if callable(compact_future_structure):
            compact_future_structure()
        release_memory()


def _build_candidate_prior(config: CandidatePriorConfig) -> Any:
    if not config.enabled:
        return _DisabledCandidatePriorTower()
    return CandidatePriorTower(config=config)


def _build_structure_tower(config: StructureTowerConfig) -> Any:
    if not config.enabled:
        return _DisabledStructureTower()
    return StructureFeatureTower(config)


def _build_graph_tower(id_map: NodeIdMap, config: GraphTowerConfig) -> Any:
    if not config.enabled:
        return _DisabledGraphTower()
    from .gnn import GraphTower

    return GraphTower(id_map=id_map, config=config)


def _build_sequence_tower(id_map: NodeIdMap, config: SequenceTowerConfig) -> Any:
    if not config.enabled:
        return _DisabledSequenceTower()
    from .sequence import SequenceTower

    return SequenceTower(id_map=id_map, config=config)


def _build_two_tower(id_map: NodeIdMap, config: TwoTowerConfig) -> Any:
    if not config.enabled:
        return _DisabledTwoTower()
    from .two_tower import TwoTower

    return TwoTower(id_map=id_map, config=config)


def _zero_scores(queries: list[TestQuery], feature_count: int) -> np.ndarray:
    if not queries:
        return np.empty((0, 0, feature_count), dtype=np.float32)
    candidate_count = len(queries[0].candidates)
    return np.zeros((len(queries), candidate_count, feature_count), dtype=np.float32)


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

    def fit(self, interactions: list[Interaction], training_config: TrainingConfig) -> TrainingReport:
        if not interactions:
            raise ValueError("training interactions are empty")

        interactions.sort(key=lambda item: item.time)
        if training_config.max_fit_events > 0 and len(interactions) > training_config.max_fit_events:
            interactions = interactions[-training_config.max_fit_events :]
        self.id_map = NodeIdMap.from_interactions(interactions)
        self.feature_names = (
            STAT_FEATURE_NAMES
            + CANDIDATE_PRIOR_FEATURE_NAMES
            + STRUCTURE_FEATURE_NAMES
            + TWO_TOWER_FEATURE_NAMES
            + GRAPH_WINDOW_NAMES
            + SEQUENCE_FEATURE_NAMES
        )
        self._fusion_hidden_dim = training_config.fusion_hidden_dim

        training_config = self._apply_auto_strategy(interactions, training_config)
        log_memory("hybrid_fit_start", enabled=training_config.verbose)
        log(f"[hybrid-fit] start events={len(interactions)}", enabled=training_config.verbose)
        fusion, fusion_result, report = self._learn_fusion(interactions, training_config)
        self.fusion = fusion
        self.fusion_result = fusion_result

        rng = np.random.default_rng(training_config.seed + 10_000)
        final_config = _config_for_selected_features(training_config, fusion_result.feature_indices)
        final_future_only = self._can_use_future_only_final_encoder()
        if final_future_only:
            final_config = replace(final_config, structure_future_only_transition_cooccur=True)
        log_event(
            "[hybrid-fit] final_encoder "
            f"prior={final_config.candidate_prior_enabled} tower={final_config.two_tower_enabled} "
            f"gnn={final_config.gnn_enabled} seq={final_config.seq_enabled} "
            f"future_only_structure={final_config.structure_future_only_transition_cooccur}",
            enabled=training_config.verbose,
        )
        log_memory("final_encoder_start", enabled=training_config.verbose)
        self.encoder = self._fit_encoder(interactions, final_config, rng, verbose=training_config.verbose)
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

    def _apply_auto_strategy(self, interactions: list[Interaction], config: TrainingConfig) -> TrainingConfig:
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

    def _dataset_profile(self, interactions: list[Interaction], config: TrainingConfig) -> DatasetProfile:
        if config.dataset_test_path is None:
            raise RuntimeError("dataset test path is required for profiling")
        if config.dataset_train_path is not None:
            return profile_dataset_paths(config.dataset_train_path, config.dataset_test_path, val_ratio=config.val_ratio)
        return profile_dataset(interactions, config.dataset_test_path, val_ratio=config.val_ratio)

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

        log_event(
            "[hybrid-fit] split "
            f"context={len(context_events)} train={len(train_events)} val={len(val_events)} "
            f"dst={len(dst_pool)}",
            enabled=config.verbose,
        )
        log_memory("split_done", enabled=config.verbose)

        supervised_encoder_config = replace(config, structure_future_only_transition_cooccur=True)
        train_encoder = self._timed_fit_encoder(
            "train_context_encoder",
            context_events,
            supervised_encoder_config,
            rng,
            config.verbose,
        )
        feature_start = perf_counter()
        log_memory("train_features_start", enabled=config.verbose)
        train_features = _build_supervised_features(train_events, train_encoder, dst_pool, config, rng)
        del train_encoder
        log_event(
            f"[hybrid-fit] train_features shape={train_features.shape} elapsed={perf_counter() - feature_start:.1f}s",
            enabled=config.verbose,
        )
        log_memory("train_features_done", enabled=config.verbose)

        val_encoder = self._timed_fit_encoder(
            "val_context_encoder",
            val_context_events,
            supervised_encoder_config,
            rng,
            config.verbose,
        )
        feature_start = perf_counter()
        log_memory("val_features_start", enabled=config.verbose)
        val_features = _build_supervised_features(val_events, val_encoder, dst_pool, config, rng)
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
            candidate_prior_config=config.candidate_prior_config(),
            dataset_profile=self.dataset_profile,
            structure_config=config.structure_config(),
            graph_config=config.graph_config(),
            sequence_config=config.sequence_config(),
            two_tower_config=config.two_tower_config(),
        )
        encoder.fit(interactions, rng=rng, verbose=verbose)
        return encoder

    def _timed_fit_encoder(
        self,
        label: str,
        interactions: list[Interaction],
        config: TrainingConfig,
        rng: np.random.Generator,
        verbose: bool,
    ) -> HybridFeatureEncoder:
        start = perf_counter()
        log_event(
            f"[hybrid-fit] {label} start events={len(interactions)} "
            f"prior={config.candidate_prior_enabled} tower={config.two_tower_enabled} "
            f"gnn={config.gnn_enabled} seq={config.seq_enabled} "
            f"future_only_structure={config.structure_future_only_transition_cooccur}",
            enabled=verbose,
        )
        log_memory(f"{label}_start", enabled=verbose)
        encoder = self._fit_encoder(interactions, config, rng, verbose=verbose)
        log_event(f"[hybrid-fit] {label} done elapsed={perf_counter() - start:.1f}s", enabled=verbose)
        log_memory(f"{label}_done", enabled=verbose)
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
    prior_end = stats_end + len(CANDIDATE_PRIOR_FEATURE_NAMES)
    structure_end = prior_end + len(STRUCTURE_FEATURE_NAMES)
    tower_end = structure_end + len(TWO_TOWER_FEATURE_NAMES)
    graph_end = tower_end + len(GRAPH_WINDOW_NAMES)
    masks = [("stats", tuple(range(min(stats_end, feature_count))))]
    if feature_count > stats_end:
        masks.append(("stats_prior", tuple(range(min(prior_end, feature_count)))))
    if feature_count > prior_end:
        masks.append(("stats_prior_structure", tuple(range(min(structure_end, feature_count)))))
    if feature_count > structure_end:
        masks.append(("stats_prior_structure_tower", tuple(range(min(tower_end, feature_count)))))
    if feature_count > tower_end:
        masks.append(("stats_prior_structure_tower_gnn", tuple(range(min(graph_end, feature_count)))))
    if feature_count > graph_end:
        masks.append(("stats_prior_structure_tower_gnn_seq", tuple(range(feature_count))))

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
    structure_end = prior_end + len(STRUCTURE_FEATURE_NAMES)
    tower_end = structure_end + len(TWO_TOWER_FEATURE_NAMES)
    graph_end = tower_end + len(GRAPH_WINDOW_NAMES)
    needs_prior = any(stats_end <= idx < prior_end for idx in feature_indices)
    needs_tower = any(structure_end <= idx < tower_end for idx in feature_indices)
    needs_graph = any(tower_end <= idx < graph_end for idx in feature_indices)
    needs_sequence = any(idx >= graph_end for idx in feature_indices)
    return replace(
        config,
        candidate_prior_enabled=config.candidate_prior_enabled and needs_prior,
        structure_enabled=config.structure_enabled and any(prior_end <= idx < structure_end for idx in feature_indices),
        two_tower_enabled=config.two_tower_enabled and needs_tower,
        gnn_enabled=config.gnn_enabled and needs_graph,
        seq_enabled=config.seq_enabled and needs_sequence,
    )


def _build_supervised_queries(
    positives: list[Interaction],
    encoder: HybridFeatureEncoder,
    dst_pool: np.ndarray,
    config: TrainingConfig,
    rng: np.random.Generator,
    candidate_pool=None,
) -> list[TestQuery]:
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
        NegativeSamplingJob(src=event.src, positive_dst=event.dst, query_time=event.time)
        for event in positives
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
    return [
        TestQuery(src=event.src, time=event.time, candidates=(event.dst, *negatives))
        for event, negatives in zip(positives, negatives_by_event)
    ]


def _build_supervised_features(
    positives: list[Interaction],
    encoder: HybridFeatureEncoder,
    dst_pool: np.ndarray,
    config: TrainingConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if not positives:
        return np.empty((0, 0, encoder.feature_dim), dtype=np.float32)

    batch_size = max(int(config.supervised_feature_batch_size), 1)
    candidate_count = int(config.num_negatives) + 1
    shape = (len(positives), candidate_count, encoder.feature_dim)
    features = _empty_feature_matrix(shape, config)
    test_values, _ = test_candidate_arrays(encoder.dataset_profile)
    candidate_pool = build_candidate_pool(dst_pool, test_values)
    start_time = perf_counter()
    for start in range(0, len(positives), batch_size):
        end = min(start + batch_size, len(positives))
        batch_events = positives[start:end]
        queries = _build_supervised_queries(batch_events, encoder, dst_pool, config, rng, candidate_pool=candidate_pool)
        features[start:end] = encoder.features_for_queries(queries)
        del queries
        if isinstance(features, np.memmap) and (start // batch_size + 1) % FEATURE_MEMMAP_FLUSH_INTERVAL == 0:
            features.flush()
        release_memory()
        if config.verbose and (end == len(positives) or end % max(batch_size * 4, 1) == 0):
            log_event(
                f"[hybrid-fit] supervised_features rows={end}/{len(positives)} "
                f"batch={len(batch_events)} elapsed={perf_counter() - start_time:.1f}s",
                enabled=True,
            )
    if isinstance(features, np.memmap):
        features.flush()
    return features


def _empty_feature_matrix(shape: tuple[int, int, int], config: TrainingConfig) -> np.ndarray:
    if not config.supervised_feature_memmap:
        return np.empty(shape, dtype=np.float32)

    temp = tempfile.NamedTemporaryFile(prefix="jgrec-supervised-features-", suffix=".dat", delete=True)
    matrix = np.memmap(temp.name, mode="w+", dtype=np.float32, shape=shape)
    _FEATURE_MEMMAP_TEMP_FILES.append(temp)
    return matrix


def _release_feature_memmaps() -> None:
    while _FEATURE_MEMMAP_TEMP_FILES:
        temp = _FEATURE_MEMMAP_TEMP_FILES.pop()
        try:
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

    def fit(self, interactions: list[Interaction], context: FitContext) -> TrainingReport:
        config = replace(
            self.config,
            seed=context.seed,
            verbose=context.verbose,
            dataset_train_path=context.dataset.train_path,
            dataset_test_path=context.dataset.test_path,
        )
        return self.impl.fit(interactions, training_config=config)

    def predict_batch(self, queries: list[TestQuery]) -> np.ndarray:
        return self.impl.predict_batch(queries)
