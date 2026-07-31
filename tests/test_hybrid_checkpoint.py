from __future__ import annotations

import builtins
import sys

import numpy as np
import pytest

from jgrec.core.types import InteractionTable
from jgrec.idmap import NodeIdMap
from jgrec.rankers.common.sparse_counts import SparseCountMap
from jgrec.rankers.hybrid.config import DEFAULT_PREDICTION_CACHE_BYTES, SourceProfileConfig, TrainingConfig
from jgrec.rankers.hybrid.source_profile import SourceProfileTower


def test_source_profile_snapshot_preserves_item2vec_embeddings() -> None:
    id_map = NodeIdMap(
        src_to_id={1: 0},
        dst_to_id={10: 0, 20: 1},
        src_values=(1,),
        dst_values=(10, 20),
    )
    tower = SourceProfileTower(id_map, SourceProfileConfig())
    tower.embeddings = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    restored = SourceProfileTower(id_map, SourceProfileConfig())
    restored.hydrate(tower.snapshot())

    np.testing.assert_array_equal(restored.embeddings, tower.embeddings)


def test_source_profile_snapshot_preserves_temporal_history() -> None:
    id_map = _id_map()
    interactions = InteractionTable.from_array(np.asarray([[1, 10, 10], [1, 20, 20]], dtype=np.int32))
    tower = SourceProfileTower(id_map, SourceProfileConfig())
    tower.index.fit(interactions, build_transitions=False, build_cooccurs=False)

    restored = SourceProfileTower(id_map, SourceProfileConfig())
    restored.hydrate(tower.snapshot())

    np.testing.assert_array_equal(
        restored.index.source_view(1, 30).visible_dsts,
        np.asarray([10, 20], dtype=np.int32),
    )


def test_source_profile_snapshot_stores_sparse_counts_as_native_csr_arrays() -> None:
    id_map = _id_map()
    tower = SourceProfileTower(id_map, SourceProfileConfig())
    tower.item_pair_counts_sparse = SparseCountMap.from_nested_dict({10: {20: 3}, 20: {10: 3}})

    snapshot = tower.snapshot()
    sparse_snapshot = snapshot["item_pair_counts"]

    assert sparse_snapshot["format"] == "csr-v1"
    for key in ("row_keys", "row_offsets", "col_indices", "values"):
        assert isinstance(sparse_snapshot[key], np.ndarray)
    restored = SourceProfileTower(id_map, SourceProfileConfig())
    restored.hydrate(snapshot)
    assert restored.item_pair_counts_sparse.get_count(10, 20) == 3
    assert restored.item_pair_counts_sparse.get_count(20, 10) == 3


def test_source_profile_hydrate_accepts_legacy_nested_sparse_counts() -> None:
    id_map = _id_map()
    tower = SourceProfileTower(id_map, SourceProfileConfig())
    snapshot = tower.snapshot()
    snapshot["item_pair_counts"] = {10: {20: 4}, 20: {10: 4}}

    tower.hydrate(snapshot)

    assert tower.item_pair_counts_sparse.get_count(10, 20) == 4


def test_legacy_training_config_without_cache_budget_uses_current_default() -> None:
    legacy_config = TrainingConfig()
    object.__delattr__(legacy_config, "prediction_cache_max_bytes")

    structure_config = legacy_config.structure_config()
    source_profile_config = legacy_config.source_profile_config()

    assert structure_config.cache_max_bytes + source_profile_config.cache_max_bytes == DEFAULT_PREDICTION_CACHE_BYTES


def test_legacy_training_config_without_negative_overrides_uses_legacy_count() -> None:
    legacy_config = TrainingConfig(num_negatives=13)
    object.__delattr__(legacy_config, "train_num_negatives")
    object.__delattr__(legacy_config, "val_num_negatives")

    assert legacy_config.resolved_train_num_negatives() == 13
    assert legacy_config.resolved_val_num_negatives() == 13


def test_legacy_training_config_without_fusion_context_keeps_raw_mlp_input() -> None:
    legacy_config = TrainingConfig()
    object.__delattr__(
        legacy_config,
        "fusion_context_transform_version",
    )

    assert legacy_config.resolved_fusion_context_transform_version() == 0
    if sys.platform != "win32":
        assert legacy_config.fusion_config().context_transform_version == 0


@pytest.mark.skipif(sys.platform == "win32", reason="Jittor model construction is verified on Linux/CUDA")
def test_graph_snapshot_preserves_final_embeddings() -> None:
    from jgrec.rankers.hybrid.config import GraphTowerConfig  # noqa: PLC0415
    from jgrec.rankers.hybrid.gnn import GraphTower  # noqa: PLC0415

    id_map = _id_map()
    tower = GraphTower(id_map, GraphTowerConfig())
    tower.user_embeddings = {"gnn_full": np.asarray([[1.0, 2.0]], dtype=np.float32)}
    tower.item_embeddings = {"gnn_full": np.asarray([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)}
    tower.seen_users = {"gnn_full": np.asarray([True])}
    tower.seen_items = {"gnn_full": np.asarray([True, False])}

    restored = GraphTower(id_map, GraphTowerConfig())
    restored.hydrate(tower.snapshot())

    np.testing.assert_array_equal(restored.user_embeddings["gnn_full"], tower.user_embeddings["gnn_full"])
    np.testing.assert_array_equal(restored.item_embeddings["gnn_full"], tower.item_embeddings["gnn_full"])
    np.testing.assert_array_equal(restored.seen_users["gnn_full"], tower.seen_users["gnn_full"])
    np.testing.assert_array_equal(restored.seen_items["gnn_full"], tower.seen_items["gnn_full"])


@pytest.mark.skipif(sys.platform == "win32", reason="Jittor model construction is verified on Linux/CUDA")
def test_sequence_snapshot_preserves_model_and_history() -> None:
    from jgrec.rankers.hybrid.config import SequenceTowerConfig  # noqa: PLC0415
    from jgrec.rankers.hybrid.sequence import SequenceTower  # noqa: PLC0415
    from jgrec.rankers.hybrid.sequence_model import GRUSequenceModel  # noqa: PLC0415

    id_map = _id_map()
    config = SequenceTowerConfig(hidden_size=4, layers=1, dropout=0.0)
    tower = SequenceTower(id_map, config)
    tower.model = GRUSequenceModel(num_items=id_map.num_dst, hidden_size=4, layers=1, dropout=0.0)
    tower.src_sequences = {
        0: (
            np.asarray([1, 2], dtype=np.int32),
            np.asarray([10, 20], dtype=np.int64),
        )
    }
    tower.seen_items = np.asarray([False, True, True])

    restored = SequenceTower(id_map, config)
    restored.hydrate(tower.snapshot())

    assert restored.model is not None
    _assert_model_states_equal(restored.model, tower.model)
    np.testing.assert_array_equal(restored.src_sequences[0][0], tower.src_sequences[0][0])
    np.testing.assert_array_equal(restored.src_sequences[0][1], tower.src_sequences[0][1])
    np.testing.assert_array_equal(restored.seen_items, tower.seen_items)


@pytest.mark.skipif(sys.platform == "win32", reason="Jittor model construction is verified on Linux/CUDA")
def test_two_tower_snapshot_preserves_model_and_temporal_state() -> None:
    from jgrec.rankers.hybrid.config import TwoTowerConfig  # noqa: PLC0415
    from jgrec.rankers.hybrid.two_tower import TwoTower, _TwoTowerModel  # noqa: PLC0415

    id_map = _id_map()
    config = TwoTowerConfig(embedding_dim=4, hidden_dim=4)
    tower = TwoTower(id_map, config)
    tower.model = _TwoTowerModel(
        num_src=id_map.num_src,
        num_dst=id_map.num_dst,
        embedding_dim=4,
        hidden_dim=4,
    )
    tower.min_time = 10
    tower.max_time = 20
    tower.graph_span = 10

    restored = TwoTower(id_map, config)
    restored.hydrate(tower.snapshot())

    assert restored.model is not None
    _assert_model_states_equal(restored.model, tower.model)
    assert (restored.min_time, restored.max_time, restored.graph_span) == (10, 20, 10)


@pytest.mark.skipif(sys.platform == "win32", reason="Jittor model construction is verified on Linux/CUDA")
def test_hybrid_snapshot_round_trips_predictions() -> None:
    import jittor as jt  # noqa: PLC0415

    from jgrec.contest_checkpoint import get_model_state  # noqa: PLC0415
    from jgrec.core.types import TestQueryArray  # noqa: PLC0415
    from jgrec.rankers.hybrid.config import TrainingConfig  # noqa: PLC0415
    from jgrec.rankers.hybrid.fusion import (  # noqa: PLC0415
        FusionMLP,
        FusionResult,
        predict_logits,
    )
    from jgrec.rankers.hybrid.ranker import TemporalHybridRanker  # noqa: PLC0415
    from jgrec.rankers.hybrid.setwise import (  # noqa: PLC0415
        setwise_context_features,
    )

    jt.flags.use_cuda = 0
    interactions = InteractionTable.from_array(
        np.asarray(
            [[src, 10 + (event_idx % 5), event_idx + 1] for event_idx, src in enumerate((1, 2, 3, 4) * 30)],
            dtype=np.int32,
        )
    )
    config = TrainingConfig(
        val_ratio=0.2,
        context_ratio=0.5,
        max_train_events=8,
        max_val_events=8,
        num_negatives=2,
        epochs=1,
        train_batch_size=8,
        auto_strategy_enabled=False,
        candidate_prior_enabled=False,
        target_window_enabled=False,
        structure_enabled=False,
        source_profile_enabled=False,
        two_tower_enabled=False,
        gnn_enabled=False,
        seq_enabled=False,
        encoder_state_cache_enabled=False,
        verbose=False,
    )
    queries = TestQueryArray(
        src=np.asarray([1, 2], dtype=np.int32),
        time=np.asarray([121, 121], dtype=np.int32),
        candidates=np.asarray([[10, 11, 12], [12, 13, 14]], dtype=np.int32),
    )
    ranker = TemporalHybridRanker(recent_window=4)
    ranker.fit(interactions, config)
    raw_features = ranker.encoder.features_for_queries(queries)
    setwise_features = setwise_context_features(raw_features)
    setwise_model = FusionMLP(
        input_dim=setwise_features.shape[-1],
        hidden_dim=4,
    )
    setwise_state = get_model_state(setwise_model)
    setwise_result = FusionResult(
        best_val_ap=0.0,
        best_val_mrr=0.0,
        state=setwise_state,
        mean=np.zeros(setwise_features.shape[-1], dtype=np.float32),
        std=np.ones(setwise_features.shape[-1], dtype=np.float32),
        feature_indices=tuple(range(setwise_features.shape[-1])),
        candidate_name="checkpoint_setwise_test",
    )
    ranker.setwise_fusion = setwise_model
    ranker.setwise_fusion_result = setwise_result
    ranker._setwise_hidden_dim = 4
    ranker.lgbm_result = None
    extra_model = FusionMLP(
        input_dim=setwise_features.shape[-1],
        hidden_dim=4,
    )
    extra_state = get_model_state(extra_model)
    extra_result = FusionResult(
        best_val_ap=0.0,
        best_val_mrr=0.0,
        state=extra_state,
        mean=np.zeros(setwise_features.shape[-1], dtype=np.float32),
        std=np.ones(setwise_features.shape[-1], dtype=np.float32),
        feature_indices=tuple(range(setwise_features.shape[-1])),
        candidate_name="checkpoint_conservative_window_test",
    )
    ranker.conservative_window_fusions = {"recent100k": extra_model}
    ranker.conservative_window_results = {"recent100k": extra_result}
    ranker.conservative_window_hidden_dims = {"recent100k": 4}
    ranker.conservative_window_config = {"alpha": 0.30}

    champion_logits = predict_logits(
        setwise_model,
        setwise_features,
        setwise_result.mean,
        setwise_result.std,
    )
    extra_logits = predict_logits(
        extra_model,
        setwise_features,
        extra_result.mean,
        extra_result.std,
    )
    champion_shifted = champion_logits - champion_logits.max(
        axis=1,
        keepdims=True,
    )
    champion_probs = np.exp(champion_shifted) / np.exp(
        champion_shifted
    ).sum(axis=1, keepdims=True)
    extra_shifted = extra_logits - extra_logits.max(axis=1, keepdims=True)
    extra_probs = np.exp(extra_shifted) / np.exp(extra_shifted).sum(
        axis=1,
        keepdims=True,
    )
    window_probs = (champion_probs + extra_probs) / 2.0
    expected = champion_probs + 0.30 * (window_probs - champion_probs)
    np.testing.assert_allclose(
        ranker.predict_batch(queries),
        expected,
        rtol=0.0,
        atol=1e-7,
    )

    restored = TemporalHybridRanker()
    snapshot = ranker.snapshot()
    assert snapshot["setwise_fusion_result"].candidate_name == (
        "checkpoint_setwise_test"
    )
    assert snapshot["conservative_window_config"] == {"alpha": 0.30}
    restored.hydrate(snapshot)

    np.testing.assert_allclose(restored.predict_batch(queries), expected, rtol=0.0, atol=1e-7)


@pytest.mark.skipif(sys.platform == "win32", reason="Jittor model construction is verified on Linux/CUDA")
def test_pure_candidate_set_snapshot_bypasses_legacy_fusion() -> None:
    import jittor as jt  # noqa: PLC0415

    from jgrec.contest_checkpoint import get_model_state  # noqa: PLC0415
    from jgrec.core.types import TestQueryArray  # noqa: PLC0415
    from jgrec.rankers.hybrid.candidate_set_transformer import (  # noqa: PLC0415
        CandidateSetEnsembleCheckpoint,
        CandidateSetFitResult,
        CandidateSetTrainingConfig,
        CandidateSetTransformer,
        CandidateSetTransformerConfig,
        predict_candidate_set_ensemble_probabilities,
    )
    from jgrec.rankers.hybrid.oof_models import (  # noqa: PLC0415
        CandidateSetMLP,
        CandidateSetMLPConfig,
        CandidateSetMLPFitResult,
        CandidateSetMLPTrainingConfig,
        PureJittorOOFStackingCheckpoint,
        predict_pure_jittor_oof_stacking_scores,
    )
    from jgrec.rankers.hybrid.oof_stacking import (  # noqa: PLC0415
        stable_expert_logit_feature_names,
    )
    from jgrec.rankers.hybrid.ranker import TemporalHybridRanker  # noqa: PLC0415

    jt.flags.use_cuda = 0
    interactions = InteractionTable.from_array(
        np.asarray(
            [
                [src, 10 + (event_idx % 5), event_idx + 1]
                for event_idx, src in enumerate((1, 2, 3, 4) * 30)
            ],
            dtype=np.int32,
        )
    )
    config = TrainingConfig(
        val_ratio=0.2,
        context_ratio=0.5,
        max_train_events=8,
        max_val_events=8,
        num_negatives=2,
        epochs=1,
        train_batch_size=8,
        auto_strategy_enabled=False,
        candidate_prior_enabled=False,
        target_window_enabled=False,
        structure_enabled=False,
        source_profile_enabled=False,
        two_tower_enabled=False,
        gnn_enabled=False,
        seq_enabled=False,
        encoder_state_cache_enabled=False,
        verbose=False,
    )
    queries = TestQueryArray(
        src=np.asarray([1, 2], dtype=np.int32),
        time=np.asarray([121, 121], dtype=np.int32),
        candidates=np.asarray(
            [[10, 11, 12], [12, 13, 14]],
            dtype=np.int32,
        ),
    )
    ranker = TemporalHybridRanker(recent_window=4)
    ranker.fit(interactions, config)
    features = ranker.encoder.features_for_queries(queries)
    model_config = CandidateSetTransformerConfig(
        input_dim=features.shape[-1],
        model_dim=8,
        heads=2,
        layers=1,
        dropout=0.0,
        feedforward_multiplier=2,
        relative_context="mean_max",
    )
    model = CandidateSetTransformer(model_config)
    state = get_model_state(model)
    result = CandidateSetFitResult(
        model_config=model_config,
        training_config=CandidateSetTrainingConfig(epochs=1),
        best_val_mrr=0.0,
        state=state,
        mean=np.zeros(features.shape[-1], dtype=np.float32),
        std=np.ones(features.shape[-1], dtype=np.float32),
        feature_names=ranker.feature_names,
        feature_provenance=("jittor",) * features.shape[-1],
        history=(),
    )
    ensemble = CandidateSetEnsembleCheckpoint(
        models=(model, model),
        results=(result, result),
        weights=(0.6, 0.4),
    )
    expected = predict_candidate_set_ensemble_probabilities(
        ensemble,
        features,
        batch_size=2,
    )

    ranker.install_candidate_set_ensemble(ensemble)
    np.testing.assert_allclose(
        ranker.predict_batch(queries),
        expected,
        rtol=0.0,
        atol=1e-7,
    )
    snapshot = ranker.snapshot()
    assert snapshot["fusion_state"] is None
    assert snapshot["fusion_result"] is None
    assert snapshot["lgbm_result"] is None
    assert snapshot["setwise_fusion_state"] is None
    assert snapshot["setwise_fusion_result"] is None

    original_import = builtins.__import__

    def block_legacy_ml_imports(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] in {"lightgbm", "sklearn"}:
            raise AssertionError(f"forbidden final inference import: {name}")
        if name in {"fusion", "fusion_lgbm"}:
            raise AssertionError(f"legacy fusion import during hydrate: {name}")
        return original_import(name, globals, locals, fromlist, level)

    restored = TemporalHybridRanker()
    builtins.__import__ = block_legacy_ml_imports
    try:
        restored.hydrate(snapshot)
        actual = restored.predict_batch(queries)
    finally:
        builtins.__import__ = original_import

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)

    setwise_config = CandidateSetMLPConfig(
        input_dim=features.shape[-1],
        hidden_dim=8,
        dropout=0.0,
    )
    setwise_model = CandidateSetMLP(setwise_config)
    setwise_result = CandidateSetMLPFitResult(
        model_config=setwise_config,
        training_config=CandidateSetMLPTrainingConfig(epochs=1),
        selection_mode="fixed_full",
        training_rows=120,
        best_val_mrr=None,
        state=get_model_state(setwise_model),
        mean=np.zeros(features.shape[-1], dtype=np.float32),
        std=np.ones(features.shape[-1], dtype=np.float32),
        feature_names=ranker.feature_names,
        feature_provenance=("jittor",) * features.shape[-1],
        history=(),
    )
    expert_names = ("cst_main", "cst_residual", "setwise_mlp")
    meta_names = stable_expert_logit_feature_names(expert_names)
    meta_config = CandidateSetMLPConfig(
        input_dim=len(meta_names),
        hidden_dim=8,
        dropout=0.0,
        relative_context="none",
    )
    meta_model = CandidateSetMLP(meta_config)
    meta_result = CandidateSetMLPFitResult(
        model_config=meta_config,
        training_config=CandidateSetMLPTrainingConfig(epochs=1),
        selection_mode="validation_best",
        training_rows=80,
        best_val_mrr=0.5,
        state=get_model_state(meta_model),
        mean=np.zeros(len(meta_names), dtype=np.float32),
        std=np.ones(len(meta_names), dtype=np.float32),
        feature_names=meta_names,
        feature_provenance=(
            ("numpy_deterministic",) * len(meta_names)
        ),
        history=(),
    )
    stacking = PureJittorOOFStackingCheckpoint(
        expert_names=expert_names,
        cst_experts=ensemble,
        setwise_mlp=(setwise_model, setwise_result),
        meta_mlp=(meta_model, meta_result),
        meta_weight=0.25,
    )
    expected_stacking = predict_pure_jittor_oof_stacking_scores(
        stacking,
        features,
        batch_size=2,
    )

    ranker.install_pure_jittor_oof_stacking(stacking)
    np.testing.assert_allclose(
        ranker.predict_batch(queries),
        expected_stacking,
        rtol=0.0,
        atol=1e-7,
    )
    stacking_snapshot = ranker.snapshot()
    assert stacking_snapshot["candidate_set_ensemble_state"] is None
    assert stacking_snapshot["oof_stacking_state"] is not None
    restored_stacking = TemporalHybridRanker()
    builtins.__import__ = block_legacy_ml_imports
    try:
        restored_stacking.hydrate(stacking_snapshot)
        actual_stacking = restored_stacking.predict_batch(queries)
    finally:
        builtins.__import__ = original_import
    np.testing.assert_allclose(
        actual_stacking,
        expected_stacking,
        rtol=0.0,
        atol=1e-7,
    )


def _id_map() -> NodeIdMap:
    return NodeIdMap(
        src_to_id={1: 0},
        dst_to_id={10: 0, 20: 1},
        src_values=(1,),
        dst_values=(10, 20),
    )


def _assert_model_states_equal(actual, expected) -> None:
    from jgrec.contest_checkpoint import get_model_state  # noqa: PLC0415

    actual_state = get_model_state(actual)
    expected_state = get_model_state(expected)
    assert actual_state.keys() == expected_state.keys()
    for key in actual_state:
        np.testing.assert_array_equal(actual_state[key], expected_state[key])
