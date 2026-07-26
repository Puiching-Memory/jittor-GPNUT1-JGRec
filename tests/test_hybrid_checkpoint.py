from __future__ import annotations

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

    from jgrec.core.types import TestQueryArray  # noqa: PLC0415
    from jgrec.rankers.hybrid.config import TrainingConfig  # noqa: PLC0415
    from jgrec.rankers.hybrid.ranker import TemporalHybridRanker  # noqa: PLC0415

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
    expected = ranker.predict_batch(queries)

    restored = TemporalHybridRanker()
    restored.hydrate(ranker.snapshot())

    np.testing.assert_allclose(restored.predict_batch(queries), expected, rtol=0.0, atol=1e-7)


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
