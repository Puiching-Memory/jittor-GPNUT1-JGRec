import importlib
import sys
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from jgrec.core.types import Interaction, InteractionTable, TestQuery
from jgrec.idmap import NodeIdMap
from jgrec.rankers.common.temporal_index import TemporalInteractionIndex
from jgrec.rankers.hybrid.auto_strategy import DatasetProfile
from jgrec.rankers.hybrid.candidate_prior import CANDIDATE_PRIOR_FEATURE_NAMES
from jgrec.rankers.hybrid.config import (
    GRAPH_WINDOW_NAMES,
    SEQUENCE_FEATURE_NAMES,
    SOURCE_PROFILE_FEATURE_NAMES,
    TARGET_WINDOW_FEATURE_NAMES,
    TWO_TOWER_FEATURE_NAMES,
    TrainingConfig,
    TwoTowerConfig,
)
from jgrec.rankers.hybrid.ranker import (
    _config_for_selected_features,
    _feature_masks,
)
from jgrec.rankers.hybrid.stats import STAT_FEATURE_NAMES
from jgrec.rankers.hybrid.structure import STRUCTURE_FEATURE_NAMES


def _interactions() -> list[Interaction]:
    return [
        Interaction(src=1, dst=10, time=10),
        Interaction(src=1, dst=20, time=20),
        Interaction(src=2, dst=20, time=30),
        Interaction(src=2, dst=30, time=40),
        Interaction(src=3, dst=10, time=50),
        Interaction(src=3, dst=30, time=60),
        Interaction(src=1, dst=30, time=70),
        Interaction(src=2, dst=10, time=80),
        Interaction(src=4, dst=40, time=90),
        Interaction(src=4, dst=20, time=100),
    ]


def _require_jittor() -> None:
    pytest.importorskip("jittor")


def test_two_tower_listwise_positive_loss_matches_group_softmax_reference():
    _require_jittor()
    import jittor as jt  # noqa: PLC0415

    from jgrec.rankers.hybrid.two_tower import (  # noqa: PLC0415
        _listwise_positive_loss,
    )

    logits = np.asarray(
        [[2.0, 1.0, -0.5], [-4.0, 0.5, 3.0]],
        dtype=np.float32,
    )
    row_max = logits.max(axis=1, keepdims=True)
    expected = np.mean(
        row_max[:, 0]
        + np.log(np.exp(logits - row_max).sum(axis=1))
        - logits[:, 0]
    )

    actual = float(
        _listwise_positive_loss(jt.array(logits, dtype=jt.float32)).item()
    )

    assert actual == pytest.approx(float(expected), abs=1e-6)


def test_two_tower_in_batch_positive_mask_treats_duplicate_destinations_as_positives():
    from jgrec.rankers.hybrid.in_batch_negatives import (  # noqa: PLC0415
        _in_batch_positive_mask,
    )

    actual = _in_batch_positive_mask(
        np.asarray([4, 7, 4, 9], dtype=np.int32)
    )

    np.testing.assert_array_equal(
        actual,
        np.asarray(
            [
                [True, False, True, False],
                [False, True, False, False],
                [True, False, True, False],
                [False, False, False, True],
            ]
        ),
    )


def test_two_tower_multi_positive_in_batch_loss_matches_numpy_reference():
    _require_jittor()
    import jittor as jt  # noqa: PLC0415

    from jgrec.rankers.hybrid.two_tower import (  # noqa: PLC0415
        _multi_positive_in_batch_loss,
    )

    logits = np.asarray(
        [
            [3.0, 0.0, 2.0],
            [-1.0, 4.0, 0.5],
            [1.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )
    positive_dst_ids = np.asarray([5, 8, 5], dtype=np.int32)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    positive_mask = positive_dst_ids[:, None] == positive_dst_ids[None, :]
    expected = -np.log((probs * positive_mask).sum(axis=1)).mean()

    actual = float(
        _multi_positive_in_batch_loss(
            jt.array(logits, dtype=jt.float32),
            positive_dst_ids,
            temperature=1.0,
        ).item()
    )

    assert actual == pytest.approx(float(expected), abs=1e-6)


def test_two_tower_full_candidate_mrr_drives_maximizing_stop_signal():
    _require_jittor()
    from jgrec.rankers.hybrid.two_tower import (  # noqa: PLC0415
        _early_stop_signal,
        _full_candidate_mrr,
    )

    scores = np.asarray(
        [[3.0, 1.0, 2.0], [0.0, 2.0, -1.0]],
        dtype=np.float32,
    )

    assert _full_candidate_mrr(scores) == pytest.approx(0.75)
    assert _early_stop_signal("mrr", val_loss=0.25, val_mrr=0.75) == pytest.approx(
        -0.75
    )
    assert _early_stop_signal("loss", val_loss=0.25, val_mrr=0.75) == pytest.approx(
        0.25
    )


def test_two_tower_config_decouples_listwise_candidate_groups_from_fusion():
    config = TrainingConfig(
        max_train_events=50_000,
        max_val_events=20_000,
        train_num_negatives=31,
        val_num_negatives=99,
        test_candidate_negative_ratio=0.25,
        two_tower_embedding_dim=64,
        two_tower_hidden_dim=64,
        two_tower_max_samples=200_000,
        two_tower_num_negatives=99,
        two_tower_test_candidate_negative_ratio=1.0,
        two_tower_objective="listwise",
        two_tower_early_stop_metric="mrr",
    )

    tower = config.two_tower_config()

    assert config.resolved_train_num_negatives() == 31
    assert config.resolved_val_num_negatives() == 99
    assert config.max_train_events == 50_000
    assert config.max_val_events == 20_000
    assert config.test_candidate_negative_ratio == pytest.approx(0.25)
    assert tower.embedding_dim == 64
    assert tower.hidden_dim == 64
    assert tower.max_samples == 200_000
    assert tower.num_negatives == 99
    assert tower.test_candidate_negative_ratio == pytest.approx(1.0)
    assert tower.objective == "listwise"
    assert tower.early_stop_metric == "mrr"


def test_two_tower_training_batch_uses_test_candidate_distribution():
    _require_jittor()
    from jgrec.rankers.hybrid.two_tower import (  # noqa: PLC0415
        _build_training_batch_for_events,
        _TowerTrainingContext,
    )

    interactions = InteractionTable.from_events(_interactions())
    id_map = NodeIdMap.from_interactions(interactions)
    index = TemporalInteractionIndex()
    index.fit(interactions, build_transitions=False, build_cooccurs=False)
    profile = DatasetProfile(
        holdout_pair_hit_rate=0.0,
        holdout_new_pair_rate=1.0,
        candidate_unseen_dst_rate=0.0,
        candidate_seen_dst_rate=1.0,
        src_history_p90=1.0,
        test_candidate_top1pct_share=0.5,
        test_candidate_total=20,
        test_candidate_counts=Counter({30: 12, 40: 8}),
    )
    context = _TowerTrainingContext.from_interactions(
        interactions=interactions,
        id_map=id_map,
        index=index,
        dataset_profile=profile,
    )

    batch = _build_training_batch_for_events(
        events=interactions.take(np.asarray([0])),
        negative_seeds=np.asarray([123], dtype=np.uint32),
        training_context=context,
        config=TwoTowerConfig(
            num_negatives=2,
            hard_negative_ratio=0.0,
            popular_negative_ratio=0.0,
            test_candidate_negative_ratio=1.0,
        ),
    )

    expected = {id_map.dst_id(30), id_map.dst_id(40)}
    assert set(batch.dst_ids[0, 1:].tolist()) == expected


def test_two_tower_scores_have_expected_shape_and_signal():
    _require_jittor()
    from jgrec.rankers.hybrid.two_tower import TwoTower  # noqa: PLC0415

    interactions = _interactions()
    interaction_table = InteractionTable.from_events(interactions)
    tower = TwoTower(
        id_map=NodeIdMap.from_interactions(interaction_table),
        config=TwoTowerConfig(
            embedding_dim=8,
            hidden_dim=8,
            epochs=1,
            batch_size=4,
            max_samples=8,
            num_negatives=2,
        ),
    )

    tower.fit(interaction_table, rng=np.random.default_rng(0), verbose=False)
    scores = tower.scores_for_queries(
        [
            TestQuery(src=1, time=110, candidates=(10, 20, 40)),
            TestQuery(src=2, time=110, candidates=(10, 30, 50)),
        ]
    )

    assert scores.shape == (2, 3, len(TWO_TOWER_FEATURE_NAMES))
    assert np.all(np.isfinite(scores))
    assert np.any(scores != 0.0)
    assert np.all(scores[:, :, 1] >= -1.0001)
    assert np.all(scores[:, :, 1] <= 1.0001)
    assert scores[1, 2, 0] == 0.0
    assert scores[1, 2, 1] == 0.0


def test_two_tower_trains_with_cosine_decay_and_in_batch_negatives():
    _require_jittor()
    from jgrec.rankers.hybrid.two_tower import TwoTower  # noqa: PLC0415

    interactions = InteractionTable.from_events(_interactions())
    tower = TwoTower(
        id_map=NodeIdMap.from_interactions(interactions),
        config=TwoTowerConfig(
            embedding_dim=8,
            hidden_dim=8,
            epochs=2,
            batch_size=4,
            max_samples=8,
            num_negatives=2,
            lr_schedule="cosine",
            min_lr_ratio=0.1,
            weight_decay=1e-4,
            in_batch_negatives=True,
            in_batch_negative_weight=0.5,
            in_batch_temperature=0.5,
        ),
    )

    tower.fit(interactions, rng=np.random.default_rng(0), verbose=False)
    scores = tower.scores_for_queries(
        [TestQuery(src=1, time=110, candidates=(10, 20, 40))]
    )

    assert scores.shape == (1, 3, len(TWO_TOWER_FEATURE_NAMES))
    assert np.all(np.isfinite(scores))
    assert np.any(scores != 0.0)


def test_two_tower_scoring_batch_size_preserves_scores():
    _require_jittor()
    from jgrec.rankers.hybrid.two_tower import TwoTower  # noqa: PLC0415

    interactions = _interactions()
    interaction_table = InteractionTable.from_events(interactions)
    queries = [
        TestQuery(src=1, time=110, candidates=(10, 20, 40)),
        TestQuery(src=2, time=110, candidates=(10, 30, 50)),
        TestQuery(src=4, time=110, candidates=(20, 30, 40)),
    ]
    common_config = {
        "embedding_dim": 8,
        "hidden_dim": 8,
        "epochs": 1,
        "batch_size": 4,
        "max_samples": 8,
        "num_negatives": 2,
    }
    tower = TwoTower(
        id_map=NodeIdMap.from_interactions(interaction_table),
        config=TwoTowerConfig(score_batch_size=32, **common_config),
    )
    tower.fit(interaction_table, rng=np.random.default_rng(0), verbose=False)
    full_scores = tower.scores_for_queries(queries)
    tower.config = replace(tower.config, score_batch_size=1)

    np.testing.assert_allclose(
        tower.scores_for_queries(queries),
        full_scores,
        rtol=1e-5,
        atol=1e-5,
    )


def test_two_tower_reuses_future_only_structure_index():
    _require_jittor()
    from jgrec.rankers.hybrid.two_tower import TwoTower  # noqa: PLC0415

    interactions = _interactions()
    interaction_table = InteractionTable.from_events(interactions)
    shared_index = TemporalInteractionIndex()
    shared_index.fit(
        interaction_table,
        build_transitions=True,
        build_cooccurs=True,
        future_only_transition_cooccur=True,
    )
    tower = TwoTower(
        id_map=NodeIdMap.from_interactions(interaction_table),
        config=TwoTowerConfig(
            embedding_dim=8,
            hidden_dim=8,
            epochs=1,
            batch_size=4,
            max_samples=8,
            num_negatives=2,
        ),
    )

    tower.fit(interaction_table, rng=np.random.default_rng(0), verbose=False, shared_index=shared_index)
    scores = tower.scores_for_queries([TestQuery(src=1, time=110, candidates=(10, 20, 40))])

    assert tower.index is shared_index
    assert shared_index.future_only
    assert shared_index.transition_times == {}
    assert shared_index.cooccur_times == {}
    assert scores.shape == (1, 3, len(TWO_TOWER_FEATURE_NAMES))
    assert np.any(scores != 0.0)


def test_disabled_two_tower_uses_zero_features_without_importing_tower_module():
    interactions = _interactions()
    interaction_table = InteractionTable.from_events(interactions)
    config = TrainingConfig(gnn_enabled=False, seq_enabled=False, two_tower_enabled=False, source_profile_enabled=False)
    sys.modules.pop("jgrec.rankers.hybrid.two_tower", None)
    ranker_module = importlib.import_module("jgrec.rankers.hybrid.ranker")

    encoder = ranker_module.HybridFeatureEncoder(
        id_map=NodeIdMap.from_interactions(interaction_table),
        recent_window=4,
        graph_config=config.graph_config(),
        sequence_config=config.sequence_config(),
        two_tower_config=config.two_tower_config(),
    )
    encoder.fit(interaction_table, rng=np.random.default_rng(0), verbose=False)
    features = encoder.features_for_queries([TestQuery(src=1, time=110, candidates=(10, 20))])

    tower_start = (
        len(STAT_FEATURE_NAMES)
        + len(CANDIDATE_PRIOR_FEATURE_NAMES)
        + len(TARGET_WINDOW_FEATURE_NAMES)
        + len(STRUCTURE_FEATURE_NAMES)
        + len(SOURCE_PROFILE_FEATURE_NAMES)
    )
    tower_end = tower_start + len(TWO_TOWER_FEATURE_NAMES)
    assert "jgrec.rankers.hybrid.two_tower" not in sys.modules
    assert features.shape[-1] == (
        len(STAT_FEATURE_NAMES)
        + len(CANDIDATE_PRIOR_FEATURE_NAMES)
        + len(TARGET_WINDOW_FEATURE_NAMES)
        + len(STRUCTURE_FEATURE_NAMES)
        + len(SOURCE_PROFILE_FEATURE_NAMES)
        + len(TWO_TOWER_FEATURE_NAMES)
        + len(GRAPH_WINDOW_NAMES)
        + len(SEQUENCE_FEATURE_NAMES)
    )
    assert np.all(features[:, :, tower_start:tower_end] == 0.0)


def test_hybrid_feature_masks_include_two_tower_groups():
    feature_count = (
        len(STAT_FEATURE_NAMES)
        + len(CANDIDATE_PRIOR_FEATURE_NAMES)
        + len(TARGET_WINDOW_FEATURE_NAMES)
        + len(STRUCTURE_FEATURE_NAMES)
        + len(SOURCE_PROFILE_FEATURE_NAMES)
        + len(TWO_TOWER_FEATURE_NAMES)
        + len(GRAPH_WINDOW_NAMES)
        + len(SEQUENCE_FEATURE_NAMES)
    )

    masks = _feature_masks(feature_count)

    assert [name for name, _ in masks][:12] == [
        "stats",
        "stats_prior",
        "stats_prior_structure",
        "stats_prior_structure_tower",
        "stats_prior_structure_tower_gnn",
        "stats_prior_structure_tower_gnn_seq",
        "stats_prior_target",
        "stats_prior_target_structure",
        "stats_prior_target_structure_profile",
        "stats_prior_target_structure_profile_tower",
        "stats_prior_target_structure_profile_tower_gnn",
        "stats_prior_target_structure_profile_tower_gnn_seq",
    ]


def test_selected_feature_config_disables_unused_two_tower():
    stats_end = len(STAT_FEATURE_NAMES)
    prior_end = stats_end + len(CANDIDATE_PRIOR_FEATURE_NAMES)
    target_end = prior_end + len(TARGET_WINDOW_FEATURE_NAMES)
    structure_end = target_end + len(STRUCTURE_FEATURE_NAMES)
    profile_end = structure_end + len(SOURCE_PROFILE_FEATURE_NAMES)
    tower_end = profile_end + len(TWO_TOWER_FEATURE_NAMES)
    feature_count = tower_end + len(GRAPH_WINDOW_NAMES) + len(SEQUENCE_FEATURE_NAMES)
    config = TrainingConfig(two_tower_enabled=True, gnn_enabled=True, seq_enabled=True)

    stats_config = _config_for_selected_features(config, tuple(range(stats_end)))
    prior_config = _config_for_selected_features(config, tuple(range(prior_end)))
    tower_config = _config_for_selected_features(config, tuple(range(tower_end)))
    tower_no_profile_config = _config_for_selected_features(
        config,
        dict(_feature_masks(feature_count))["stats_prior_structure_tower"],
    )

    assert not stats_config.candidate_prior_enabled
    assert not stats_config.target_window_enabled
    assert not stats_config.source_profile_enabled
    assert not stats_config.two_tower_enabled
    assert not stats_config.gnn_enabled
    assert not stats_config.seq_enabled
    assert prior_config.candidate_prior_enabled
    assert not prior_config.target_window_enabled
    assert not prior_config.source_profile_enabled
    assert not prior_config.two_tower_enabled
    assert tower_config.source_profile_enabled
    assert tower_config.two_tower_enabled
    assert not tower_config.gnn_enabled
    assert not tower_config.seq_enabled
    assert not tower_no_profile_config.target_window_enabled
    assert not tower_no_profile_config.source_profile_enabled
    assert tower_no_profile_config.structure_enabled
    assert tower_no_profile_config.two_tower_enabled


def test_champion_structure_tower_gnn_mask_excludes_experimental_towers():
    feature_count = (
        len(STAT_FEATURE_NAMES)
        + len(CANDIDATE_PRIOR_FEATURE_NAMES)
        + len(TARGET_WINDOW_FEATURE_NAMES)
        + len(STRUCTURE_FEATURE_NAMES)
        + len(SOURCE_PROFILE_FEATURE_NAMES)
        + len(TWO_TOWER_FEATURE_NAMES)
        + len(GRAPH_WINDOW_NAMES)
        + len(SEQUENCE_FEATURE_NAMES)
    )
    selected = dict(_feature_masks(feature_count))["stats_prior_structure_tower_gnn"]

    selected_config = _config_for_selected_features(
        TrainingConfig(
            target_window_enabled=True,
            source_profile_enabled=True,
            structure_enabled=True,
            two_tower_enabled=True,
            gnn_enabled=True,
            seq_enabled=True,
        ),
        selected,
    )

    assert not selected_config.target_window_enabled
    assert not selected_config.source_profile_enabled
    assert selected_config.structure_enabled
    assert selected_config.two_tower_enabled
    assert selected_config.gnn_enabled
    assert not selected_config.seq_enabled


def test_feature_masks_respect_disabled_experimental_towers():
    feature_count = (
        len(STAT_FEATURE_NAMES)
        + len(CANDIDATE_PRIOR_FEATURE_NAMES)
        + len(TARGET_WINDOW_FEATURE_NAMES)
        + len(STRUCTURE_FEATURE_NAMES)
        + len(SOURCE_PROFILE_FEATURE_NAMES)
        + len(TWO_TOWER_FEATURE_NAMES)
        + len(GRAPH_WINDOW_NAMES)
        + len(SEQUENCE_FEATURE_NAMES)
    )

    masks = _feature_masks(
        feature_count,
        config=TrainingConfig(target_window_enabled=False, source_profile_enabled=False),
    )

    assert [name for name, _ in masks][:6] == [
        "stats",
        "stats_prior",
        "stats_prior_structure",
        "stats_prior_structure_tower",
        "stats_prior_structure_tower_gnn",
        "stats_prior_structure_tower_gnn_seq",
    ]
