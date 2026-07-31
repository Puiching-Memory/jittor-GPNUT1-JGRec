from collections import Counter
from dataclasses import replace

import numpy as np

from jgrec.core.types import Interaction, InteractionTable
from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid.auto_strategy import DatasetProfile
from jgrec.rankers.hybrid.config import TrainingConfig
from jgrec.rankers.hybrid.ranker import TemporalHybridRanker
from jgrec.rankers.hybrid.supervised_feature_cache import SupervisedFeatureCache, supervised_feature_cache_key


def _interactions(count: int = 24) -> InteractionTable:
    return InteractionTable.from_events(
        [Interaction(src=index % 3 + 1, dst=index % 5 + 10, time=index + 1) for index in range(count)]
    )


def test_cache_key_reuses_features_across_fusion_only_changes():
    interactions = _interactions()
    base = TrainingConfig(
        epochs=5,
        train_batch_size=128,
        selection_metric="ap",
        early_stop_patience=2,
        fusion_hidden_dim=32,
        fusion_mode="mlp",
    )
    tuned_fusion = replace(
        base,
        epochs=50,
        train_batch_size=1024,
        selection_metric="mrr",
        early_stop_patience=8,
        fusion_hidden_dim=256,
        fusion_mode="ensemble",
    )

    base_key = supervised_feature_cache_key(
        interactions,
        base,
        recent_window=32,
        feature_names=("stats", "structure"),
    )
    tuned_key = supervised_feature_cache_key(
        interactions,
        tuned_fusion,
        recent_window=32,
        feature_names=("stats", "structure"),
    )

    assert tuned_key == base_key


def test_cache_round_trip_loads_read_only_memmaps(tmp_path):
    cache = SupervisedFeatureCache(tmp_path)
    train = (0.25 * np.arange(24, dtype=np.float32)).reshape(2, 3, 4)
    val = (0.5 * np.arange(32, dtype=np.float32)).reshape(2, 4, 4)

    assert cache.load("feature-key") is None

    cache.save("feature-key", train, val)
    loaded = cache.load("feature-key")

    assert loaded is not None
    loaded_train, loaded_val = loaded
    assert isinstance(loaded_train, np.memmap)
    assert isinstance(loaded_val, np.memmap)
    assert not loaded_train.flags.writeable
    assert not loaded_val.flags.writeable
    np.testing.assert_array_equal(loaded_train, train)
    np.testing.assert_array_equal(loaded_val, val)


def test_cache_round_trip_loads_candidate_identity_sidecars(tmp_path):
    cache = SupervisedFeatureCache(tmp_path)
    train = np.ones((2, 3, 4), dtype=np.float32)
    val = np.ones((2, 4, 4), dtype=np.float32)
    train_candidates = np.asarray([[10, 20, 30], [11, 21, 31]], dtype=np.int32)
    val_candidates = np.asarray([[12, 22, 32, 42], [13, 23, 33, 43]], dtype=np.int32)

    cache.save(
        "feature-key",
        train,
        val,
        train_candidates=train_candidates,
        val_candidates=val_candidates,
    )
    loaded = cache.load_candidate_ids("feature-key")

    assert loaded is not None
    loaded_train, loaded_val = loaded
    assert isinstance(loaded_train, np.memmap)
    assert isinstance(loaded_val, np.memmap)
    assert not loaded_train.flags.writeable
    assert not loaded_val.flags.writeable
    np.testing.assert_array_equal(loaded_train, train_candidates)
    np.testing.assert_array_equal(loaded_val, val_candidates)


def test_cache_rejects_candidate_identity_shape_mismatch_and_keeps_legacy_optional(tmp_path):
    cache = SupervisedFeatureCache(tmp_path)
    train = np.ones((2, 3, 4), dtype=np.float32)
    val = np.ones((2, 4, 4), dtype=np.float32)

    cache.save("legacy-key", train, val)
    assert cache.load("legacy-key") is not None
    assert cache.load_candidate_ids("legacy-key") is None

    with np.testing.assert_raises_regex(ValueError, "train candidate IDs"):
        cache.save(
            "bad-key",
            train,
            val,
            train_candidates=np.ones((2, 2), dtype=np.int32),
            val_candidates=np.ones((2, 4), dtype=np.int32),
        )
    assert cache.load("bad-key") is None


def test_cache_key_changes_with_feature_inputs():
    interactions = _interactions()
    config = TrainingConfig(structure_cooccur_history_limit=32)
    profile = DatasetProfile(
        holdout_pair_hit_rate=0.5,
        holdout_new_pair_rate=0.5,
        candidate_unseen_dst_rate=0.25,
        candidate_seen_dst_rate=0.75,
        src_history_p90=10.0,
        test_candidate_top1pct_share=0.2,
        test_candidate_total=3,
        test_candidate_counts=Counter({10: 2, 11: 1}),
    )

    def key(events=interactions, feature_config=config, dataset_profile=profile):
        return supervised_feature_cache_key(
            events,
            feature_config,
            recent_window=32,
            feature_names=("stats", "structure"),
            dataset_profile=dataset_profile,
        )

    assert key(feature_config=replace(config, structure_cooccur_history_limit=64)) != key()
    assert key(events=_interactions(25)) != key()
    assert key(dataset_profile=replace(profile, candidate_unseen_dst_rate=0.5)) != key()
    assert key(dataset_profile=replace(profile, test_candidate_counts=Counter({10: 1, 11: 2}))) != key()


def test_corrupt_cache_entry_is_treated_as_a_miss(tmp_path):
    cache = SupervisedFeatureCache(tmp_path)
    train = np.ones((2, 3, 4), dtype=np.float32)
    val = np.ones((2, 4, 4), dtype=np.float32)
    cache.save("feature-key", train, val)
    next(tmp_path.glob("*.train.npy")).write_bytes(b"not-a-numpy-array")

    assert cache.load("feature-key") is None


def test_learn_fusion_reuses_cached_features_without_refitting_supervised_encoders(tmp_path, monkeypatch):
    interactions = _interactions(120)
    config = TrainingConfig(
        val_ratio=0.25,
        context_ratio=0.5,
        num_negatives=1,
        train_num_negatives=2,
        val_num_negatives=4,
        epochs=1,
        max_train_events=0,
        max_val_events=0,
        encoder_state_cache_enabled=False,
        auto_strategy_enabled=False,
        supervised_feature_cache_dir=tmp_path,
        fusion_mode="mlp",
        verbose=False,
    )
    ranker = TemporalHybridRanker()
    ranker.id_map = NodeIdMap.from_interactions(interactions)
    ranker.feature_names = ("feature",)
    feature_builds: list[str] = []
    fusion_inputs: list[tuple[np.ndarray, np.ndarray]] = []
    fusion_rng_samples: list[int] = []

    class DummyEncoder:
        feature_dim = 1

    class DummyFusionResult:
        best_val_ap = 0.1
        best_val_mrr = 0.2
        feature_indices = (0,)
        candidate_name = "dummy"

    def fake_build_supervised_features(positives, encoder, dst_pool, split_config, rng, label):
        feature_builds.append(label)
        rng.integers(0, 2**31 - 1, size=len(positives) * split_config.num_negatives)
        fill = 1.0 if label == "train_features" else 2.0
        return np.full(
            (len(positives), split_config.num_negatives + 1, encoder.feature_dim),
            fill,
            dtype=np.float32,
        )

    def fake_fit_best_fusion(**kwargs):
        fusion_inputs.append((np.asarray(kwargs["train_features"]).copy(), np.asarray(kwargs["val_features"]).copy()))
        fusion_rng_samples.append(int(kwargs["rng"].integers(0, 2**31 - 1)))
        return object(), DummyFusionResult()

    monkeypatch.setattr(ranker, "_timed_fit_encoder", lambda *args, **kwargs: DummyEncoder())
    monkeypatch.setattr(ranker, "_fit_best_fusion", fake_fit_best_fusion)
    monkeypatch.setattr("jgrec.rankers.hybrid.ranker._build_supervised_features", fake_build_supervised_features)

    ranker._learn_fusion(interactions, config)
    assert feature_builds == ["train_features", "val_features"]

    monkeypatch.setattr(
        ranker,
        "_timed_fit_encoder",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache hit refit an encoder")),
    )
    monkeypatch.setattr(
        "jgrec.rankers.hybrid.ranker._build_supervised_features",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache hit rebuilt features")),
    )

    ranker._learn_fusion(interactions, replace(config, fusion_hidden_dim=128, selection_metric="mrr"))

    assert feature_builds == ["train_features", "val_features"]
    assert len(fusion_inputs) == 2
    np.testing.assert_array_equal(fusion_inputs[1][0], fusion_inputs[0][0])
    np.testing.assert_array_equal(fusion_inputs[1][1], fusion_inputs[0][1])
    assert fusion_rng_samples[1] == fusion_rng_samples[0]
