import numpy as np

from jgrec.core.types import Interaction, InteractionTable
from jgrec.idmap import NodeIdMap
from jgrec.rankers.common.temporal_index import TemporalInteractionIndex
from jgrec.rankers.hybrid.config import TrainingConfig
from jgrec.rankers.hybrid.ranker import TemporalHybridRanker
from jgrec.rankers.hybrid.sampling import (
    NegativeSamplingContext,
    NegativeSamplingJob,
    sample_mixed_negatives,
    sample_mixed_negatives_batch,
)


def _context(interactions: list[Interaction]) -> NegativeSamplingContext:
    interaction_table = InteractionTable.from_events(interactions)
    index = TemporalInteractionIndex()
    index.fit(interaction_table)
    return NegativeSamplingContext(
        index=index,
        dst_values=NodeIdMap.from_interactions(interaction_table).dst_values,
    )


def test_mixed_negative_sampling_prefers_recent_and_structural_hard_candidates():
    context = _context(
        [
            Interaction(src=1, dst=10, time=10),
            Interaction(src=1, dst=20, time=20),
            Interaction(src=2, dst=20, time=30),
            Interaction(src=2, dst=30, time=40),
            Interaction(src=3, dst=10, time=50),
            Interaction(src=3, dst=30, time=60),
        ]
    )

    negatives = sample_mixed_negatives(
        src=1,
        positive_dst=50,
        query_time=70,
        context=context,
        dst_pool=np.asarray([10, 20, 30, 40, 50], dtype=np.int64),
        num_negatives=2,
        rng=np.random.default_rng(7),
        hard_negative_ratio=1.0,
        popular_negative_ratio=0.0,
    )

    assert len(negatives) == 2
    assert set(negatives) & {10, 20}
    assert 30 in negatives
    assert 50 not in negatives


def test_mixed_negative_sampling_is_reproducible_for_fixed_seed():
    context = _context(
        [
            Interaction(src=1, dst=10, time=10),
            Interaction(src=1, dst=20, time=20),
            Interaction(src=2, dst=20, time=30),
            Interaction(src=2, dst=30, time=40),
            Interaction(src=3, dst=40, time=50),
            Interaction(src=4, dst=60, time=60),
        ]
    )
    dst_pool = np.asarray([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int64)

    first = sample_mixed_negatives(
        src=1,
        positive_dst=50,
        query_time=70,
        context=context,
        dst_pool=dst_pool,
        num_negatives=4,
        rng=np.random.default_rng(11),
        hard_negative_ratio=0.5,
        popular_negative_ratio=0.25,
    )
    second = sample_mixed_negatives(
        src=1,
        positive_dst=50,
        query_time=70,
        context=context,
        dst_pool=dst_pool,
        num_negatives=4,
        rng=np.random.default_rng(11),
        hard_negative_ratio=0.5,
        popular_negative_ratio=0.25,
    )

    assert first == second


def test_mixed_negative_sampling_avoids_positive_until_pool_exhaustion():
    context = _context(
        [
            Interaction(src=1, dst=10, time=10),
            Interaction(src=1, dst=20, time=20),
            Interaction(src=2, dst=20, time=30),
            Interaction(src=2, dst=30, time=40),
            Interaction(src=3, dst=40, time=50),
        ]
    )

    negatives = sample_mixed_negatives(
        src=1,
        positive_dst=50,
        query_time=60,
        context=context,
        dst_pool=np.asarray([10, 20, 30, 40, 50, 60], dtype=np.int64),
        num_negatives=4,
        rng=np.random.default_rng(3),
        hard_negative_ratio=0.5,
        popular_negative_ratio=0.25,
    )
    exhausted = sample_mixed_negatives(
        src=1,
        positive_dst=50,
        query_time=60,
        context=context,
        dst_pool=np.asarray([50], dtype=np.int64),
        num_negatives=2,
        rng=np.random.default_rng(3),
        hard_negative_ratio=0.5,
        popular_negative_ratio=0.25,
    )

    assert 50 not in negatives
    assert exhausted == (50, 50)


def test_batched_negative_sampling_matches_single_process_default():
    context = _context(
        [
            Interaction(src=1, dst=10, time=10),
            Interaction(src=1, dst=20, time=20),
            Interaction(src=2, dst=20, time=30),
            Interaction(src=2, dst=30, time=40),
            Interaction(src=3, dst=40, time=50),
            Interaction(src=4, dst=60, time=60),
        ]
    )
    dst_pool = np.asarray([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int64)
    jobs = [
        NegativeSamplingJob(src=1, positive_dst=50, query_time=70),
        NegativeSamplingJob(src=2, positive_dst=60, query_time=80),
    ]

    batched = sample_mixed_negatives_batch(
        jobs=jobs,
        context=context,
        dst_pool=dst_pool,
        num_negatives=4,
        rng=np.random.default_rng(11),
        hard_negative_ratio=0.5,
        popular_negative_ratio=0.25,
        workers=0,
    )
    manual_rng = np.random.default_rng(11)
    manual = [
        sample_mixed_negatives(
            src=job.src,
            positive_dst=job.positive_dst,
            query_time=job.query_time,
            context=context,
            dst_pool=dst_pool,
            num_negatives=4,
            rng=manual_rng,
            hard_negative_ratio=0.5,
            popular_negative_ratio=0.25,
        )
        for job in jobs
    ]

    assert batched == manual


def test_parallel_negative_sampling_is_stable_across_worker_counts():
    context = _context(
        [
            Interaction(src=1, dst=10, time=10),
            Interaction(src=1, dst=20, time=20),
            Interaction(src=2, dst=20, time=30),
            Interaction(src=2, dst=30, time=40),
            Interaction(src=3, dst=40, time=50),
            Interaction(src=4, dst=60, time=60),
            Interaction(src=5, dst=70, time=70),
            Interaction(src=6, dst=80, time=80),
        ]
    )
    dst_pool = np.asarray([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=np.int64)
    jobs = [
        NegativeSamplingJob(src=src, positive_dst=positive_dst, query_time=100 + idx)
        for idx, (src, positive_dst) in enumerate([(1, 50), (2, 60), (3, 70), (4, 80)])
    ]

    two_workers = sample_mixed_negatives_batch(
        jobs=jobs,
        context=context,
        dst_pool=dst_pool,
        num_negatives=4,
        rng=np.random.default_rng(17),
        hard_negative_ratio=0.5,
        popular_negative_ratio=0.25,
        workers=2,
    )
    three_workers = sample_mixed_negatives_batch(
        jobs=jobs,
        context=context,
        dst_pool=dst_pool,
        num_negatives=4,
        rng=np.random.default_rng(17),
        hard_negative_ratio=0.5,
        popular_negative_ratio=0.25,
        workers=3,
    )

    assert two_workers == three_workers
    assert [len(negatives) for negatives in two_workers] == [4, 4, 4, 4]
    for job, negatives in zip(jobs, two_workers, strict=True):
        assert job.positive_dst not in negatives


def test_learn_fusion_uses_visible_dst_pools_for_train_and_val(monkeypatch):
    interactions = InteractionTable.from_events(
        [
            Interaction(src=idx, dst=dst, time=idx)
            for idx, dst in enumerate(
                [
                    *([10, 20, 30] * 15),
                    *([40, 50, 60] * 15),
                    *([70, 80] * 15),
                ],
                start=1,
            )
        ]
    )
    config = TrainingConfig(
        val_ratio=0.25,
        context_ratio=0.5,
        num_negatives=1,
        epochs=1,
        max_train_events=0,
        max_val_events=0,
        encoder_state_cache_enabled=False,
        test_candidate_negative_ratio=1.0,
        auto_strategy_enabled=False,
        verbose=False,
    )
    ranker = TemporalHybridRanker()
    ranker.id_map = NodeIdMap.from_interactions(interactions)
    ranker.feature_names = ("feature",)
    captured: list[tuple[str, tuple[int, ...], float]] = []

    class DummyEncoder:
        feature_dim = 1

    def fake_fit_encoder(*args, **kwargs):
        return DummyEncoder()

    def fake_build_supervised_features(positives, encoder, dst_pool, config, rng, label):
        captured.append((label, tuple(int(value) for value in dst_pool), config.test_candidate_negative_ratio))
        return np.zeros((len(positives), config.num_negatives + 1, encoder.feature_dim), dtype=np.float32)

    class DummyFusionResult:
        best_val_ap = 0.1
        best_val_mrr = 0.2
        feature_indices = (0,)
        candidate_name = "dummy"

    monkeypatch.setattr(ranker, "_timed_fit_encoder", fake_fit_encoder)
    monkeypatch.setattr(ranker, "_fit_best_fusion", lambda **kwargs: (object(), DummyFusionResult()))
    monkeypatch.setattr("jgrec.rankers.hybrid.ranker._build_supervised_features", fake_build_supervised_features)

    ranker._learn_fusion(interactions, config)

    assert captured == [
        ("train_features", (10, 20, 30, 40, 50, 60, 70, 80), 1.0),
        ("val_features", (10, 20, 30, 40, 50, 60, 70, 80), 1.0),
    ]
