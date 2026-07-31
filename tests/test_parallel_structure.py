from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pytest

from jgrec.core.types import Interaction, InteractionTable, TestQueryArray
from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid.config import (
    SourceProfileConfig,
    StructureTowerConfig,
)
from jgrec.rankers.hybrid.parallel_structure import (
    ForkedStructureFeatureTower,
    select_parallel_worker_trial,
    validate_exact_parallel_features,
)
from jgrec.rankers.hybrid.source_profile import SourceProfileTower
from jgrec.rankers.hybrid.structure import StructureFeatureTower


def _fitted_future_structure_tower() -> StructureFeatureTower:
    interactions = InteractionTable.from_events(
        [
            Interaction(src=1, dst=10, time=10),
            Interaction(src=2, dst=10, time=20),
            Interaction(src=3, dst=20, time=30),
            Interaction(src=4, dst=20, time=40),
            Interaction(src=10, dst=30, time=50),
            Interaction(src=20, dst=40, time=60),
            Interaction(src=1, dst=20, time=70),
            Interaction(src=2, dst=30, time=80),
            Interaction(src=3, dst=40, time=90),
            Interaction(src=4, dst=10, time=100),
        ]
    )
    tower = StructureFeatureTower(
        StructureTowerConfig(
            future_only_transition_cooccur=True,
            cache_max_bytes=4 * 1024 * 1024,
        )
    )
    tower.fit(
        interactions,
        rng=np.random.default_rng(0),
        verbose=False,
    )
    return tower


def _fitted_source_profile_tower() -> SourceProfileTower:
    interactions = InteractionTable.from_events(
        [
            Interaction(src=1, dst=10, time=10),
            Interaction(src=2, dst=10, time=20),
            Interaction(src=3, dst=20, time=30),
            Interaction(src=4, dst=20, time=40),
            Interaction(src=1, dst=20, time=50),
            Interaction(src=2, dst=30, time=60),
            Interaction(src=3, dst=40, time=70),
            Interaction(src=4, dst=10, time=80),
            Interaction(src=1, dst=30, time=90),
            Interaction(src=2, dst=40, time=100),
        ]
    )
    tower = SourceProfileTower(
        NodeIdMap.from_interactions(interactions),
        SourceProfileConfig(
            item2vec_enabled=False,
            epochs=0,
            cache_max_bytes=4 * 1024 * 1024,
        ),
    )
    tower.fit(
        interactions,
        rng=np.random.default_rng(0),
        verbose=False,
    )
    return tower


@pytest.mark.skipif(
    "fork" not in mp.get_all_start_methods(),
    reason="formal acceleration is Linux fork-only",
)
def test_forked_structure_features_are_exact_and_use_multiple_processes():
    tower = _fitted_future_structure_tower()
    queries = TestQueryArray(
        src=np.asarray([1, 2, 3, 4, 1, 2, 3, 4], dtype=np.int32),
        time=np.asarray([110, 111, 112, 113, 114, 115, 116, 117], dtype=np.int32),
        candidates=np.asarray(
            [
                [10, 20, 30, 40],
                [20, 30, 40, 10],
                [30, 40, 10, 20],
                [40, 10, 20, 30],
                [20, 10, 40, 30],
                [30, 20, 10, 40],
                [40, 30, 20, 10],
                [10, 40, 30, 20],
            ],
            dtype=np.int32,
        ),
    )
    expected = tower.features_for_query_array(queries)

    with ForkedStructureFeatureTower(tower, worker_count=2) as parallel:
        actual = parallel.features_for_query_array(queries)
        worker_pids = parallel.worker_pids

    np.testing.assert_array_equal(actual, expected)
    assert len(worker_pids) == 2


@pytest.mark.skipif(
    "fork" not in mp.get_all_start_methods(),
    reason="formal acceleration is Linux fork-only",
)
def test_forked_structure_pool_also_preserves_source_profile_scores():
    structure = _fitted_future_structure_tower()
    source_profile = _fitted_source_profile_tower()
    queries = TestQueryArray(
        src=np.asarray([1, 2, 3, 4, 1, 2, 3, 4], dtype=np.int32),
        time=np.asarray([110, 111, 112, 113, 114, 115, 116, 117], dtype=np.int32),
        candidates=np.asarray(
            [
                [10, 20, 30, 40],
                [20, 30, 40, 10],
                [30, 40, 10, 20],
                [40, 10, 20, 30],
                [20, 10, 40, 30],
                [30, 20, 10, 40],
                [40, 30, 20, 10],
                [10, 40, 30, 20],
            ],
            dtype=np.int32,
        ),
    )
    expected_structure = structure.features_for_query_array(queries)
    expected_profile = source_profile.scores_for_query_array(queries)

    with ForkedStructureFeatureTower(
        structure,
        worker_count=2,
        source_profile=source_profile,
    ) as parallel:
        actual_structure = parallel.features_for_query_array(queries)
        actual_profile = parallel.source_profile.scores_for_query_array(queries)

    np.testing.assert_array_equal(actual_structure, expected_structure)
    np.testing.assert_array_equal(actual_profile, expected_profile)


def test_exact_parallel_feature_gate_reports_speed_and_worker_utilization():
    sequential = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    report = validate_exact_parallel_features(
        sequential,
        sequential.copy(),
        sequential_seconds=3.0,
        parallel_seconds=1.5,
        minimum_speedup=1.5,
        worker_pids={101, 202},
    )

    assert report["exact_equal"] is True
    assert report["speedup"] == pytest.approx(2.0)
    assert report["worker_count_observed"] == 2


def test_exact_parallel_feature_gate_rejects_any_value_drift():
    sequential = np.zeros((2, 3, 4), dtype=np.float32)
    parallel = sequential.copy()
    parallel[1, 2, 3] = np.nextafter(
        np.float32(0.0),
        np.float32(1.0),
    )

    with pytest.raises(ValueError, match="not exactly equal"):
        validate_exact_parallel_features(
            sequential,
            parallel,
            sequential_seconds=3.0,
            parallel_seconds=1.0,
            minimum_speedup=1.5,
            worker_pids={101, 202},
        )


def test_exact_parallel_feature_gate_rejects_insufficient_speedup():
    features = np.zeros((2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="speedup"):
        validate_exact_parallel_features(
            features,
            features.copy(),
            sequential_seconds=3.0,
            parallel_seconds=2.5,
            minimum_speedup=1.5,
            worker_pids={101, 202},
        )


def test_exact_parallel_feature_gate_requires_multiple_workers():
    features = np.zeros((2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="multiple worker"):
        validate_exact_parallel_features(
            features,
            features.copy(),
            sequential_seconds=3.0,
            parallel_seconds=1.0,
            minimum_speedup=1.5,
            worker_pids={101},
        )


def test_parallel_worker_trial_selects_exact_faster_eight_worker_arm():
    baseline = {
        "status": "passed",
        "parallel_seconds": 10.0,
        "sequential_sha256": "a" * 64,
        "parallel_sha256": "a" * 64,
        "worker_count_observed": 4,
    }
    trial = {
        "status": "passed",
        "parallel_seconds": 8.0,
        "sequential_sha256": "a" * 64,
        "parallel_sha256": "a" * 64,
        "worker_count_observed": 8,
    }

    decision = select_parallel_worker_trial(
        baseline_report=baseline,
        trial_report=trial,
        baseline_worker_count=4,
        trial_worker_count=8,
        minimum_incremental_speedup=1.10,
        available_memory_bytes=12 * 1024**3,
        minimum_memory_reserve_bytes=8 * 1024**3,
    )

    assert decision["selected_worker_count"] == 8
    assert decision["incremental_speedup"] == pytest.approx(1.25)
    assert decision["fallback_reason"] is None


@pytest.mark.parametrize(
    ("trial_seconds", "available_gib", "reason"),
    [
        (9.5, 12, "incremental_speedup"),
        (8.0, 7, "memory_reserve"),
    ],
)
def test_parallel_worker_trial_falls_back_to_four_when_gate_fails(
    trial_seconds: float,
    available_gib: int,
    reason: str,
):
    baseline = {
        "status": "passed",
        "parallel_seconds": 10.0,
        "sequential_sha256": "a" * 64,
        "parallel_sha256": "a" * 64,
        "worker_count_observed": 4,
    }
    trial = {
        "status": "passed",
        "parallel_seconds": trial_seconds,
        "sequential_sha256": "a" * 64,
        "parallel_sha256": "a" * 64,
        "worker_count_observed": 8,
    }

    decision = select_parallel_worker_trial(
        baseline_report=baseline,
        trial_report=trial,
        baseline_worker_count=4,
        trial_worker_count=8,
        minimum_incremental_speedup=1.10,
        available_memory_bytes=available_gib * 1024**3,
        minimum_memory_reserve_bytes=8 * 1024**3,
    )

    assert decision["selected_worker_count"] == 4
    assert decision["fallback_reason"] == reason


def test_parallel_worker_trial_rejects_cross_arm_byte_drift():
    baseline = {
        "status": "passed",
        "parallel_seconds": 10.0,
        "sequential_sha256": "a" * 64,
        "parallel_sha256": "a" * 64,
        "worker_count_observed": 4,
    }
    trial = {
        "status": "passed",
        "parallel_seconds": 8.0,
        "sequential_sha256": "a" * 64,
        "parallel_sha256": "b" * 64,
        "worker_count_observed": 8,
    }

    with pytest.raises(ValueError, match="byte hashes differ"):
        select_parallel_worker_trial(
            baseline_report=baseline,
            trial_report=trial,
            baseline_worker_count=4,
            trial_worker_count=8,
            minimum_incremental_speedup=1.10,
            available_memory_bytes=12 * 1024**3,
            minimum_memory_reserve_bytes=8 * 1024**3,
        )
