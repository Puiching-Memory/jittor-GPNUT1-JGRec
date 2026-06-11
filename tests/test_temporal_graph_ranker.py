from pathlib import Path

import numpy as np
import pytest

import jgrec.rankers.temporal_graph.ranker as ranker_module
from jgrec.core.types import DatasetPaths, FitContext, InteractionTable
from jgrec.rankers.temporal_graph.ranker import TemporalGraphRanker, TemporalGraphTrainingConfig


def _minimal_interactions() -> InteractionTable:
    return InteractionTable.from_array(
        np.asarray(
            [
                [1, 100, 10],
                [1, 101, 11],
                [2, 100, 12],
                [2, 102, 13],
            ],
            dtype=np.int32,
        )
    )


def _fit_context() -> FitContext:
    dataset = DatasetPaths(
        name="dummy",
        root=Path("data/dummy"),
        train_path=Path("data/dummy/train.csv"),
        test_path=Path("data/dummy/test.csv"),
    )
    return FitContext(dataset=dataset, seed=777, verbose=False)


def test_fit_configures_jittor_cuda_and_seed_before_building_training_graph(monkeypatch) -> None:
    class FakeFlags:
        use_cuda = 0

    class FakeJittor:
        flags = FakeFlags()

        def __init__(self) -> None:
            self.seeds: list[int] = []

        def set_global_seed(self, seed: int) -> None:
            self.seeds.append(seed)

    def stop_before_graph_build(*args, **kwargs):
        raise RuntimeError("stop before graph build")

    fake_jittor = FakeJittor()
    monkeypatch.setattr(ranker_module, "jt", fake_jittor)
    monkeypatch.setattr(
        ranker_module.TemporalNodeMap,
        "from_interactions_and_test",
        stop_before_graph_build,
    )

    with pytest.raises(RuntimeError, match="stop before graph build"):
        TemporalGraphRanker().fit(
            _minimal_interactions(),
            TemporalGraphTrainingConfig(seed=777, verbose=False),
            _fit_context(),
        )

    assert fake_jittor.flags.use_cuda == 1
    assert fake_jittor.seeds == [777]


def test_fit_always_enables_cuda_before_building_training_graph(monkeypatch) -> None:
    class FakeFlags:
        use_cuda = 0

    class FakeJittor:
        flags = FakeFlags()

        def __init__(self) -> None:
            self.seeds: list[int] = []

        def set_global_seed(self, seed: int) -> None:
            self.seeds.append(seed)

    def stop_before_graph_build(*args, **kwargs):
        raise RuntimeError("stop before graph build")

    fake_jittor = FakeJittor()
    monkeypatch.setattr(ranker_module, "jt", fake_jittor)
    monkeypatch.setattr(
        ranker_module.TemporalNodeMap,
        "from_interactions_and_test",
        stop_before_graph_build,
    )

    with pytest.raises(RuntimeError, match="stop before graph build"):
        TemporalGraphRanker().fit(
            _minimal_interactions(),
            TemporalGraphTrainingConfig(seed=777, verbose=False),
            _fit_context(),
        )

    assert fake_jittor.flags.use_cuda == 1
    assert fake_jittor.seeds == [777]


def test_fit_builds_selection_candidate_prior_from_train_split_only(monkeypatch) -> None:
    class FakeFlags:
        use_cuda = 0

    class FakeJittor:
        flags = FakeFlags()

        def set_global_seed(self, seed: int) -> None:
            pass

    class FakeNodeMap:
        num_nodes = 10
        num_dst = 4

        @classmethod
        def from_interactions_and_test(cls, interactions, test_path):
            return cls()

        def dst_ids(self, values):
            return np.asarray(values, dtype=np.int32)

        def src_ids(self, values):
            return np.asarray(values, dtype=np.int32)

    class FakeModel:
        config = type("Config", (), {"history_len": 2, "candidate_history_len": 2})()

    prior_calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    train_prior_holder: dict[str, object] = {}

    def fake_temporal_data_from_interactions(interactions, node_map):
        return object()

    def fake_temporal_loader_api():
        return None, lambda data, strategy, seed: object()

    def fake_safe_neighbor_sampler(sampler):
        return sampler

    def fake_test_candidate_index(self, context):
        return object()

    def fake_build_model(self, time_span):
        return FakeModel()

    def fake_from_test_candidates(candidate_index, train_dst_ids, train_times=None, recent_feature_group="none"):
        marker = object()
        prior_calls.append((tuple(int(value) for value in train_dst_ids), tuple(int(value) for value in train_times)))
        return marker

    def fake_train_listwise(**kwargs):
        train_prior_holder["value"] = kwargs["candidate_prior_index"]
        return type(
            "Result",
            (),
            {
                "best_val_ap": 0.1,
                "best_val_mrr": 0.2,
                "best_epoch": 1,
                "state": {},
            },
        )()

    monkeypatch.setattr(ranker_module, "jt", FakeJittor())
    monkeypatch.setattr(ranker_module, "TemporalNodeMap", FakeNodeMap)
    monkeypatch.setattr(ranker_module, "temporal_data_from_interactions", fake_temporal_data_from_interactions)
    monkeypatch.setattr(ranker_module, "temporal_loader_api", fake_temporal_loader_api)
    monkeypatch.setattr(ranker_module, "safe_neighbor_sampler", fake_safe_neighbor_sampler)
    monkeypatch.setattr(TemporalGraphRanker, "_test_candidate_index", fake_test_candidate_index)
    monkeypatch.setattr(TemporalGraphRanker, "_build_model", fake_build_model)
    monkeypatch.setattr(ranker_module.CandidatePriorIndex, "from_test_candidates", fake_from_test_candidates)
    monkeypatch.setattr(ranker_module, "train_listwise", fake_train_listwise)

    interactions = InteractionTable.from_array(
        np.asarray(
            [
                [1, 100, 10],
                [1, 101, 11],
                [2, 102, 12],
                [2, 103, 13],
                [3, 104, 14],
                [3, 105, 15],
                [4, 106, 16],
                [4, 107, 17],
                [5, 108, 18],
                [5, 109, 19],
            ],
            dtype=np.int32,
        )
    )

    TemporalGraphRanker().fit(
        interactions,
        TemporalGraphTrainingConfig(
            seed=777,
            verbose=False,
            val_ratio=0.2,
            refit_full=False,
            candidate_recent_feature_group="recency_rank",
        ),
        _fit_context(),
    )

    assert prior_calls == [
        (
            tuple(range(100, 108)),
            tuple(range(10, 18)),
        )
    ]
    assert train_prior_holder["value"] is not None
