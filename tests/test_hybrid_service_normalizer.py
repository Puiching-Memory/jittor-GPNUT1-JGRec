from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Hybrid ranker imports Jittor; integration is verified on Linux.",
)
def test_ranker_recalibrates_raw_and_setwise_heads_without_changing_state() -> None:
    from jgrec.core.types import TestQueryArray  # noqa: PLC0415
    from jgrec.rankers.hybrid.fusion import FusionResult  # noqa: PLC0415
    from jgrec.rankers.hybrid.ranker import TemporalHybridRanker  # noqa: PLC0415
    from jgrec.rankers.hybrid.setwise import (  # noqa: PLC0415
        setwise_context_features,
    )

    class _Encoder:
        @staticmethod
        def features_for_query_array(queries: TestQueryArray) -> np.ndarray:
            sources = np.broadcast_to(
                queries.src[:, None],
                queries.candidates.shape,
            )
            return np.stack(
                (
                    sources,
                    queries.candidates,
                    sources + queries.candidates,
                ),
                axis=-1,
            ).astype(np.float32)

        @staticmethod
        def clear_batch_caches() -> None:
            return

    def _result(
        *,
        feature_indices: tuple[int, ...],
        name: str,
    ) -> FusionResult:
        feature_dim = len(feature_indices)
        return FusionResult(
            best_val_ap=0.1,
            best_val_mrr=0.2,
            state={"weight": np.ones((feature_dim, 1), dtype=np.float32)},
            mean=np.zeros(feature_dim, dtype=np.float32),
            std=np.ones(feature_dim, dtype=np.float32),
            feature_indices=feature_indices,
            candidate_name=name,
        )

    queries = TestQueryArray(
        src=np.asarray([1, 3, 5], dtype=np.int32),
        time=np.asarray([100, 101, 102], dtype=np.int32),
        candidates=np.asarray(
            [[10, 20], [30, 40], [50, 60]],
            dtype=np.int32,
        ),
    )
    raw_features = _Encoder.features_for_query_array(queries)
    setwise_features = setwise_context_features(raw_features)
    raw_indices = (0, 2)
    setwise_indices = (0, 4, 8)

    ranker = TemporalHybridRanker()
    ranker.encoder = _Encoder()
    ranker.fusion_result = _result(
        feature_indices=raw_indices,
        name="raw-head",
    )
    ranker.setwise_fusion_result = _result(
        feature_indices=setwise_indices,
        name="setwise-head",
    )
    raw_state = ranker.fusion_result.state
    setwise_state = ranker.setwise_fusion_result.state

    report = ranker.recalibrate_service_normalizers(
        queries,
        batch_size=1,
    )

    assert set(report) == {"fusion", "setwise"}
    assert report["fusion"]["count"] == 6
    assert report["setwise"]["count"] == 6
    assert ranker.fusion_result.state is raw_state
    assert ranker.setwise_fusion_result.state is setwise_state
    assert ranker.fusion_result.feature_indices == raw_indices
    assert ranker.setwise_fusion_result.feature_indices == setwise_indices
    np.testing.assert_allclose(
        ranker.fusion_result.mean,
        raw_features[..., raw_indices].reshape((-1, 2)).mean(axis=0),
    )
    np.testing.assert_allclose(
        ranker.fusion_result.std,
        raw_features[..., raw_indices].reshape((-1, 2)).std(axis=0),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        ranker.setwise_fusion_result.mean,
        setwise_features[..., setwise_indices].reshape((-1, 3)).mean(axis=0),
    )
    np.testing.assert_allclose(
        ranker.setwise_fusion_result.std,
        setwise_features[..., setwise_indices].reshape((-1, 3)).std(axis=0),
        rtol=1e-6,
    )


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Hybrid ranker imports Jittor; integration is verified on Linux.",
)
def test_ranker_recalibrates_contextual_base_head_in_serving_input_space() -> None:
    from jgrec.core.types import TestQueryArray  # noqa: PLC0415
    from jgrec.rankers.hybrid.fusion import FusionResult  # noqa: PLC0415
    from jgrec.rankers.hybrid.ranker import TemporalHybridRanker  # noqa: PLC0415
    from jgrec.rankers.hybrid.setwise import (  # noqa: PLC0415
        setwise_context_features,
    )

    class _Encoder:
        @staticmethod
        def features_for_query_array(queries: TestQueryArray) -> np.ndarray:
            sources = np.broadcast_to(
                queries.src[:, None],
                queries.candidates.shape,
            )
            return np.stack(
                (sources, queries.candidates),
                axis=-1,
            ).astype(np.float32)

        @staticmethod
        def clear_batch_caches() -> None:
            return

    queries = TestQueryArray(
        src=np.asarray([1, 3], dtype=np.int32),
        time=np.asarray([100, 101], dtype=np.int32),
        candidates=np.asarray([[10, 20], [30, 40]], dtype=np.int32),
    )
    raw_features = _Encoder.features_for_query_array(queries)
    contextual = setwise_context_features(raw_features)
    ranker = TemporalHybridRanker()
    ranker.encoder = _Encoder()
    ranker.fusion_result = FusionResult(
        best_val_ap=0.1,
        best_val_mrr=0.2,
        state={"weight": np.ones((6, 1), dtype=np.float32)},
        mean=np.zeros(6, dtype=np.float32),
        std=np.ones(6, dtype=np.float32),
        feature_indices=(0, 1),
        candidate_name="contextual-base",
    )

    report = ranker.recalibrate_service_normalizers(
        queries,
        batch_size=1,
    )

    expected = contextual.reshape((-1, contextual.shape[-1]))
    assert report["fusion"]["feature_dim"] == 6
    np.testing.assert_allclose(
        ranker.fusion_result.mean,
        expected.mean(axis=0),
    )
    expected_std = expected.std(axis=0)
    expected_std[expected_std < 1e-6] = 1.0
    np.testing.assert_allclose(
        ranker.fusion_result.std,
        expected_std,
        rtol=1e-6,
    )


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Hybrid ranker imports Jittor; integration is verified on Linux.",
)
def test_ranker_service_recalibration_rejects_empty_queries() -> None:
    from jgrec.core.types import TestQueryArray  # noqa: PLC0415
    from jgrec.rankers.hybrid.ranker import TemporalHybridRanker  # noqa: PLC0415

    ranker = TemporalHybridRanker()

    with pytest.raises(ValueError, match="non-empty"):
        ranker.recalibrate_service_normalizers(
            TestQueryArray.from_queries([]),
        )


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Hybrid ranker imports Jittor; integration is verified on Linux.",
)
def test_fit_recalibrates_after_final_encoder_refit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jgrec.core.types import (  # noqa: PLC0415
        InteractionTable,
        TestQueryArray,
        TrainingReport,
    )
    from jgrec.rankers.hybrid.config import TrainingConfig  # noqa: PLC0415
    from jgrec.rankers.hybrid.fusion import FusionResult  # noqa: PLC0415
    from jgrec.rankers.hybrid.ranker import TemporalHybridRanker  # noqa: PLC0415

    class _FinalEncoder:
        @staticmethod
        def features_for_query_array(queries: TestQueryArray) -> np.ndarray:
            sources = np.broadcast_to(
                queries.src[:, None],
                queries.candidates.shape,
            )
            return np.stack(
                (sources, queries.candidates),
                axis=-1,
            ).astype(np.float32)

        @staticmethod
        def clear_batch_caches() -> None:
            return

    test_path = tmp_path / "test.csv"
    header = ["src", "time", *(f"c{idx}" for idx in range(1, 101))]
    rows = [
        [1, 100, *range(1, 101)],
        [3, 101, *range(101, 201)],
    ]
    test_path.write_text(
        "\n".join(",".join(str(value) for value in row) for row in [header, *rows]),
        encoding="utf-8",
    )
    interactions = InteractionTable.from_array(
        np.asarray(
            [[1, 10, 1], [3, 20, 2]],
            dtype=np.int32,
        )
    )
    state = {"weight": np.ones((2, 1), dtype=np.float32)}
    fusion_result = FusionResult(
        best_val_ap=0.1,
        best_val_mrr=0.2,
        state=state,
        mean=np.zeros(2, dtype=np.float32),
        std=np.ones(2, dtype=np.float32),
        feature_indices=(0, 1),
        candidate_name="pre-refit-head",
    )
    config = TrainingConfig(
        auto_strategy_enabled=False,
        candidate_prior_enabled=False,
        dataset_test_path=test_path,
        service_normalizer_calibration_enabled=True,
        service_normalizer_calibration_batch_size=1,
        verbose=False,
    )
    ranker = TemporalHybridRanker()
    monkeypatch.setattr(
        ranker,
        "_learn_fusion",
        lambda _interactions, selected_config: (
            object(),
            fusion_result,
            None,
            TrainingReport(model_name="hybrid"),
            None,
            selected_config,
        ),
    )
    monkeypatch.setattr(
        ranker,
        "_final_encoder_cache",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        ranker,
        "_fit_encoder",
        lambda *_args, **_kwargs: _FinalEncoder(),
    )

    report = ranker.fit(interactions, config)

    assert ranker.fusion_result.state is state
    np.testing.assert_allclose(
        ranker.fusion_result.mean,
        np.asarray([2.0, 100.5], dtype=np.float32),
    )
    assert report.metrics["service_normalizer_head_count"] == 1.0
    assert report.metrics["service_normalizer_candidate_rows"] == 200.0
    assert report.metrics["service_normalizer_max_abs_mean_shift_in_training_std"] > 0.0


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Hybrid ranker imports Jittor; integration is verified on Linux.",
)
def test_hybrid_no_refit_full_reuses_validation_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jgrec.core.types import (  # noqa: PLC0415
        InteractionTable,
        TrainingReport,
    )
    from jgrec.rankers.hybrid.config import TrainingConfig  # noqa: PLC0415
    from jgrec.rankers.hybrid.fusion import FusionResult  # noqa: PLC0415
    from jgrec.rankers.hybrid.ranker import TemporalHybridRanker  # noqa: PLC0415

    validation_encoder = object()
    result = FusionResult(
        best_val_ap=0.1,
        best_val_mrr=0.2,
        state={"weight": np.ones((1, 1), dtype=np.float32)},
        mean=np.zeros(1, dtype=np.float32),
        std=np.ones(1, dtype=np.float32),
        feature_indices=(0,),
        candidate_name="validation-head",
    )
    interactions = InteractionTable.from_array(
        np.asarray([[1, 10, 1], [2, 20, 2]], dtype=np.int32)
    )
    ranker = TemporalHybridRanker()

    def learn_fusion(_interactions, selected_config):
        ranker.encoder = validation_encoder
        return (
            object(),
            result,
            None,
            TrainingReport(model_name="hybrid"),
            None,
            selected_config,
        )

    monkeypatch.setattr(ranker, "_learn_fusion", learn_fusion)
    monkeypatch.setattr(
        ranker,
        "_fit_encoder",
        lambda *_args, **_kwargs: pytest.fail(
            "--no-refit-full unexpectedly refitted an encoder"
        ),
    )
    monkeypatch.setattr(
        ranker,
        "_final_encoder_cache",
        lambda **_kwargs: pytest.fail(
            "--no-refit-full unexpectedly built a final encoder cache"
        ),
    )

    report = ranker.fit(
        interactions,
        TrainingConfig(
            auto_strategy_enabled=False,
            refit_full=False,
            verbose=False,
        ),
    )

    assert ranker.encoder is validation_encoder
    assert report.metrics["refit_full"] == 0.0


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Hybrid ranker imports Jittor; integration is verified on Linux.",
)
def test_hybrid_no_refit_full_restores_only_train_end_after_feature_cache_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jgrec.core.types import (  # noqa: PLC0415
        InteractionTable,
        TrainingReport,
    )
    from jgrec.rankers.hybrid.config import TrainingConfig  # noqa: PLC0415
    from jgrec.rankers.hybrid.fusion import FusionResult  # noqa: PLC0415
    from jgrec.rankers.hybrid.ranker import TemporalHybridRanker  # noqa: PLC0415

    interactions = InteractionTable.from_array(
        np.asarray(
            [[index, index + 1000, index] for index in range(100)],
            dtype=np.int32,
        )
    )
    result = FusionResult(
        best_val_ap=0.1,
        best_val_mrr=0.2,
        state={"weight": np.ones((1, 1), dtype=np.float32)},
        mean=np.zeros(1, dtype=np.float32),
        std=np.ones(1, dtype=np.float32),
        feature_indices=(0,),
        candidate_name="cached-head",
    )
    ranker = TemporalHybridRanker()
    fitted_event_counts: list[int] = []
    restored_encoder = object()

    monkeypatch.setattr(
        ranker,
        "_learn_fusion",
        lambda _interactions, selected_config: (
            object(),
            result,
            None,
            TrainingReport(model_name="hybrid"),
            None,
            selected_config,
        ),
    )

    def fit_encoder(prefix, *_args, **_kwargs):
        fitted_event_counts.append(len(prefix))
        return restored_encoder

    monkeypatch.setattr(ranker, "_fit_encoder", fit_encoder)
    monkeypatch.setattr(
        ranker,
        "_final_encoder_cache",
        lambda **_kwargs: pytest.fail(
            "--no-refit-full unexpectedly built a final encoder cache"
        ),
    )

    ranker.fit(
        interactions,
        TrainingConfig(
            val_ratio=0.2,
            auto_strategy_enabled=False,
            refit_full=False,
            verbose=False,
        ),
    )

    assert fitted_event_counts == [80]
    assert ranker.encoder is restored_encoder
