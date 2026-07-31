from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_dataset2_champion_errors.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_dataset2_champion_errors_under_test",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def test_ranking_summary_uses_stable_candidate_order_for_ties() -> None:
    probabilities = np.asarray(
        [
            [0.9, 0.8, 0.7],
            [0.4, 0.8, 0.3],
            [0.2, 0.8, 0.4],
            [0.5, 0.5, 0.4],
        ],
        dtype=np.float64,
    )

    ranks, reciprocal_ranks = analysis.ranking_arrays(probabilities)
    summary = analysis.ranking_summary(ranks, reciprocal_ranks)

    assert ranks.tolist() == [1, 2, 3, 1]
    assert summary["query_count"] == 4
    assert summary["mrr"] == pytest.approx((1.0 + 0.5 + 1.0 / 3.0 + 1.0) / 4.0)
    assert summary["hit_at_1"] == 0.5
    assert summary["positive_rank_p50"] == 2
    assert summary["positive_rank_p90"] == 3


def test_causal_history_excludes_events_at_the_query_timestamp() -> None:
    history = analysis.causal_history_descriptors(
        train_src=np.asarray([1, 1, 2, 1]),
        train_dst=np.asarray([10, 11, 10, 10]),
        train_time=np.asarray([1, 3, 3, 5]),
        query_src=np.asarray([1, 1, 1, 1]),
        query_dst=np.asarray([10, 11, 11, 10]),
        query_time=np.asarray([3, 3, 4, 6]),
    )

    assert history["repeat_edge"].tolist() == [True, False, True, True]
    assert history["source_activity"].tolist() == [1, 1, 2, 3]
    assert history["positive_dst_popularity"].tolist() == [1, 0, 1, 3]


def test_quantile_segments_keep_equal_values_together_and_cover_all_rows() -> None:
    values = np.asarray([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    result = analysis.quantile_segments(values)
    masks = [segment["mask"] for segment in result["segments"]]
    assignments = np.stack(masks, axis=1).sum(axis=1)

    assert len(masks) == 4
    assert np.all(assignments == 1)
    assert next(index for index, mask in enumerate(masks) if mask[0]) == next(
        index for index, mask in enumerate(masks) if mask[1]
    )
    assert result["thresholds"]["p25"] <= result["thresholds"]["p50"]
    assert result["thresholds"]["p50"] <= result["thresholds"]["p75"]


def test_positive_feature_percentiles_and_spearman_detect_signal_and_inactivity() -> None:
    features = np.zeros((5, 5, 2), dtype=np.float32)
    for row in range(5):
        features[row, :, 0] = np.arange(5, dtype=np.float32)
        features[row, 0, 0], features[row, row, 0] = (
            features[row, row, 0],
            features[row, 0, 0],
        )
        features[row, :, 1] = 7.0

    descriptors = analysis.positive_feature_percentiles(features, chunk_rows=2)
    reciprocal_ranks = np.asarray([0.2, 0.4, 0.6, 0.8, 1.0])

    assert descriptors[:, 0].tolist() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert descriptors[:, 1].tolist() == pytest.approx([0.5] * 5)
    assert analysis.spearman_rank_correlation(descriptors[:, 0], reciprocal_ranks, min_rows=3) == pytest.approx(1.0)
    assert analysis.spearman_rank_correlation(descriptors[:, 1], reciprocal_ranks, min_rows=3) is None


def test_priority_segments_require_below_overall_mrr_and_use_volume_times_gap() -> None:
    records = [
        {
            "dimension": "a",
            "segment": "large_low",
            "query_count": 60,
            "query_share": 0.6,
            "mrr": 0.4,
        },
        {
            "dimension": "a",
            "segment": "small_low",
            "query_count": 20,
            "query_share": 0.2,
            "mrr": 0.2,
        },
        {
            "dimension": "b",
            "segment": "large_high",
            "query_count": 80,
            "query_share": 0.8,
            "mrr": 0.7,
        },
    ]

    selected = analysis.select_priority_segments(
        records,
        overall_mrr=0.6,
        minimum_rows=10,
        limit=2,
    )

    assert [row["segment"] for row in selected] == ["large_low", "small_low"]
    assert selected[0]["priority_score"] == pytest.approx(0.6 * (0.6 - 0.4))
    assert selected[1]["priority_score"] == pytest.approx(0.2 * (0.6 - 0.2))


def test_correlation_diagnostic_keeps_query_level_raw_feature_signal() -> None:
    row_count = 5
    feature_count = len(analysis.FEATURE_NAMES)
    positive_values = np.zeros((row_count, feature_count), dtype=np.float32)
    positive_values[:, 0] = np.arange(row_count, dtype=np.float32)
    candidate_relative = np.full(
        (row_count, feature_count),
        0.5,
        dtype=np.float32,
    )
    reciprocal_ranks = np.arange(1, row_count + 1, dtype=np.float64)

    diagnostic = analysis._correlation_diagnostic(
        priority={
            "dimension": "synthetic",
            "segment": "all",
            "query_count": row_count,
            "query_share": 1.0,
            "mrr": float(reciprocal_ranks.mean()),
            "_mask": np.ones(row_count, dtype=bool),
        },
        positive_feature_values=positive_values,
        feature_descriptors=candidate_relative,
        reciprocal_ranks=reciprocal_ranks,
        minimum_rows=3,
    )
    first_feature = next(row for row in diagnostic["feature_correlations"] if row["feature_index"] == 0)

    assert first_feature["positive_value_spearman_rho"] == pytest.approx(1.0)
    assert first_feature["candidate_relative_spearman_rho"] is None


def test_analyze_arrays_emits_hypothesis_only_segment_and_family_report() -> None:
    row_count = 12
    candidate_count = 4
    feature_count = len(analysis.FEATURE_NAMES)
    val_src = np.asarray([1, 2] * (row_count // 2), dtype=np.int64)
    val_dst = np.arange(100, 100 + row_count, dtype=np.int64)
    val_time = np.arange(10, 10 + row_count, dtype=np.int64)
    candidates = np.column_stack(
        (
            val_dst,
            np.arange(200, 200 + row_count),
            np.arange(300, 300 + row_count),
            np.arange(400, 400 + row_count),
        )
    )
    probabilities = np.zeros((row_count, candidate_count), dtype=np.float64)
    for row in range(row_count):
        positive_rank = 1 + row % candidate_count
        probabilities[row, 0] = candidate_count - positive_rank + 0.5
        probabilities[row, 1:] = np.arange(candidate_count - 1, 0, -1, dtype=np.float64)

    rng = np.random.default_rng(7)
    features = rng.normal(size=(row_count, candidate_count, feature_count)).astype(np.float32)
    candidate_test_frequency_index = analysis.FEATURE_NAMES.index("candidate_test_freq")
    features[:, 0, candidate_test_frequency_index] = np.arange(row_count)
    train_src = np.asarray([1, 2, 1, 2, 1, 2], dtype=np.int64)
    train_dst = np.asarray([100, 101, 102, 103, 104, 105], dtype=np.int64)
    train_time = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int64)

    report = analysis.analyze_arrays(
        probabilities=probabilities,
        candidates=candidates,
        features=features,
        val_src=val_src,
        val_dst=val_dst,
        val_time=val_time,
        train_src=train_src,
        train_dst=train_dst,
        train_time=train_time,
        diagnostic_segment_limit=2,
        minimum_segment_rows=2,
        correlation_minimum_rows=2,
        chunk_rows=3,
    )

    assert report["status"] == "hypothesis_generation_only"
    assert "weight selection" in report["discipline"]["prohibited_uses"]
    assert report["overall"]["query_count"] == row_count
    assert set(report["segment_dimensions"]) == {
        "repeat_edge",
        "source_activity_quantile",
        "positive_dst_popularity_quantile",
        "candidate_test_freq_quantile",
        "time_slice",
    }
    for diagnostic in report["high_volume_low_mrr_diagnostics"]:
        assert len(diagnostic["feature_correlations"]) == feature_count
        assert len(diagnostic["family_correlations"]) == len(analysis.FEATURE_FAMILIES)
        assert diagnostic["weakest_signal_family"] in analysis.FEATURE_FAMILIES


def test_analyze_arrays_rejects_misaligned_positive_candidates() -> None:
    feature_count = len(analysis.FEATURE_NAMES)

    with pytest.raises(ValueError, match="candidate column zero"):
        analysis.analyze_arrays(
            probabilities=np.ones((2, 2), dtype=np.float64),
            candidates=np.asarray([[9, 1], [8, 2]], dtype=np.int64),
            features=np.zeros((2, 2, feature_count), dtype=np.float32),
            val_src=np.asarray([1, 2]),
            val_dst=np.asarray([1, 2]),
            val_time=np.asarray([10, 11]),
            train_src=np.asarray([1]),
            train_dst=np.asarray([1]),
            train_time=np.asarray([1]),
            diagnostic_segment_limit=1,
            minimum_segment_rows=1,
            correlation_minimum_rows=2,
            chunk_rows=1,
        )
