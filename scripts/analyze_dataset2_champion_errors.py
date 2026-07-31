"""Read-only, hypothesis-only error profiling for the fixed Dataset2 champion.

This script never trains a model, scans a weight, or mutates an input asset. Its
output is descriptive evidence for hypotheses that must be tested on a fresh
fold before any model or weight selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.rankers.hybrid.candidate_prior import (
    CANDIDATE_PRIOR_FEATURE_NAMES,
)
from jgrec.rankers.hybrid.config import (
    GRAPH_WINDOW_NAMES,
    SEQUENCE_FEATURE_NAMES,
    SOURCE_PROFILE_FEATURE_NAMES,
    TARGET_WINDOW_FEATURE_NAMES,
    TWO_TOWER_FEATURE_NAMES,
)
from jgrec.rankers.hybrid.stats import STAT_FEATURE_NAMES
from jgrec.rankers.hybrid.structure import STRUCTURE_FEATURE_NAMES

DEFAULT_PROBABILITIES = Path("result/dataset2_partial_listwise_expert_blend_20260728/champion-probabilities.npy")
DEFAULT_CACHE_PREFIX = Path("cache/supervised_features/dataset2_joint_recent200k_full100_val_seed60_20260725")
DEFAULT_TRAIN_CSV = Path("data/dataset2/train.csv")
EXPECTED_QUERY_SHAPE = (20_000, 100)
EXPECTED_CHAMPION_MRR = 0.5485470648527594
EXPECTED_ASSET_SHA256 = {
    "probabilities": ("0a39d5f4d2ba8eedb2966a91075afce0a9be469cea89913e6f2eb71078a85983"),
    "candidates": ("dec159209d9c6913825591b585afa0689b7b7323912543204ca6190dad4e4a95"),
    "features": ("7c2cfb763a2803fa7b7bd754dc7f44fb40bedfa15c0015f2c1ca9bcd717ecbcf"),
    "src": "1de31b37ad2eeaa4091fdbcbd8a59aec1ad43f03ad5b875ac75b41fa8bf18b83",
    "dst": "fe7134b5b63da4afd36cc7a906a035c3ce089030c420dc1afe5203699b5d8eac",
    "time": "b08f07610f59905ec2d1d1366c4c8c188a80a2e1621e440361d1b70ebb9c18d7",
    "train_csv": ("da50210bbd0581de89971d907eb7ca590a4fc0364b41ee50730ae2a13d8de4a0"),
}

FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "stats": STAT_FEATURE_NAMES,
    "candidate_prior": CANDIDATE_PRIOR_FEATURE_NAMES,
    "target_window": TARGET_WINDOW_FEATURE_NAMES,
    "structure": STRUCTURE_FEATURE_NAMES,
    "source_profile": SOURCE_PROFILE_FEATURE_NAMES,
    "two_tower": TWO_TOWER_FEATURE_NAMES,
    "gnn": GRAPH_WINDOW_NAMES,
    "sequence": SEQUENCE_FEATURE_NAMES,
}
FEATURE_NAMES = tuple(feature for family_features in FEATURE_FAMILIES.values() for feature in family_features)
if len(FEATURE_NAMES) != 66 or len(set(FEATURE_NAMES)) != 66:
    raise RuntimeError("the frozen Dataset2 feature schema must contain 66 unique columns")


def ranking_arrays(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return positive ranks and reciprocal ranks with stable candidate-order ties."""
    values = np.asarray(probabilities)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("probabilities must have shape [queries, candidates>=2]")
    if not np.all(np.isfinite(values)):
        raise ValueError("probabilities contain non-finite values")
    ranks = 1 + np.count_nonzero(values[:, 1:] > values[:, 0:1], axis=1)
    ranks = ranks.astype(np.int32, copy=False)
    return ranks, 1.0 / ranks.astype(np.float64)


def ranking_summary(
    ranks: np.ndarray,
    reciprocal_ranks: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, int | float | None]:
    rank_values = np.asarray(ranks)
    rr_values = np.asarray(reciprocal_ranks, dtype=np.float64)
    if rank_values.ndim != 1 or rr_values.ndim != 1 or rank_values.shape != rr_values.shape:
        raise ValueError("ranks and reciprocal ranks must be aligned one-dimensional arrays")
    if mask is not None:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != rank_values.shape:
            raise ValueError("ranking mask does not align with ranks")
        rank_values = rank_values[selected]
        rr_values = rr_values[selected]
    rows = int(rank_values.size)
    if rows == 0:
        return {
            "query_count": 0,
            "mrr": None,
            "hit_at_1": None,
            "positive_rank_p50": None,
            "positive_rank_p90": None,
        }
    p50, p90 = np.quantile(rank_values, (0.50, 0.90), method="higher")
    return {
        "query_count": rows,
        "mrr": float(rr_values.mean()),
        "hit_at_1": float(np.mean(rank_values == 1)),
        "positive_rank_p50": int(p50),
        "positive_rank_p90": int(p90),
    }


def causal_history_descriptors(
    *,
    train_src: np.ndarray,
    train_dst: np.ndarray,
    train_time: np.ndarray,
    query_src: np.ndarray,
    query_dst: np.ndarray,
    query_time: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build rolling history descriptors from train rows with time < query time."""
    train_src_values = np.asarray(train_src, dtype=np.int64)
    train_dst_values = np.asarray(train_dst, dtype=np.int64)
    train_time_values = np.asarray(train_time, dtype=np.int64)
    query_src_values = np.asarray(query_src, dtype=np.int64)
    query_dst_values = np.asarray(query_dst, dtype=np.int64)
    query_time_values = np.asarray(query_time, dtype=np.int64)
    if (
        train_src_values.ndim != 1
        or train_dst_values.shape != train_src_values.shape
        or train_time_values.shape != train_src_values.shape
    ):
        raise ValueError("train src/dst/time arrays must be aligned and one-dimensional")
    if (
        query_src_values.ndim != 1
        or query_dst_values.shape != query_src_values.shape
        or query_time_values.shape != query_src_values.shape
    ):
        raise ValueError("query src/dst/time arrays must be aligned and one-dimensional")

    row_count = int(query_src_values.size)
    repeat_edge = np.zeros(row_count, dtype=bool)
    source_activity = np.zeros(row_count, dtype=np.int64)
    dst_popularity = np.zeros(row_count, dtype=np.int64)
    if row_count == 0:
        return {
            "repeat_edge": repeat_edge,
            "source_activity": source_activity,
            "positive_dst_popularity": dst_popularity,
        }

    relevant_sources = {int(value) for value in query_src_values}
    relevant_destinations = {int(value) for value in query_dst_values}
    relevant_pairs = {(int(src), int(dst)) for src, dst in zip(query_src_values, query_dst_values, strict=True)}
    source_counts: dict[int, int] = {}
    destination_counts: dict[int, int] = {}
    seen_pairs: set[tuple[int, int]] = set()

    train_order = np.argsort(train_time_values, kind="stable")
    query_order = np.argsort(query_time_values, kind="stable")
    train_position = 0
    for raw_query_index in query_order:
        query_index = int(raw_query_index)
        cutoff = int(query_time_values[query_index])
        while train_position < train_order.size and int(train_time_values[int(train_order[train_position])]) < cutoff:
            train_index = int(train_order[train_position])
            src = int(train_src_values[train_index])
            dst = int(train_dst_values[train_index])
            if src in relevant_sources:
                source_counts[src] = source_counts.get(src, 0) + 1
            if dst in relevant_destinations:
                destination_counts[dst] = destination_counts.get(dst, 0) + 1
            pair = (src, dst)
            if pair in relevant_pairs:
                seen_pairs.add(pair)
            train_position += 1

        src = int(query_src_values[query_index])
        dst = int(query_dst_values[query_index])
        repeat_edge[query_index] = (src, dst) in seen_pairs
        source_activity[query_index] = source_counts.get(src, 0)
        dst_popularity[query_index] = destination_counts.get(dst, 0)

    return {
        "repeat_edge": repeat_edge,
        "source_activity": source_activity,
        "positive_dst_popularity": dst_popularity,
    }


def quantile_segments(values: np.ndarray) -> dict[str, Any]:
    """Return tie-preserving quartile masks based on P25/P50/P75 thresholds."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("quantile values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError("quantile values contain non-finite values")
    p25, p50, p75 = np.quantile(array, (0.25, 0.50, 0.75), method="linear")
    masks = (
        array <= p25,
        (array > p25) & (array <= p50),
        (array > p50) & (array <= p75),
        array > p75,
    )
    intervals = (
        f"value <= {p25:.12g}",
        f"{p25:.12g} < value <= {p50:.12g}",
        f"{p50:.12g} < value <= {p75:.12g}",
        f"value > {p75:.12g}",
    )
    return {
        "thresholds": {
            "p25": float(p25),
            "p50": float(p50),
            "p75": float(p75),
        },
        "segments": [
            {"name": f"q{index + 1}", "interval": intervals[index], "mask": mask} for index, mask in enumerate(masks)
        ],
    }


def positive_feature_percentiles(
    features: np.ndarray,
    *,
    chunk_rows: int,
) -> np.ndarray:
    """Reduce [query,candidate,feature] to positive-vs-negative mid-percentiles."""
    values = np.asarray(features)
    if values.ndim != 3 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("features must have shape [queries, candidates>=2, features]")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    row_count, candidate_count, feature_count = values.shape
    output = np.empty((row_count, feature_count), dtype=np.float32)
    negative_count = candidate_count - 1
    for start in range(0, row_count, chunk_rows):
        stop = min(start + chunk_rows, row_count)
        block = np.asarray(values[start:stop])
        if not np.all(np.isfinite(block)):
            raise ValueError("validation features contain non-finite values")
        positive = block[:, 0:1, :]
        negatives = block[:, 1:, :]
        lower = np.count_nonzero(negatives < positive, axis=1)
        equal = np.count_nonzero(negatives == positive, axis=1)
        output[start:stop] = (lower + 0.5 * equal) / negative_count
    return output


def spearman_rank_correlation(
    left: np.ndarray,
    right: np.ndarray,
    *,
    min_rows: int,
) -> float | None:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.ndim != 1 or right_values.shape != left_values.shape:
        raise ValueError("Spearman inputs must be aligned one-dimensional arrays")
    if min_rows < 2:
        raise ValueError("min_rows must be at least two")
    finite = np.isfinite(left_values) & np.isfinite(right_values)
    if int(finite.sum()) < min_rows:
        return None
    left_ranks = _average_ranks(left_values[finite])
    right_ranks = _average_ranks(right_values[finite])
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = float(np.sqrt(np.dot(left_centered, left_centered) * np.dot(right_centered, right_centered)))
    if denominator == 0.0:
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


def select_priority_segments(
    records: list[dict[str, Any]],
    *,
    overall_mrr: float,
    minimum_rows: int,
    limit: int,
) -> list[dict[str, Any]]:
    """Rank eligible segments by query_share * (overall_mrr - segment_mrr)."""
    if minimum_rows < 1:
        raise ValueError("minimum_rows must be positive")
    if limit < 0:
        raise ValueError("limit cannot be negative")
    eligible: list[dict[str, Any]] = []
    for record in records:
        segment_mrr = record.get("mrr")
        if segment_mrr is None or int(record["query_count"]) < minimum_rows or float(segment_mrr) >= overall_mrr:
            continue
        selected = dict(record)
        selected["mrr_gap_to_overall"] = float(overall_mrr - float(segment_mrr))
        selected["priority_score"] = float(float(record["query_share"]) * selected["mrr_gap_to_overall"])
        eligible.append(selected)
    eligible.sort(
        key=lambda row: (
            -float(row["priority_score"]),
            -int(row["query_count"]),
            str(row["dimension"]),
            str(row["segment"]),
        )
    )
    return eligible[:limit]


def analyze_arrays(
    *,
    probabilities: np.ndarray,
    candidates: np.ndarray,
    features: np.ndarray,
    val_src: np.ndarray,
    val_dst: np.ndarray,
    val_time: np.ndarray,
    train_src: np.ndarray,
    train_dst: np.ndarray,
    train_time: np.ndarray,
    diagnostic_segment_limit: int,
    minimum_segment_rows: int,
    correlation_minimum_rows: int,
    chunk_rows: int,
) -> dict[str, Any]:
    probabilities_values = np.asarray(probabilities)
    candidate_values = np.asarray(candidates)
    feature_values = np.asarray(features)
    src_values = np.asarray(val_src)
    dst_values = np.asarray(val_dst)
    time_values = np.asarray(val_time)
    _validate_aligned_assets(
        probabilities=probabilities_values,
        candidates=candidate_values,
        features=feature_values,
        val_src=src_values,
        val_dst=dst_values,
        val_time=time_values,
    )
    if minimum_segment_rows < 1:
        raise ValueError("minimum_segment_rows must be positive")
    if correlation_minimum_rows < 2:
        raise ValueError("correlation_minimum_rows must be at least two")
    if diagnostic_segment_limit < 0:
        raise ValueError("diagnostic_segment_limit cannot be negative")

    ranks, reciprocal_ranks = ranking_arrays(probabilities_values)
    overall = ranking_summary(ranks, reciprocal_ranks)
    overall_mrr = float(overall["mrr"])
    history = causal_history_descriptors(
        train_src=train_src,
        train_dst=train_dst,
        train_time=train_time,
        query_src=src_values,
        query_dst=dst_values,
        query_time=time_values,
    )
    feature_descriptors = positive_feature_percentiles(
        feature_values,
        chunk_rows=chunk_rows,
    )
    positive_feature_values = np.asarray(
        feature_values[:, 0, :],
        dtype=np.float32,
    )
    row_count = int(probabilities_values.shape[0])
    segment_dimensions, internal_records = _build_segment_dimensions(
        ranks=ranks,
        reciprocal_ranks=reciprocal_ranks,
        repeat_edge=history["repeat_edge"],
        source_activity=history["source_activity"],
        positive_dst_popularity=history["positive_dst_popularity"],
        candidate_test_freq=np.asarray(
            feature_values[:, 0, FEATURE_NAMES.index("candidate_test_freq")],
            dtype=np.float64,
        ),
        query_time=time_values,
    )
    priority_segments = select_priority_segments(
        internal_records,
        overall_mrr=overall_mrr,
        minimum_rows=minimum_segment_rows,
        limit=diagnostic_segment_limit,
    )
    diagnostics = [
        _correlation_diagnostic(
            priority=priority,
            positive_feature_values=positive_feature_values,
            feature_descriptors=feature_descriptors,
            reciprocal_ranks=reciprocal_ranks,
            minimum_rows=correlation_minimum_rows,
        )
        for priority in priority_segments
    ]

    cached_repeat = np.asarray(feature_values[:, 0, FEATURE_NAMES.index("pair_strength")]) > 0.0
    repeat_agreement = cached_repeat == history["repeat_edge"]
    positive_scores = probabilities_values[:, 0:1]
    tie_queries = np.any(probabilities_values[:, 1:] == positive_scores, axis=1)
    return {
        "schema_version": 1,
        "status": "hypothesis_generation_only",
        "discipline": {
            "allowed_use": ("descriptive error profiling and generation of hypotheses to pre-register on a fresh fold"),
            "prohibited_uses": [
                "weight selection",
                "model selection",
                "threshold tuning",
                "feature selection",
            ],
            "required_next_step": (
                "Any hypothesis suggested here must be frozen and evaluated on "
                "a newly created fold before it can influence a model decision."
            ),
            "validation_reuse_warning": (
                "This 20k validation has been repeatedly consumed and is not an unbiased selection set."
            ),
        },
        "definitions": {
            "positive_candidate_column": 0,
            "rank": (
                "1 + count(candidate_score > positive_score); equal scores keep "
                "the existing candidate order, so the positive at column 0 wins ties"
            ),
            "rank_percentiles": "integer empirical quantiles using numpy method='higher'",
            "history": (
                "rolling causal history from train.csv rows with event time "
                "strictly less than each query time; events at equal time are excluded"
            ),
            "quartiles": (
                "P25/P50/P75 threshold bins; equal values are never split, so "
                "bin sizes may differ and bins may be empty"
            ),
            "candidate_test_freq": ("cached candidate_test_freq value for the positive candidate at column 0"),
            "feature_correlation_descriptor": (
                "Primary: cached raw feature value at positive candidate column 0. "
                "Secondary candidate-relative view: fraction of 99 negatives below "
                "the positive plus half the tied-negative fraction. Both use "
                "Spearman correlation against champion reciprocal rank in-segment."
            ),
            "weakest_signal_family": (
                "family with the smallest median absolute raw-positive feature "
                "Spearman, treating constant/insufficient feature correlations as "
                "zero; this is association evidence, not causal ablation evidence"
            ),
        },
        "overall": {
            **overall,
            "positive_score_tie_query_count": int(tie_queries.sum()),
            "positive_score_tie_query_share": float(tie_queries.mean()),
        },
        "history_cache_alignment": {
            "causal_repeat_query_count": int(history["repeat_edge"].sum()),
            "cached_pair_strength_repeat_query_count": int(cached_repeat.sum()),
            "agreement_query_count": int(repeat_agreement.sum()),
            "agreement_share": float(repeat_agreement.mean()),
            "note": (
                "A disagreement is possible because rolling causal history includes "
                "earlier validation-time train rows while the frozen cache may use a "
                "fixed fit prefix."
            ),
        },
        "feature_schema": {
            "feature_count": len(FEATURE_NAMES),
            "feature_names": list(FEATURE_NAMES),
            "families": {family: list(names) for family, names in FEATURE_FAMILIES.items()},
        },
        "segment_dimensions": segment_dimensions,
        "high_volume_low_mrr_rule": {
            "eligibility": (f"query_count >= {minimum_segment_rows} and segment MRR < overall MRR"),
            "priority_score": "query_share * (overall_mrr - segment_mrr)",
            "maximum_segments": diagnostic_segment_limit,
            "correlation_minimum_rows": correlation_minimum_rows,
        },
        "high_volume_low_mrr_diagnostics": diagnostics,
        "query_count": row_count,
    }


def _build_segment_dimensions(
    *,
    ranks: np.ndarray,
    reciprocal_ranks: np.ndarray,
    repeat_edge: np.ndarray,
    source_activity: np.ndarray,
    positive_dst_popularity: np.ndarray,
    candidate_test_freq: np.ndarray,
    query_time: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row_count = int(ranks.size)
    dimensions: dict[str, Any] = {}
    records: list[dict[str, Any]] = []

    repeat_segments = [
        {"name": "new_edge", "mask": ~repeat_edge},
        {"name": "repeat_edge", "mask": repeat_edge},
    ]
    _add_dimension(
        dimensions,
        records,
        name="repeat_edge",
        definition="Whether the positive (src,dst) occurs in rolling causal history.",
        segments=repeat_segments,
        ranks=ranks,
        reciprocal_ranks=reciprocal_ranks,
        row_count=row_count,
    )

    for name, definition, values in (
        (
            "source_activity_quantile",
            "Quartile of causal source event count before the query.",
            source_activity,
        ),
        (
            "positive_dst_popularity_quantile",
            "Quartile of causal positive-destination event count before the query.",
            positive_dst_popularity,
        ),
        (
            "candidate_test_freq_quantile",
            "Quartile of cached positive-candidate candidate_test_freq.",
            candidate_test_freq,
        ),
    ):
        quartiles = quantile_segments(values)
        _add_dimension(
            dimensions,
            records,
            name=name,
            definition=definition,
            segments=quartiles["segments"],
            ranks=ranks,
            reciprocal_ranks=reciprocal_ranks,
            row_count=row_count,
            metadata={"thresholds": quartiles["thresholds"]},
        )

    time_segments: list[dict[str, Any]] = []
    for slice_index, indices in enumerate(np.array_split(np.arange(row_count), 3)):
        mask = np.zeros(row_count, dtype=bool)
        mask[indices] = True
        start = int(indices[0]) if indices.size else 0
        stop = int(indices[-1]) + 1 if indices.size else start
        time_segments.append(
            {
                "name": f"slice_{slice_index}",
                "mask": mask,
                "start_row": start,
                "stop_row": stop,
                "start_time": int(query_time[start]) if indices.size else None,
                "end_time": int(query_time[stop - 1]) if indices.size else None,
            }
        )
    _add_dimension(
        dimensions,
        records,
        name="time_slice",
        definition="Three contiguous equal-count slices in chronological cache row order.",
        segments=time_segments,
        ranks=ranks,
        reciprocal_ranks=reciprocal_ranks,
        row_count=row_count,
    )
    return dimensions, records


def _add_dimension(
    dimensions: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    name: str,
    definition: str,
    segments: list[dict[str, Any]],
    ranks: np.ndarray,
    reciprocal_ranks: np.ndarray,
    row_count: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    output_segments = []
    for segment in segments:
        mask = np.asarray(segment["mask"], dtype=bool)
        metrics = ranking_summary(ranks, reciprocal_ranks, mask)
        record = {
            "dimension": name,
            "segment": str(segment["name"]),
            **metrics,
            "query_share": float(metrics["query_count"] / row_count),
            "_mask": mask,
        }
        record.update({key: value for key, value in segment.items() if key not in {"name", "mask"}})
        records.append(record)
        output_segments.append({key: value for key, value in record.items() if key != "_mask"})
    dimensions[name] = {
        "definition": definition,
        **(metadata or {}),
        "segments": output_segments,
    }


def _correlation_diagnostic(
    *,
    priority: dict[str, Any],
    positive_feature_values: np.ndarray,
    feature_descriptors: np.ndarray,
    reciprocal_ranks: np.ndarray,
    minimum_rows: int,
) -> dict[str, Any]:
    mask = np.asarray(priority["_mask"], dtype=bool)
    rr = reciprocal_ranks[mask]
    positive_values = positive_feature_values[mask]
    descriptors = feature_descriptors[mask]
    feature_rows = []
    for feature_index, feature_name in enumerate(FEATURE_NAMES):
        raw_values = positive_values[:, feature_index]
        relative_values = descriptors[:, feature_index]
        positive_value_rho = spearman_rank_correlation(
            raw_values,
            rr,
            min_rows=minimum_rows,
        )
        candidate_relative_rho = spearman_rank_correlation(
            relative_values,
            rr,
            min_rows=minimum_rows,
        )
        feature_rows.append(
            {
                "feature": feature_name,
                "feature_index": feature_index,
                "family": _feature_family(feature_name),
                "spearman_rho": positive_value_rho,
                "absolute_spearman_rho": (abs(positive_value_rho) if positive_value_rho is not None else None),
                "positive_value_spearman_rho": positive_value_rho,
                "positive_value_unique_values": int(np.unique(raw_values).size),
                "positive_value_p50": float(np.quantile(raw_values, 0.50, method="linear")),
                "candidate_relative_spearman_rho": candidate_relative_rho,
                "candidate_relative_absolute_spearman_rho": (
                    abs(candidate_relative_rho) if candidate_relative_rho is not None else None
                ),
                "positive_percentile_unique_values": int(np.unique(relative_values).size),
                "positive_percentile_p50": float(np.quantile(relative_values, 0.50, method="linear")),
            }
        )
    family_rows = []
    for family, family_features in FEATURE_FAMILIES.items():
        family_set = set(family_features)
        rows = [row for row in feature_rows if row["feature"] in family_set]
        usable = [row for row in rows if row["spearman_rho"] is not None]
        relative_usable = [row for row in rows if row["candidate_relative_spearman_rho"] is not None]
        all_absolute = [
            float(row["absolute_spearman_rho"]) if row["absolute_spearman_rho"] is not None else 0.0 for row in rows
        ]
        all_relative_absolute = [
            (
                float(row["candidate_relative_absolute_spearman_rho"])
                if row["candidate_relative_absolute_spearman_rho"] is not None
                else 0.0
            )
            for row in rows
        ]
        usable_rhos = [float(row["spearman_rho"]) for row in usable]
        strongest = max(usable, key=lambda row: float(row["absolute_spearman_rho"])) if usable else None
        family_rows.append(
            {
                "family": family,
                "feature_count": len(rows),
                "usable_feature_count": len(usable),
                "inactive_or_insufficient_feature_count": len(rows) - len(usable),
                "median_spearman_rho": (float(np.median(usable_rhos)) if usable_rhos else None),
                "median_absolute_spearman_all_features": float(np.median(all_absolute)),
                "median_absolute_candidate_relative_spearman_all_features": float(np.median(all_relative_absolute)),
                "maximum_absolute_spearman": (max(all_absolute) if all_absolute else 0.0),
                "candidate_relative_usable_feature_count": len(relative_usable),
                "negative_spearman_feature_count": sum(rho < 0.0 for rho in usable_rhos),
                "strongest_feature": (
                    {
                        "feature": strongest["feature"],
                        "spearman_rho": strongest["spearman_rho"],
                    }
                    if strongest is not None
                    else None
                ),
            }
        )
    family_rows.sort(
        key=lambda row: (
            float(row["median_absolute_spearman_all_features"]),
            str(row["family"]),
        )
    )
    feature_rows.sort(
        key=lambda row: (
            -(float(row["absolute_spearman_rho"]) if row["absolute_spearman_rho"] is not None else -1.0),
            str(row["feature"]),
        )
    )
    return {
        **{key: value for key, value in priority.items() if key != "_mask"},
        "correlation_basis": (
            "Primary Spearman(raw positive feature value, champion reciprocal "
            "rank); secondary Spearman(positive-vs-negative mid-percentile, "
            "champion reciprocal rank)"
        ),
        "weakest_signal_family": str(family_rows[0]["family"]),
        "weakest_signal_family_evidence": (
            "smallest median absolute raw-positive feature Spearman, with "
            "undefined constant/insufficient features scored as zero"
        ),
        "family_correlations": family_rows,
        "feature_correlations": feature_rows,
    }


def _feature_family(feature_name: str) -> str:
    for family, names in FEATURE_FAMILIES.items():
        if feature_name in names:
            return family
    raise KeyError(feature_name)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    unique, inverse, counts = np.unique(
        np.asarray(values, dtype=np.float64),
        return_inverse=True,
        return_counts=True,
    )
    del unique
    starts = np.cumsum(np.concatenate(([0], counts[:-1])))
    average = starts + (counts - 1) / 2.0 + 1.0
    return average[inverse]


def _validate_aligned_assets(
    *,
    probabilities: np.ndarray,
    candidates: np.ndarray,
    features: np.ndarray,
    val_src: np.ndarray,
    val_dst: np.ndarray,
    val_time: np.ndarray,
) -> None:
    if probabilities.ndim != 2 or probabilities.shape[0] == 0 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have shape [queries, candidates>=2]")
    if candidates.shape != probabilities.shape:
        raise ValueError("candidate sidecar shape differs from probabilities")
    if features.shape != (*probabilities.shape, len(FEATURE_NAMES)):
        raise ValueError("feature cache must align with probabilities and contain the frozen 66 columns")
    expected_sidecar_shape = (probabilities.shape[0],)
    if any(np.asarray(values).shape != expected_sidecar_shape for values in (val_src, val_dst, val_time)):
        raise ValueError("src/dst/time sidecars do not align with probabilities")
    if not np.array_equal(candidates[:, 0], val_dst):
        raise ValueError("candidate column zero must equal the positive dst sidecar")
    if np.any(np.diff(val_time.astype(np.int64, copy=False)) < 0):
        raise ValueError("validation rows must be chronological for time slices")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities contain non-finite values")
    if np.any(probabilities < 0.0):
        raise ValueError("champion probabilities cannot be negative")
    sorted_candidates = np.sort(candidates, axis=1)
    if np.any(np.diff(sorted_candidates, axis=1) == 0):
        raise ValueError("candidate rows must contain unique identifiers")


def read_train_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), None)
    required = ("src", "dst", "time")
    if header is None or any(name not in header for name in required):
        raise ValueError(f"{path} must contain src,dst,time columns")
    data = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=tuple(header.index(name) for name in required),
        dtype=np.int64,
        ndmin=2,
    )
    if data.shape[0] == 0:
        raise ValueError(f"{path} contains no interactions")
    return data[:, 0], data[:, 1], data[:, 2]


def _cache_path(prefix: Path, sidecar: str) -> Path:
    return Path(f"{prefix}.val-{sidecar}.npy")


def _feature_cache_path(prefix: Path) -> Path:
    return Path(f"{prefix}.val.npy")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only Dataset2 champion error profiling. Output is hypothesis-only "
            "and must not be used for weight or model selection."
        )
    )
    parser.add_argument(
        "--probabilities",
        type=Path,
        default=DEFAULT_PROBABILITIES,
    )
    parser.add_argument(
        "--cache-prefix",
        type=Path,
        default=DEFAULT_CACHE_PREFIX,
        help=("Prefix for .val.npy and .val-candidates/src/dst/time.npy assets."),
    )
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--diagnostic-segments", type=int, default=5)
    parser.add_argument("--minimum-segment-rows", type=int, default=500)
    parser.add_argument("--correlation-minimum-rows", type=int, default=100)
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument(
        "--skip-asset-lock",
        action="store_true",
        help=(
            "Skip the fixed hashes, 20k x 100 shape, and exact champion-MRR lock. "
            "Intended only for synthetic smoke tests, never for the requested report."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = {
        "probabilities": args.probabilities,
        "features": _feature_cache_path(args.cache_prefix),
        "candidates": _cache_path(args.cache_prefix, "candidates"),
        "src": _cache_path(args.cache_prefix, "src"),
        "dst": _cache_path(args.cache_prefix, "dst"),
        "time": _cache_path(args.cache_prefix, "time"),
        "train_csv": args.train_csv,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required read-only assets are missing: {missing}")

    hashes = {name: _sha256(path) for name, path in paths.items()}
    if not args.skip_asset_lock:
        for name, expected in EXPECTED_ASSET_SHA256.items():
            actual = hashes[name]
            if actual != expected:
                raise ValueError(f"{name} SHA-256 mismatch: expected={expected} actual={actual}")

    probabilities = np.load(
        paths["probabilities"],
        mmap_mode="r",
        allow_pickle=False,
    )
    features = np.load(paths["features"], mmap_mode="r", allow_pickle=False)
    candidates = np.load(paths["candidates"], mmap_mode="r", allow_pickle=False)
    val_src = np.load(paths["src"], mmap_mode="r", allow_pickle=False)
    val_dst = np.load(paths["dst"], mmap_mode="r", allow_pickle=False)
    val_time = np.load(paths["time"], mmap_mode="r", allow_pickle=False)
    train_src, train_dst, train_time = read_train_csv(paths["train_csv"])
    if not args.skip_asset_lock and tuple(probabilities.shape) != EXPECTED_QUERY_SHAPE:
        raise ValueError(f"fixed champion shape mismatch: {probabilities.shape} != {EXPECTED_QUERY_SHAPE}")

    report = analyze_arrays(
        probabilities=probabilities,
        candidates=candidates,
        features=features,
        val_src=val_src,
        val_dst=val_dst,
        val_time=val_time,
        train_src=train_src,
        train_dst=train_dst,
        train_time=train_time,
        diagnostic_segment_limit=args.diagnostic_segments,
        minimum_segment_rows=args.minimum_segment_rows,
        correlation_minimum_rows=args.correlation_minimum_rows,
        chunk_rows=args.chunk_rows,
    )
    actual_mrr = float(report["overall"]["mrr"])
    if not args.skip_asset_lock and abs(actual_mrr - EXPECTED_CHAMPION_MRR) > 1e-12:
        raise RuntimeError(
            f"champion metric lock failed: expected={EXPECTED_CHAMPION_MRR:.16f} actual={actual_mrr:.16f}"
        )
    row_sums = np.asarray(probabilities, dtype=np.float64).sum(axis=1)
    report["inputs"] = {
        "fixed_asset_lock_enforced": not args.skip_asset_lock,
        "paths": {name: str(path.resolve()) for name, path in paths.items()},
        "sha256": hashes,
        "shapes": {
            "probabilities": list(probabilities.shape),
            "features": list(features.shape),
            "candidates": list(candidates.shape),
            "src": list(val_src.shape),
            "dst": list(val_dst.shape),
            "time": list(val_time.shape),
            "train": [int(train_src.size), 3],
        },
        "probability_row_sum_max_abs_error": float(np.max(np.abs(row_sums - 1.0))),
        "expected_champion_mrr": (EXPECTED_CHAMPION_MRR if not args.skip_asset_lock else None),
        "champion_mrr_absolute_error": (abs(actual_mrr - EXPECTED_CHAMPION_MRR) if not args.skip_asset_lock else None),
    }
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
