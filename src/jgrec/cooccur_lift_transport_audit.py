"""Zero-label transport audit for the Dataset2 cooccur-lift auxiliary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.core.io import read_interactions, read_test_queries

QUANTILES = (0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
POPULARITY_BUCKET_EDGES = (
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
)


def collapse_summary(
    short_lift: np.ndarray,
    *,
    chunk_rows: int,
) -> dict[str, Any]:
    values = np.asarray(short_lift)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("short lift must have shape [rows, candidates]")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    rows, candidates = values.shape
    zero_cells = 0
    zero_rows = 0
    constant_rows = 0
    zero_fraction = np.empty(rows, dtype=np.float64)
    minimum = math.inf
    maximum = -math.inf
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        block = np.asarray(values[start:stop])
        if not np.all(np.isfinite(block)):
            raise ValueError("short lift contains non-finite values")
        zeros = block == 0.0
        zero_cells += int(np.count_nonzero(zeros))
        zero_rows += int(np.count_nonzero(np.all(zeros, axis=1)))
        constant_rows += int(
            np.count_nonzero(np.ptp(block, axis=1) == 0.0)
        )
        zero_fraction[start:stop] = np.mean(zeros, axis=1)
        minimum = min(minimum, float(block.min()))
        maximum = max(maximum, float(block.max()))
    cell_count = rows * candidates
    return {
        "rows": rows,
        "candidates_per_row": candidates,
        "candidate_cells": cell_count,
        "exact_zero_cells": zero_cells,
        "exact_zero_cell_rate": zero_cells / cell_count,
        "all_exact_zero_rows": zero_rows,
        "all_exact_zero_row_rate": zero_rows / rows,
        "constant_rows": constant_rows,
        "constant_row_rate": constant_rows / rows,
        "row_exact_zero_fraction": _describe(zero_fraction),
        "minimum": minimum,
        "maximum": maximum,
    }


def first_layer_lift_intervention_summary(
    lift_features: np.ndarray,
    *,
    linear1_weight: np.ndarray,
    std: np.ndarray,
    chunk_rows: int,
) -> dict[str, Any]:
    """Measure exact first-layer perturbations from zeroing each lift signal."""
    lift = np.asarray(lift_features)
    weight = np.asarray(linear1_weight, dtype=np.float32)
    scale = np.asarray(std, dtype=np.float32)
    if lift.ndim != 3 or lift.shape[2] != 2 or lift.shape[0] == 0:
        raise ValueError("lift features must have shape [rows,candidates,2]")
    if weight.ndim != 2 or weight.shape[1] != 195:
        raise ValueError("linear1 weight must have shape [hidden,195]")
    if scale.shape != (195,) or np.any(scale <= 0.0):
        raise ValueError("normalizer std must be positive with shape [195]")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")

    columns = {
        "full": np.asarray([63, 128, 193], dtype=np.int64),
        "short": np.asarray([64, 129, 194], dtype=np.int64),
    }
    effective = {
        name: weight[:, indices] / scale[indices][np.newaxis, :]
        for name, indices in columns.items()
    }
    row_energy = {
        name: np.empty(lift.shape[0], dtype=np.float64)
        for name in columns
    }
    row_maximum = {
        name: np.empty(lift.shape[0], dtype=np.float64)
        for name in columns
    }
    total_energy = dict.fromkeys(columns, 0.0)
    cross_dot = 0.0
    for start in range(0, lift.shape[0], chunk_rows):
        stop = min(start + chunk_rows, lift.shape[0])
        block = np.asarray(lift[start:stop], dtype=np.float32)
        contributions: dict[str, np.ndarray] = {}
        for signal_index, name in enumerate(("full", "short")):
            raw = block[:, :, signal_index]
            context = np.stack(
                (
                    raw,
                    raw - raw.mean(
                        axis=1,
                        keepdims=True,
                        dtype=np.float32,
                    ),
                    raw - raw.max(axis=1, keepdims=True),
                ),
                axis=2,
            )
            delta = np.einsum(
                "rcf,hf->rch",
                context,
                effective[name],
                optimize=True,
            )
            squared = delta * delta
            energy = np.sum(
                squared,
                axis=(1, 2),
                dtype=np.float64,
            )
            row_energy[name][start:stop] = energy
            row_maximum[name][start:stop] = np.max(
                np.abs(delta),
                axis=(1, 2),
            )
            total_energy[name] += float(energy.sum())
            contributions[name] = delta
        cross_dot += float(
            np.sum(
                contributions["full"] * contributions["short"],
                dtype=np.float64,
            )
        )

    separate_energy = total_energy["full"] + total_energy["short"]
    output: dict[str, Any] = {}
    denominator = lift.shape[1] * weight.shape[0]
    for name in ("full", "short"):
        rms = np.sqrt(row_energy[name] / denominator)
        output[name] = {
            "preactivation_energy": total_energy[name],
            "exactly_zero_rows": int(
                np.count_nonzero(row_energy[name] == 0.0)
            ),
            "exactly_zero_row_rate": float(
                np.mean(row_energy[name] == 0.0)
            ),
            "row_rms_preactivation_delta": _describe(rms),
            "row_max_abs_preactivation_delta": _describe(
                row_maximum[name]
            ),
            "effective_weight_l2_by_context_channel": [
                float(value)
                for value in np.linalg.norm(
                    effective[name],
                    axis=0,
                )
            ],
        }
    output["full_energy_fraction_of_separate_sum"] = (
        total_energy["full"] / separate_energy
        if separate_energy > 0.0
        else 0.0
    )
    output["short_energy_fraction_of_separate_sum"] = (
        total_energy["short"] / separate_energy
        if separate_energy > 0.0
        else 0.0
    )
    energy_product = math.sqrt(
        total_energy["full"] * total_energy["short"]
    )
    output["full_short_preactivation_cosine"] = (
        cross_dot / energy_product if energy_product > 0.0 else 0.0
    )
    return output


def probability_distribution_summary(
    probabilities: np.ndarray,
    *,
    chunk_rows: int,
) -> dict[str, Any]:
    maximum, entropy, row_sum_error = _probability_row_metrics(
        probabilities,
        chunk_rows=chunk_rows,
    )
    return {
        "rows": int(maximum.size),
        "candidates_per_row": int(np.asarray(probabilities).shape[1]),
        "row_max_probability": _describe(maximum),
        "normalized_entropy": _describe(entropy),
        "maximum_row_sum_error": row_sum_error,
    }


def popularity_distribution_summary(
    candidates: np.ndarray,
    destination_counts: np.ndarray,
    *,
    chunk_rows: int,
) -> dict[str, Any]:
    candidate_values = np.asarray(candidates)
    counts = np.asarray(destination_counts, dtype=np.int64)
    if candidate_values.ndim != 2 or candidate_values.shape[0] == 0:
        raise ValueError("candidates must have shape [rows, candidates]")
    if counts.ndim != 1 or np.any(counts < 0):
        raise ValueError("destination counts must be nonnegative and dense")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    histogram = np.zeros(int(counts.max(initial=0)) + 1, dtype=np.int64)
    for start in range(0, candidate_values.shape[0], chunk_rows):
        block = np.asarray(
            candidate_values[start : start + chunk_rows],
            dtype=np.int64,
        )
        valid = (block >= 0) & (block < counts.size)
        popularity = np.zeros(block.shape, dtype=np.int64)
        popularity[valid] = counts[block[valid]]
        histogram += np.bincount(
            popularity.reshape(-1),
            minlength=histogram.size,
        )
    return _popularity_summary_from_histogram(
        histogram,
        rows=int(candidate_values.shape[0]),
        candidates_per_row=int(candidate_values.shape[1]),
    )


def top1_change_summary(
    champion: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, int | float]:
    champion_values = np.asarray(champion)
    candidate_values = np.asarray(candidate)
    if (
        champion_values.ndim != 2
        or champion_values.shape != candidate_values.shape
        or champion_values.shape[0] == 0
    ):
        raise ValueError("top1 inputs must be aligned nonempty matrices")
    changed = int(
        np.count_nonzero(
            np.argmax(champion_values, axis=1)
            != np.argmax(candidate_values, axis=1)
        )
    )
    rows = int(champion_values.shape[0])
    return {
        "rows": rows,
        "top1_changed_rows": changed,
        "top1_change_rate": changed / rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run read-only, zero-label temporal-support and candidate-"
            "transport audits for cooccur_lift_aux_expert_v1."
        )
    )
    parser.add_argument("--test-lift", required=True, type=Path)
    parser.add_argument("--external-lift", required=True, type=Path)
    parser.add_argument("--test-auxiliary", required=True, type=Path)
    parser.add_argument("--external-auxiliary", required=True, type=Path)
    parser.add_argument("--external-baseline", required=True, type=Path)
    parser.add_argument("--external-candidate", required=True, type=Path)
    parser.add_argument("--validation-candidates", required=True, type=Path)
    parser.add_argument("--validation-times", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--champion-zip", required=True, type=Path)
    parser.add_argument("--candidate-zip", required=True, type=Path)
    parser.add_argument("--auxiliary-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--training-time-max", required=True, type=int)
    parser.add_argument("--short-window", required=True, type=float)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    train = read_interactions(args.train_csv)
    test_queries = read_test_queries(args.test_csv)
    validation_candidates = np.load(
        args.validation_candidates,
        mmap_mode="r",
        allow_pickle=False,
    )
    validation_times = np.load(
        args.validation_times,
        mmap_mode="r",
        allow_pickle=False,
    )
    strict_external = np.asarray(validation_times) > args.training_time_max
    if int(strict_external.sum()) == 0:
        raise ValueError("strict external mask is empty")

    test_lift = np.load(args.test_lift, mmap_mode="r", allow_pickle=False)
    external_lift = np.load(
        args.external_lift,
        mmap_mode="r",
        allow_pickle=False,
    )
    external_auxiliary = np.load(
        args.external_auxiliary,
        mmap_mode="r",
        allow_pickle=False,
    )
    test_auxiliary = np.load(
        args.test_auxiliary,
        mmap_mode="r",
        allow_pickle=False,
    )
    external_baseline = np.load(
        args.external_baseline,
        mmap_mode="r",
        allow_pickle=False,
    )
    external_candidate = np.load(
        args.external_candidate,
        mmap_mode="r",
        allow_pickle=False,
    )
    if external_auxiliary.shape[0] != int(strict_external.sum()):
        raise ValueError("external probabilities do not match strict rows")
    if external_lift.shape[0] != validation_times.shape[0]:
        raise ValueError("external lift does not match validation rows")
    if test_lift.shape[:2] != test_queries.candidates.shape:
        raise ValueError("test lift does not match test candidates")

    dst_counts = np.bincount(
        np.asarray(train.dst, dtype=np.int64),
        minlength=max(
            int(np.max(train.dst, initial=0)),
            int(np.max(test_queries.candidates, initial=0)),
            int(np.max(validation_candidates, initial=0)),
        )
        + 1,
    )
    external_short = np.asarray(external_lift[strict_external, :, 1])
    test_short = test_lift[:, :, 1]
    external_max, external_entropy, _ = _probability_row_metrics(
        external_auxiliary,
        chunk_rows=args.chunk_rows,
    )
    test_max, test_entropy, _ = _probability_row_metrics(
        test_auxiliary,
        chunk_rows=args.chunk_rows,
    )

    validation_popularity = popularity_distribution_summary(
        validation_candidates,
        dst_counts,
        chunk_rows=args.chunk_rows,
    )
    validation_negative_popularity = popularity_distribution_summary(
        validation_candidates[:, 1:],
        dst_counts,
        chunk_rows=args.chunk_rows,
    )
    test_popularity = popularity_distribution_summary(
        test_queries.candidates,
        dst_counts,
        chunk_rows=args.chunk_rows,
    )
    with np.load(args.auxiliary_model, allow_pickle=False) as model:
        linear1_weight = np.asarray(
            model["state__linear1.weight"],
            dtype=np.float32,
        )
        auxiliary_std = np.asarray(model["std"], dtype=np.float32)
    audit_a = {
        "external_strict": collapse_summary(
            external_short,
            chunk_rows=args.chunk_rows,
        ),
        "test": collapse_summary(
            test_short,
            chunk_rows=args.chunk_rows,
        ),
        "time_support": time_support_summary(
            train_time=np.asarray(train.time),
            external_time=np.asarray(validation_times)[strict_external],
            test_time=np.asarray(test_queries.time),
            frozen_training_time_max=args.training_time_max,
            short_window=args.short_window,
        ),
    }
    audit_a["majority_test_rows_collapsed"] = bool(
        audit_a["test"]["all_exact_zero_row_rate"] > 0.5
    )
    audit_b = {
        "auxiliary_output_distribution": {
            "external_strict": probability_distribution_summary(
                external_auxiliary,
                chunk_rows=args.chunk_rows,
            ),
            "test": probability_distribution_summary(
                test_auxiliary,
                chunk_rows=args.chunk_rows,
            ),
            "row_max_shift_external_to_test": _distribution_shift(
                external_max,
                test_max,
            ),
            "normalized_entropy_shift_external_to_test": (
                _distribution_shift(external_entropy, test_entropy)
            ),
        },
        "candidate_popularity": {
            "reference": "full train.csv destination event counts",
            "validation_all_candidates": validation_popularity,
            "validation_negative_candidates_only": (
                validation_negative_popularity
            ),
            "test_all_candidates": test_popularity,
            "shift_validation_all_to_test": _popularity_shift(
                validation_popularity,
                test_popularity,
            ),
            "shift_validation_negatives_to_test": _popularity_shift(
                validation_negative_popularity,
                test_popularity,
            ),
        },
        "top1_change": {
            "external_baseline_to_candidate": top1_change_summary(
                external_baseline,
                external_candidate,
            ),
            "online_package_champion_to_candidate": (
                _top1_change_from_zip_members(
                    args.champion_zip,
                    args.candidate_zip,
                    member="dataset2.csv",
                )
            ),
            "high_risk_threshold_from_request": 0.20,
        },
    }
    online_change = audit_b["top1_change"][
        "online_package_champion_to_candidate"
    ]["top1_change_rate"]
    audit_b["top1_change"]["online_exceeds_20_percent"] = bool(
        online_change >= 0.20
    )
    audit_c = {
        "definition": (
            "Exact change in first-layer pre-activation when the full or "
            "short lift signal and its centered/max-difference context "
            "channels are set to zero; later ReLU layers are not evaluated."
        ),
        "external_strict": first_layer_lift_intervention_summary(
            external_lift[strict_external],
            linear1_weight=linear1_weight,
            std=auxiliary_std,
            chunk_rows=args.chunk_rows,
        ),
        "test": first_layer_lift_intervention_summary(
            test_lift,
            linear1_weight=linear1_weight,
            std=auxiliary_std,
            chunk_rows=args.chunk_rows,
        ),
    }
    input_paths = {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, Path) and key != "output"
    }
    report = {
        "schema_version": 1,
        "status": "complete_zero_label_read_only_audit",
        "integration_id": "cooccur_lift_aux_expert_v1",
        "discipline": {
            "labels_read": False,
            "external_metrics_read_for_selection": False,
            "leaderboard_read_for_selection": False,
            "weights_or_models_changed": False,
            "purpose": "transport-risk diagnosis only",
        },
        "audit_a_temporal_support": audit_a,
        "audit_b_candidate_transport": audit_b,
        "audit_c_full_vs_short_first_layer_attribution": audit_c,
        "inputs": {
            key: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for key, path in input_paths.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(
            report,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def _probability_row_metrics(
    probabilities: np.ndarray,
    *,
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.asarray(probabilities)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("probabilities must be a nonempty matrix")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    maximum = np.empty(values.shape[0], dtype=np.float64)
    entropy = np.empty(values.shape[0], dtype=np.float64)
    row_sum_error = 0.0
    denominator = math.log(values.shape[1])
    for start in range(0, values.shape[0], chunk_rows):
        stop = min(start + chunk_rows, values.shape[0])
        block = np.asarray(values[start:stop], dtype=np.float64)
        if not np.all(np.isfinite(block)) or np.any(block < 0.0):
            raise ValueError("probabilities must be finite and nonnegative")
        maximum[start:stop] = block.max(axis=1)
        logarithm = np.zeros_like(block)
        np.log(block, out=logarithm, where=block > 0.0)
        entropy[start:stop] = (
            -np.sum(block * logarithm, axis=1) / denominator
        )
        row_sum_error = max(
            row_sum_error,
            float(np.max(np.abs(block.sum(axis=1) - 1.0))),
        )
    return maximum, entropy, row_sum_error


def _describe(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(array, QUANTILES, method="linear")
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std()),
        **{
            f"p{int(quantile * 100):02d}": float(value)
            for quantile, value in zip(QUANTILES, quantiles, strict=True)
        },
    }


def _popularity_summary_from_histogram(
    histogram: np.ndarray,
    *,
    rows: int,
    candidates_per_row: int,
) -> dict[str, Any]:
    counts = np.asarray(histogram, dtype=np.int64)
    total = int(counts.sum())
    values = np.arange(counts.size, dtype=np.float64)
    mean = float(np.dot(values, counts) / total)
    variance = float(np.dot((values - mean) ** 2, counts) / total)
    log_values = np.log1p(values)
    log_mean = float(np.dot(log_values, counts) / total)
    log_variance = float(
        np.dot((log_values - log_mean) ** 2, counts) / total
    )
    cumulative = np.cumsum(counts)
    quantile_values = {
        f"p{int(quantile * 100):02d}": float(
            np.searchsorted(
                cumulative,
                quantile * max(total - 1, 0) + 1,
                side="left",
            )
        )
        for quantile in QUANTILES
    }
    bucket_counts = []
    labels = []
    for lower, upper in zip(
        POPULARITY_BUCKET_EDGES,
        (*POPULARITY_BUCKET_EDGES[1:], None),
        strict=True,
    ):
        stop = counts.size if upper is None else min(upper, counts.size)
        bucket_counts.append(int(counts[lower:stop].sum()))
        labels.append(
            f"[{lower},inf)" if upper is None else f"[{lower},{upper})"
        )
    nonzero = np.flatnonzero(counts)
    return {
        "rows": rows,
        "candidates_per_row": candidates_per_row,
        "candidate_cells": total,
        "unseen_candidate_rate": float(counts[0] / total),
        "raw_train_dst_count": {
            "mean": mean,
            "standard_deviation": math.sqrt(variance),
            "maximum": float(nonzero[-1]) if nonzero.size else 0.0,
            **quantile_values,
        },
        "log1p_train_dst_count": {
            "mean": log_mean,
            "standard_deviation": math.sqrt(log_variance),
        },
        "fixed_bucket_labels": labels,
        "fixed_bucket_rates": [
            count / total for count in bucket_counts
        ],
    }


def _distribution_shift(
    reference: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    left = np.sort(np.asarray(reference, dtype=np.float64))
    right = np.sort(np.asarray(target, dtype=np.float64))
    pooled = np.sort(np.concatenate((left, right)))
    cdf_left = np.searchsorted(left, pooled, side="right") / left.size
    cdf_right = np.searchsorted(right, pooled, side="right") / right.size
    pooled_std = math.sqrt(
        (float(left.var()) + float(right.var())) / 2.0
    )
    mean_delta = float(right.mean() - left.mean())
    return {
        "mean_delta_target_minus_reference": mean_delta,
        "standardized_mean_delta": (
            mean_delta / pooled_std if pooled_std > 0.0 else 0.0
        ),
        "kolmogorov_smirnov_distance": float(
            np.max(np.abs(cdf_left - cdf_right))
        ),
    }


def _popularity_shift(
    reference: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, float]:
    left = np.asarray(reference["fixed_bucket_rates"], dtype=np.float64)
    right = np.asarray(target["fixed_bucket_rates"], dtype=np.float64)
    midpoint = (left + right) / 2.0
    left_positive = left > 0.0
    right_positive = right > 0.0
    left_kl = np.sum(
        left[left_positive]
        * np.log(left[left_positive] / midpoint[left_positive])
    )
    right_kl = np.sum(
        right[right_positive]
        * np.log(right[right_positive] / midpoint[right_positive])
    )
    return {
        "jensen_shannon_divergence_nats": float(
            (left_kl + right_kl) / 2.0
        ),
        "total_variation_distance": float(
            0.5 * np.sum(np.abs(left - right))
        ),
        "unseen_rate_delta": float(
            target["unseen_candidate_rate"]
            - reference["unseen_candidate_rate"]
        ),
        "mean_log1p_popularity_delta": float(
            target["log1p_train_dst_count"]["mean"]
            - reference["log1p_train_dst_count"]["mean"]
        ),
    }


def time_support_summary(
    *,
    train_time: np.ndarray,
    external_time: np.ndarray,
    test_time: np.ndarray,
    frozen_training_time_max: int,
    short_window: float,
) -> dict[str, Any]:
    train_values = np.asarray(train_time, dtype=np.int64)
    external_values = np.asarray(external_time, dtype=np.int64)
    test_values = np.asarray(test_time, dtype=np.int64)
    full_train_time_max = int(train_values.max())
    return {
        "train_time_min": int(train_values.min()),
        "full_train_time_max": full_train_time_max,
        "frozen_training_time_max": int(frozen_training_time_max),
        "short_window_seconds": float(short_window),
        "short_window_days": float(short_window / 86400.0),
        "external_query_time": _describe(external_values),
        "test_query_time": _describe(test_values),
        "external_gap_from_frozen_origin_days": _describe(
            (external_values - frozen_training_time_max) / 86400.0
        ),
        "test_gap_from_frozen_origin_days": _describe(
            (test_values - frozen_training_time_max) / 86400.0
        ),
        "external_gap_from_full_train_end_days": _describe(
            (external_values - full_train_time_max) / 86400.0
        ),
        "test_gap_from_full_train_end_days": _describe(
            (test_values - full_train_time_max) / 86400.0
        ),
        "external_rows_whose_short_window_starts_after_frozen_origin": float(
            np.mean(
                external_values - short_window
                > frozen_training_time_max
            )
        ),
        "test_rows_whose_short_window_starts_after_frozen_origin": float(
            np.mean(test_values - short_window > frozen_training_time_max)
        ),
        "external_rows_whose_short_window_starts_after_full_train": float(
            np.mean(external_values - short_window > full_train_time_max)
        ),
        "test_rows_whose_short_window_starts_after_full_train": float(
            np.mean(test_values - short_window > full_train_time_max)
        ),
    }


def _top1_change_from_zip_members(
    champion_zip: Path,
    candidate_zip: Path,
    *,
    member: str,
) -> dict[str, int | float]:
    rows = 0
    changed = 0
    with (
        zipfile.ZipFile(champion_zip) as champion_archive,
        zipfile.ZipFile(candidate_zip) as candidate_archive,
        champion_archive.open(member) as champion_file,
        candidate_archive.open(member) as candidate_file,
    ):
        while True:
            champion_line = champion_file.readline()
            candidate_line = candidate_file.readline()
            if not champion_line and not candidate_line:
                break
            if not champion_line or not candidate_line:
                raise ValueError("ZIP member row counts differ")
            champion_values = np.fromstring(
                champion_line.decode("ascii"),
                sep=",",
                dtype=np.float64,
            )
            candidate_values = np.fromstring(
                candidate_line.decode("ascii"),
                sep=",",
                dtype=np.float64,
            )
            if (
                champion_values.shape != (100,)
                or candidate_values.shape != (100,)
            ):
                raise ValueError("ZIP member must have 100 columns")
            changed += int(
                int(np.argmax(champion_values))
                != int(np.argmax(candidate_values))
            )
            rows += 1
    return {
        "rows": rows,
        "top1_changed_rows": changed,
        "top1_change_rate": changed / rows,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
