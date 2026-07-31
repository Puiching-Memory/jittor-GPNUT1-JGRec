from __future__ import annotations

import argparse
import hashlib
import json
import time
from itertools import pairwise
from pathlib import Path

import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.core.io import read_interactions, read_test_queries
from jgrec.rankers.hybrid.oof_hard_negatives import contiguous_oof_folds
from jgrec.rankers.hybrid.ranker import _sample_events
from jgrec.rankers.hybrid.two_hop_decay_proxy import (
    accumulate_required_cooccurrence_events,
    canonical_item_pair,
    passes_two_hop_proxy_gate,
    recent_unique_targets,
    tie_neutral_mrr,
    two_hop_scores,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate raw versus time-decayed two-hop evidence on a frozen Dataset2 proxy."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--proxy-queries", type=int, default=2_000)
    parser.add_argument("--negative-count", type=int, default=31)
    parser.add_argument("--source-history-limit", type=int, default=64)
    parser.add_argument("--cooccur-history-limit", type=int, default=128)
    parser.add_argument("--tau-span-ratio", type=float, default=0.05)
    parser.add_argument("--candidate-seed-offset", type=int, default=20_000)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite two-hop proxy: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    frozen_path = args.output_dir / "frozen-config.json"
    report_path = args.output_dir / "two-hop-decay-proxy-report.json"
    started = time.time()

    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    config = state["config"]
    interactions = read_interactions(args.train_csv).sort_by_time()
    original_rows = len(interactions)
    if config.max_fit_events > 0 and len(interactions) > config.max_fit_events:
        interactions = interactions.tail(config.max_fit_events)
    n_events = len(interactions)
    val_size = max(1, int(n_events * config.val_ratio))
    train_end = max(2, n_events - val_size)
    context_end = max(1, min(train_end - 1, int(train_end * config.context_ratio)))
    rng = np.random.default_rng(config.seed)
    sampled_train = _sample_events(
        interactions[context_end:train_end], config.max_train_events, rng
    )
    sampled_val = _sample_events(interactions[train_end:], config.max_val_events, rng)
    if len(sampled_val) != 20_000:
        raise ValueError(f"expected 20,000 reconstructed validation rows, got {len(sampled_val)}")
    if args.proxy_queries > len(sampled_val):
        raise ValueError("proxy query count exceeds reconstructed validation rows")

    proxy_indices = np.linspace(
        0, len(sampled_val) - 1, args.proxy_queries, dtype=np.int64
    )
    if np.unique(proxy_indices).size != args.proxy_queries:
        raise RuntimeError("evenly spaced proxy indices are not unique")
    proxy_events = sampled_val.take(proxy_indices)
    candidate_matrix = _sample_candidate_matrix(
        positives=proxy_events.dst,
        test_csv=args.test_csv,
        negative_count=args.negative_count,
        seed=int(config.seed) + args.candidate_seed_offset,
    )
    if any(np.unique(row).size != row.size for row in candidate_matrix):
        raise RuntimeError("proxy candidate rows are not unique")

    prefix = interactions[:train_end]
    order = np.lexsort((prefix.time, prefix.src))
    sorted_src = prefix.src[order]
    sorted_dst = prefix.dst[order]
    sorted_time = prefix.time[order]
    source_histories = _query_source_histories(
        sorted_src,
        sorted_dst,
        proxy_events.src,
        args.source_history_limit,
    )
    required_pairs = _required_pairs(source_histories, candidate_matrix)
    prefix_span = int(prefix.time.max()) - int(prefix.time.min())
    tau = max(1.0, args.tau_span_ratio * prefix_span)
    folds = contiguous_oof_folds(row_count=args.proxy_queries, fold_count=3)
    frozen = {
        "status": "frozen_before_cooccurrence_collection_and_scoring",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "train_csv": str(args.train_csv.resolve()),
        "train_csv_sha256": _sha256_file(args.train_csv),
        "test_csv": str(args.test_csv.resolve()),
        "test_csv_sha256": _sha256_file(args.test_csv),
        "original_interaction_rows": original_rows,
        "fit_interaction_rows": n_events,
        "context_end": context_end,
        "train_end": train_end,
        "sampled_train_rows": len(sampled_train),
        "sampled_validation_rows": len(sampled_val),
        "proxy_query_count": args.proxy_queries,
        "proxy_index_sha256": _sha256_array(proxy_indices),
        "proxy_time_range": [int(proxy_events.time.min()), int(proxy_events.time.max())],
        "negative_count": args.negative_count,
        "candidate_sampling": "unique weighted sample from public test candidate frequencies",
        "candidate_seed": int(config.seed) + args.candidate_seed_offset,
        "candidate_matrix_shape": list(candidate_matrix.shape),
        "candidate_matrix_sha256": _sha256_array(candidate_matrix),
        "source_history_limit": args.source_history_limit,
        "cooccur_history_limit": args.cooccur_history_limit,
        "required_pair_count": len(required_pairs),
        "required_pair_commutative_checksum": _commutative_pair_checksum(required_pairs),
        "prefix_time_range": [int(prefix.time.min()), int(prefix.time.max())],
        "prefix_time_span": prefix_span,
        "tau_span_ratio": args.tau_span_ratio,
        "tau": tau,
        "seed": int(config.seed),
        "validation_slices": [
            [fold.holdout.start, fold.holdout.stop] for fold in folds
        ],
        "tie_policy": "average rank among equal scores",
        "thresholds": {
            "min_query_coverage": 0.20,
            "min_full_mrr_delta": 0.01,
            "require_every_slice_improvement": True,
        },
    }
    _write_json(frozen_path, frozen)
    print(json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True), flush=True)

    pair_event_times: dict[tuple[int, int], list[int]] = {}
    boundaries = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(sorted_src[1:] != sorted_src[:-1]) + 1,
            np.array([len(sorted_src)], dtype=np.int64),
        )
    )
    group_count = len(boundaries) - 1
    for group_index, (left, right) in enumerate(
        pairwise(boundaries), start=1
    ):
        accumulate_required_cooccurrence_events(
            sorted_dst[left:right],
            sorted_time[left:right],
            required_pairs,
            pair_event_times,
            history_limit=args.cooccur_history_limit,
        )
        if group_index % 10_000 == 0 or group_index == group_count:
            event_count = sum(len(values) for values in pair_event_times.values())
            print(
                "[two-hop-proxy] "
                f"groups={group_index}/{group_count} "
                f"matched_pairs={len(pair_event_times)} events={event_count} "
                f"seconds={time.time() - started:.1f}",
                flush=True,
            )

    compact_event_times = {
        pair: np.asarray(values, dtype=np.int32)
        for pair, values in pair_event_times.items()
    }
    raw_scores = np.zeros(candidate_matrix.shape, dtype=np.float64)
    decayed_scores = np.zeros(candidate_matrix.shape, dtype=np.float64)
    for row_index, (query_time, history, candidates) in enumerate(
        zip(
            proxy_events.time,
            source_histories,
            candidate_matrix,
            strict=True,
        )
    ):
        raw_row, decayed_row = two_hop_scores(
            query_time=int(query_time),
            source_history=history,
            candidates=candidates,
            pair_event_times=compact_event_times,
            tau=tau,
        )
        raw_scores[row_index] = raw_row
        decayed_scores[row_index] = decayed_row

    query_nonzero = np.any(raw_scores > 0, axis=1)
    coverage = float(np.mean(query_nonzero))
    positive_coverage = float(np.mean(raw_scores[:, 0] > 0))
    raw_full = tie_neutral_mrr(raw_scores)
    decayed_full = tie_neutral_mrr(decayed_scores)
    raw_slices = [tie_neutral_mrr(raw_scores[fold.holdout]) for fold in folds]
    decayed_slices = [
        tie_neutral_mrr(decayed_scores[fold.holdout]) for fold in folds
    ]
    passed = passes_two_hop_proxy_gate(
        coverage=coverage,
        baseline_mrr=raw_full,
        candidate_mrr=decayed_full,
        baseline_slice_mrrs=raw_slices,
        candidate_slice_mrrs=decayed_slices,
    )
    report = {
        "status": "passed" if passed else "rejected",
        "gate_passed": passed,
        "frozen_config": frozen,
        "coverage": {
            "query_with_any_nonzero_score_share": coverage,
            "positive_nonzero_score_share": positive_coverage,
            "matched_required_pair_count": len(compact_event_times),
            "matched_required_pair_share": (
                len(compact_event_times) / len(required_pairs) if required_pairs else 0.0
            ),
            "cooccurrence_event_count": sum(
                len(values) for values in compact_event_times.values()
            ),
        },
        "raw_count": {
            "full_tie_neutral_mrr": raw_full,
            "slice_tie_neutral_mrrs": raw_slices,
        },
        "time_decayed": {
            "full_tie_neutral_mrr": decayed_full,
            "full_delta": decayed_full - raw_full,
            "slice_tie_neutral_mrrs": decayed_slices,
            "slice_deltas": [
                candidate - baseline
                for baseline, candidate in zip(raw_slices, decayed_slices, strict=True)
            ],
        },
        "elapsed_seconds": time.time() - started,
    }
    _write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 2


def _sample_candidate_matrix(
    *,
    positives: np.ndarray,
    test_csv: Path,
    negative_count: int,
    seed: int,
) -> np.ndarray:
    test_queries = read_test_queries(test_csv)
    values, counts = np.unique(test_queries.candidates, return_counts=True)
    probabilities = counts.astype(np.float64)
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(seed)
    output = np.empty((len(positives), negative_count + 1), dtype=np.int32)
    output[:, 0] = positives
    for row_index, positive in enumerate(positives):
        eligible = values != int(positive)
        eligible_values = values[eligible]
        eligible_probabilities = probabilities[eligible]
        eligible_probabilities /= eligible_probabilities.sum()
        output[row_index, 1:] = rng.choice(
            eligible_values,
            size=negative_count,
            replace=False,
            p=eligible_probabilities,
        )
    return output


def _query_source_histories(
    sorted_src: np.ndarray,
    sorted_dst: np.ndarray,
    query_sources: np.ndarray,
    limit: int,
) -> list[np.ndarray]:
    histories: list[np.ndarray] = []
    for source in query_sources:
        left = int(np.searchsorted(sorted_src, source, side="left"))
        right = int(np.searchsorted(sorted_src, source, side="right"))
        histories.append(recent_unique_targets(sorted_dst[left:right], limit))
    return histories


def _required_pairs(
    source_histories: list[np.ndarray], candidate_matrix: np.ndarray
) -> set[tuple[int, int]]:
    required: set[tuple[int, int]] = set()
    for history, candidates in zip(source_histories, candidate_matrix, strict=True):
        for history_target in history:
            history_int = int(history_target)
            for candidate in candidates:
                candidate_int = int(candidate)
                if history_int != candidate_int:
                    required.add(canonical_item_pair(history_int, candidate_int))
    return required


def _commutative_pair_checksum(pairs: set[tuple[int, int]]) -> str:
    xor_value = 0
    sum_value = 0
    mask = (1 << 64) - 1
    for left, right in pairs:
        value = ((int(left) & 0xFFFFFFFF) << 32) | (int(right) & 0xFFFFFFFF)
        value = (value + 0x9E3779B97F4A7C15) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
        xor_value ^= value
        sum_value = (sum_value + value) & mask
    return f"xor64:{xor_value:016x}:sum64:{sum_value:016x}"


def _sha256_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
