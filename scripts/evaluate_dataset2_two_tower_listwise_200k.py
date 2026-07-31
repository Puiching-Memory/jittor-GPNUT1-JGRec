from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from jgrec.contest_checkpoint import get_model_state
from jgrec.core.io import read_interactions
from jgrec.core.memory import release_memory
from jgrec.core.types import TestQueryArray
from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid.auto_strategy import profile_dataset
from jgrec.rankers.hybrid.config import TwoTowerConfig
from jgrec.rankers.hybrid.full100_training import passes_full100_gate
from jgrec.rankers.hybrid.two_tower import TwoTower


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the frozen Dataset2 Two-Tower validation scores with a "
            "200k/full-100/listwise candidate."
        )
    )
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--full100-prefix", required=True, type=Path)
    parser.add_argument("--cache-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--context-ratio", type=float, default=0.75)
    parser.add_argument("--validation-queries", type=int, default=20_000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--sampling-workers", type=int, default=16)
    parser.add_argument("--candidate-max-samples", type=int, default=200_000)
    parser.add_argument("--candidate-negatives", type=int, default=99)
    parser.add_argument("--min-full-delta", type=float, default=0.002)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite Two-Tower experiment: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True)
    started = time.time()

    cache_report = json.loads(args.cache_report.read_text(encoding="utf-8"))
    if cache_report.get("status") != "complete":
        raise ValueError("full-100 cache report is incomplete")
    feature_names = tuple(str(name) for name in cache_report["feature_names"])
    tower_feature_index = feature_names.index("two_tower_dot")
    prefix = str(args.full100_prefix)
    feature_path = Path(f"{prefix}.train.npy")
    candidate_path = Path(f"{prefix}.train-candidates.npy")
    src_path = Path(f"{prefix}.train-src.npy")
    dst_path = Path(f"{prefix}.train-dst.npy")
    time_path = Path(f"{prefix}.train-time.npy")
    source_features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    source_candidates = np.load(
        candidate_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    source_src = np.load(src_path, mmap_mode="r", allow_pickle=False)
    source_dst = np.load(dst_path, mmap_mode="r", allow_pickle=False)
    source_time = np.load(time_path, mmap_mode="r", allow_pickle=False)
    if list(source_features.shape) != cache_report["train_shape"]:
        raise ValueError("full-100 feature shape differs from its report")
    if list(source_candidates.shape) != cache_report["candidate_shape"]:
        raise ValueError("full-100 candidate shape differs from its report")
    if source_features.shape[:2] != source_candidates.shape:
        raise ValueError("full-100 features and candidates do not align")
    if source_features.shape[2] != len(feature_names):
        raise ValueError("cache feature schema does not match its report")
    if source_candidates.shape[1] != 100:
        raise ValueError("the frozen external validation must have 100 candidates")
    if not (
        len(source_src)
        == len(source_dst)
        == len(source_time)
        == source_candidates.shape[0]
    ):
        raise ValueError("full-100 event arrays do not align")
    if args.validation_queries > source_candidates.shape[0]:
        raise ValueError("requested validation rows exceed the full-100 cache")

    interactions = read_interactions(args.train_csv).sort_by_time()
    val_size = max(1, int(len(interactions) * args.val_ratio))
    train_end = max(2, len(interactions) - val_size)
    context_end = max(
        1,
        min(train_end - 1, int(train_end * args.context_ratio)),
    )
    train_prefix = interactions[:context_end]
    validation_indices = np.linspace(
        0,
        source_candidates.shape[0] - 1,
        args.validation_queries,
        dtype=np.int64,
    )
    if np.unique(validation_indices).size != args.validation_queries:
        raise RuntimeError("validation indices are not unique")
    validation_candidates = np.asarray(
        source_candidates[validation_indices],
        dtype=np.int32,
    )
    validation_src = np.asarray(source_src[validation_indices], dtype=np.int32)
    validation_dst = np.asarray(source_dst[validation_indices], dtype=np.int32)
    validation_time = np.asarray(source_time[validation_indices], dtype=np.int32)
    if not np.array_equal(
        validation_candidates[:, 0],
        validation_dst,
    ):
        raise ValueError(
            "cached validation positives do not match cached events"
        )
    if any(np.unique(row).size != 100 for row in validation_candidates):
        raise ValueError("cached validation candidate rows are not unique")
    validation_queries = TestQueryArray(
        src=validation_src,
        time=validation_time,
        candidates=validation_candidates,
    )
    baseline_scores = np.asarray(
        source_features[validation_indices, :, tower_feature_index],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(baseline_scores)):
        raise ValueError("frozen baseline Two-Tower scores are non-finite")

    candidate_config = TwoTowerConfig(
        enabled=True,
        embedding_dim=64,
        hidden_dim=64,
        epochs=args.epochs,
        early_stop_patience=args.patience,
        early_stop_val_ratio=0.1,
        batch_size=args.batch_size,
        score_batch_size=2048,
        max_samples=args.candidate_max_samples,
        lr=1e-3,
        weight_decay=0.0,
        num_negatives=args.candidate_negatives,
        hard_negative_ratio=0.5,
        popular_negative_ratio=0.25,
        test_candidate_negative_ratio=1.0,
        objective="listwise",
        early_stop_metric="mrr",
        negative_sampling_workers=args.sampling_workers,
    )
    slices = _three_slices(len(validation_queries))
    frozen = {
        "status": "frozen_before_candidate_training",
        "scope": "Dataset2 Two-Tower only; no package generation",
        "train_csv": str(args.train_csv.resolve()),
        "train_csv_sha256": _sha256_file(args.train_csv),
        "test_csv": str(args.test_csv.resolve()),
        "test_csv_sha256": _sha256_file(args.test_csv),
        "full100_cache_report": str(args.cache_report.resolve()),
        "full100_cache_report_sha256": _sha256_file(args.cache_report),
        "source_features_sha256": cache_report["artifacts"]["features"][
            "sha256"
        ],
        "source_candidates_sha256": cache_report["artifacts"]["candidates"][
            "sha256"
        ],
        "interaction_rows": len(interactions),
        "train_prefix_rows": len(train_prefix),
        "context_end": context_end,
        "validation_rows": len(validation_queries),
        "validation_candidates": 100,
        "validation_time_range": [
            int(validation_time.min()),
            int(validation_time.max()),
        ],
        "validation_indices_sha256": _sha256_array(validation_indices),
        "validation_slices": [
            [part.start, part.stop] for part in slices
        ],
        "baseline": {
            "source": (
                "frozen supervised validation cache generated by the current "
                "50k/31/BCE/loss-early-stop tower"
            ),
            "feature_name": "two_tower_dot",
            "feature_index": tower_feature_index,
        },
        "candidate": asdict(candidate_config),
        "seed": args.seed,
        "gate": {
            "minimum_full_mrr_delta": args.min_full_delta,
            "all_three_slices_non_decreasing": True,
        },
    }
    _write_json(args.output_dir / "frozen-config.json", frozen)
    print(json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True), flush=True)

    baseline = _evaluate_scores(baseline_scores, slices)
    baseline["score_sha256"] = _sha256_array(baseline_scores)
    _write_json(args.output_dir / "baseline-metrics.json", baseline)
    print(json.dumps({"baseline": baseline}, indent=2), flush=True)
    del source_features, baseline_scores

    profile = profile_dataset(
        interactions,
        args.test_csv,
        val_ratio=args.val_ratio,
    )
    phase_started = time.time()
    jt.sync_all()
    jt.clean()
    jt.set_global_seed(args.seed)
    candidate_tower = TwoTower(
        id_map=NodeIdMap.from_interactions(train_prefix),
        config=candidate_config,
        dataset_profile=profile,
    )
    candidate_tower.fit(
        train_prefix,
        rng=np.random.default_rng(args.seed),
        verbose=True,
    )
    if candidate_tower.model is None:
        raise RuntimeError("candidate Two-Tower did not produce a model")
    model_path = args.output_dir / "candidate-model.npz"
    np.savez_compressed(
        model_path,
        **get_model_state(candidate_tower.model),
    )
    candidate_scores = candidate_tower.scores_for_query_array(
        validation_queries
    )[:, :, 0]
    if not np.all(np.isfinite(candidate_scores)):
        raise RuntimeError("candidate produced non-finite validation scores")
    candidate = _evaluate_scores(candidate_scores, slices)
    candidate.update(
        {
            "config": asdict(candidate_config),
            "score_sha256": _sha256_array(candidate_scores),
            "model_path": str(model_path.resolve()),
            "model_sha256": _sha256_file(model_path),
            "elapsed_seconds": time.time() - phase_started,
        }
    )
    _write_json(args.output_dir / "candidate-metrics.json", candidate)
    print(json.dumps({"candidate": candidate}, indent=2), flush=True)

    baseline_slices = tuple(float(value) for value in baseline["slice_mrrs"])
    candidate_slices = tuple(float(value) for value in candidate["slice_mrrs"])
    passed = passes_full100_gate(
        baseline_full_mrr=float(baseline["full_mrr"]),
        candidate_full_mrr=float(candidate["full_mrr"]),
        baseline_slice_mrrs=baseline_slices,
        candidate_slice_mrrs=candidate_slices,
        min_full_delta=args.min_full_delta,
    )
    slice_deltas = [
        new - old
        for old, new in zip(
            baseline_slices,
            candidate_slices,
            strict=True,
        )
    ]
    report = {
        "status": "passed" if passed else "rejected",
        "gate_passed": passed,
        "full_reranker_integration_authorized": passed,
        "package_generated": False,
        "frozen_config": frozen,
        "baseline": baseline,
        "candidate": candidate,
        "delta": {
            "full_mrr": float(candidate["full_mrr"] - baseline["full_mrr"]),
            "slice_mrrs": slice_deltas,
        },
        "gate": {
            "minimum_full_mrr_delta": args.min_full_delta,
            "full_delta_passed": bool(
                candidate["full_mrr"]
                - baseline["full_mrr"]
                + 1e-12
                >= args.min_full_delta
            ),
            "all_slices_non_decreasing": bool(
                all(delta >= 0.0 for delta in slice_deltas)
            ),
        },
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.output_dir / "evaluation-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    del candidate_tower, candidate_scores
    release_memory()
    return 0 if passed else 2


def _evaluate_scores(
    scores: np.ndarray,
    slices: tuple[slice, slice, slice],
) -> dict[str, Any]:
    return {
        "full_mrr": _mrr(scores),
        "slice_mrrs": [_mrr(scores[part]) for part in slices],
        "score_shape": list(scores.shape),
    }


def _three_slices(row_count: int) -> tuple[slice, slice, slice]:
    first = row_count // 3
    second = (row_count * 2) // 3
    return (
        slice(0, first),
        slice(first, second),
        slice(second, row_count),
    )


def _mrr(scores: np.ndarray) -> float:
    values = np.asarray(scores)
    ranks = 1 + np.sum(values[:, 1:] > values[:, :1], axis=1)
    return float(np.mean(1.0 / ranks))


def _sha256_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
