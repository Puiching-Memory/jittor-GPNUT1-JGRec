from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, replace
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
from jgrec.rankers.hybrid.tower_optimization_experiment import (
    TWO_TOWER_SCREEN_ARMS,
    paired_rank_movements,
    positive_ranks,
    ranking_metrics,
    two_tower_screen_config,
    two_tower_screen_gate,
)
from jgrec.rankers.hybrid.two_tower import TwoTower


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one frozen Dataset2 Two-Tower optimizer/in-batch screen arm."
        )
    )
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--full100-prefix", required=True, type=Path)
    parser.add_argument("--cache-report", required=True, type=Path)
    parser.add_argument("--control-model", required=True, type=Path)
    parser.add_argument("--control-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--arm",
        required=True,
        choices=TWO_TOWER_SCREEN_ARMS,
    )
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--context-ratio", type=float, default=0.75)
    parser.add_argument("--validation-queries", type=int, default=20_000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--sampling-workers", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=200_000)
    parser.add_argument("--negatives", type=int, default=99)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.time()
    jt.flags.use_cuda = 1

    cache_report = _read_json(args.cache_report)
    if cache_report.get("status") != "complete":
        raise ValueError("full-100 cache report is incomplete")
    prefix = str(args.full100_prefix)
    candidates = np.load(
        Path(f"{prefix}.train-candidates.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    source_src = np.load(
        Path(f"{prefix}.train-src.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    source_dst = np.load(
        Path(f"{prefix}.train-dst.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    source_time = np.load(
        Path(f"{prefix}.train-time.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    if candidates.ndim != 2 or candidates.shape[1] != 100:
        raise ValueError("the frozen validation must have 100 candidates")
    if not (
        len(source_src)
        == len(source_dst)
        == len(source_time)
        == candidates.shape[0]
    ):
        raise ValueError("validation arrays do not align")
    if args.validation_queries > candidates.shape[0]:
        raise ValueError("requested validation rows exceed the frozen cache")

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
        candidates.shape[0] - 1,
        args.validation_queries,
        dtype=np.int64,
    )
    if np.unique(validation_indices).size != args.validation_queries:
        raise RuntimeError("validation indices are not unique")
    validation_candidates = np.asarray(
        candidates[validation_indices],
        dtype=np.int32,
    )
    validation_src = np.asarray(
        source_src[validation_indices],
        dtype=np.int32,
    )
    validation_dst = np.asarray(
        source_dst[validation_indices],
        dtype=np.int32,
    )
    validation_time = np.asarray(
        source_time[validation_indices],
        dtype=np.int32,
    )
    if not np.array_equal(validation_candidates[:, 0], validation_dst):
        raise ValueError("validation positives must be candidate column zero")
    if any(np.unique(row).size != 100 for row in validation_candidates):
        raise ValueError("validation candidate rows must be unique")
    queries = TestQueryArray(
        src=validation_src,
        time=validation_time,
        candidates=validation_candidates,
    )
    slices = _three_slices(len(queries))
    profile = profile_dataset(
        interactions,
        args.test_csv,
        val_ratio=args.val_ratio,
    )
    id_map = NodeIdMap.from_interactions(train_prefix)

    base_config = TwoTowerConfig(
        enabled=True,
        embedding_dim=64,
        hidden_dim=64,
        epochs=args.epochs,
        early_stop_patience=args.patience,
        early_stop_val_ratio=0.1,
        batch_size=args.batch_size,
        score_batch_size=2048,
        max_samples=args.max_samples,
        lr=1e-3,
        weight_decay=0.0,
        num_negatives=args.negatives,
        hard_negative_ratio=0.5,
        popular_negative_ratio=0.25,
        test_candidate_negative_ratio=1.0,
        objective="listwise",
        early_stop_metric="mrr",
        negative_sampling_workers=args.sampling_workers,
        in_batch_negative_weight=1.0,
        in_batch_temperature=1.0,
    )
    control_config = two_tower_screen_config(base_config, "control")
    candidate_config = two_tower_screen_config(base_config, args.arm)
    frozen = {
        "status": "frozen_before_candidate_training",
        "scope": (
            "Dataset2 standalone Two-Tower Stage 1 screen; "
            "no final integration, external opening, or package generation"
        ),
        "arm": args.arm,
        "seed": args.seed,
        "train_csv": _file_record(args.train_csv),
        "test_csv": _file_record(args.test_csv),
        "cache_report": _file_record(args.cache_report),
        "control_model": _file_record(args.control_model),
        "control_report": _file_record(args.control_report),
        "validation_indices_sha256": _sha256_array(validation_indices),
        "validation_rows": len(queries),
        "validation_candidates": queries.candidate_count,
        "validation_time_range": [
            int(validation_time.min()),
            int(validation_time.max()),
        ],
        "validation_slices": [
            [part.start, part.stop] for part in slices
        ],
        "interaction_rows": len(interactions),
        "train_prefix_rows": len(train_prefix),
        "context_end": context_end,
        "control_config": asdict(control_config),
        "candidate_config": asdict(candidate_config),
        "gate": {
            "full_metrics_non_decreasing": [
                "mrr",
                "hit_at_1",
                "hit_at_3",
                "hit_at_10",
                "ndcg_at_10",
            ],
            "full_mean_rank_non_increasing": True,
            "each_slice_non_decreasing": ["mrr", "ndcg_at_10"],
            "improved_queries_strictly_exceed_worsened": True,
        },
    }
    _write_json(args.output_dir / "frozen-config.json", frozen)
    print(json.dumps(frozen, indent=2, sort_keys=True), flush=True)

    control_tower = _load_control_tower(
        model_path=args.control_model,
        interactions=train_prefix,
        id_map=id_map,
        config=control_config,
        profile=profile,
        seed=args.seed,
    )
    control_scores = control_tower.scores_for_query_array(queries)[:, :, 0]
    control_evaluation = _evaluate(control_scores, slices)
    _verify_control_reproduction(
        control_evaluation,
        _read_json(args.control_report),
    )
    np.save(args.output_dir / "control-scores.npy", control_scores)
    del control_tower
    release_memory()

    phase_started = time.time()
    jt.sync_all()
    jt.clean()
    jt.set_global_seed(args.seed)
    candidate_tower = TwoTower(
        id_map=id_map,
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
    candidate_scores = candidate_tower.scores_for_query_array(queries)[
        :,
        :,
        0,
    ]
    if not np.all(np.isfinite(candidate_scores)):
        raise RuntimeError("candidate scores are non-finite")
    np.save(args.output_dir / "candidate-scores.npy", candidate_scores)
    candidate_evaluation = _evaluate(candidate_scores, slices)
    movements = paired_rank_movements(
        positive_ranks(control_scores),
        positive_ranks(candidate_scores),
    )
    slice_movements = [
        paired_rank_movements(
            positive_ranks(control_scores[part]),
            positive_ranks(candidate_scores[part]),
        )
        for part in slices
    ]
    gate = two_tower_screen_gate(
        control_evaluation["full"],
        candidate_evaluation["full"],
        control_evaluation["slices"],
        candidate_evaluation["slices"],
        movements,
    )
    report = {
        "status": "complete",
        "arm": args.arm,
        "stage1_gate_passed": gate["passed"],
        "exact_integrated_rolling_authorized": gate["passed"],
        "external_opened": False,
        "package_generated": False,
        "frozen_config": frozen,
        "control": {
            **control_evaluation,
            "score_sha256": _sha256_array(control_scores),
        },
        "candidate": {
            **candidate_evaluation,
            "score_sha256": _sha256_array(candidate_scores),
            "model_path": str(model_path.resolve()),
            "model_sha256": _sha256_file(model_path),
            "training_elapsed_seconds": time.time() - phase_started,
        },
        "delta": {
            "full": _metric_delta(
                candidate_evaluation["full"],
                control_evaluation["full"],
            ),
            "slices": [
                _metric_delta(candidate_part, control_part)
                for control_part, candidate_part in zip(
                    control_evaluation["slices"],
                    candidate_evaluation["slices"],
                    strict=True,
                )
            ],
        },
        "query_movements": movements,
        "slice_query_movements": slice_movements,
        "gate": gate,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.output_dir / "evaluation-report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    del candidate_tower, control_scores, candidate_scores
    release_memory()
    return 0


def _load_control_tower(
    *,
    model_path: Path,
    interactions: Any,
    id_map: NodeIdMap,
    config: TwoTowerConfig,
    profile: Any,
    seed: int,
) -> TwoTower:
    tower = TwoTower(
        id_map=id_map,
        config=replace(config, enabled=False),
        dataset_profile=profile,
    )
    tower.fit(
        interactions,
        rng=np.random.default_rng(seed),
        verbose=False,
    )
    tower.config = config
    with np.load(model_path, allow_pickle=False) as archive:
        model_state = {
            key: np.asarray(archive[key], dtype=np.float32)
            for key in archive.files
        }
    tower.hydrate(
        {
            "model_state": model_state,
            "index": tower.index,
            "min_time": tower.min_time,
            "max_time": tower.max_time,
            "graph_span": tower.graph_span,
        }
    )
    return tower


def _evaluate(
    scores: np.ndarray,
    slices: tuple[slice, slice, slice],
) -> dict[str, Any]:
    return {
        "full": ranking_metrics(scores),
        "slices": [ranking_metrics(scores[part]) for part in slices],
        "score_shape": list(scores.shape),
    }


def _verify_control_reproduction(
    actual: dict[str, Any],
    historical_report: dict[str, Any],
) -> None:
    expected = historical_report["candidate"]
    if abs(actual["full"]["mrr"] - float(expected["full_mrr"])) > 1e-8:
        raise RuntimeError("frozen control full MRR reproduction failed")
    for index, expected_mrr in enumerate(expected["slice_mrrs"]):
        if (
            abs(
                actual["slices"][index]["mrr"]
                - float(expected_mrr)
            )
            > 1e-8
        ):
            raise RuntimeError(
                f"frozen control slice {index} MRR reproduction failed"
            )


def _three_slices(row_count: int) -> tuple[slice, slice, slice]:
    first = row_count // 3
    second = (row_count * 2) // 3
    return (
        slice(0, first),
        slice(first, second),
        slice(second, row_count),
    )


def _metric_delta(
    candidate: dict[str, float],
    control: dict[str, float],
) -> dict[str, float]:
    return {
        key: float(candidate[key] - control[key])
        for key in control
    }


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _sha256_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
