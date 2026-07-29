from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from jgrec.core.io import read_interactions
from jgrec.core.memory import release_memory
from jgrec.rankers.hybrid.bounded_source_decoder import (
    bounded_source_decoder_audit,
    load_bounded_source_decoder_checkpoint,
    predict_bounded_source_decoder_logits,
)
from jgrec.rankers.hybrid.multi_horizon_oof import (
    HORIZON_NAMES,
    MultiHorizonOOFPiece,
    assemble_multi_horizon_oof,
    audit_multi_horizon_oof,
    canonical_multi_horizon_slices,
)
from jgrec.rankers.hybrid.source_conditioned_training import (
    load_source_conditioned_checkpoint,
    predict_source_conditioned_logits,
)
from jgrec.rankers.hybrid.source_sequence_cache import (
    SourceConditionedFold,
    SourceSequenceRows,
    build_causal_source_sequences,
)

CAP = 0.10
EXPECTED_COVERAGE = {
    "short": 120_091,
    "medium": 81_184,
    "long": 40_196,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate strict multi-horizon OOF residuals from existing "
            "Jittor bounded source decoders."
        ),
    )
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument("--sequence-cache-dir", required=True, type=Path)
    parser.add_argument("--base-result-dir", required=True, type=Path)
    parser.add_argument("--decoder-result-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--predict-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    started = time.time()
    _configure_device(args.device, args.seed)
    if args.predict_batch_size <= 0:
        raise ValueError("--predict-batch-size must be positive")
    if args.output_dir.exists():
        manifest_path = args.output_dir / "manifest.json"
        if manifest_path.exists():
            _verify_existing(args.output_dir, _read_json(manifest_path))
            print(json.dumps(_read_json(manifest_path), indent=2), flush=True)
            return 0
        raise FileExistsError(f"output directory is not complete: {args.output_dir}")

    build_dir = args.output_dir.with_name(f"{args.output_dir.name}.building")
    if build_dir.exists():
        raise FileExistsError(f"stale build directory exists: {build_dir}")
    build_dir.mkdir(parents=True)

    context = _load_context(args)
    folds = context["folds"]
    slices = canonical_multi_horizon_slices(folds)
    by_origin = _generate_origins(args, context, slices)
    pieces = _build_pieces(slices, by_origin, context["times"])
    artifact = assemble_multi_horizon_oof(
        pieces,
        row_count=int(context["features"].shape[0]),
        candidate_count=int(context["features"].shape[1]),
    )
    audit = audit_multi_horizon_oof(artifact, cap=CAP)
    if audit["coverage_rows"] != EXPECTED_COVERAGE:
        raise RuntimeError(
            "multi-horizon coverage differs: "
            f"{audit['coverage_rows']} != {EXPECTED_COVERAGE}"
        )
    if not audit["passed"]:
        raise RuntimeError(f"multi-horizon audit failed: {audit}")

    arrays = {
        "residuals": ("residuals.npy", artifact.residuals),
        "base_logits": ("base-logits.npy", artifact.base_logits),
        "corrected_logits": (
            "corrected-logits.npy",
            artifact.corrected_logits,
        ),
        "valid_mask": ("valid-mask.npy", artifact.valid_mask),
        "origin_index": ("origin-index.npy", artifact.origin_index),
        "gap_days": ("gap-days.npy", artifact.gap_days),
    }
    artifact_records = {}
    for key, (name, values) in arrays.items():
        build_path = build_dir / name
        _save_array_atomic(build_path, values)
        artifact_records[key] = _array_record(
            build_path,
            args.output_dir / name,
            values,
        )
        print(
            f"[multi-horizon] saved {name} shape={values.shape}",
            flush=True,
        )

    metrics = _metrics_report(artifact, slices)
    _write_json_atomic(build_dir / "audit.json", audit)
    _write_json_atomic(build_dir / "metrics.json", metrics)
    manifest = {
        "status": "complete",
        "protocol": "dataset2_bounded_source_multi_horizon_oof_v1",
        "created_at_unix": int(time.time()),
        "elapsed_seconds": float(time.time() - started),
        "horizon_axis": list(HORIZON_NAMES),
        "cap": CAP,
        "candidate_positive_column": 0,
        "strict_oof_rule": (
            "CST-A and bounded decoder are frozen at origin; "
            "score query time is after train_time_max; source history uses "
            "event_time < origin history_time_limit"
        ),
        "invalid_row_rule": (
            "base/corrected/residual are zero, origin is -1, gap is NaN; "
            "valid-mask is authoritative"
        ),
        "folds": [asdict(fold) for fold in folds],
        "slices": [asdict(row) for row in slices],
        "coverage_rows": audit["coverage_rows"],
        "gap_day_ranges": metrics["gap_day_ranges"],
        "artifacts": artifact_records,
        "audit": str((args.output_dir / "audit.json").resolve()),
        "metrics": str((args.output_dir / "metrics.json").resolve()),
        "inputs": _input_manifest(args, context, by_origin),
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
        "new_training_performed": False,
        "submission_generated": False,
    }
    _write_json_atomic(build_dir / "manifest.json", manifest)
    os.replace(build_dir, args.output_dir)
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(args.output_dir.resolve()),
                "coverage_rows": audit["coverage_rows"],
                "gap_day_ranges": metrics["gap_day_ranges"],
                "horizon_metrics": metrics["horizon_metrics"],
                "elapsed_seconds": time.time() - started,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def _load_context(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.sequence_cache_dir / "fold-manifest.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("status") != "complete"
        or manifest.get("trainable_frameworks") != ["jittor"]
        or manifest.get("non_jittor_trainable_models") != []
    ):
        raise ValueError("source sequence manifest is invalid")
    prefix = str(args.train_cache_prefix)
    paths = {
        "features": Path(f"{prefix}.train.npy"),
        "candidates": Path(f"{prefix}.train-candidates.npy"),
        "src": Path(f"{prefix}.train-src.npy"),
        "time": Path(f"{prefix}.train-time.npy"),
        "dst": Path(f"{prefix}.train-dst.npy"),
        "row_indices": Path(f"{prefix}.train-row-indices.npy"),
    }
    arrays = {
        key: np.load(path, mmap_mode="r", allow_pickle=False)
        for key, path in paths.items()
    }
    features = arrays["features"]
    expected_rows = int(manifest["train_rows"])
    expected_candidates = int(manifest["candidate_count"])
    if (
        features.shape[:2] != (expected_rows, expected_candidates)
        or arrays["candidates"].shape != features.shape[:2]
        or any(
            arrays[key].shape != (expected_rows,)
            for key in ("src", "time", "dst", "row_indices")
        )
        or not np.array_equal(
            np.asarray(arrays["candidates"][:, 0]),
            np.asarray(arrays["dst"]),
        )
    ):
        raise ValueError("multi-horizon feature cache contract differs")

    interactions = read_interactions(args.train_csv).sort_by_time()
    row_indices = np.asarray(arrays["row_indices"], dtype=np.int64)
    if (
        row_indices.size == 0
        or int(row_indices[-1]) >= len(interactions)
        or not np.array_equal(interactions.src[row_indices], arrays["src"])
        or not np.array_equal(interactions.dst[row_indices], arrays["dst"])
        or not np.array_equal(
            interactions.time[row_indices].astype(np.int64),
            arrays["time"],
        )
    ):
        raise ValueError("train.csv does not reproduce cached query rows")

    folds = tuple(
        SourceConditionedFold(
            index=int(row["index"]),
            train_rows=tuple(int(value) for value in row["train_rows"]),
            score_rows=tuple(int(value) for value in row["score_rows"]),
            role=str(row["role"]),
            train_time_max=int(row["train_time_max"]),
            score_time_min=int(row["score_time_min"]),
            score_time_max=int(row["score_time_max"]),
        )
        for row in manifest["folds"]
    )
    return {
        **arrays,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "paths": paths,
        "folds": folds,
        "interactions": interactions,
        "times": arrays["time"],
        "features": features,
        "num_items": int(manifest["num_items"]),
        "max_length": int(manifest["max_length"]),
    }


def _generate_origins(
    args: argparse.Namespace,
    context: dict[str, Any],
    slices: tuple[Any, ...],
) -> dict[int, dict[str, Any]]:
    by_origin: dict[int, dict[str, Any]] = {}
    for fold in context["folds"]:
        origin_slices = [
            row for row in slices if row.origin_index == fold.index
        ]
        if not origin_slices:
            continue
        score_start = min(row.score_rows[0] for row in origin_slices)
        score_stop = max(row.score_rows[1] for row in origin_slices)
        query_src = context["src"][score_start:score_stop]
        query_times = context["time"][score_start:score_stop]
        print(
            f"[multi-horizon] origin={fold.index} "
            f"train=[0,{fold.train_rows[1]}) "
            f"score=[{score_start},{score_stop}) "
            f"history_limit={fold.score_time_min}",
            flush=True,
        )
        sequences = build_causal_source_sequences(
            context["interactions"],
            query_src=query_src,
            query_time=query_times,
            max_length=context["max_length"],
            history_time_limit=int(fold.score_time_min),
        )
        sequence_replay = _verify_short_sequence_replay(
            args.sequence_cache_dir,
            fold,
            sequences,
        )

        base_checkpoint = (
            args.base_result_dir
            / "folds"
            / "variant-A"
            / f"fold-{fold.index}"
            / "model.npz"
        )
        base_model, base_result = load_source_conditioned_checkpoint(
            base_checkpoint
        )
        if (
            base_result.training_rows != int(fold.train_rows[1])
            or base_result.trainable_frameworks != ("jittor",)
            or base_result.non_jittor_trainable_models
        ):
            raise ValueError(
                f"origin {fold.index} frozen CST checkpoint differs"
            )
        base_logits = predict_source_conditioned_logits(
            base_model,
            context["features"][score_start:score_stop],
            context["candidates"][score_start:score_stop],
            sequences,
            mean=base_result.mean,
            std=base_result.std,
            batch_size=args.predict_batch_size,
        )
        del base_model
        release_memory()

        decoder_checkpoint = (
            args.decoder_result_dir
            / "folds"
            / f"cap-{CAP:.2f}"
            / f"fold-{fold.index}"
            / "model.npz"
        )
        decoder, decoder_result = load_bounded_source_decoder_checkpoint(
            decoder_checkpoint
        )
        if (
            decoder_result.training_rows != int(fold.train_rows[1])
            or not math.isclose(decoder.config.cap, CAP)
            or decoder_result.trainable_frameworks != ("jittor",)
            or decoder_result.non_jittor_trainable_models
        ):
            raise ValueError(
                f"origin {fold.index} bounded decoder checkpoint differs"
            )
        support = _candidate_support(
            context["dst"][: int(fold.train_rows[1])],
            context["candidates"][score_start:score_stop],
            context["num_items"],
        )
        corrected_logits = predict_bounded_source_decoder_logits(
            decoder,
            base_logits,
            context["candidates"][score_start:score_stop],
            sequences,
            support,
            batch_size=args.predict_batch_size,
        )
        residual_audit = bounded_source_decoder_audit(
            base_logits,
            corrected_logits,
            sequences.lengths,
            cap=CAP,
        )
        if not residual_audit["passed"]:
            raise RuntimeError(
                f"origin {fold.index} bounded residual audit failed"
            )
        short_replay = _verify_short_logit_replay(
            args,
            fold,
            score_start,
            base_logits,
            corrected_logits,
        )
        by_origin[int(fold.index)] = {
            "score_start": score_start,
            "score_stop": score_stop,
            "base_logits": base_logits,
            "corrected_logits": corrected_logits,
            "sequence_replay": sequence_replay,
            "short_logit_replay": short_replay,
            "residual_audit": residual_audit,
            "base_checkpoint": base_checkpoint,
            "decoder_checkpoint": decoder_checkpoint,
            "source_empty_rows": int(np.sum(sequences.lengths == 0)),
        }
        print(
            f"[multi-horizon] origin={fold.index} "
            f"max_residual={residual_audit['max_absolute_residual']:.8f} "
            f"short_replay={short_replay['max_corrected_error']:.3g}",
            flush=True,
        )
        del decoder, support, sequences
        release_memory()
    return by_origin


def _build_pieces(
    slices: tuple[Any, ...],
    by_origin: dict[int, dict[str, Any]],
    times: Any,
) -> list[MultiHorizonOOFPiece]:
    pieces = []
    for row in slices:
        origin = by_origin[row.origin_index]
        offset_start = int(row.score_rows[0]) - int(origin["score_start"])
        offset_stop = int(row.score_rows[1]) - int(origin["score_start"])
        pieces.append(
            MultiHorizonOOFPiece(
                horizon=row.horizon,
                origin_index=row.origin_index,
                score_rows=row.score_rows,
                train_stop=row.train_stop,
                train_time_max=row.train_time_max,
                history_time_limit=row.history_time_limit,
                query_times=times[row.score_rows[0] : row.score_rows[1]],
                base_logits=origin["base_logits"][
                    offset_start:offset_stop
                ],
                corrected_logits=origin["corrected_logits"][
                    offset_start:offset_stop
                ],
            )
        )
    return pieces


def _verify_short_sequence_replay(
    directory: Path,
    fold: SourceConditionedFold,
    sequences: SourceSequenceRows,
) -> dict[str, Any]:
    prefix = f"fold-{fold.index}-score-frozen"
    cached = _load_sequence_rows(directory, prefix)
    short_rows = int(fold.score_rows[1] - fold.score_rows[0])
    passed = bool(
        np.array_equal(sequences.items[:short_rows], cached.items)
        and np.array_equal(
            sequences.time_buckets[:short_rows],
            cached.time_buckets,
        )
        and np.array_equal(sequences.lengths[:short_rows], cached.lengths)
    )
    if not passed:
        raise RuntimeError(
            f"origin {fold.index} source sequence replay differs"
        )
    return {
        "passed": True,
        "rows": short_rows,
        "items_exact": True,
        "time_buckets_exact": True,
        "lengths_exact": True,
    }


def _verify_short_logit_replay(
    args: argparse.Namespace,
    fold: SourceConditionedFold,
    combined_start: int,
    base_logits: np.ndarray,
    corrected_logits: np.ndarray,
) -> dict[str, float | bool | int]:
    short_rows = int(fold.score_rows[1] - fold.score_rows[0])
    offset = int(fold.score_rows[0]) - combined_start
    reference_base = np.load(
        args.base_result_dir
        / "folds"
        / "variant-A"
        / f"fold-{fold.index}"
        / "score-logits.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    reference_corrected = np.load(
        args.decoder_result_dir
        / "folds"
        / f"cap-{CAP:.2f}"
        / f"fold-{fold.index}"
        / "score-logits.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    base_error = float(
        np.max(
            np.abs(
                base_logits[offset : offset + short_rows]
                - reference_base
            )
        )
    )
    corrected_error = float(
        np.max(
            np.abs(
                corrected_logits[offset : offset + short_rows]
                - reference_corrected
            )
        )
    )
    replay_tolerance = 5e-6
    passed = bool(
        base_error <= replay_tolerance
        and corrected_error <= replay_tolerance
    )
    if not passed:
        raise RuntimeError(
            f"origin {fold.index} short logits do not replay: "
            f"base={base_error} corrected={corrected_error}"
        )
    return {
        "passed": passed,
        "rows": short_rows,
        "tolerance": replay_tolerance,
        "max_base_error": base_error,
        "max_corrected_error": corrected_error,
    }


def _metrics_report(artifact: Any, slices: tuple[Any, ...]) -> dict[str, Any]:
    slice_metrics = []
    for row in slices:
        horizon_index = HORIZON_NAMES.index(row.horizon)
        start, stop = row.score_rows
        base = artifact.base_logits[horizon_index, start:stop]
        corrected = artifact.corrected_logits[horizon_index, start:stop]
        base_rr = _row_reciprocal_rank(base)
        corrected_rr = _row_reciprocal_rank(corrected)
        reward = corrected_rr - base_rr
        gaps = artifact.gap_days[horizon_index, start:stop]
        slice_metrics.append(
            {
                "horizon": row.horizon,
                "origin_index": row.origin_index,
                "target_fold_index": row.target_fold_index,
                "score_rows": list(row.score_rows),
                "rows": int(stop - start),
                "gap_days_min": float(np.min(gaps)),
                "gap_days_max": float(np.max(gaps)),
                "base_mrr": float(np.mean(base_rr)),
                "corrected_mrr": float(np.mean(corrected_rr)),
                "delta_mrr": float(np.mean(reward)),
                "row_gain_fraction": float(np.mean(reward > 0.0)),
                "row_loss_fraction": float(np.mean(reward < 0.0)),
                "row_unchanged_fraction": float(np.mean(reward == 0.0)),
            }
        )

    horizon_metrics = {}
    gap_ranges = {}
    for horizon_index, horizon in enumerate(HORIZON_NAMES):
        selected = artifact.valid_mask[horizon_index]
        base = artifact.base_logits[horizon_index, selected]
        corrected = artifact.corrected_logits[horizon_index, selected]
        base_rr = _row_reciprocal_rank(base)
        corrected_rr = _row_reciprocal_rank(corrected)
        reward = corrected_rr - base_rr
        gaps = artifact.gap_days[horizon_index, selected]
        horizon_metrics[horizon] = {
            "rows": int(np.sum(selected)),
            "base_mrr": float(np.mean(base_rr)),
            "corrected_mrr": float(np.mean(corrected_rr)),
            "delta_mrr": float(np.mean(reward)),
            "row_gain_fraction": float(np.mean(reward > 0.0)),
            "row_loss_fraction": float(np.mean(reward < 0.0)),
        }
        gap_ranges[horizon] = {
            "min": float(np.min(gaps)),
            "max": float(np.max(gaps)),
        }
    return {
        "slice_metrics": slice_metrics,
        "horizon_metrics": horizon_metrics,
        "gap_day_ranges": gap_ranges,
        "pairwise_residual_disagreement": _pairwise_disagreement(artifact),
    }


def _pairwise_disagreement(artifact: Any) -> list[dict[str, Any]]:
    rows = []
    for left in range(len(HORIZON_NAMES)):
        for right in range(left + 1, len(HORIZON_NAMES)):
            common = (
                artifact.valid_mask[left] & artifact.valid_mask[right]
            )
            left_values = artifact.residuals[left, common].astype(
                np.float64,
                copy=False,
            )
            right_values = artifact.residuals[right, common].astype(
                np.float64,
                copy=False,
            )
            left_flat = left_values.reshape(-1)
            right_flat = right_values.reshape(-1)
            left_norm = float(np.linalg.norm(left_flat))
            right_norm = float(np.linalg.norm(right_flat))
            cosine = (
                float(np.dot(left_flat, right_flat) / (left_norm * right_norm))
                if left_norm > 0.0 and right_norm > 0.0
                else 0.0
            )
            correlation = (
                float(np.corrcoef(left_flat, right_flat)[0, 1])
                if left_flat.size > 1
                else 0.0
            )
            rows.append(
                {
                    "left": HORIZON_NAMES[left],
                    "right": HORIZON_NAMES[right],
                    "common_rows": int(np.sum(common)),
                    "flattened_cosine": cosine,
                    "flattened_correlation": correlation,
                    "mean_row_l2_distance": float(
                        np.mean(
                            np.linalg.norm(
                                left_values - right_values,
                                axis=1,
                            )
                        )
                    ),
                    "max_residual_candidate_agreement": float(
                        np.mean(
                            np.argmax(left_values, axis=1)
                            == np.argmax(right_values, axis=1)
                        )
                    ),
                }
            )
    return rows


def _row_reciprocal_rank(scores: Any) -> np.ndarray:
    values = np.asarray(scores)
    positive = values[:, :1]
    better = np.sum(values > positive, axis=1)
    tied = np.sum(values == positive, axis=1) - 1
    return 1.0 / (1.0 + better + 0.5 * tied)


def _candidate_support(
    history_dst: Any,
    candidate_ids: Any,
    num_items: int,
) -> np.ndarray:
    history = np.asarray(history_dst, dtype=np.int64)
    candidates = np.asarray(candidate_ids)
    if (
        history.ndim != 1
        or candidates.ndim != 2
        or np.any(history < 0)
        or np.any(history > num_items)
        or np.any(candidates < 0)
        or np.any(candidates > num_items)
    ):
        raise ValueError("multi-horizon support IDs are invalid")
    counts = np.bincount(history, minlength=num_items + 1).astype(
        np.float32,
        copy=False,
    )
    return counts[candidates]


def _load_sequence_rows(directory: Path, prefix: str) -> SourceSequenceRows:
    return SourceSequenceRows(
        items=np.load(
            directory / f"{prefix}-items.npy",
            mmap_mode="r",
            allow_pickle=False,
        ),
        time_buckets=np.load(
            directory / f"{prefix}-time-buckets.npy",
            mmap_mode="r",
            allow_pickle=False,
        ),
        lengths=np.load(
            directory / f"{prefix}-lengths.npy",
            mmap_mode="r",
            allow_pickle=False,
        ),
    )


def _input_manifest(
    args: argparse.Namespace,
    context: dict[str, Any],
    by_origin: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    checkpoints = []
    for origin_index, row in sorted(by_origin.items()):
        checkpoints.append(
            {
                "origin_index": origin_index,
                "base_checkpoint": str(
                    row["base_checkpoint"].resolve()
                ),
                "base_checkpoint_sha256": _sha256(
                    row["base_checkpoint"]
                ),
                "decoder_checkpoint": str(
                    row["decoder_checkpoint"].resolve()
                ),
                "decoder_checkpoint_sha256": _sha256(
                    row["decoder_checkpoint"]
                ),
                "source_sequence_replay": row["sequence_replay"],
                "short_logit_replay": row["short_logit_replay"],
                "origin_residual_audit": row["residual_audit"],
                "source_empty_rows": row["source_empty_rows"],
            }
        )
    return {
        "train_csv": str(args.train_csv.resolve()),
        "train_csv_sha256": _sha256(args.train_csv),
        "train_cache_prefix": str(args.train_cache_prefix.resolve()),
        "train_cache_files": {
            key: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for key, path in context["paths"].items()
        },
        "sequence_manifest": str(context["manifest_path"].resolve()),
        "sequence_manifest_sha256": _sha256(context["manifest_path"]),
        "base_result_dir": str(args.base_result_dir.resolve()),
        "decoder_result_dir": str(args.decoder_result_dir.resolve()),
        "checkpoints": checkpoints,
    }


def _configure_device(device: str, seed: int) -> None:
    if device == "cuda":
        if not jt.has_cuda:
            raise RuntimeError("CUDA requested but Jittor has no CUDA")
        jt.flags.use_cuda = 1
    else:
        jt.flags.use_cuda = 0
    jt.set_global_seed(seed)


def _array_record(
    build_path: Path,
    final_path: Path,
    values: np.ndarray,
) -> dict[str, Any]:
    return {
        "path": str(final_path.resolve()),
        "sha256": _sha256(build_path),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "bytes": int(values.nbytes),
    }


def _save_array_atomic(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _verify_existing(output_dir: Path, manifest: dict[str, Any]) -> None:
    if (
        manifest.get("status") != "complete"
        or manifest.get("trainable_frameworks") != ["jittor"]
        or manifest.get("non_jittor_trainable_models") != []
    ):
        raise ValueError("existing multi-horizon manifest is invalid")
    for row in manifest["artifacts"].values():
        path = Path(row["path"])
        if (
            path.parent != output_dir.resolve()
            or not path.exists()
            or _sha256(path) != row["sha256"]
        ):
            raise ValueError(f"existing multi-horizon artifact differs: {path}")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
