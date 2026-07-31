from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.core.io import read_interactions
from jgrec.rankers.hybrid.cooccur_lift import (
    load_frozen_cooccur_lift_config,
)
from jgrec.rankers.hybrid.cooccur_lift_native import (
    materialize_compact_cooccur_lift,
)
from jgrec.rankers.hybrid.full100_training import (
    validate_joint_cache_reports,
)

TRAIN_SHAPE = (200_000, 100, 63)
VALIDATION_SHAPE = (20_000, 100, 63)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize fresh causal cooccur-lift features for a K512 "
            "200k cache and prove any reused query-aligned sidecars are exact."
        )
    )
    parser.add_argument("--frozen-config", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument("--train-cache-report", required=True, type=Path)
    parser.add_argument(
        "--validation-cache-prefix",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--validation-cache-report",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--reference-train-cache-prefix",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--reference-validation-cache-prefix",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--reference-train-short-none",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--reference-validation-short-none",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--reference-prior-external",
        required=True,
        type=Path,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.time()

    train_report = _read_json(args.train_cache_report)
    validation_report = _read_json(args.validation_cache_report)
    _validate_cache_report(
        train_report,
        expected_shape=TRAIN_SHAPE,
        expected_role="train",
    )
    _validate_cache_report(
        validation_report,
        expected_shape=VALIDATION_SHAPE,
        expected_role="validation",
    )
    joint_lineage = _validate_joint_lineage(
        train_report,
        validation_report,
    )

    train_paths = _cache_paths(args.train_cache_prefix, split="train")
    validation_paths = _cache_paths(
        args.validation_cache_prefix,
        split="val",
    )
    reference_train_paths = _cache_paths(
        args.reference_train_cache_prefix,
        split="train",
    )
    reference_validation_paths = _cache_paths(
        args.reference_validation_cache_prefix,
        split="val",
    )
    _validate_report_artifacts(train_report, train_paths)
    _validate_report_artifacts(validation_report, validation_paths)

    aligned_fields = (
        "candidates",
        "sources",
        "destinations",
        "times",
        "row_indices",
    )
    alignment: dict[str, Any] = {}
    for split, current, reference in (
        ("train", train_paths, reference_train_paths),
        ("validation", validation_paths, reference_validation_paths),
    ):
        for field in aligned_fields:
            current_path = current[field]
            reference_path = reference[field]
            if not _arrays_equal(current_path, reference_path):
                raise ValueError(
                    f"{split} {field} differs from frozen query alignment"
                )
            alignment[f"{split}_{field}"] = {
                "current_sha256": _sha256(current_path),
                "reference_sha256": _sha256(reference_path),
                "exact": True,
            }

    train_short_none = np.load(
        args.reference_train_short_none,
        mmap_mode="r",
        allow_pickle=False,
    )
    validation_short_none = np.load(
        args.reference_validation_short_none,
        mmap_mode="r",
        allow_pickle=False,
    )
    prior_external = np.load(
        args.reference_prior_external,
        mmap_mode="r",
        allow_pickle=False,
    )
    if train_short_none.shape != TRAIN_SHAPE[:2]:
        raise ValueError("reference train short-none shape differs")
    if validation_short_none.shape != VALIDATION_SHAPE[:2]:
        raise ValueError("reference validation short-none shape differs")
    if prior_external.shape != VALIDATION_SHAPE[:2]:
        raise ValueError("reference prior external shape differs")
    if not np.all(np.isfinite(train_short_none)):
        raise ValueError("reference train short-none is non-finite")
    if not np.all(np.isfinite(validation_short_none)):
        raise ValueError("reference validation short-none is non-finite")
    if not np.all(np.isfinite(prior_external)):
        raise ValueError("reference prior external is non-finite")

    features = np.load(
        train_paths["features"],
        mmap_mode="r",
        allow_pickle=False,
    )
    candidates = np.load(
        train_paths["candidates"],
        mmap_mode="r",
        allow_pickle=False,
    )
    sources = np.load(
        train_paths["sources"],
        mmap_mode="r",
        allow_pickle=False,
    )
    destinations = np.load(
        train_paths["destinations"],
        mmap_mode="r",
        allow_pickle=False,
    )
    event_time = np.load(
        train_paths["times"],
        mmap_mode="r",
        allow_pickle=False,
    )
    if features.shape != TRAIN_SHAPE:
        raise ValueError("K512 train feature shape differs")
    if not np.array_equal(candidates[:, 0], destinations):
        raise ValueError("K512 positive candidate differs from destination")

    config = load_frozen_cooccur_lift_config(args.frozen_config)
    interactions = read_interactions(args.train_csv).sort_by_time()
    time_span = int(interactions.time[-1]) - int(interactions.time[0])
    short_window = float(time_span) * config.short_window_ratio
    lift_path = args.output_dir / "lift-features.npy"
    popularity_path = (
        args.output_dir / "positive-dst-causal-popularity.npy"
    )
    native = materialize_compact_cooccur_lift(
        interactions=interactions,
        sources=sources,
        candidates=candidates,
        destinations=destinations,
        event_time=event_time,
        short_window=short_window,
        lift_path=lift_path,
        positive_popularity_path=popularity_path,
        progress_path=args.output_dir / "materialization-progress.json",
        work_dir=args.output_dir / "native-work",
    )
    lift = np.load(lift_path, mmap_mode="r", allow_pickle=False)
    if lift.shape != (*TRAIN_SHAPE[:2], 2):
        raise ValueError("fresh near lift shape differs")
    if not np.all(np.isfinite(lift)):
        raise ValueError("fresh near lift is non-finite")

    report = {
        "schema_version": 1,
        "protocol": "dataset2_k512_near_assets_v1",
        "status": "complete",
        "external_scores_read": False,
        "train_cache_report": str(args.train_cache_report.resolve()),
        "train_cache_report_sha256": _sha256(args.train_cache_report),
        "validation_cache_report": str(
            args.validation_cache_report.resolve()
        ),
        "validation_cache_report_sha256": _sha256(
            args.validation_cache_report
        ),
        "joint_build": train_report["joint_build"],
        "joint_cache_validation": joint_lineage,
        "prediction_limits": train_report["prediction_limits"],
        "query_alignment": alignment,
        "reused_sidecars": {
            "train_short_none": _descriptor(
                args.reference_train_short_none
            ),
            "validation_short_none": _descriptor(
                args.reference_validation_short_none
            ),
            "prior_external": _descriptor(
                args.reference_prior_external
            ),
        },
        "fresh_artifacts": {
            "lift_features": _descriptor(lift_path),
            "positive_popularity": _descriptor(popularity_path),
        },
        "native_materializer": native,
        "realized_short_window": short_window,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.output_dir / "near-assets-report.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def _validate_cache_report(
    report: dict[str, Any],
    *,
    expected_shape: tuple[int, int, int],
    expected_role: str,
) -> None:
    if report.get("status") != "complete":
        raise ValueError(f"{expected_role} cache report is incomplete")
    shape_key = (
        "train_shape" if expected_role == "train" else "validation_shape"
    )
    if tuple(report.get(shape_key, ())) != expected_shape:
        raise ValueError(f"{expected_role} cache shape differs")
    limits = report.get("prediction_limits")
    if (
        not isinstance(limits, dict)
        or limits.get("structure_predict_neighbor_limit") != 512
        or limits.get("source_profile_predict_history_limit") != 512
    ):
        raise ValueError(f"{expected_role} cache is not K512")


def _validate_joint_lineage(
    train_report: dict[str, Any],
    validation_report: dict[str, Any],
) -> dict[str, Any]:
    lineage = validate_joint_cache_reports(
        train_report,
        validation_report,
    )
    if (
        train_report.get("checkpoint_sha256")
        != validation_report.get("checkpoint_sha256")
    ):
        raise ValueError("joint cache checkpoint differs")
    return lineage


def _validate_report_artifacts(
    report: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("cache report lacks artifacts")
    report_names = {
        "features": "features",
        "candidates": "candidates",
        "sources": "src",
        "destinations": "dst",
        "times": "time",
        "row_indices": "row_indices",
    }
    for path_name, report_name in report_names.items():
        descriptor = artifacts.get(report_name)
        path = paths[path_name]
        if (
            not isinstance(descriptor, dict)
            or not path.is_file()
            or Path(str(descriptor.get("path", ""))).resolve()
            != path.resolve()
            or int(descriptor.get("bytes", -1)) != path.stat().st_size
            or descriptor.get("sha256") != _sha256(path)
        ):
            raise ValueError(f"cache artifact differs: {path_name}")


def _arrays_equal(first_path: Path, second_path: Path) -> bool:
    first = np.load(first_path, mmap_mode="r", allow_pickle=False)
    second = np.load(second_path, mmap_mode="r", allow_pickle=False)
    if first.shape != second.shape or first.dtype != second.dtype:
        return False
    rows = int(first.shape[0]) if first.ndim else 1
    for start in range(0, rows, 4096):
        stop = min(start + 4096, rows)
        if not np.array_equal(first[start:stop], second[start:stop]):
            return False
    return True


def _cache_paths(prefix: Path, *, split: str) -> dict[str, Path]:
    base = str(prefix)
    return {
        "features": Path(f"{base}.{split}.npy"),
        "candidates": Path(f"{base}.{split}-candidates.npy"),
        "sources": Path(f"{base}.{split}-src.npy"),
        "destinations": Path(f"{base}.{split}-dst.npy"),
        "times": Path(f"{base}.{split}-time.npy"),
        "row_indices": Path(f"{base}.{split}-row-indices.npy"),
    }


def _descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
