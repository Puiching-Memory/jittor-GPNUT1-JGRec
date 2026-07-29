from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

A2_INTEGRATION_ID = "listwise_mlp_exact_current_champion_v1"
A2_WEIGHTS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)


@dataclass(frozen=True)
class RollingFoldSpec:
    fold_id: str
    train_rows: tuple[int, int]
    score_rows: tuple[int, int]


A2_ROLLING_FOLDS = (
    RollingFoldSpec("fold-0", (0, 79_909), (79_909, 118_816)),
    RollingFoldSpec("fold-1", (0, 118_816), (118_816, 159_804)),
    RollingFoldSpec("fold-2", (0, 159_804), (159_804, 200_000)),
)


def validate_rolling_folds(
    folds: tuple[RollingFoldSpec, ...],
    *,
    row_count: int,
) -> tuple[RollingFoldSpec, ...]:
    if len(folds) < 3:
        raise ValueError("rolling protocol requires at least three folds")
    previous_score_stop: int | None = None
    previous_train_stop: int | None = None
    fold_ids: set[str] = set()
    for fold in folds:
        if not fold.fold_id or fold.fold_id in fold_ids:
            raise ValueError("rolling fold ids must be unique and non-empty")
        fold_ids.add(fold.fold_id)
        train_start, train_stop = fold.train_rows
        score_start, score_stop = fold.score_rows
        if train_start != 0:
            raise ValueError("rolling training ranges must be expanding prefixes")
        if not 0 < train_stop <= score_start < score_stop <= row_count:
            raise ValueError(f"{fold.fold_id} must train strictly before its score range")
        if previous_train_stop is not None and train_stop <= previous_train_stop:
            raise ValueError("rolling training origins must increase")
        if previous_score_stop is not None and score_start != previous_score_stop:
            raise ValueError("rolling score ranges must be contiguous")
        previous_train_stop = train_stop
        previous_score_stop = score_stop
    return folds


def validate_frozen_a2_weights(
    frozen_config: dict[str, Any],
) -> tuple[float, ...]:
    if frozen_config.get("status") != "frozen_before_any_partial_blend_metric":
        raise ValueError("A2 source config was not frozen before metrics")
    try:
        weights = tuple(float(value) for value in frozen_config["weights"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("A2 source config has invalid weights") from error
    if weights != A2_WEIGHTS:
        raise ValueError(f"weights differ from the predeclared A2 weights {A2_WEIGHTS}")
    return weights


def materialize_fold_candidates(
    *,
    output_dir: Path,
    fold_id: str,
    integration_id: str,
    train_time_max: int,
    score_time_min: int,
    score_time_max: int,
    baseline_scores: np.ndarray,
    auxiliary_scores: np.ndarray,
    candidate_ids: np.ndarray,
    weights: tuple[float, ...],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    if not fold_id:
        raise ValueError("fold_id must be non-empty")
    if not integration_id:
        raise ValueError("integration_id must be non-empty")
    if not train_time_max < score_time_min <= score_time_max:
        raise ValueError("fold score times must be strictly after training")
    baseline = _validated_probabilities(
        baseline_scores,
        label="baseline_scores",
    )
    auxiliary = _validated_probabilities(
        auxiliary_scores,
        label="auxiliary_scores",
    )
    if baseline.shape != auxiliary.shape:
        raise ValueError("baseline and auxiliary score matrices must have identical shapes")
    candidates = np.asarray(candidate_ids)
    if candidates.shape != baseline.shape:
        raise ValueError("candidate_ids shape must match score matrices")
    if not np.issubdtype(candidates.dtype, np.integer):
        raise ValueError("candidate_ids must contain integers")
    validated_weights = _validated_weights(weights)
    candidate_fingerprint = hashlib.sha256(np.ascontiguousarray(candidates).tobytes(order="C")).hexdigest()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.",
        dir=output_dir.parent,
    ) as temporary:
        staging = Path(temporary) / output_dir.name
        staging.mkdir()
        baseline_path = staging / "baseline.npy"
        auxiliary_path = staging / "auxiliary.npy"
        np.save(baseline_path, baseline, allow_pickle=False)
        np.save(auxiliary_path, auxiliary, allow_pickle=False)
        candidate_descriptors: dict[str, dict[str, Any]] = {}
        for weight in validated_weights:
            weight_key = _weight_key(weight)
            candidate_path = staging / f"candidate-w{weight_key}.npy"
            blended = (1.0 - weight) * baseline + weight * auxiliary
            np.save(candidate_path, blended, allow_pickle=False)
            candidate_descriptors[weight_key] = {
                "path": str((output_dir / candidate_path.name).resolve()),
                "sha256": _sha256(candidate_path),
                "integration_id": integration_id,
                "candidate_fingerprint": candidate_fingerprint,
            }
        entry = {
            "fold_id": fold_id,
            "train_time_max": int(train_time_max),
            "score_time_min": int(score_time_min),
            "score_time_max": int(score_time_max),
            "candidate_fingerprint": candidate_fingerprint,
            "baseline": {
                "path": str((output_dir / baseline_path.name).resolve()),
                "sha256": _sha256(baseline_path),
            },
            "auxiliary": {
                "path": str((output_dir / auxiliary_path.name).resolve()),
                "sha256": _sha256(auxiliary_path),
            },
            "candidates": candidate_descriptors,
        }
        _write_json(staging / "fold-score-report.json", entry)
        staging.replace(output_dir)
    return entry


def build_rolling_selection_manifest(
    *,
    integration_id: str,
    fold_entries: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite: {output_path}")
    if len(fold_entries) < 3:
        raise ValueError("selection manifest requires at least three folds")
    for entry in fold_entries:
        for weight, descriptor in entry["candidates"].items():
            if descriptor.get("integration_id") != integration_id:
                raise ValueError(f"{entry['fold_id']} weight {weight} integration_id differs")
    manifest = {
        "protocol": "exact_integrated_rolling_weight_selection_v1",
        "integration_id": integration_id,
        "positive_candidate_column": 0,
        "folds": fold_entries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, manifest, exclusive=True)
    return manifest


def _validated_probabilities(
    scores: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError(f"{label} must be a non-empty 2D matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain only finite values")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{label} must contain probabilities in [0, 1]")
    if not np.allclose(
        values.sum(axis=1),
        1.0,
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError(f"{label} must be row-normalized")
    return values


def _validated_weights(weights: tuple[float, ...]) -> tuple[float, ...]:
    values = tuple(float(value) for value in weights)
    if (
        not values
        or len(values) != len(set(values))
        or any(not np.isfinite(value) or not 0.0 < value <= 1.0 for value in values)
    ):
        raise ValueError("weights must be unique finite values in (0, 1]")
    return values


def _weight_key(weight: float) -> str:
    return format(float(weight), "g")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(
    path: Path,
    payload: dict[str, Any],
    *,
    exclusive: bool = False,
) -> None:
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
