from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from jgrec.rankers.hybrid.partial_listwise_blend import (
    blend_partial_listwise,
)


def ranking_metric_panel(scores: np.ndarray) -> dict[str, float]:
    values = _validated_scores(scores, label="scores")
    ranks = 1 + np.sum(values[:, 1:] > values[:, :1], axis=1)
    return {
        "mrr": float(np.mean(1.0 / ranks)),
        "hit_at_1": float(np.mean(ranks <= 1)),
        "hit_at_3": float(np.mean(ranks <= 3)),
        "hit_at_5": float(np.mean(ranks <= 5)),
        "hit_at_10": float(np.mean(ranks <= 10)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
    }


def materialize_submission_member_scores(
    *,
    source_zip: Path,
    member_name: str,
    output_path: Path,
    expected_zip_sha256: str,
    expected_member_sha256: str,
    expected_shape: tuple[int, int],
) -> dict[str, Any]:
    source_zip = Path(source_zip)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite: {output_path}")
    _require_hash(source_zip, expected_zip_sha256, label="source ZIP")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_path.stem}.",
        dir=output_path.parent,
    ) as temporary:
        staging = Path(temporary)
        member_path = staging / member_name
        member_path.parent.mkdir(parents=True, exist_ok=True)
        _extract_unique_member(source_zip, member_name, member_path)
        _require_hash(
            member_path,
            expected_member_sha256,
            label=f"{member_name} source member",
        )
        _validate_csv_matrix(
            member_path,
            expected_rows=int(expected_shape[0]),
            expected_columns=int(expected_shape[1]),
        )
        scores = np.loadtxt(
            member_path,
            delimiter=",",
            dtype=np.float64,
            ndmin=2,
        )
        _require_probability_matrix(
            scores,
            label=f"{member_name} source scores",
            require_row_normalized=True,
        )
        staged_output = staging / output_path.name
        np.save(staged_output, scores, allow_pickle=False)
        staged_output.replace(output_path)
    return {
        "status": "passed",
        "source_zip": str(source_zip.resolve()),
        "source_zip_sha256": _sha256(source_zip),
        "source_member": member_name,
        "source_member_sha256": expected_member_sha256,
        "output_path": str(output_path.resolve()),
        "output_sha256": _sha256(output_path),
        "shape": list(expected_shape),
        "maximum_row_sum_error": float(
            np.max(np.abs(scores.sum(axis=1) - 1.0))
        ),
    }


def build_partial_listwise_submission(
    *,
    champion_zip: Path,
    expert_scores_path: Path,
    output_dir: Path,
    auxiliary_weight: float,
    expected_rows: Mapping[str, int],
    expected_columns: int,
    expected_champion_zip_sha256: str,
    expected_dataset1_sha256: str,
    expected_dataset2_sha256: str,
    expert_name: str,
    expert_model_sha256: str,
    candidate_manifest_sha256: str,
    selection_lock_sha256: str,
    expert_score_transform: str = "normalized_descending_midrank",
    expert_source_sha256: str | None = None,
    dataset2_mode: str = "partial_listwise_two_tower_blend",
) -> dict[str, Any]:
    champion_zip = Path(champion_zip)
    expert_scores_path = Path(expert_scores_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    if not expert_name:
        raise ValueError("expert_name must not be empty")
    if not dataset2_mode:
        raise ValueError("dataset2_mode must not be empty")
    _require_hash(
        champion_zip,
        expected_champion_zip_sha256,
        label="champion ZIP",
    )
    expert = np.load(expert_scores_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (
        int(expected_rows["dataset2"]),
        int(expected_columns),
    )
    if expert.shape != expected_shape:
        raise ValueError(
            f"expert scores have shape {expert.shape}, expected {expected_shape}"
        )
    _require_probability_matrix(
        expert,
        label="expert scores",
        require_row_normalized=True,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.",
        dir=output_dir.parent,
    ) as temporary:
        staging = Path(temporary)
        csv_dir = staging / "csv"
        csv_dir.mkdir()
        dataset1_path = csv_dir / "dataset1.csv"
        champion_dataset2_path = staging / "champion-dataset2.csv"
        dataset2_path = csv_dir / "dataset2.csv"
        _extract_unique_member(
            champion_zip,
            "dataset1.csv",
            dataset1_path,
        )
        _extract_unique_member(
            champion_zip,
            "dataset2.csv",
            champion_dataset2_path,
        )
        _require_hash(
            dataset1_path,
            expected_dataset1_sha256,
            label="Dataset1 champion member",
        )
        _require_hash(
            champion_dataset2_path,
            expected_dataset2_sha256,
            label="Dataset2 champion member",
        )
        _validate_csv_matrix(
            dataset1_path,
            expected_rows=int(expected_rows["dataset1"]),
            expected_columns=expected_columns,
        )
        _validate_csv_matrix(
            champion_dataset2_path,
            expected_rows=int(expected_rows["dataset2"]),
            expected_columns=expected_columns,
        )

        champion = np.loadtxt(
            champion_dataset2_path,
            delimiter=",",
            dtype=np.float64,
            ndmin=2,
        )
        _require_probability_matrix(
            champion,
            label="champion Dataset2 scores",
            require_row_normalized=False,
        )
        blended = blend_partial_listwise(
            champion,
            expert,
            auxiliary_weight=auxiliary_weight,
        )
        np.savetxt(dataset2_path, blended, delimiter=",", fmt="%.8f")
        _validate_csv_matrix(
            dataset2_path,
            expected_rows=int(expected_rows["dataset2"]),
            expected_columns=expected_columns,
        )
        persisted = np.loadtxt(
            dataset2_path,
            delimiter=",",
            dtype=np.float64,
            ndmin=2,
        )
        maximum_rounding_error = float(
            np.max(np.abs(persisted - blended))
        )
        if maximum_rounding_error > 5.000001e-9:
            raise RuntimeError(
                "persisted Dataset2 scores differ from the frozen blend "
                f"formula: max_error={maximum_rounding_error}"
            )

        result_zip = staging / "result.zip"
        with zipfile.ZipFile(
            result_zip,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.write(dataset1_path, arcname="dataset1.csv")
            archive.write(dataset2_path, arcname="dataset2.csv")
        with zipfile.ZipFile(result_zip) as archive:
            if archive.namelist() != ["dataset1.csv", "dataset2.csv"]:
                raise RuntimeError("candidate ZIP has unexpected members")
            if (
                _sha256_stream(archive.open("dataset1.csv"))
                != expected_dataset1_sha256
            ):
                raise RuntimeError(
                    "candidate ZIP did not preserve Dataset1 bytes"
                )

        weight = float(auxiliary_weight)
        report: dict[str, Any] = {
            "status": "online_candidate",
            "submission_authorized_by_user": True,
            "promotion_authorized": False,
            "promotion_threshold": 1.3557002251184347,
            "champion": {
                "zip": str(champion_zip.resolve()),
                "zip_sha256": _sha256(champion_zip),
            },
            "dataset1": {
                "mode": "byte_identical_champion_member",
                "rows": int(expected_rows["dataset1"]),
                "columns": int(expected_columns),
                "sha256": _sha256(dataset1_path),
            },
            "dataset2": {
                "mode": dataset2_mode,
                "rows": int(expected_rows["dataset2"]),
                "columns": int(expected_columns),
                "auxiliary_weight": weight,
                "formula": (
                    f"candidate = {1.0 - weight:.2f} * champion "
                    f"+ {weight:.2f} * expert"
                ),
                "champion_member_sha256": expected_dataset2_sha256,
                "sha256": _sha256(dataset2_path),
                "maximum_csv_rounding_error": maximum_rounding_error,
            },
            "expert": {
                "name": expert_name,
                "scores_path": str(expert_scores_path.resolve()),
                "scores_sha256": _sha256(expert_scores_path),
                "model_sha256": expert_model_sha256,
                "source_artifact_sha256": expert_source_sha256,
                "score_transform": expert_score_transform,
                "candidate_manifest_sha256": candidate_manifest_sha256,
                "selection_lock_sha256": selection_lock_sha256,
            },
            "result_zip": str((output_dir / "result.zip").resolve()),
            "result_zip_bytes": result_zip.stat().st_size,
            "result_zip_sha256": _sha256(result_zip),
            "zip_members": ["dataset1.csv", "dataset2.csv"],
        }
        (staging / "candidate-report.json").write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        champion_dataset2_path.unlink()
        staging.replace(output_dir)
    return report


def _extract_unique_member(
    source_zip: Path,
    member_name: str,
    output_path: Path,
) -> None:
    with zipfile.ZipFile(source_zip) as archive:
        matches = [
            info
            for info in archive.infolist()
            if info.filename == member_name and not info.is_dir()
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{source_zip} must contain exactly one {member_name}"
            )
        with archive.open(matches[0]) as source, output_path.open("xb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)


def _validate_csv_matrix(
    path: Path,
    *,
    expected_rows: int,
    expected_columns: int,
) -> None:
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        rows = 0
        for rows, row in enumerate(reader, start=1):
            if len(row) != expected_columns:
                raise ValueError(
                    f"{path}:{rows} has {len(row)} columns, "
                    f"expected {expected_columns}"
                )
            values = np.asarray(row, dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{path}:{rows} contains non-finite values")
            if np.any((values < 0.0) | (values > 1.0)):
                raise ValueError(
                    f"{path}:{rows} contains probability outside [0, 1]"
                )
    if rows != expected_rows:
        raise ValueError(f"{path} has {rows} rows, expected {expected_rows}")


def _require_probability_matrix(
    values: np.ndarray,
    *,
    label: str,
    require_row_normalized: bool,
) -> None:
    matrix = _validated_scores(values, label=label)
    if np.any((matrix < 0.0) | (matrix > 1.0)):
        raise ValueError(f"{label} contains probability outside [0, 1]")
    if require_row_normalized and not np.allclose(
        matrix.sum(axis=1),
        1.0,
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError(f"{label} must be row-normalized")


def _validated_scores(scores: np.ndarray, *, label: str) -> np.ndarray:
    values = np.asarray(scores)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError(f"{label} must be a non-empty 2D candidate matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain only finite values")
    return values


def _require_hash(path: Path, expected: str, *, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch: actual={actual} expected={expected}"
        )


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256_stream(handle)


def _sha256_stream(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()
