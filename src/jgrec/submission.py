from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .core.io import count_csv_data_rows
from .core.types import DatasetPaths, DatasetResult, TrainingReport


def write_zip(results: list[DatasetResult], zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for result in results:
            zf.write(result.output_path, arcname=result.output_path.name)


def validate_submission_file(csv_path: Path, expected_rows: int | None = None) -> None:
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        first_row = next(reader, None)
    rows = 0 if first_row is None else count_csv_data_rows(csv_path) + 1
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"{csv_path} has {rows} rows, expected {expected_rows}")
    if rows == 0:
        return
    if len(first_row) != 100:
        raise ValueError(f"{csv_path}:1 has {len(first_row)} columns, expected 100")

    try:
        data = np.loadtxt(csv_path, delimiter=",", dtype=np.float64, ndmin=2)
    except ValueError as exc:
        message = str(exc)
        if "the number of columns changed" in message:
            raise ValueError(f"{csv_path} has inconsistent column count") from exc
        raise
    if data.shape != (rows, 100):
        if data.shape[1] != 100:
            raise ValueError(f"{csv_path} has {data.shape[1]} columns, expected 100")
        raise ValueError(f"{csv_path} has {data.shape[0]} rows, expected {rows}")
    if np.any((data < 0.0) | (data > 1.0)):
        raise ValueError(f"{csv_path} contains probability outside [0, 1]")


def expected_test_rows(dataset: DatasetPaths) -> int:
    return count_csv_data_rows(dataset.test_path)


def compose_submission_package(
    *,
    dataset1_source_zip: Path,
    dataset2_source_zip: Path,
    output_dir: Path,
    expected_rows: Mapping[str, int],
    expected_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Compose two exact CSV members into one validated submission package."""

    csv_dir = output_dir / "csv"
    zip_path = output_dir / "result.zip"
    report_path = output_dir / "composition-report.json"
    outputs = {
        "dataset1": csv_dir / "dataset1.csv",
        "dataset2": csv_dir / "dataset2.csv",
    }
    for target in (*outputs.values(), zip_path, report_path):
        if target.exists():
            raise FileExistsError(
                f"refusing to overwrite existing composition artifact: {target}"
            )

    sources = {
        "dataset1": Path(dataset1_source_zip),
        "dataset2": Path(dataset2_source_zip),
    }
    source_member_hashes: dict[str, str] = {}
    for dataset, source_zip in sources.items():
        member_name = f"{dataset}.csv"
        member_hash = _zip_member_sha256(source_zip, member_name)
        expected_hash = str(expected_sha256[dataset])
        if member_hash != expected_hash:
            raise ValueError(
                f"{source_zip}:{member_name} has sha256 {member_hash}, "
                f"expected {expected_hash}"
            )
        source_member_hashes[dataset] = member_hash

    csv_dir.mkdir(parents=True, exist_ok=True)
    results: list[DatasetResult] = []
    report: dict[str, Any] = {}
    for dataset, source_zip in sources.items():
        member_name = f"{dataset}.csv"
        output_path = outputs[dataset]
        _extract_unique_zip_member(source_zip, member_name, output_path)
        row_count = int(expected_rows[dataset])
        validate_submission_file(output_path, expected_rows=row_count)
        output_hash = _sha256(output_path)
        if output_hash != source_member_hashes[dataset]:
            raise RuntimeError(
                f"materialized {dataset} CSV hash differs from its source member"
            )
        results.append(
            DatasetResult(
                name=dataset,
                rows=row_count,
                output_path=output_path,
                training_report=TrainingReport(),
            )
        )
        report[dataset] = {
            "source_zip": str(source_zip.resolve()),
            "source_zip_sha256": _sha256(source_zip),
            "source_member": member_name,
            "source_member_sha256": source_member_hashes[dataset],
            "output_csv": str(output_path.resolve()),
            "rows": row_count,
            "sha256": output_hash,
        }

    write_zip(results, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()
        expected_members = ["dataset1.csv", "dataset2.csv"]
        if members != expected_members:
            raise RuntimeError(
                f"composed ZIP members differ: {members} != {expected_members}"
            )
        for dataset in sources:
            member_hash = _zip_member_sha256(zip_path, f"{dataset}.csv")
            if member_hash != source_member_hashes[dataset]:
                raise RuntimeError(
                    f"composed ZIP {dataset} member hash differs from its source"
                )

    report.update(
        {
            "result_zip": str(zip_path.resolve()),
            "result_zip_bytes": zip_path.stat().st_size,
            "result_zip_sha256": _sha256(zip_path),
            "zip_members": ["dataset1.csv", "dataset2.csv"],
            "status": "complete",
        }
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _extract_unique_zip_member(
    source_zip: Path,
    member_name: str,
    output_path: Path,
) -> None:
    with zipfile.ZipFile(source_zip) as archive:
        matches = [
            info for info in archive.infolist() if info.filename == member_name
        ]
        if len(matches) != 1 or matches[0].is_dir():
            raise ValueError(
                f"{source_zip} must contain exactly one file named {member_name}"
            )
        with archive.open(matches[0]) as source, output_path.open("xb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)


def _zip_member_sha256(source_zip: Path, member_name: str) -> str:
    with zipfile.ZipFile(source_zip) as archive:
        matches = [
            info for info in archive.infolist() if info.filename == member_name
        ]
        if len(matches) != 1 or matches[0].is_dir():
            raise ValueError(
                f"{source_zip} must contain exactly one file named {member_name}"
            )
        with archive.open(matches[0]) as handle:
            return _sha256_stream(handle)


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256_stream(handle)


def _sha256_stream(handle: Any) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()
