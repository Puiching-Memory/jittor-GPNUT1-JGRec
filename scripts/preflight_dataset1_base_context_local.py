from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, BinaryIO

EXPECTED_TRAIN_FEATURE_SHA256 = (
    "a8f4b5d71dedd1b5aa89a9f0c40e1501afc882929d036ddaaa73076e5be6a6ef"
)
EXPECTED_TRAIN_TIME_SHA256 = (
    "4461a25bb14e859de2c927f428a28a651cf4a3586e51b8f4103cd52de082be6a"
)
EXPECTED_EXTERNAL_FEATURE_SHA256 = (
    "43f32c3430eec82180314f889cdfe94b9a8fdb9dc3fd338f7b22f3fa44ad6906"
)
EXPECTED_EXTERNAL_TIME_SHA256 = (
    "2f5a68329f5e75f26acd77bf22e578bb47f51261aaf3c30ee4dca7aefbd89f96"
)
EXPECTED_CHAMPION_ZIP_SHA256 = (
    "085da277f6f20429a2f9e4872438de2f7dca672eea41ba5a8e7fe1d99fb50730"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit all locally available metadata and immutable package "
            "inputs without opening external score arrays."
        )
    )
    parser.add_argument("--rolling-manifest", required=True, type=Path)
    parser.add_argument("--train-cache-report", required=True, type=Path)
    parser.add_argument(
        "--external-cache-report",
        required=True,
        type=Path,
    )
    parser.add_argument("--champion-result-zip", required=True, type=Path)
    parser.add_argument(
        "--champion-verification-report",
        required=True,
        type=Path,
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    rolling = _read_json(args.rolling_manifest)
    train = _read_json(args.train_cache_report)
    external = _read_json(args.external_cache_report)
    champion_verification = _read_json(
        args.champion_verification_report
    )
    folds = _validated_folds(rolling)
    _validate_cache_reports(rolling, train, external)
    _require_hash(
        args.champion_result_zip,
        EXPECTED_CHAMPION_ZIP_SHA256,
        "tie-safe champion package",
    )
    if (
        champion_verification.get("sha256")
        != EXPECTED_CHAMPION_ZIP_SHA256
        or champion_verification.get("datasets", {})
        .get("dataset1", {})
        .get("rows_with_exact_ties")
        != 0
        or champion_verification.get("datasets", {})
        .get("dataset2", {})
        .get("rows_with_exact_ties")
        != 0
    ):
        raise ValueError("tie-safe champion verification report is invalid")
    zip_members = _inspect_zip(args.champion_result_zip)
    report = {
        "status": "ready_for_remote_rolling",
        "external_arrays_opened": False,
        "external_metrics_read": False,
        "package_generated": False,
        "rolling_manifest": str(args.rolling_manifest.resolve()),
        "rolling_manifest_sha256": _sha256(args.rolling_manifest),
        "train_cache_report": str(
            args.train_cache_report.resolve()
        ),
        "train_cache_report_sha256": _sha256(
            args.train_cache_report
        ),
        "external_cache_report": str(
            args.external_cache_report.resolve()
        ),
        "external_cache_report_sha256": _sha256(
            args.external_cache_report
        ),
        "folds": folds,
        "selection_fold_count": 3,
        "rolling_train_feature_sha256": EXPECTED_TRAIN_FEATURE_SHA256,
        "rolling_train_time_sha256": EXPECTED_TRAIN_TIME_SHA256,
        "external_feature_sha256": EXPECTED_EXTERNAL_FEATURE_SHA256,
        "external_time_sha256": EXPECTED_EXTERNAL_TIME_SHA256,
        "external_time_bounds": [
            int(external["split"]["validation_time_min"]),
            int(external["split"]["validation_time_max"]),
        ],
        "external_gap": {
            "training_time_max": int(
                rolling["folds"][-1]["time_boundary"][
                    "score_time_max"
                ]
            ),
            "score_time_min": int(
                external["split"]["validation_time_min"]
            ),
            "actual_gap": int(
                external["split"]["validation_time_min"]
                - rolling["folds"][-1]["time_boundary"][
                    "score_time_max"
                ]
            ),
        },
        "champion_result_zip": str(
            args.champion_result_zip.resolve()
        ),
        "champion_result_zip_sha256": EXPECTED_CHAMPION_ZIP_SHA256,
        "champion_verification_report_sha256": _sha256(
            args.champion_verification_report
        ),
        "champion_zip_members": zip_members,
        "local_limitations": [
            "5.04GB rolling feature array is remote-only",
            "504MB external feature array is remote-only",
            "current approximately 5GB contest checkpoint is remote-only",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _validated_folds(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    protocol = manifest.get("protocol", {})
    folds = manifest.get("folds", [])
    if (
        manifest.get("status") != "frozen_before_training"
        or manifest.get("dataset_name") != "dataset1"
        or int(protocol.get("train_window_rows", -1)) != 100_000
        or int(protocol.get("score_rows", -1)) != 25_000
        or int(protocol.get("selection_fold_count", -1)) != 3
        or len(folds) != 4
    ):
        raise ValueError("rolling manifest contract differs")
    summaries: list[dict[str, Any]] = []
    previous_score_max: int | None = None
    for index, fold in enumerate(folds):
        if int(fold["index"]) != index:
            raise ValueError("rolling fold indices are not contiguous")
        train_start, train_stop = (
            int(value) for value in fold["train_rows"]
        )
        score_start, score_stop = (
            int(value) for value in fold["score_rows"]
        )
        boundary = fold["time_boundary"]
        train_max = int(boundary["train_time_max"])
        score_min = int(boundary["score_time_min"])
        score_max = int(boundary["score_time_max"])
        if (
            train_stop - train_start != 100_000
            or score_stop - score_start != 25_000
            or train_stop != score_start
            or not train_max < score_min <= score_max
            or (
                previous_score_max is not None
                and score_min <= previous_score_max
            )
        ):
            raise ValueError(f"rolling fold {index} is not causal")
        previous_score_max = score_max
        summaries.append(
            {
                "index": index,
                "role": str(fold["role"]),
                "train_rows": [train_start, train_stop],
                "score_rows": [score_start, score_stop],
                "train_time_max": train_max,
                "score_time_min": score_min,
                "score_time_max": score_max,
            }
        )
    return summaries


def _validate_cache_reports(
    rolling: dict[str, Any],
    train: dict[str, Any],
    external: dict[str, Any],
) -> None:
    if (
        train.get("status") != "complete"
        or train.get("dataset_name") != "dataset1"
        or train.get("train_shape") != [200_000, 100, 63]
        or train["artifacts"]["features"]["sha256"]
        != EXPECTED_TRAIN_FEATURE_SHA256
        or train["artifacts"]["time"]["sha256"]
        != EXPECTED_TRAIN_TIME_SHA256
        or rolling["source"]["features_sha256"]
        != EXPECTED_TRAIN_FEATURE_SHA256
        or rolling["source"]["times_sha256"]
        != EXPECTED_TRAIN_TIME_SHA256
    ):
        raise ValueError("rolling cache metadata differs")
    if (
        external.get("status") != "complete"
        or external.get("dataset_name") != "dataset1"
        or external.get("validation_shape") != [20_000, 100, 63]
        or external["artifacts"]["features"]["sha256"]
        != EXPECTED_EXTERNAL_FEATURE_SHA256
        or external["artifacts"]["time"]["sha256"]
        != EXPECTED_EXTERNAL_TIME_SHA256
        or external.get("train_feature_sha256")
        != EXPECTED_TRAIN_FEATURE_SHA256
        or external.get("joint_build", {}).get("id")
        != train.get("joint_build", {}).get("id")
    ):
        raise ValueError("external cache metadata differs")
    training_max = int(
        rolling["folds"][-1]["time_boundary"]["score_time_max"]
    )
    if int(external["split"]["validation_time_min"]) <= training_max:
        raise ValueError("external holdout does not follow rolling data")


def _inspect_zip(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(path, "r") as archive:
        if archive.testzip() is not None:
            raise ValueError("champion ZIP failed CRC validation")
        if set(archive.namelist()) != {"dataset1.csv", "dataset2.csv"}:
            raise ValueError("champion ZIP has unexpected members")
        for name in sorted(archive.namelist()):
            with archive.open(name, "r") as handle:
                member_hash, rows, size = _hash_rows(handle)
            output[name] = {
                "sha256": member_hash,
                "rows": rows,
                "uncompressed_bytes": size,
            }
    if (
        output["dataset1.csv"]["rows"] != 61_051
        or output["dataset2.csv"]["rows"] != 153_420
    ):
        raise ValueError("champion ZIP row counts differ")
    return output


def _hash_rows(handle: BinaryIO) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    rows = 0
    size = 0
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
        rows += block.count(b"\n")
        size += len(block)
    return digest.hexdigest(), rows, size


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch: actual={actual} expected={expected}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
