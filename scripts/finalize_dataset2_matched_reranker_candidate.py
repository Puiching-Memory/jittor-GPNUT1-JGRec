from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

from jgrec.contest_checkpoint import (
    load_checkpoint_dataset,
    load_checkpoint_metadata,
)
from jgrec.core.io import discover_datasets
from jgrec.rankers.hybrid.fusion_analysis import authorized_setwise_weight
from jgrec.submission import expected_test_rows, validate_submission_file


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize an already-generated matched-reranker package after "
            "validating its checkpoint, CSV files, and selected weight."
        )
    )
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--evaluation-report", required=True, type=Path)
    parser.add_argument("--output-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    started = time.time()
    evaluation = json.loads(args.evaluation_report.read_text(encoding="utf-8"))
    winner = evaluation.get("winner")
    if (
        evaluation.get("status") != "passed"
        or not evaluation.get("gate_passed")
        or not evaluation.get("package_authorized")
        or winner not in {"lightgbm", "setwise"}
    ):
        raise RuntimeError("evaluation does not authorize this package")

    state = load_checkpoint_dataset(args.output_checkpoint, "dataset2")
    stored_lgbm = state.get("lgbm_result")
    if stored_lgbm is None:
        raise RuntimeError("output checkpoint has no Dataset2 LightGBM expert")
    expected_weight = (
        authorized_setwise_weight(evaluation) if winner == "setwise" else None
    )
    stored_weight = float(stored_lgbm.mlp_weight)
    if expected_weight is not None and abs(stored_weight - expected_weight) > 1e-12:
        raise RuntimeError(
            f"checkpoint weight {stored_weight} differs from {expected_weight}"
        )

    metadata = load_checkpoint_metadata(args.output_checkpoint)
    metadata_weight = metadata.get("dataset2_setwise_weight")
    if expected_weight is not None and (
        metadata_weight is None
        or abs(float(metadata_weight) - expected_weight) > 1e-12
    ):
        raise RuntimeError("checkpoint metadata has the wrong Setwise weight")

    datasets = {
        dataset.name: dataset for dataset in discover_datasets(args.data_dir)
    }
    csv_dir = args.output_dir / "csv"
    dataset1_output = csv_dir / "dataset1.csv"
    dataset2_output = csv_dir / "dataset2.csv"
    zip_path = args.output_dir / "result.zip"
    dataset1_rows = expected_test_rows(datasets["dataset1"])
    dataset2_rows = expected_test_rows(datasets["dataset2"])
    validate_submission_file(dataset1_output, expected_rows=dataset1_rows)
    validate_submission_file(dataset2_output, expected_rows=dataset2_rows)
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)

    report = {
        "status": "complete",
        "winner": winner,
        "setwise_weight": expected_weight,
        "stored_checkpoint_blend_weight": stored_weight,
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": _sha256(args.source_checkpoint),
        "evaluation_report": str(args.evaluation_report.resolve()),
        "evaluation_report_sha256": _sha256(args.evaluation_report),
        "offline_gate": evaluation[winner],
        "output_checkpoint": str(args.output_checkpoint.resolve()),
        "output_checkpoint_bytes": args.output_checkpoint.stat().st_size,
        "output_checkpoint_sha256": _sha256(args.output_checkpoint),
        "dataset1_mode": "byte_copy_from_online_champion",
        "dataset1_rows": dataset1_rows,
        "dataset1_sha256": _sha256(dataset1_output),
        "dataset2_mode": f"matched_{winner}_with_rebuilt_final_encoder",
        "dataset2_rows": dataset2_rows,
        "dataset2_sha256": _sha256(dataset2_output),
        "result_zip": str(zip_path.resolve()),
        "result_zip_bytes": zip_path.stat().st_size,
        "result_zip_sha256": _sha256(zip_path),
        "finalize_seconds": time.time() - started,
    }
    _write_json_atomic(args.output_dir / "candidate-report.json", report)
    shutil.copyfile(
        args.evaluation_report,
        args.output_dir / "evaluation-report.json",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
