from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from jgrec.core.cuda import require_jittor_cuda
from jgrec.rankers.hybrid.rolling_origin import (
    RollingOriginFold,
    passes_rolling_origin_gate,
)
from jgrec.rankers.hybrid.time_ramp import apply_time_ramp
from train_select_dataset1_rolling_origin_setwise import (
    _mrr,
    _validate_manifest,
    train_and_score_fold,
)

MINIMUM_OVERALL_MEAN_DELTA = 0.0002


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True, type=Path)
    parser.add_argument("--selection-report", required=True, type=Path)
    parser.add_argument(
        "--selection-report-sha256",
        required=True,
        type=Path,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    manifest_hash = _require_hash_sidecar(
        args.manifest,
        args.manifest_sha256,
        label="manifest",
    )
    selection_hash = _require_hash_sidecar(
        args.selection_report,
        args.selection_report_sha256,
        label="selection report",
    )
    manifest = _read_json(args.manifest)
    selection = _read_json(args.selection_report)
    _validate_manifest(manifest)
    if (
        not selection.get("gate_passed")
        or not selection.get("gate_fold_unlocked")
        or selection.get("gate_fold_metrics_read")
        or selection.get("manifest_sha256") != manifest_hash
    ):
        raise RuntimeError(
            "selection report did not authorize the rolling gate"
        )
    selected_name = selection["selection"]["selected_name"]
    selected_power = selection.get("selected_power")
    if (
        selected_name is None
        or selected_power is None
        or selected_name != f"gamma_{float(selected_power):.1f}"
    ):
        raise ValueError("selection report has no valid locked power")
    selected_trial = next(
        trial
        for trial in selection["selection"]["trials"]
        if trial["name"] == selected_name
    )
    gate_payloads = [
        payload
        for payload in manifest["folds"]
        if payload["role"] == "gate"
    ]
    if len(gate_payloads) != 1:
        raise ValueError("frozen protocol requires one gate fold")
    payload = gate_payloads[0]
    gate_fold = RollingOriginFold(
        index=int(payload["index"]),
        train_rows=tuple(int(value) for value in payload["train_rows"]),
        score_rows=tuple(int(value) for value in payload["score_rows"]),
        role="gate",
    )

    feature_path = Path(manifest["source"]["features"])
    time_path = Path(manifest["source"]["times"])
    feature_hash_before = _sha256(feature_path)
    time_hash_before = _sha256(time_path)
    if (
        feature_hash_before != manifest["source"]["features_sha256"]
        or time_hash_before != manifest["source"]["times_sha256"]
    ):
        raise ValueError("rolling-origin gate source hash differs")

    require_jittor_cuda(jt)
    args.output_dir.mkdir(parents=True)
    frozen = {
        "status": "frozen_before_forward_training",
        "manifest_sha256": manifest_hash,
        "selection_report_sha256": selection_hash,
        "selected_name": selected_name,
        "selected_power": float(selected_power),
        "gate_fold": gate_fold.index,
        "minimum_overall_mean_delta": (
            MINIMUM_OVERALL_MEAN_DELTA
        ),
        "selection_reopened": False,
    }
    _write_json_atomic(args.output_dir / "frozen-config.json", frozen)
    print(json.dumps(frozen, indent=2), flush=True)

    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    times = np.load(time_path, mmap_mode="r", allow_pickle=False)
    fold_report, raw_scores, setwise_scores = train_and_score_fold(
        features=features,
        times=times,
        fold=gate_fold,
        output_dir=args.output_dir,
        training=manifest["protocol"]["head_training"],
    )
    score_start, score_stop = gate_fold.score_rows
    candidate = apply_time_ramp(
        raw_scores,
        setwise_scores,
        times[score_start:score_stop],
        power=float(selected_power),
    )
    baseline_forward_mrr = _mrr(raw_scores)
    candidate_forward_mrr = _mrr(candidate)
    gate = passes_rolling_origin_gate(
        selection_fold_deltas=tuple(
            float(value)
            for value in selected_trial["fold_deltas"]
        ),
        baseline_forward_mrr=baseline_forward_mrr,
        candidate_forward_mrr=candidate_forward_mrr,
        minimum_overall_mean_delta=MINIMUM_OVERALL_MEAN_DELTA,
    )
    candidate_path = args.output_dir / (
        f"fold-{gate_fold.index:02d}-locked-ramp.npy"
    )
    np.save(
        candidate_path,
        np.asarray(candidate, dtype=np.float32),
    )
    source_hashes_unchanged = bool(
        _sha256(feature_path) == feature_hash_before
        and _sha256(time_path) == time_hash_before
    )
    if not source_hashes_unchanged:
        raise RuntimeError("rolling-origin source artifacts changed")
    report = {
        "status": "passed" if gate.passed else "rejected",
        "gate_passed": gate.passed,
        "level2_full_pipeline_authorized": gate.passed,
        "package_authorized": False,
        "selection_reopened": False,
        "manifest_sha256": manifest_hash,
        "selection_report_sha256": selection_hash,
        "selected_name": selected_name,
        "selected_power": float(selected_power),
        "gate": asdict(gate),
        "gate_fold": {
            **fold_report,
            "locked_ramp_mrr": candidate_forward_mrr,
            "locked_ramp_delta_vs_raw": (
                candidate_forward_mrr - baseline_forward_mrr
            ),
            "locked_ramp_sha256": _sha256(candidate_path),
        },
        "source_hashes_unchanged": source_hashes_unchanged,
        "decision": (
            "continue_to_level2_full_pipeline_rolling_origin"
            if gate.passed
            else "stop_before_level2"
        ),
    }
    _write_json_atomic(args.output_dir / "evaluation-report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if gate.passed else 2


def _require_hash_sidecar(
    path: Path,
    sidecar: Path,
    *,
    label: str,
) -> str:
    expected = sidecar.read_text(encoding="ascii").split()[0]
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} hash differs from sidecar")
    return actual


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
