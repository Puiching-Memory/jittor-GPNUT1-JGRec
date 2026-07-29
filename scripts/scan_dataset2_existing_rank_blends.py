from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.core.memory import release_memory
from jgrec.rankers.hybrid.fusion import build_fusion_from_state, predict_logits
from jgrec.rankers.hybrid.fusion_analysis import (
    ranking_mrr_three_slices,
    scan_rank_blend_on_prefix,
)
from jgrec.rankers.hybrid.fusion_lgbm import predict_logits_lgbm
from jgrec.rankers.hybrid.oof_hard_negatives import passes_temporal_mrr_gate

METHODS = ("probability", "row_zscore", "rank_percentile")
EXPECTED_ROWS = 20_000
EXPECTED_CANDIDATES = 100
EXPECTED_FEATURES = 63
SELECTION_STOP = 13_334


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scan frozen Dataset2 rank blends on one champion validation tensor, "
            "selecting on the first two chronological thirds."
        )
    )
    parser.add_argument("--champion-checkpoint", required=True, type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--val-features", required=True, type=Path)
    parser.add_argument("--full100-model", required=True, type=Path)
    parser.add_argument("--matched32-model", required=True, type=Path)
    parser.add_argument("--oof-model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--expected-champion-mrr",
        type=float,
        default=0.5428303297309955,
    )
    parser.add_argument("--min-full-delta", type=float, default=0.001)
    parser.add_argument("--external-mlp-weight", type=float, default=0.07)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    frozen_path = args.output_dir / "frozen-config.json"
    report_path = args.output_dir / "rank-blend-report.json"
    val_features = np.load(args.val_features, mmap_mode="r", allow_pickle=False)
    expected_shape = (EXPECTED_ROWS, EXPECTED_CANDIDATES, EXPECTED_FEATURES)
    if val_features.shape != expected_shape:
        raise ValueError(
            f"champion validation tensor shape mismatch: {val_features.shape}"
        )
    if not 0.0 <= args.external_mlp_weight <= 1.0:
        raise ValueError("external MLP weight must be between zero and one")

    frozen = {
        "status": "frozen_before_scoring",
        "champion_checkpoint": str(args.champion_checkpoint.resolve()),
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "validation_tensor": str(args.val_features.resolve()),
        "validation_tensor_sha256": _sha256(args.val_features),
        "validation_shape": list(val_features.shape),
        "alignment": (
            "Every included model is evaluated on the exact same champion "
            "validation feature tensor; models requiring different learned "
            "tower features are excluded."
        ),
        "excluded": {
            "twotower200k_exploratory": (
                "different learned tower feature tensor and no candidate-ID "
                "sidecar for the historical validation caches"
            )
        },
        "methods": list(METHODS),
        "weights": {
            "start": 0.0,
            "stop": 1.0,
            "step": 0.01,
            "reference": "current champion",
        },
        "selection_rows": [0, SELECTION_STOP],
        "final_rows": [SELECTION_STOP, EXPECTED_ROWS],
        "selection_uses_final_rows": False,
        "gate": {
            "min_full_delta": args.min_full_delta,
            "all_three_slices_non_decreasing": True,
        },
        "external_mlp_weight": args.external_mlp_weight,
        "alternates": {
            "source_checkpoint_stored": str(args.source_checkpoint.resolve()),
            "full100_fixed007": str(args.full100_model.resolve()),
            "matched32_fixed007": str(args.matched32_model.resolve()),
            "oof_hardneg_fixed007": str(args.oof_model.resolve()),
        },
    }
    _write_json_atomic(frozen_path, frozen)
    print(json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True), flush=True)

    (
        champion_mlp,
        champion,
        feature_names,
        feature_indices,
    ) = _score_champion(args.champion_checkpoint, val_features)
    baseline = ranking_mrr_three_slices(champion)
    if abs(float(baseline["full"]) - args.expected_champion_mrr) > 1e-12:
        raise RuntimeError(
            "champion/cache alignment failed: "
            f"actual={baseline['full']:.16f} "
            f"expected={args.expected_champion_mrr:.16f}"
        )
    print(f"[rank-blend] champion full_mrr={baseline['full']:.8f}", flush=True)

    alternates: dict[str, np.ndarray] = {}
    source_state = load_checkpoint_dataset(args.source_checkpoint, "dataset2")
    if tuple(str(name) for name in source_state["feature_names"]) != feature_names:
        raise ValueError("source checkpoint feature schema differs from champion")
    source_lgbm = source_state.get("lgbm_result")
    if source_lgbm is None:
        raise ValueError("source checkpoint has no Dataset2 LightGBM result")
    if tuple(int(index) for index in source_lgbm.feature_indices) != feature_indices:
        raise ValueError("source checkpoint selected feature columns differ")
    source_probs = _softmax(
        predict_logits_lgbm(source_lgbm.model_text, _select(val_features, feature_indices))
    )
    alternates["source_checkpoint_stored"] = (
        float(source_lgbm.mlp_weight) * champion_mlp
        + (1.0 - float(source_lgbm.mlp_weight)) * source_probs
    )
    del source_state, source_lgbm, source_probs
    gc.collect()
    release_memory()

    selected = _select(val_features, feature_indices)
    for name, path in (
        ("full100_fixed007", args.full100_model),
        ("matched32_fixed007", args.matched32_model),
        ("oof_hardneg_fixed007", args.oof_model),
    ):
        model_text = path.read_text(encoding="utf-8")
        lgbm_probs = _softmax(predict_logits_lgbm(model_text, selected))
        alternates[name] = (
            args.external_mlp_weight * champion_mlp
            + (1.0 - args.external_mlp_weight) * lgbm_probs
        )
        del model_text, lgbm_probs
        gc.collect()

    trials: list[dict[str, Any]] = []
    for alternate_index, (name, alternate) in enumerate(alternates.items()):
        for method_index, method in enumerate(METHODS):
            scan = scan_rank_blend_on_prefix(
                champion,
                alternate,
                selection_stop=SELECTION_STOP,
                method=method,
            )
            slice_keys = ("slice_0", "slice_1", "slice_2")
            slice_deltas = [
                float(scan.mrr[key] - baseline[key]) for key in slice_keys
            ]
            trial = {
                "alternate_index": alternate_index,
                "method_index": method_index,
                "alternate": name,
                **asdict(scan),
                "full_delta": float(scan.mrr["full"] - baseline["full"]),
                "slice_deltas": slice_deltas,
            }
            trials.append(trial)
            print(
                f"[rank-blend] alternate={name} method={method} "
                f"champion_w={scan.reference_weight:.2f} "
                f"selection={scan.selection_mrr:.8f} "
                f"full={scan.mrr['full']:.8f}",
                flush=True,
            )

    winner = max(
        trials,
        key=lambda trial: (
            trial["selection_mrr"],
            trial["reference_weight"],
            -trial["alternate_index"],
            -trial["method_index"],
        ),
    )
    candidate_slices = tuple(
        float(winner["mrr"][key])
        for key in ("slice_0", "slice_1", "slice_2")
    )
    baseline_slices = tuple(
        float(baseline[key]) for key in ("slice_0", "slice_1", "slice_2")
    )
    passed = passes_temporal_mrr_gate(
        candidate_slices=candidate_slices,
        baseline_slices=baseline_slices,
        candidate_full_mrr=float(winner["mrr"]["full"]),
        baseline_full_mrr=float(baseline["full"]),
        min_full_delta=args.min_full_delta,
    )
    report = {
        "status": "passed" if passed else "rejected",
        "gate_passed": passed,
        "cache_build_required": not passed,
        "frozen_config": frozen,
        "feature_names": list(feature_names),
        "feature_indices": list(feature_indices),
        "baseline": baseline,
        "winner": winner,
        "gate": {
            "min_full_delta": args.min_full_delta,
            "full_delta_passed": bool(
                winner["full_delta"] + 1e-12 >= args.min_full_delta
            ),
            "all_three_slices_non_decreasing": bool(
                all(delta >= 0.0 for delta in winner["slice_deltas"])
            ),
        },
        "trials": trials,
    }
    _write_json_atomic(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 2


def _score_champion(
    checkpoint: Path,
    val_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[int, ...]]:
    state = load_checkpoint_dataset(checkpoint, "dataset2")
    feature_names = tuple(str(name) for name in state["feature_names"])
    if len(feature_names) != val_features.shape[-1]:
        raise ValueError("champion checkpoint feature schema differs from validation")
    fusion_result = state["fusion_result"]
    feature_indices = tuple(int(index) for index in fusion_result.feature_indices)
    lgbm_result = state.get("lgbm_result")
    if lgbm_result is None:
        raise ValueError("champion checkpoint has no Dataset2 LightGBM result")
    if tuple(int(index) for index in lgbm_result.feature_indices) != feature_indices:
        raise ValueError("champion MLP and LightGBM feature selections differ")
    selected = _select(val_features, feature_indices)
    model = build_fusion_from_state(
        input_dim=len(feature_indices),
        hidden_dim=int(state["fusion_hidden_dim"]),
        state=state["fusion_state"],
    )
    mlp = _softmax(
        predict_logits(
            model,
            selected,
            fusion_result.mean,
            fusion_result.std,
        )
    )
    lgbm = _softmax(predict_logits_lgbm(lgbm_result.model_text, selected))
    ensemble = (
        float(lgbm_result.mlp_weight) * mlp
        + (1.0 - float(lgbm_result.mlp_weight)) * lgbm
    )
    del state, model, lgbm
    gc.collect()
    release_memory()
    return mlp, ensemble, feature_names, feature_indices


def _select(features: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    if indices == tuple(range(features.shape[-1])):
        return features
    return features[..., indices]


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
