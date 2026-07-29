from __future__ import annotations

import argparse
import gc
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.core.memory import release_memory
from jgrec.rankers.hybrid.fusion import build_fusion_from_state, predict_logits
from jgrec.rankers.hybrid.fusion_analysis import ranking_mrr_slices, scan_probability_blend
from jgrec.rankers.hybrid.fusion_lgbm import predict_logits_lgbm


@dataclass(frozen=True)
class CheckpointProbabilities:
    mlp: np.ndarray
    lgbm: np.ndarray
    stored_ensemble: np.ndarray
    stored_mlp_weight: float
    feature_names: tuple[str, ...]
    selected_features: tuple[int, ...]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two hybrid checkpoints on one cached validation tensor.")
    parser.add_argument("--dataset", required=True, choices=("dataset1", "dataset2"))
    parser.add_argument("--val-features", required=True, type=Path)
    parser.add_argument("--old-checkpoint", required=True, type=Path)
    parser.add_argument("--new-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    val_features = np.load(args.val_features, mmap_mode="r", allow_pickle=False)
    if val_features.ndim != 3 or val_features.shape[0] < 2 or val_features.shape[1] != 100:
        raise ValueError(f"expected validation shape (queries, 100, features), got {val_features.shape}")
    val_identity = {
        "path": str(args.val_features.resolve()),
        "shape": list(val_features.shape),
        "dtype": val_features.dtype.str,
        "manifest_sha256": _manifest_sha256(args.val_features),
    }

    print(f"[{args.dataset}] scoring old checkpoint on {tuple(val_features.shape)}", flush=True)
    old = _score_checkpoint(args.old_checkpoint, args.dataset, val_features)
    print(f"[{args.dataset}] scoring new checkpoint on the same validation tensor", flush=True)
    new = _score_checkpoint(args.new_checkpoint, args.dataset, val_features)
    if old.feature_names != new.feature_names:
        raise ValueError("checkpoint feature names differ")

    old_fine = scan_probability_blend(old.mlp, old.lgbm)
    new_fine = scan_probability_blend(new.mlp, new.lgbm)
    old_new_stored = scan_probability_blend(new.stored_ensemble, old.stored_ensemble)
    new_fine_probs = (
        new_fine.reference_weight * new.mlp + (1.0 - new_fine.reference_weight) * new.lgbm
    )
    new_fine_old = scan_probability_blend(new_fine_probs, old.stored_ensemble)

    report: dict[str, Any] = {
        "dataset": args.dataset,
        "validation": val_identity,
        "old_checkpoint": str(args.old_checkpoint.resolve()),
        "new_checkpoint": str(args.new_checkpoint.resolve()),
        "feature_names": list(new.feature_names),
        "old": _component_report(old),
        "new": _component_report(new),
        "scans": {
            "old_mlp_reference_vs_lgbm": asdict(old_fine),
            "new_mlp_reference_vs_lgbm": asdict(new_fine),
            "new_stored_reference_vs_old_stored": asdict(old_new_stored),
            "new_fine_reference_vs_old_stored": asdict(new_fine_old),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    _print_report(report)
    return 0


def _score_checkpoint(
    checkpoint: Path,
    dataset: str,
    val_features: np.ndarray,
) -> CheckpointProbabilities:
    state = load_checkpoint_dataset(checkpoint, dataset)
    feature_names = tuple(state["feature_names"])
    if len(feature_names) != val_features.shape[-1]:
        raise ValueError(
            f"feature count mismatch for {checkpoint}: checkpoint={len(feature_names)} cache={val_features.shape[-1]}"
        )
    fusion_result = state["fusion_result"]
    indices = tuple(int(index) for index in fusion_result.feature_indices)
    selected = _select_columns(val_features, indices)
    model = build_fusion_from_state(
        input_dim=len(indices),
        hidden_dim=int(state["fusion_hidden_dim"]),
        state=state["fusion_state"],
    )
    mlp = _softmax(predict_logits(model, selected, fusion_result.mean, fusion_result.std))
    lgbm_result = state.get("lgbm_result")
    if lgbm_result is None:
        raise ValueError(f"checkpoint has no LightGBM fusion state: {checkpoint}")
    if tuple(int(index) for index in lgbm_result.feature_indices) != indices:
        raise ValueError("MLP and LightGBM feature selections differ")
    lgbm = _softmax(predict_logits_lgbm(lgbm_result.model_text, selected))
    stored_weight = float(lgbm_result.mlp_weight)
    stored_ensemble = stored_weight * mlp + (1.0 - stored_weight) * lgbm
    del model, state, selected
    gc.collect()
    release_memory()
    return CheckpointProbabilities(
        mlp=mlp,
        lgbm=lgbm,
        stored_ensemble=stored_ensemble,
        stored_mlp_weight=stored_weight,
        feature_names=feature_names,
        selected_features=indices,
    )


def _component_report(probabilities: CheckpointProbabilities) -> dict[str, Any]:
    return {
        "selected_features": list(probabilities.selected_features),
        "stored_mlp_weight": probabilities.stored_mlp_weight,
        "mlp_mrr": ranking_mrr_slices(probabilities.mlp),
        "lgbm_mrr": ranking_mrr_slices(probabilities.lgbm),
        "stored_ensemble_mrr": ranking_mrr_slices(probabilities.stored_ensemble),
    }


def _select_columns(features: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    if not indices:
        raise ValueError("checkpoint selected no fusion features")
    start, stop = indices[0], indices[-1] + 1
    if indices == tuple(range(start, stop)):
        return features[..., start:stop]
    return features[..., indices]


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _manifest_sha256(val_path: Path) -> str | None:
    manifest_path = val_path.with_suffix(".json")
    if not manifest_path.exists():
        manifest_path = val_path.parent / val_path.name.replace(".val.npy", ".json")
    if not manifest_path.exists():
        return None
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def _print_report(report: dict[str, Any]) -> None:
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
