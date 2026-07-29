from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from jgrec.rankers.hybrid.candidate_set_transformer import (
    compare_candidate_set_to_baseline,
    load_candidate_set_checkpoint,
    load_candidate_set_ensemble_checkpoint,
    predict_candidate_set_ensemble_probabilities,
    save_candidate_set_ensemble_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Package independently trained pure-Jittor candidate-set "
            "experts into one fixed-probability ensemble checkpoint."
        )
    )
    parser.add_argument(
        "--expert",
        action="append",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--weight",
        action="append",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--validation-features",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--champion-validation-scores",
        required=True,
        type=Path,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
    )
    args = parser.parse_args()

    if len(args.expert) != len(args.weight):
        raise ValueError("every expert needs exactly one weight")
    if len(args.expert) < 2:
        raise ValueError("an ensemble needs at least two experts")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite ensemble: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True)
    jt.flags.use_cuda = 1 if args.device == "cuda" else 0

    expert_paths = tuple(Path(path) for path in args.expert)
    experts = tuple(
        load_candidate_set_checkpoint(path)
        for path in expert_paths
    )
    model_path = (
        args.output_dir / "candidate-set-transformer-ensemble.npz"
    )
    save_candidate_set_ensemble_checkpoint(
        model_path,
        experts,
        weights=tuple(args.weight),
    )
    ensemble = load_candidate_set_ensemble_checkpoint(model_path)

    validation_features = np.load(
        args.validation_features,
        mmap_mode="r",
        allow_pickle=False,
    )
    champion_scores = np.load(
        args.champion_validation_scores,
        mmap_mode="r",
        allow_pickle=False,
    )
    if validation_features.ndim != 3:
        raise ValueError(
            "validation features must have query/candidate/feature axes"
        )
    if validation_features.shape[:2] != champion_scores.shape:
        raise ValueError(
            "champion scores do not align with validation features"
        )
    feature_names = ensemble.results[0].feature_names
    if validation_features.shape[-1] != len(feature_names):
        raise ValueError(
            "validation feature width does not match the checkpoint"
        )

    probabilities = predict_candidate_set_ensemble_probabilities(
        ensemble,
        validation_features,
        batch_size=args.batch_size,
    )
    probability_path = args.output_dir / "validation-probabilities.npy"
    _save_array_atomic(probability_path, probabilities)
    positives = np.zeros(probabilities.shape[0], dtype=np.int32)
    comparison = compare_candidate_set_to_baseline(
        probabilities,
        champion_scores,
        positive_indices=positives,
    )
    deltas = comparison["delta_vs_baseline"]
    gate_passed = bool(
        deltas["full"] > 0.0
        and all(
            deltas[f"slice_{index}"] >= 0.0
            for index in range(3)
        )
    )
    report = {
        "status": "passed" if gate_passed else "rejected",
        "protocol": (
            "first_two_time_slices_select_fixed_probability_weight;"
            "third_time_slice_forward_gate"
        ),
        "blend": "fixed_probability",
        "weights": list(ensemble.weights),
        "comparison": comparison,
        "gate": {
            "passed": gate_passed,
            "full_improved": deltas["full"] > 0.0,
            "all_three_slices_non_decreasing": all(
                deltas[f"slice_{index}"] >= 0.0
                for index in range(3)
            ),
        },
        "checkpoint": {
            "path": str(model_path.resolve()),
            "sha256": _sha256(model_path),
            "trainable_frameworks": list(
                ensemble.trainable_frameworks
            ),
            "non_jittor_trainable_models": list(
                ensemble.non_jittor_trainable_models
            ),
            "external_ml_runtime_dependencies": [],
        },
        "experts": [
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "model_config": {
                    "model_dim": result.model_config.model_dim,
                    "heads": result.model_config.heads,
                    "layers": result.model_config.layers,
                    "relative_context": (
                        result.model_config.relative_context
                    ),
                    "pointwise_residual_dim": (
                        result.model_config.pointwise_residual_dim
                    ),
                },
            }
            for path, result in zip(
                expert_paths,
                ensemble.results,
                strict=True,
            )
        ],
        "validation": {
            "features": str(args.validation_features.resolve()),
            "features_sha256": _sha256(args.validation_features),
            "champion_scores": str(
                args.champion_validation_scores.resolve()
            ),
            "champion_scores_sha256": _sha256(
                args.champion_validation_scores
            ),
            "probabilities": str(probability_path.resolve()),
            "probabilities_sha256": _sha256(probability_path),
        },
        "champion_role": "comparison_only_no_training_no_blend",
        "rank_ensemble": "forbidden_due_to_exact_tie_mrr_inflation",
    }
    report_path = args.output_dir / "evaluation-report.json"
    _write_json_atomic(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if gate_passed else 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_array_atomic(path: Path, values: np.ndarray) -> None:
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


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
