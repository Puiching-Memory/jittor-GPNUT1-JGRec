from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from jgrec.core.cuda import require_jittor_cuda
from jgrec.rankers.hybrid.oof_models import (
    CandidateSetMLPConfig,
    CandidateSetMLPTrainingConfig,
    fit_candidate_set_mlp,
    load_candidate_set_mlp_checkpoint,
    predict_candidate_set_mlp_logits,
    save_candidate_set_mlp_checkpoint,
)
from jgrec.rankers.hybrid.oof_stacking import (
    STABLE_EXPERT_LOGIT_FEATURE_VERSION,
    StableExpertLogitFeatureView,
    stable_expert_logit_feature_names,
    stable_expert_logit_features,
    tie_neutral_mrr,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain the pure-Jittor OOF meta learner after freezing the "
            "tie-stable logit feature contract."
        )
    )
    parser.add_argument(
        "--base-experiment-dir",
        required=True,
        type=Path,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--champion-validation-scores",
        required=True,
        type=Path,
    )
    parser.add_argument("--meta-epochs", type=int, default=12)
    parser.add_argument("--meta-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--minimum-full-delta", type=float, default=2e-4)
    args = parser.parse_args()

    require_jittor_cuda(jt)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.time()

    frozen = _read_json(args.base_experiment_dir / "frozen-config.json")
    expert_names = tuple(str(name) for name in frozen["expert_names"])
    folds = tuple(frozen["folds"])
    meta_train_fold_count = int(frozen["meta_train_fold_count"])
    score_start = int(folds[0]["score_rows"][0])
    meta_stop = int(
        folds[meta_train_fold_count - 1]["score_rows"][1]
    )
    validation_start = int(
        folds[meta_train_fold_count]["score_rows"][0]
    )
    if meta_stop != validation_start:
        raise RuntimeError("OOF meta train/validation rows are not contiguous")

    oof_path = args.base_experiment_dir / "oof-expert-logits.npy"
    oof_logits = np.load(oof_path, mmap_mode="r", allow_pickle=False)
    if (
        oof_logits.ndim != 3
        or oof_logits.shape[0] != len(expert_names)
        or not np.all(
            np.isfinite(oof_logits[:, score_start:, :])
        )
    ):
        raise ValueError("base OOF logits are incomplete or misaligned")
    train_view = StableExpertLogitFeatureView(
        oof_logits,
        row_start=score_start,
        row_stop=meta_stop,
    )
    validation_view = StableExpertLogitFeatureView(
        oof_logits,
        row_start=validation_start,
        row_stop=int(oof_logits.shape[1]),
    )
    feature_names = stable_expert_logit_feature_names(expert_names)
    model, result = fit_candidate_set_mlp(
        train_view,
        np.zeros(train_view.shape[0], dtype=np.int32),
        validation_features=validation_view,
        validation_positive_indices=np.zeros(
            validation_view.shape[0],
            dtype=np.int32,
        ),
        model_config=CandidateSetMLPConfig(
            input_dim=train_view.shape[-1],
            hidden_dim=64,
            dropout=0.05,
            relative_context="none",
        ),
        training_config=CandidateSetMLPTrainingConfig(
            epochs=args.meta_epochs,
            batch_size=args.meta_batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            early_stop_patience=3,
        ),
        feature_names=feature_names,
        feature_provenance=tuple(
            "numpy_deterministic" for _ in feature_names
        ),
        verbose=True,
    )
    meta_path = args.output_dir / "meta-stacking-mlp.npz"
    save_candidate_set_mlp_checkpoint(meta_path, model, result)

    oof_validation_logits = np.asarray(
        oof_logits[:, validation_start:, :],
        dtype=np.float32,
    )
    meta_validation_logits = predict_candidate_set_mlp_logits(
        model,
        validation_view,
        mean=result.mean,
        std=result.std,
        batch_size=args.meta_batch_size,
    )
    oof_consensus = _expert_consensus_percentile(
        oof_validation_logits,
        batch_size=args.meta_batch_size,
    )
    oof_meta_percentile = _single_score_percentile(
        meta_validation_logits,
        batch_size=args.meta_batch_size,
    )
    scan = _scan_meta_weights(oof_meta_percentile, oof_consensus)
    selected = max(
        scan,
        key=lambda row: (
            row["metrics"]["full"],
            min(
                row["metrics"][f"slice_{index}"]
                for index in range(3)
            ),
            -row["meta_weight"],
        ),
    )
    meta_weight = float(selected["meta_weight"])
    oof_selected = (
        meta_weight * oof_meta_percentile
        + (1.0 - meta_weight) * oof_consensus
    ).astype(np.float32)
    _save_array_atomic(
        args.output_dir / "meta-validation-logits.npy",
        meta_validation_logits,
    )
    _save_array_atomic(
        args.output_dir / "meta-validation-selected-scores.npy",
        oof_selected,
    )
    meta_report = {
        "status": "complete",
        "stable_feature_version": STABLE_EXPERT_LOGIT_FEATURE_VERSION,
        "meta_train_rows": [score_start, meta_stop],
        "meta_validation_rows": [
            validation_start,
            int(oof_logits.shape[1]),
        ],
        "feature_names": list(feature_names),
        "best_val_mrr": result.best_val_mrr,
        "history": list(result.history),
        "expert_metrics": {
            name: _mrr_three_slices(oof_validation_logits[index])
            for index, name in enumerate(expert_names)
        },
        "consensus_metrics": _mrr_three_slices(oof_consensus),
        "raw_meta_metrics": _mrr_three_slices(meta_validation_logits),
        "blend_scan": scan,
        "selected": selected,
        "selected_metrics": _mrr_three_slices(oof_selected),
        "metric_protocol": "tie_neutral_average_rank",
        "meta_checkpoint": str(meta_path.resolve()),
        "meta_checkpoint_sha256": _sha256(meta_path),
        "source_oof_logits": str(oof_path.resolve()),
        "source_oof_logits_sha256": _sha256(oof_path),
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
    }
    _write_json_atomic(args.output_dir / "meta-report.json", meta_report)

    full_expert_path = (
        args.base_experiment_dir / "full-validation-expert-logits.npy"
    )
    full_expert_logits = np.load(
        full_expert_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    champion = np.load(
        args.champion_validation_scores,
        mmap_mode="r",
        allow_pickle=False,
    )
    if full_expert_logits.shape[1:] != champion.shape:
        raise ValueError("full expert logits and champion scores misalign")
    full_stable_view = StableExpertLogitFeatureView(full_expert_logits)
    full_meta_logits = predict_candidate_set_mlp_logits(
        model,
        full_stable_view,
        mean=result.mean,
        std=result.std,
        batch_size=args.meta_batch_size,
    )
    full_consensus = _expert_consensus_percentile(
        full_expert_logits,
        batch_size=args.meta_batch_size,
    )
    full_meta_percentile = _single_score_percentile(
        full_meta_logits,
        batch_size=args.meta_batch_size,
    )
    full_selected = (
        meta_weight * full_meta_percentile
        + (1.0 - meta_weight) * full_consensus
    ).astype(np.float32)

    loaded_model, loaded_result = load_candidate_set_mlp_checkpoint(
        meta_path
    )
    replay_meta_logits = predict_candidate_set_mlp_logits(
        loaded_model,
        full_stable_view,
        mean=loaded_result.mean,
        std=loaded_result.std,
        batch_size=args.meta_batch_size,
    )
    replay_meta_percentile = _single_score_percentile(
        replay_meta_logits,
        batch_size=args.meta_batch_size,
    )
    replay_selected = (
        meta_weight * replay_meta_percentile
        + (1.0 - meta_weight) * full_consensus
    ).astype(np.float32)
    replay_ranking_rows = _different_ranking_rows(
        full_selected,
        replay_selected,
    )

    _save_array_atomic(
        args.output_dir / "full-validation-meta-logits.npy",
        full_meta_logits,
    )
    selected_path = (
        args.output_dir / "full-validation-selected-scores.npy"
    )
    _save_array_atomic(selected_path, full_selected)
    comparison = _compare_three_slices(full_selected, champion)
    deltas = comparison["delta_vs_baseline"]
    score_gate_passed = bool(
        deltas["full"] + 1e-12 >= args.minimum_full_delta
        and all(deltas[f"slice_{index}"] >= 0.0 for index in range(3))
    )
    replay_passed = replay_ranking_rows == 0
    gate_passed = score_gate_passed and replay_passed
    evaluation = {
        "status": "passed" if gate_passed else "rejected",
        "stable_feature_version": STABLE_EXPERT_LOGIT_FEATURE_VERSION,
        "metric_protocol": "tie_neutral_average_rank",
        "meta_weight": meta_weight,
        "full_expert_metrics": {
            name: _mrr_three_slices(full_expert_logits[index])
            for index, name in enumerate(expert_names)
        },
        "consensus_metrics": _mrr_three_slices(full_consensus),
        "comparison": comparison,
        "gate": {
            "passed": gate_passed,
            "score_gate_passed": score_gate_passed,
            "replay_passed": replay_passed,
            "minimum_full_delta": args.minimum_full_delta,
            "all_three_slices_non_decreasing": all(
                deltas[f"slice_{index}"] >= 0.0 for index in range(3)
            ),
        },
        "replay": {
            "max_abs_meta_logit_error": float(
                np.max(np.abs(full_meta_logits - replay_meta_logits))
            ),
            "max_abs_selected_score_error": float(
                np.max(np.abs(full_selected - replay_selected))
            ),
            "different_ranking_rows": replay_ranking_rows,
        },
        "tie_leakage_audit": {
            "candidate_optimistic_mrr": _optimistic_mrr(full_selected),
            "candidate_tie_neutral_mrr": _mrr(full_selected),
            "positive_negative_exact_ties": int(
                np.sum(full_selected[:, 1:] == full_selected[:, :1])
            ),
        },
        "selected_scores": str(selected_path.resolve()),
        "selected_scores_sha256": _sha256(selected_path),
        "source_full_expert_logits": str(full_expert_path.resolve()),
        "source_full_expert_logits_sha256": _sha256(full_expert_path),
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
        "elapsed_seconds": time.time() - started,
    }
    _write_json_atomic(
        args.output_dir / "evaluation-report.json",
        evaluation,
    )
    print(json.dumps(evaluation, ensure_ascii=False, indent=2), flush=True)
    return 0


def _scan_meta_weights(
    meta_percentile: np.ndarray,
    consensus: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "meta_weight": float(weight),
            "metrics": _mrr_three_slices(
                float(weight) * meta_percentile
                + (1.0 - float(weight)) * consensus
            ),
        }
        for weight in np.linspace(0.0, 1.0, 21)
    ]


def _expert_consensus_percentile(
    expert_logits: Any,
    *,
    batch_size: int,
) -> np.ndarray:
    output = np.empty(expert_logits.shape[1:], dtype=np.float32)
    for start in range(0, int(expert_logits.shape[1]), batch_size):
        stop = min(start + batch_size, int(expert_logits.shape[1]))
        stable = stable_expert_logit_features(
            np.asarray(expert_logits[:, start:stop, :])
        )
        output[start:stop] = stable[..., -5]
    return output


def _single_score_percentile(
    scores: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    output = np.empty_like(scores, dtype=np.float32)
    for start in range(0, scores.shape[0], batch_size):
        stop = min(start + batch_size, scores.shape[0])
        output[start:stop] = stable_expert_logit_features(
            scores[None, start:stop, :]
        )[..., 0]
    return output


def _mrr(scores: np.ndarray) -> float:
    return tie_neutral_mrr(
        scores,
        np.zeros(scores.shape[0], dtype=np.int32),
    )


def _mrr_three_slices(scores: np.ndarray) -> dict[str, float]:
    boundaries = np.linspace(0, scores.shape[0], 4, dtype=np.int64)
    return {
        "full": _mrr(scores),
        **{
            f"slice_{index}": _mrr(
                scores[boundaries[index] : boundaries[index + 1]]
            )
            for index in range(3)
        },
    }


def _compare_three_slices(
    candidate: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, Any]:
    candidate_metrics = _mrr_three_slices(candidate)
    baseline_metrics = _mrr_three_slices(baseline)
    return {
        "protocol": "comparison_only_no_blend_tie_neutral",
        "candidate": candidate_metrics,
        "baseline": baseline_metrics,
        "delta_vs_baseline": {
            key: candidate_metrics[key] - baseline_metrics[key]
            for key in candidate_metrics
        },
    }


def _optimistic_mrr(scores: np.ndarray) -> float:
    ranks = 1 + np.sum(scores[:, 1:] > scores[:, :1], axis=1)
    return float(np.mean(1.0 / ranks))


def _different_ranking_rows(
    left: np.ndarray,
    right: np.ndarray,
) -> int:
    return int(
        np.sum(
            np.any(
                np.argsort(-left, axis=1, kind="stable")
                != np.argsort(-right, axis=1, kind="stable"),
                axis=1,
            )
        )
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
