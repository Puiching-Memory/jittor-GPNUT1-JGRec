from __future__ import annotations

import os
from pathlib import Path

import jittor as jt
import numpy as np

from jgrec.rankers.hybrid.candidate_set_transformer import (
    CandidateSetEnsembleCheckpoint,
    load_candidate_set_checkpoint,
    predict_candidate_set_logits,
)
from jgrec.rankers.hybrid.oof_models import (
    PureJittorOOFStackingCheckpoint,
    hydrate_pure_jittor_oof_stacking,
    load_candidate_set_mlp_checkpoint,
    predict_candidate_set_mlp_logits,
    predict_pure_jittor_oof_stacking_scores,
    snapshot_pure_jittor_oof_stacking,
)
from jgrec.rankers.hybrid.oof_stacking import (
    stable_expert_logit_features,
)

ROOT = Path("result/dataset2_pure_jittor_oof_stacking_20260726")
CACHE = Path(
    "cache/supervised_features/"
    "dataset2_joint_recent200k_full100_val_seed60_20260725.val.npy"
)
CHAMPION = Path(
    "result/dataset2_conservative_window_blend_20260726/"
    "artifacts/validation-conservative-blend.npy"
)


def main() -> None:
    jt.flags.use_cuda = int(os.environ.get("OOF_DIAG_CUDA", "1"))
    features = np.load(CACHE, mmap_mode="r", allow_pickle=False)
    expected_experts = np.load(
        ROOT / "full-validation-expert-logits.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    expected_selected = np.load(
        ROOT / "full-validation-selected-scores.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    champion = np.load(CHAMPION, mmap_mode="r", allow_pickle=False)
    cst_pairs = tuple(
        load_candidate_set_checkpoint(ROOT / "full-experts" / f"{name}.npz")
        for name in ("cst_main", "cst_residual")
    )
    setwise_pair = load_candidate_set_mlp_checkpoint(
        ROOT / "full-experts/setwise_mlp.npz"
    )
    meta_pair = load_candidate_set_mlp_checkpoint(
        ROOT / "meta-stacking-mlp.npz"
    )
    actual_experts = []
    for index, (name, pair) in enumerate(
        zip(("cst_main", "cst_residual"), cst_pairs, strict=True)
    ):
        actual = predict_candidate_set_logits(
            pair[0],
            features,
            mean=pair[1].mean,
            std=pair[1].std,
            batch_size=512,
        )
        actual_experts.append(actual)
        print(
            name,
            "max_logit_error",
            float(np.max(np.abs(actual - expected_experts[index]))),
            "ranking_equal",
            bool(
                np.array_equal(
                    np.argsort(-actual, axis=1, kind="stable"),
                    np.argsort(
                        -expected_experts[index],
                        axis=1,
                        kind="stable",
                    ),
                )
            ),
            flush=True,
        )
        _print_logit_scale_diagnostics(
            name,
            actual,
            expected_experts[index],
        )
    setwise_actual = predict_candidate_set_mlp_logits(
        setwise_pair[0],
        features,
        mean=setwise_pair[1].mean,
        std=setwise_pair[1].std,
        batch_size=512,
    )
    actual_experts.append(setwise_actual)
    print(
        "setwise_mlp",
        "max_logit_error",
        float(np.max(np.abs(setwise_actual - expected_experts[2]))),
        "ranking_equal",
        bool(
            np.array_equal(
                np.argsort(-setwise_actual, axis=1, kind="stable"),
                np.argsort(-expected_experts[2], axis=1, kind="stable"),
            )
        ),
        flush=True,
    )
    _print_logit_scale_diagnostics(
        "setwise_mlp",
        setwise_actual,
        expected_experts[2],
    )
    stacking = PureJittorOOFStackingCheckpoint(
        expert_names=("cst_main", "cst_residual", "setwise_mlp"),
        cst_experts=CandidateSetEnsembleCheckpoint(
            models=tuple(pair[0] for pair in cst_pairs),
            results=tuple(pair[1] for pair in cst_pairs),
            weights=(0.5, 0.5),
        ),
        setwise_mlp=setwise_pair,
        meta_mlp=meta_pair,
        meta_weight=0.25,
    )
    direct = predict_pure_jittor_oof_stacking_scores(
        stacking,
        features,
        batch_size=512,
    )
    hydrated = hydrate_pure_jittor_oof_stacking(
        snapshot_pure_jittor_oof_stacking(stacking)
    )
    replay = predict_pure_jittor_oof_stacking_scores(
        hydrated,
        features,
        batch_size=512,
    )
    direct_parts = _prediction_parts(stacking, features)
    replay_parts = _prediction_parts(hydrated, features)
    for name in ("stable", "consensus", "meta_logits", "meta_percentile"):
        left = direct_parts[name]
        right = replay_parts[name]
        print(
            f"part_{name}",
            "max_abs_error",
            float(np.max(np.abs(left - right))),
            "equal",
            bool(np.array_equal(left, right)),
            flush=True,
        )
    for name, scores in (
        ("consensus", direct_parts["consensus"]),
        ("meta_logits", direct_parts["meta_logits"]),
        ("meta_percentile", direct_parts["meta_percentile"]),
        *(
            (expert_name, expert_logits)
            for expert_name, expert_logits in zip(
                ("cst_main", "cst_residual", "setwise_mlp"),
                actual_experts,
                strict=True,
            )
        ),
    ):
        print(
            f"metric_{name}",
            "optimistic_mrr",
            _optimistic_mrr(scores),
            "tie_neutral_mrr",
            _tie_neutral_mrr(scores),
            "positive_ties",
            int(np.sum(scores[:, 1:] == scores[:, :1])),
            flush=True,
        )
    for name, actual in (("direct", direct), ("hydrated", replay)):
        print(
            name,
            "max_score_error",
            float(np.max(np.abs(actual - expected_selected))),
            "ranking_equal",
            bool(
                np.array_equal(
                    np.argsort(-actual, axis=1, kind="stable"),
                    np.argsort(-expected_selected, axis=1, kind="stable"),
                )
            ),
            flush=True,
        )
        print(
            name,
            "optimistic_mrr",
            _optimistic_mrr(actual),
            "tie_neutral_mrr",
            _tie_neutral_mrr(actual),
            "delta_vs_champion_tie_neutral",
            _tie_neutral_mrr(actual) - _tie_neutral_mrr(champion),
            "positive_ties",
            int(np.sum(actual[:, 1:] == actual[:, :1])),
            flush=True,
        )
    print(
        "direct_vs_hydrated",
        float(np.max(np.abs(direct - replay))),
        "different_ranking_rows",
        int(
            np.sum(
                np.any(
                    np.argsort(-direct, axis=1, kind="stable")
                    != np.argsort(-replay, axis=1, kind="stable"),
                    axis=1,
                )
            )
        ),
        flush=True,
    )


def _prediction_parts(
    stacking: PureJittorOOFStackingCheckpoint,
    features: np.ndarray,
) -> dict[str, np.ndarray]:
    logits = [
        predict_candidate_set_logits(
            model,
            features,
            mean=result.mean,
            std=result.std,
            batch_size=512,
        )
        for model, result in zip(
            stacking.cst_experts.models,
            stacking.cst_experts.results,
            strict=True,
        )
    ]
    logits.append(
        predict_candidate_set_mlp_logits(
            stacking.setwise_mlp[0],
            features,
            mean=stacking.setwise_mlp[1].mean,
            std=stacking.setwise_mlp[1].std,
            batch_size=512,
        )
    )
    stable = stable_expert_logit_features(np.stack(logits, axis=0))
    meta_logits = predict_candidate_set_mlp_logits(
        stacking.meta_mlp[0],
        stable,
        mean=stacking.meta_mlp[1].mean,
        std=stacking.meta_mlp[1].std,
        batch_size=512,
    )
    meta_percentile = stable_expert_logit_features(
        meta_logits[None, ...]
    )[..., 0]
    return {
        "stable": stable,
        "consensus": stable[..., -5],
        "meta_logits": meta_logits,
        "meta_percentile": meta_percentile,
    }


def _optimistic_mrr(scores: np.ndarray) -> float:
    ranks = 1 + np.sum(scores[:, 1:] > scores[:, :1], axis=1)
    return float(np.mean(1.0 / ranks))


def _print_logit_scale_diagnostics(
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
) -> None:
    work = np.asarray(actual, dtype=np.float64)
    median = np.median(work, axis=1, keepdims=True)
    mad = np.median(np.abs(work - median), axis=1)
    span = np.ptp(work, axis=1)
    row_error = np.max(
        np.abs(work - np.asarray(expected, dtype=np.float64)),
        axis=1,
    )
    print(
        name,
        "span_quantiles",
        np.quantile(span, [0.0, 0.01, 0.1, 0.5, 0.9]).tolist(),
        "mad_quantiles",
        np.quantile(mad, [0.0, 0.01, 0.1, 0.5, 0.9]).tolist(),
        "error_over_mad_rows",
        int(np.sum(row_error > np.maximum(mad, 1e-12) * 0.1)),
        flush=True,
    )


def _tie_neutral_mrr(scores: np.ndarray) -> float:
    greater = np.sum(scores[:, 1:] > scores[:, :1], axis=1)
    equal = np.sum(scores[:, 1:] == scores[:, :1], axis=1)
    return float(np.mean(1.0 / (1.0 + greater + 0.5 * equal)))


if __name__ == "__main__":
    main()
