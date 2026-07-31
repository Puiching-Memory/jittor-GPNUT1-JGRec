from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class Dataset2LGBMTrial:
    grid_index: int
    name: str
    params: dict[str, Any]
    best_iteration: int
    lgbm_tune_mrr: float
    blend_tune_mrr: float
    mlp_weight: float
    model_text: str


def chronological_validation_slices(row_count: int, *, tune_rows: int) -> tuple[slice, slice]:
    if tune_rows <= 0 or tune_rows >= row_count:
        raise ValueError("tune_rows must leave non-empty tune and pseudo-B slices")
    return slice(0, tune_rows), slice(tune_rows, row_count)


def select_tune_winner(trials: list[Dataset2LGBMTrial]) -> Dataset2LGBMTrial:
    if not trials:
        raise ValueError("at least one Dataset2 LightGBM trial is required")
    return max(
        trials,
        key=lambda trial: (
            trial.blend_tune_mrr,
            trial.lgbm_tune_mrr,
            -trial.grid_index,
        ),
    )


def predeclared_dataset2_lgbm_grid(
    *,
    seed: int,
    num_threads: int,
) -> tuple[tuple[str, dict[str, Any]], ...]:
    base: dict[str, Any] = {
        "objective": "lambdarank",
        "metric": "None",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambdarank_truncation_level": 30,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "feature_pre_filter": False,
        "deterministic": True,
        "force_col_wise": True,
        "seed": seed,
        "num_threads": num_threads,
        "verbose": -1,
    }
    overrides: tuple[tuple[str, dict[str, Any]], ...] = (
        ("baseline", {}),
        ("lr003", {"learning_rate": 0.03}),
        ("lr008", {"learning_rate": 0.08}),
        ("leaves31", {"num_leaves": 31}),
        ("leaves127", {"num_leaves": 127}),
        ("minchild50", {"min_child_samples": 50}),
        ("minchild100", {"min_child_samples": 100}),
        ("trunc10", {"lambdarank_truncation_level": 10}),
        ("trunc50", {"lambdarank_truncation_level": 50}),
        (
            "leaves31_minchild50_trunc20",
            {"num_leaves": 31, "min_child_samples": 50, "lambdarank_truncation_level": 20},
        ),
        (
            "leaves127_minchild50_trunc20",
            {"num_leaves": 127, "min_child_samples": 50, "lambdarank_truncation_level": 20},
        ),
        (
            "regularized_trunc20",
            {
                "min_child_samples": 50,
                "lambdarank_truncation_level": 20,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.9,
                "lambda_l2": 1.0,
            },
        ),
    )
    return tuple((name, {**base, **changes}) for name, changes in overrides)


def passes_robustness_gate(
    *,
    candidate_tune_mrr: float,
    candidate_pseudo_b_mrr: float,
    baseline_tune_mrr: float,
    baseline_pseudo_b_mrr: float,
) -> bool:
    return candidate_tune_mrr > baseline_tune_mrr and candidate_pseudo_b_mrr > baseline_pseudo_b_mrr


def apply_tuned_lgbm_result(
    current,
    *,
    model_text: str,
    report: dict[str, Any],
    allow_rejected_report: bool = False,
):
    if not report.get("gate_passed") and not allow_rejected_report:
        raise ValueError("Dataset2 LightGBM tuning report did not pass the robustness gate")
    if not model_text.strip():
        raise ValueError("tuned Dataset2 LightGBM model is empty")
    winner = report["winner"]
    return replace(
        current,
        best_val_mrr=float(winner["lgbm_tune_mrr"]),
        model_text=model_text,
        candidate_name=f"lgbm_tuned_{winner['name']}",
        mlp_weight=float(winner["mlp_weight"]),
    )
