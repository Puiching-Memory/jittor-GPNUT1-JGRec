import pytest

from jgrec.rankers.hybrid.fusion_lgbm import LGBMFusionResult
from jgrec.rankers.hybrid.lgbm_tuning import (
    Dataset2LGBMTrial,
    apply_tuned_lgbm_result,
    chronological_validation_slices,
    passes_robustness_gate,
    predeclared_dataset2_lgbm_grid,
    select_tune_winner,
)


def test_chronological_validation_slices_keep_late_rows_untouched():
    tune, pseudo_b = chronological_validation_slices(20_000, tune_rows=10_000)

    assert tune == slice(0, 10_000)
    assert pseudo_b == slice(10_000, 20_000)
    assert tune.stop == pseudo_b.start


def test_select_tune_winner_uses_only_tune_metrics_and_stable_grid_order():
    trials = [
        Dataset2LGBMTrial(
            grid_index=0,
            name="baseline",
            params={"num_leaves": 63},
            best_iteration=80,
            lgbm_tune_mrr=0.52,
            blend_tune_mrr=0.53,
            mlp_weight=0.10,
            model_text="baseline-model",
        ),
        Dataset2LGBMTrial(
            grid_index=1,
            name="larger",
            params={"num_leaves": 127},
            best_iteration=70,
            lgbm_tune_mrr=0.54,
            blend_tune_mrr=0.55,
            mlp_weight=0.08,
            model_text="larger-model",
        ),
        Dataset2LGBMTrial(
            grid_index=2,
            name="same-score-later",
            params={"num_leaves": 31},
            best_iteration=65,
            lgbm_tune_mrr=0.54,
            blend_tune_mrr=0.55,
            mlp_weight=0.07,
            model_text="later-model",
        ),
    ]

    winner = select_tune_winner(trials)

    assert winner.name == "larger"
    assert "pseudo_b" not in Dataset2LGBMTrial.__dataclass_fields__


def test_predeclared_grid_is_bounded_reproducible_and_starts_with_baseline():
    grid = predeclared_dataset2_lgbm_grid(seed=60, num_threads=16)

    assert len(grid) == 12
    assert grid[0][0] == "baseline"
    assert len({name for name, _ in grid}) == len(grid)
    assert all(params["objective"] == "lambdarank" for _, params in grid)
    assert all(params["metric"] == "None" for _, params in grid)
    assert all(params["seed"] == 60 for _, params in grid)
    assert all(params["num_threads"] == 16 for _, params in grid)


def test_robustness_gate_requires_strict_improvement_on_both_time_slices():
    assert passes_robustness_gate(
        candidate_tune_mrr=0.531,
        candidate_pseudo_b_mrr=0.511,
        baseline_tune_mrr=0.530,
        baseline_pseudo_b_mrr=0.510,
    )
    assert not passes_robustness_gate(
        candidate_tune_mrr=0.531,
        candidate_pseudo_b_mrr=0.509,
        baseline_tune_mrr=0.530,
        baseline_pseudo_b_mrr=0.510,
    )


def test_apply_tuned_lgbm_result_requires_passed_gate_and_preserves_feature_contract():
    current = LGBMFusionResult(
        best_val_ap=0.4,
        best_val_mrr=0.5,
        model_text="old-model",
        feature_indices=(1, 2, 3),
        candidate_name="lgbm_all",
        mlp_weight=0.1,
    )
    report = {
        "gate_passed": True,
        "winner": {
            "name": "lr003",
            "lgbm_tune_mrr": 0.574497,
            "mlp_weight": 0.07,
        },
    }

    updated = apply_tuned_lgbm_result(current, model_text="new-model", report=report)

    assert updated.model_text == "new-model"
    assert updated.feature_indices == current.feature_indices
    assert updated.best_val_mrr == pytest.approx(0.574497)
    assert updated.mlp_weight == pytest.approx(0.07)
    assert updated.candidate_name == "lgbm_tuned_lr003"

    with pytest.raises(ValueError, match="did not pass"):
        apply_tuned_lgbm_result(current, model_text="new-model", report={"gate_passed": False})


def test_apply_tuned_lgbm_result_allows_explicit_exploratory_override():
    current = LGBMFusionResult(
        best_val_ap=0.4,
        best_val_mrr=0.5,
        model_text="old-model",
        feature_indices=(4, 5, 6),
        candidate_name="lgbm_all",
        mlp_weight=0.3,
    )
    rejected_report = {
        "gate_passed": False,
        "winner": {
            "name": "minchild50",
            "lgbm_tune_mrr": 0.5714301876,
            "mlp_weight": 0.19,
        },
    }

    updated = apply_tuned_lgbm_result(
        current,
        model_text="exploratory-model",
        report=rejected_report,
        allow_rejected_report=True,
    )

    assert updated.model_text == "exploratory-model"
    assert updated.feature_indices == current.feature_indices
    assert updated.best_val_mrr == pytest.approx(0.5714301876)
    assert updated.mlp_weight == pytest.approx(0.19)
    assert updated.candidate_name == "lgbm_tuned_minchild50"
