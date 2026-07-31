import sys
from types import SimpleNamespace

import numpy as np
import pytest

from jgrec.rankers.hybrid import fusion as fusion_module
from jgrec.rankers.hybrid import fusion_lgbm as fusion_lgbm_module
from jgrec.rankers.hybrid.config import TrainingConfig
from jgrec.rankers.hybrid.fusion_lgbm import (
    LGBMFusionResult,
    _full_candidate_mrr,
    fit_fusion_lgbm,
)
from jgrec.rankers.hybrid.ranker import (
    _expert_blend_calibration,
    _feature_masks,
    _find_ensemble_weight,
    _select_expert_features,
)


def test_full_candidate_mrr_averages_reciprocal_rank_per_query():
    scores = np.asarray(
        [
            [3.0, 2.0, 1.0],
            [2.0, 3.0, 1.0],
            [1.0, 3.0, 2.0],
        ],
        dtype=np.float64,
    )

    actual = _full_candidate_mrr(scores.ravel(), candidate_count=3)

    assert actual == np.mean([1.0, 1.0 / 2.0, 1.0 / 3.0])


def test_mrr_selection_uses_full_candidate_metric_for_early_stopping(monkeypatch):
    captured: dict[str, object] = {}
    validation_predictions = np.asarray([3.0, 2.0, 1.0, 2.0, 3.0, 1.0])

    class DummyBooster:
        best_iteration = 7

        def predict(self, features, num_iteration=None):
            captured["predict_num_iteration"] = num_iteration
            return validation_predictions

        def model_to_string(self):
            return "dummy-model"

    def fake_dataset(data, **kwargs):
        return {"data": data, **kwargs}

    def fake_early_stopping(stopping_rounds, **kwargs):
        captured["early_stopping"] = {"stopping_rounds": stopping_rounds, **kwargs}
        return "early-stopping-callback"

    def fake_train(params, train_set, **kwargs):
        captured["params"] = params
        captured["train_kwargs"] = kwargs
        feval = kwargs.get("feval")
        captured["metric_result"] = None if feval is None else feval(validation_predictions, None)
        return DummyBooster()

    fake_lgb = SimpleNamespace(
        Dataset=fake_dataset,
        early_stopping=fake_early_stopping,
        log_evaluation=lambda period: "log-callback",
        train=fake_train,
    )
    monkeypatch.setitem(sys.modules, "lightgbm", fake_lgb)

    train_features = np.zeros((2, 3, 2), dtype=np.float32)
    val_features = np.zeros((2, 3, 2), dtype=np.float32)
    result = fit_fusion_lgbm(
        train_features,
        val_features,
        selection_metric="mrr",
        verbose=False,
    )

    assert captured["params"]["metric"] == "None"
    assert captured["metric_result"] == ("full_candidate_mrr", 0.75, True)
    assert captured["early_stopping"]["first_metric_only"] is True
    assert captured["predict_num_iteration"] == 7
    assert result.best_val_mrr == 0.75


def test_mlp_and_lgbm_select_their_own_feature_indices():
    features = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

    mlp_features, lgbm_features = _select_expert_features(
        features,
        mlp_indices=(0, 2),
        lgbm_indices=(0, 1, 2, 3),
    )

    np.testing.assert_array_equal(mlp_features, features[..., (0, 2)])
    np.testing.assert_array_equal(lgbm_features, features)


def test_ensemble_weight_search_uses_configured_rrf_scores(monkeypatch):
    mlp_logits = np.asarray([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    lgbm_logits = np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    monkeypatch.setattr(
        fusion_module,
        "predict_logits",
        lambda *_args, **_kwargs: mlp_logits,
    )
    monkeypatch.setattr(
        fusion_lgbm_module,
        "predict_logits_lgbm",
        lambda *_args, **_kwargs: lgbm_logits,
    )
    lgbm_result = LGBMFusionResult(
        best_val_ap=0.1,
        best_val_mrr=0.2,
        model_text="model",
        feature_indices=(0,),
        candidate_name="lgbm",
    )

    weight, calibration = _find_ensemble_weight(
        object(),
        SimpleNamespace(mean=np.zeros(1), std=np.ones(1)),
        lgbm_result,
        np.zeros((2, 3, 1), dtype=np.float32),
        (0,),
        TrainingConfig(
            selection_metric="mrr",
            fusion_mode="ensemble",
            expert_blend_mode="rrf",
            expert_rrf_k=30.0,
            verbose=False,
        ),
    )

    assert weight == 0.5
    assert calibration.mode == "rrf"
    assert calibration.rrf_k == 30.0


def test_frozen_ensemble_weight_bypasses_single_split_weight_search(
    monkeypatch,
):
    mlp_logits = np.asarray([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    lgbm_logits = np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    monkeypatch.setattr(
        fusion_module,
        "predict_logits",
        lambda *_args, **_kwargs: mlp_logits,
    )
    monkeypatch.setattr(
        fusion_lgbm_module,
        "predict_logits_lgbm",
        lambda *_args, **_kwargs: lgbm_logits,
    )
    lgbm_result = LGBMFusionResult(
        best_val_ap=0.1,
        best_val_mrr=0.2,
        model_text="model",
        feature_indices=(0,),
        candidate_name="lgbm",
    )

    weight, calibration = _find_ensemble_weight(
        object(),
        SimpleNamespace(mean=np.zeros(1), std=np.ones(1)),
        lgbm_result,
        np.zeros((2, 3, 1), dtype=np.float32),
        (0,),
        TrainingConfig(
            fusion_mode="ensemble",
            expert_blend_mode="rrf",
            frozen_ensemble_mlp_weight=0.3,
            verbose=False,
        ),
    )

    assert weight == pytest.approx(0.3)
    assert calibration.mode == "rrf"


def test_frozen_feature_candidate_disables_single_split_mask_search():
    config = TrainingConfig(
        frozen_fusion_feature_candidate="stats_prior_structure_tower_gnn"
    )

    masks = _feature_masks(1_000, config=config)

    assert [name for name, _indices in masks] == [
        "stats_prior_structure_tower_gnn"
    ]

    with pytest.raises(ValueError, match="frozen fusion feature candidate"):
        _feature_masks(
            1_000,
            config=TrainingConfig(
                frozen_fusion_feature_candidate="does_not_exist"
            ),
        )


def test_legacy_lgbm_result_without_metadata_resolves_probability_blend():
    calibration = _expert_blend_calibration(
        SimpleNamespace()
    )

    assert calibration.mode == "probability"
    assert calibration.mlp_temperature == 1.0
    assert calibration.lgbm_temperature == 1.0
