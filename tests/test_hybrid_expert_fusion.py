import numpy as np
import pytest

from jgrec.rankers.hybrid.expert_fusion import (
    ExpertBlendCalibration,
    blend_expert_logits,
    fit_positive_column_temperature,
    positive_column_nll,
)
from jgrec.rankers.hybrid.fusion_lgbm import LGBMFusionResult


def test_rrf_is_invariant_to_monotonic_expert_rescaling():
    mlp = np.asarray([[3.0, 1.0, 2.0], [0.0, 4.0, 2.0]])
    lgbm = np.asarray([[0.2, 0.9, 0.4], [8.0, 1.0, 3.0]])
    calibration = ExpertBlendCalibration(mode="rrf", rrf_k=60.0)

    reference = blend_expert_logits(mlp, lgbm, 0.4, calibration=calibration)
    rescaled = blend_expert_logits(
        100.0 * mlp + 7.0,
        np.exp(lgbm),
        0.4,
        calibration=calibration,
    )

    np.testing.assert_allclose(rescaled, reference, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(reference.sum(axis=1), 1.0)


def test_temperature_calibration_reduces_overconfident_validation_nll():
    logits = np.asarray(
        [
            [10.0, 0.0, -1.0],
            [-10.0, 0.0, -1.0],
            [10.0, 0.0, -1.0],
            [-10.0, 0.0, -1.0],
        ]
    )

    temperature = fit_positive_column_temperature(logits)

    assert temperature > 1.0
    assert positive_column_nll(logits, temperature) < positive_column_nll(logits)


@pytest.mark.parametrize("mode", ["probability", "temperature", "rrf"])
def test_all_expert_blend_modes_accept_per_query_weights_and_normalize(mode):
    mlp = np.asarray([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
    lgbm = np.asarray([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])

    probabilities = blend_expert_logits(
        mlp,
        lgbm,
        np.asarray([1.0, 0.0]),
        calibration=ExpertBlendCalibration(
            mode=mode,
            mlp_temperature=2.0,
            lgbm_temperature=0.5,
        ),
    )

    assert probabilities.shape == mlp.shape
    assert np.all(probabilities >= 0.0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_expert_blend_rejects_unknown_mode():
    logits = np.asarray([[1.0, 0.0]])

    with pytest.raises(ValueError, match="blend mode"):
        blend_expert_logits(
            logits,
            logits,
            0.5,
            calibration=ExpertBlendCalibration(mode="unknown"),
        )


def test_lgbm_checkpoint_metadata_defaults_to_legacy_probability_blend():
    result = LGBMFusionResult(
        best_val_ap=0.1,
        best_val_mrr=0.2,
        model_text="model",
        feature_indices=(0,),
        candidate_name="legacy",
    )

    assert result.blend_mode == "probability"
    assert result.mlp_temperature == 1.0
    assert result.lgbm_temperature == 1.0
    assert result.rrf_k == 60.0
