from __future__ import annotations

import jittor as jt
import numpy as np
import pytest

from jgrec.rankers.hybrid.candidate_set_transformer import (
    CandidateSetEnsembleCheckpoint,
    CandidateSetFitResult,
    CandidateSetTrainingConfig,
    CandidateSetTransformer,
    CandidateSetTransformerConfig,
)
from jgrec.rankers.hybrid.oof_models import (
    CandidateSetMLP,
    CandidateSetMLPConfig,
    CandidateSetMLPFitResult,
    CandidateSetMLPTrainingConfig,
    PureJittorOOFStackingCheckpoint,
    fit_candidate_set_mlp,
    hydrate_candidate_set_mlp,
    hydrate_pure_jittor_oof_stacking,
    load_candidate_set_mlp_checkpoint,
    predict_candidate_set_mlp_logits,
    predict_pure_jittor_oof_stacking_scores,
    save_candidate_set_mlp_checkpoint,
    snapshot_candidate_set_mlp,
    snapshot_pure_jittor_oof_stacking,
)
from jgrec.rankers.hybrid.oof_stacking import (
    stable_expert_logit_feature_names,
)


def test_candidate_set_mlp_is_candidate_permutation_equivariant() -> None:
    jt.flags.use_cuda = 0
    jt.set_seed(7)
    model = CandidateSetMLP(
        CandidateSetMLPConfig(
            input_dim=3,
            hidden_dim=8,
            dropout=0.0,
            relative_context="mean_max",
        )
    )
    features = np.asarray(
        [
            [
                [1.0, 0.0, -1.0],
                [0.0, 2.0, 1.0],
                [3.0, 1.0, 0.0],
                [-2.0, 0.5, 4.0],
            ],
            [
                [2.0, 1.0, 0.0],
                [4.0, -1.0, 2.0],
                [0.0, 3.0, 1.0],
                [1.0, 0.0, -2.0],
            ],
        ],
        dtype=np.float32,
    )
    permutation = np.asarray([2, 0, 3, 1])

    model.eval()
    with jt.no_grad():
        scores = model(jt.array(features, dtype=jt.float32)).numpy()
        permuted = model(
            jt.array(features[:, permutation], dtype=jt.float32)
        ).numpy()

    assert scores.shape == (2, 4)
    np.testing.assert_allclose(
        permuted,
        scores[:, permutation],
        rtol=0.0,
        atol=1e-6,
    )


def test_fixed_candidate_set_mlp_training_uses_all_rows_and_is_pure_jittor(
    tmp_path,
) -> None:
    jt.flags.use_cuda = 0
    features = np.asarray(
        [
            [[2.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            [[0.0, 1.0], [3.0, 0.0], [-2.0, 1.0]],
            [[-1.0, 0.0], [0.0, 1.0], [2.5, 0.0]],
            [[2.2, 0.0], [1.0, 1.0], [-2.0, 0.0]],
        ],
        dtype=np.float32,
    )
    positives = np.asarray([0, 1, 2, 0], dtype=np.int32)

    model, result = fit_candidate_set_mlp(
        features,
        positives,
        model_config=CandidateSetMLPConfig(
            input_dim=2,
            hidden_dim=8,
            dropout=0.0,
        ),
        training_config=CandidateSetMLPTrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=0.01,
            seed=11,
        ),
        feature_names=("signal", "context"),
        feature_provenance=("numpy_deterministic", "jittor"),
        verbose=False,
    )
    scores = predict_candidate_set_mlp_logits(
        model,
        features,
        mean=result.mean,
        std=result.std,
        batch_size=2,
    )

    assert scores.shape == (4, 3)
    assert result.selection_mode == "fixed_full"
    assert result.training_rows == 4
    assert result.best_val_mrr is None
    assert result.trainable_frameworks == ("jittor",)
    assert result.non_jittor_trainable_models == ()

    checkpoint = tmp_path / "setwise-mlp.npz"
    save_candidate_set_mlp_checkpoint(checkpoint, model, result)
    restored_model, restored_result = load_candidate_set_mlp_checkpoint(
        checkpoint
    )
    restored_scores = predict_candidate_set_mlp_logits(
        restored_model,
        features,
        mean=restored_result.mean,
        std=restored_result.std,
        batch_size=2,
    )
    np.testing.assert_allclose(
        restored_scores,
        scores,
        rtol=0.0,
        atol=1e-6,
    )
    hydrated_model, hydrated_result = hydrate_candidate_set_mlp(
        snapshot_candidate_set_mlp(model, result)
    )
    hydrated_scores = predict_candidate_set_mlp_logits(
        hydrated_model,
        features,
        mean=hydrated_result.mean,
        std=hydrated_result.std,
        batch_size=2,
    )
    np.testing.assert_allclose(
        hydrated_scores,
        scores,
        rtol=0.0,
        atol=1e-6,
    )
    with pytest.raises(FileExistsError):
        save_candidate_set_mlp_checkpoint(checkpoint, model, result)


def test_pure_jittor_oof_stacking_snapshot_replays_scores() -> None:
    jt.flags.use_cuda = 0
    raw_names = ("signal", "context")
    raw_provenance = ("numpy_deterministic", "jittor")
    cst_config = CandidateSetTransformerConfig(
        input_dim=2,
        model_dim=4,
        heads=2,
        layers=1,
        dropout=0.0,
    )
    cst_training = CandidateSetTrainingConfig(epochs=1, batch_size=2)
    cst_pairs = []
    for seed in (21, 22):
        jt.set_seed(seed)
        cst_model = CandidateSetTransformer(cst_config)
        cst_pairs.append(
            (
                cst_model,
                CandidateSetFitResult(
                    model_config=cst_config,
                    training_config=cst_training,
                    best_val_mrr=0.5,
                    state={},
                    mean=np.zeros(2, dtype=np.float32),
                    std=np.ones(2, dtype=np.float32),
                    feature_names=raw_names,
                    feature_provenance=raw_provenance,
                    history=(),
                    training_rows=4,
                ),
            )
        )
    setwise_config = CandidateSetMLPConfig(
        input_dim=2,
        hidden_dim=8,
        dropout=0.0,
    )
    setwise_model = CandidateSetMLP(setwise_config)
    setwise_result = CandidateSetMLPFitResult(
        model_config=setwise_config,
        training_config=CandidateSetMLPTrainingConfig(
            epochs=1,
            batch_size=2,
        ),
        selection_mode="fixed_full",
        training_rows=4,
        best_val_mrr=None,
        state={},
        mean=np.zeros(2, dtype=np.float32),
        std=np.ones(2, dtype=np.float32),
        feature_names=raw_names,
        feature_provenance=raw_provenance,
        history=(),
    )
    expert_names = ("cst_main", "cst_residual", "setwise_mlp")
    meta_names = stable_expert_logit_feature_names(expert_names)
    meta_config = CandidateSetMLPConfig(
        input_dim=len(meta_names),
        hidden_dim=8,
        dropout=0.0,
        relative_context="none",
    )
    meta_model = CandidateSetMLP(meta_config)
    meta_result = CandidateSetMLPFitResult(
        model_config=meta_config,
        training_config=CandidateSetMLPTrainingConfig(
            epochs=1,
            batch_size=2,
        ),
        selection_mode="validation_best",
        training_rows=4,
        best_val_mrr=0.5,
        state={},
        mean=np.zeros(len(meta_names), dtype=np.float32),
        std=np.ones(len(meta_names), dtype=np.float32),
        feature_names=meta_names,
        feature_provenance=tuple(
            "numpy_deterministic" for _ in meta_names
        ),
        history=(),
    )
    stacking = PureJittorOOFStackingCheckpoint(
        expert_names=expert_names,
        cst_experts=CandidateSetEnsembleCheckpoint(
            models=tuple(pair[0] for pair in cst_pairs),
            results=tuple(pair[1] for pair in cst_pairs),
            weights=(0.5, 0.5),
        ),
        setwise_mlp=(setwise_model, setwise_result),
        meta_mlp=(meta_model, meta_result),
        meta_weight=0.25,
    )
    features = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)

    expected = predict_pure_jittor_oof_stacking_scores(
        stacking,
        features,
        batch_size=2,
    )
    snapshot = snapshot_pure_jittor_oof_stacking(stacking)
    assert snapshot["stable_feature_version"] == 2
    restored = hydrate_pure_jittor_oof_stacking(snapshot)
    actual = predict_pure_jittor_oof_stacking_scores(
        restored,
        features,
        batch_size=2,
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)
    incompatible = {**snapshot, "stable_feature_version": 1}
    with pytest.raises(ValueError, match="snapshot format differs"):
        hydrate_pure_jittor_oof_stacking(incompatible)
