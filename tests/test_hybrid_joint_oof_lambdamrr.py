import importlib.util
import sys
from pathlib import Path

import jittor as jt
import numpy as np

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_dataset2_joint_oof_lambdamrr.py"
_SPEC = importlib.util.spec_from_file_location("train_dataset2_joint_oof_lambdamrr_under_test", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_train_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _train_module
_SPEC.loader.exec_module(_train_module)
JointCandidateFeatureView = _train_module.JointCandidateFeatureView

from jgrec.rankers.hybrid.joint_oof_lambdamrr import (  # noqa: E402
    JointOOFLambdaMRRConfig,
    JointOOFLambdaMRRModel,
    JointOOFLambdaMRRTrainingConfig,
    bounded_joint_topk_alternatives,
    fit_joint_oof_lambdamrr,
    joint_router_lambdamrr_loss,
    load_joint_oof_lambdamrr_checkpoint,
    predict_joint_oof_lambdamrr,
    save_joint_oof_lambdamrr_checkpoint,
)


def test_joint_loss_backpropagates_to_shared_router_and_rank_heads():
    jt.set_seed(60)
    model = JointOOFLambdaMRRModel(
        JointOOFLambdaMRRConfig(
            row_input_dim=3,
            candidate_input_dim=2,
            hidden_dim=8,
            dropout=0.0,
        )
    )
    row_features = jt.array(
        np.array(
            [[0.2, -0.1, 0.4], [-0.3, 0.5, 0.1]],
            dtype=np.float32,
        )
    )
    candidate_features = jt.array(
        np.array(
            [
                [[0.4, 0.1], [0.2, -0.3], [-0.1, 0.5]],
                [[0.1, -0.2], [-0.4, 0.3], [0.6, 0.2]],
            ],
            dtype=np.float32,
        )
    )
    route_targets = jt.array(
        np.array([[0.10, -0.05], [-0.02, 0.08]], dtype=np.float32)
    )
    base_group_scores = jt.array(
        np.array(
            [
                [[0.3, 0.5, 0.1], [0.4, 0.6, 0.2]],
                [[0.2, 0.4, 0.1], [0.3, 0.5, 0.0]],
            ],
            dtype=np.float32,
        )
    )
    pair_weights = jt.array(
        np.array(
            [
                [[0.5, 0.2], [0.4, 0.1]],
                [[0.3, 0.2], [0.6, 0.2]],
            ],
            dtype=np.float32,
        )
    )

    route_predictions, candidate_residuals = model(
        row_features,
        candidate_features,
    )
    adjusted_scores = (
        base_group_scores + candidate_residuals.permute(0, 2, 1)
    )
    total, route_loss, rank_loss = joint_router_lambdamrr_loss(
        route_predictions,
        route_targets,
        adjusted_scores,
        pair_weights,
        route_loss_weight=1.0,
        rank_loss_weight=1.0,
    )
    gradients = jt.grad(
        total,
        [
            model.row_input.weight,
            model.route_output.weight,
            model.candidate_output.weight,
        ],
    )

    assert float(route_loss.item()) > 0.0
    assert float(rank_loss.item()) > 0.0
    assert all(np.any(np.abs(gradient.numpy()) > 0.0) for gradient in gradients)


def test_joint_alternatives_bound_total_horizon_and_lambda_correction():
    default = np.array(
        [[0.8, 0.6, 0.4, 0.2], [0.7, 0.5, 0.3, 0.1]],
        dtype=np.float32,
    )
    medium = default + np.array(
        [[-0.01, 0.02, -0.01, 0.0], [0.01, -0.02, 0.01, 0.0]],
        dtype=np.float32,
    )
    long = default + np.array(
        [[0.02, -0.01, -0.01, 0.0], [-0.01, 0.02, -0.01, 0.0]],
        dtype=np.float32,
    )
    residuals = np.array(
        [
            [[-2.0, 3.0], [4.0, -5.0], [1.0, 2.0], [9.0, 9.0]],
            [[3.0, -2.0], [-4.0, 5.0], [2.0, 1.0], [8.0, 8.0]],
        ],
        dtype=np.float32,
    )

    alternatives = bounded_joint_topk_alternatives(
        default,
        (medium, long),
        residuals,
        top_k=3,
        cap=0.02,
    )

    for alternative in alternatives:
        delta = alternative.scores - default
        assert np.array_equal(
            alternative.scores[~alternative.topk_mask],
            default[~alternative.topk_mask],
        )
        assert float(np.max(np.abs(delta))) <= 0.020002
        np.testing.assert_allclose(delta.sum(axis=1), 0.0, atol=2e-7)


def test_joint_fit_checkpoint_replays_both_outputs(tmp_path):
    rng = np.random.default_rng(60)
    rows, candidates = 48, 6
    row_features = rng.normal(size=(rows, 4)).astype(np.float32)
    candidate_features = rng.normal(
        size=(rows, candidates, 3)
    ).astype(np.float32)
    default = rng.normal(size=(rows, candidates)).astype(np.float32)
    default[:, 0] += 0.2
    medium = default + 0.02 * rng.normal(
        size=default.shape
    ).astype(np.float32)
    long = default + 0.02 * rng.normal(
        size=default.shape
    ).astype(np.float32)
    route_targets = np.column_stack(
        (
            0.01 * row_features[:, 0],
            -0.01 * row_features[:, 1],
        )
    ).astype(np.float32)
    model_config = JointOOFLambdaMRRConfig(
        row_input_dim=4,
        candidate_input_dim=3,
        hidden_dim=8,
        dropout=0.0,
    )
    training_config = JointOOFLambdaMRRTrainingConfig(
        epochs=2,
        batch_size=12,
        learning_rate=0.003,
        weight_decay=0.0,
        reward_scale=10.0,
        nonzero_weight=2.0,
        route_loss_weight=1.0,
        rank_loss_weight=0.2,
        seed=60,
    )

    model, result = fit_joint_oof_lambdamrr(
        row_features,
        candidate_features,
        default,
        (medium, long),
        route_targets,
        top_k=3,
        cap=0.02,
        model_config=model_config,
        training_config=training_config,
        row_feature_names=("r0", "r1", "r2", "r3"),
        candidate_feature_names=("c0", "c1", "c2"),
        verbose=False,
    )
    before = predict_joint_oof_lambdamrr(
        model,
        row_features,
        candidate_features,
        row_mean=result.row_mean,
        row_std=result.row_std,
        candidate_mean=result.candidate_mean,
        candidate_std=result.candidate_std,
        reward_scale=result.training_config.reward_scale,
        batch_size=12,
    )
    checkpoint = tmp_path / "joint.npz"
    save_joint_oof_lambdamrr_checkpoint(checkpoint, model, result)
    loaded, loaded_result = load_joint_oof_lambdamrr_checkpoint(checkpoint)
    after = predict_joint_oof_lambdamrr(
        loaded,
        row_features,
        candidate_features,
        row_mean=loaded_result.row_mean,
        row_std=loaded_result.row_std,
        candidate_mean=loaded_result.candidate_mean,
        candidate_std=loaded_result.candidate_std,
        reward_scale=loaded_result.training_config.reward_scale,
        batch_size=12,
    )

    assert result.training_rows == rows
    assert all(row["route_loss"] > 0.0 for row in result.history)
    assert all(row["rank_loss"] > 0.0 for row in result.history)
    assert result.trainable_frameworks == ("jittor",)
    assert result.non_jittor_trainable_models == ()
    np.testing.assert_allclose(after[0], before[0], atol=1e-6)
    np.testing.assert_allclose(after[1], before[1], atol=1e-6)


def test_joint_candidate_features_follow_candidate_permutation():
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(3, 5, 2)).astype(np.float32)
    default = rng.normal(size=(3, 5)).astype(np.float32)
    residuals = rng.normal(size=(3, 3, 5)).astype(np.float32)
    medium = default + 0.01 * rng.normal(
        size=default.shape
    ).astype(np.float32)
    long = default + 0.01 * rng.normal(
        size=default.shape
    ).astype(np.float32)
    before = JointCandidateFeatureView(
        raw,
        ("a", "b"),
        default,
        residuals,
        (medium, long),
    )
    permutation = np.array([3, 0, 4, 1, 2])
    after = JointCandidateFeatureView(
        raw[:, permutation],
        ("a", "b"),
        default[:, permutation],
        residuals[:, :, permutation],
        (medium[:, permutation], long[:, permutation]),
    )

    assert before.feature_names == after.feature_names
    np.testing.assert_allclose(
        after[:],
        before[:, permutation],
        atol=1e-6,
    )
