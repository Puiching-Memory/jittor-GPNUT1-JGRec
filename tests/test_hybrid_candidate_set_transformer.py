from __future__ import annotations

import jittor as jt
import numpy as np
import pytest

from jgrec.rankers.hybrid.candidate_set_transformer import (
    CandidateSetTrainingConfig,
    CandidateSetTransformer,
    CandidateSetTransformerConfig,
    candidate_relative_features,
    candidate_set_listwise_loss,
    compare_candidate_set_to_baseline,
    fit_candidate_set_transformer,
    fit_candidate_set_transformer_fixed,
    load_candidate_set_checkpoint,
    load_candidate_set_ensemble_checkpoint,
    predict_candidate_set_ensemble_probabilities,
    predict_candidate_set_logits,
    save_candidate_set_checkpoint,
    save_candidate_set_ensemble_checkpoint,
)


def test_fixed_candidate_set_transformer_training_uses_every_row() -> None:
    jt.flags.use_cuda = 0
    features = np.asarray(
        [
            [[2.0, 0.0], [-1.0, 0.0], [0.0, 1.0]],
            [[-1.0, 0.0], [2.0, 0.0], [0.0, 1.0]],
            [[-1.0, 0.0], [0.0, 1.0], [2.0, 0.0]],
            [[2.5, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    positives = np.asarray([0, 1, 2, 0], dtype=np.int32)

    _, result = fit_candidate_set_transformer_fixed(
        features,
        positives,
        model_config=CandidateSetTransformerConfig(
            input_dim=2,
            model_dim=4,
            heads=2,
            layers=1,
            dropout=0.0,
        ),
        training_config=CandidateSetTrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=0.01,
            seed=64,
        ),
        feature_names=("signal", "context"),
        feature_provenance=("numpy_deterministic", "jittor"),
        verbose=False,
    )

    assert result.selection_mode == "fixed_full"
    assert result.training_rows == 4
    assert result.best_val_mrr != result.best_val_mrr


def test_candidate_set_transformer_scores_are_candidate_permutation_equivariant() -> None:
    jt.flags.use_cuda = 0
    jt.set_seed(60)
    config = CandidateSetTransformerConfig(
        input_dim=6,
        model_dim=12,
        heads=3,
        layers=2,
        dropout=0.0,
    )
    model = CandidateSetTransformer(config)
    model.eval()
    rng = np.random.default_rng(60)
    features = rng.normal(size=(2, 7, 6)).astype(np.float32)
    permutation = np.asarray([4, 0, 6, 2, 5, 1, 3])

    with jt.no_grad():
        original = np.asarray(
            model(jt.array(features, dtype=jt.float32)).numpy(),
            dtype=np.float32,
        )
        permuted = np.asarray(
            model(
                jt.array(features[:, permutation], dtype=jt.float32)
            ).numpy(),
            dtype=np.float32,
        )

    assert original.shape == (2, 7)
    np.testing.assert_allclose(
        permuted,
        original[:, permutation],
        rtol=0.0,
        atol=1e-5,
    )


def test_masked_candidates_neither_receive_scores_nor_affect_valid_candidates() -> None:
    jt.flags.use_cuda = 0
    jt.set_seed(61)
    model = CandidateSetTransformer(
        CandidateSetTransformerConfig(
            input_dim=3,
            model_dim=8,
            heads=2,
            layers=1,
            dropout=0.0,
        )
    )
    model.eval()
    features = np.asarray(
        [
            [
                [1.0, 0.0, 0.5],
                [0.0, 1.0, -0.5],
                [0.5, 0.5, 0.0],
                [2.0, -3.0, 5.0],
            ]
        ],
        dtype=np.float32,
    )
    changed = features.copy()
    changed[:, 3, :] = np.asarray([1000.0, -2000.0, 3000.0])
    mask = np.asarray([[True, True, True, False]])

    with jt.no_grad():
        original = np.asarray(
            model(
                jt.array(features, dtype=jt.float32),
                jt.array(mask),
            ).numpy(),
            dtype=np.float32,
        )
        modified = np.asarray(
            model(
                jt.array(changed, dtype=jt.float32),
                jt.array(mask),
            ).numpy(),
            dtype=np.float32,
        )

    np.testing.assert_allclose(
        modified[:, :3],
        original[:, :3],
        rtol=0.0,
        atol=1e-5,
    )
    assert np.all(original[:, 3] < -1e8)
    assert np.all(modified[:, 3] < -1e8)


def test_candidate_set_listwise_loss_supports_arbitrary_positive_positions_and_masks() -> None:
    logits = np.asarray(
        [
            [0.5, 2.0, -1.0, 8.0],
            [1.5, -0.5, 3.0, 0.0],
        ],
        dtype=np.float32,
    )
    positive_indices = np.asarray([1, 2], dtype=np.int32)
    mask = np.asarray(
        [
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    expected_rows = []
    for row, positive, valid in zip(
        logits,
        positive_indices,
        mask,
        strict=True,
    ):
        valid_logits = row[valid]
        row_max = valid_logits.max()
        expected_rows.append(
            row_max
            + np.log(np.exp(valid_logits - row_max).sum())
            - row[positive]
        )

    actual = float(
        candidate_set_listwise_loss(
            jt.array(logits, dtype=jt.float32),
            jt.array(positive_indices, dtype=jt.int32),
            jt.array(mask),
        ).item()
    )

    assert actual == pytest.approx(np.mean(expected_rows), abs=1e-6)


def test_pure_jittor_trainer_learns_a_small_candidate_ranking_problem() -> None:
    jt.flags.use_cuda = 0
    rng = np.random.default_rng(62)
    query_count = 24
    candidate_count = 5
    positives = np.arange(query_count, dtype=np.int32) % candidate_count
    features = rng.normal(
        scale=0.05,
        size=(query_count, candidate_count, 2),
    ).astype(np.float32)
    features[..., 0] = -1.0
    features[np.arange(query_count), positives, 0] = 3.0
    model_config = CandidateSetTransformerConfig(
        input_dim=2,
        model_dim=8,
        heads=2,
        layers=1,
        dropout=0.0,
        feedforward_multiplier=2,
    )
    training_config = CandidateSetTrainingConfig(
        epochs=8,
        batch_size=8,
        learning_rate=0.01,
        weight_decay=0.0,
        seed=62,
    )

    model, result = fit_candidate_set_transformer(
        features,
        positives,
        features,
        positives,
        model_config=model_config,
        training_config=training_config,
        feature_names=("positive_signal", "noise"),
        feature_provenance=("numpy_deterministic", "numpy_deterministic"),
        verbose=False,
    )
    scores = predict_candidate_set_logits(
        model,
        features,
        mean=result.mean,
        std=result.std,
        batch_size=8,
    )

    assert len(result.history) == training_config.epochs
    assert result.history[-1]["train_loss"] < result.history[0]["train_loss"]
    np.testing.assert_array_equal(scores.argmax(axis=1), positives)
    assert result.best_val_mrr == pytest.approx(1.0)
    assert result.trainable_frameworks == ("jittor",)
    assert result.non_jittor_trainable_models == ()


def test_candidate_set_checkpoint_round_trip_has_only_jittor_trainable_provenance(
    tmp_path,
) -> None:
    jt.flags.use_cuda = 0
    features = np.asarray(
        [
            [[2.0, 0.0], [-1.0, 0.1], [-1.0, -0.1]],
            [[-1.0, 0.1], [2.0, 0.0], [-1.0, -0.1]],
            [[-1.0, 0.1], [-1.0, -0.1], [2.0, 0.0]],
        ],
        dtype=np.float32,
    )
    positives = np.asarray([0, 1, 2], dtype=np.int32)
    model, result = fit_candidate_set_transformer(
        features,
        positives,
        features,
        positives,
        model_config=CandidateSetTransformerConfig(
            input_dim=2,
            model_dim=4,
            heads=2,
            layers=1,
            dropout=0.0,
        ),
        training_config=CandidateSetTrainingConfig(
            epochs=1,
            batch_size=3,
            learning_rate=0.01,
            seed=63,
        ),
        feature_names=("deterministic_signal", "jittor_score"),
        feature_provenance=("numpy_deterministic", "jittor"),
        verbose=False,
    )
    expected = predict_candidate_set_logits(
        model,
        features,
        mean=result.mean,
        std=result.std,
        batch_size=3,
    )
    checkpoint_path = tmp_path / "candidate-set-transformer.npz"

    save_candidate_set_checkpoint(checkpoint_path, model, result)
    restored_model, restored_result = load_candidate_set_checkpoint(
        checkpoint_path
    )
    actual = predict_candidate_set_logits(
        restored_model,
        features,
        mean=restored_result.mean,
        std=restored_result.std,
        batch_size=3,
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)
    assert restored_result.trainable_frameworks == ("jittor",)
    assert restored_result.non_jittor_trainable_models == ()
    assert b"lightgbm" not in checkpoint_path.read_bytes().lower()
    assert b"sklearn" not in checkpoint_path.read_bytes().lower()


def test_champion_scores_are_comparison_only_and_never_blended() -> None:
    positives = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int32)
    baseline = np.zeros((6, 3), dtype=np.float32)
    candidate = np.zeros((6, 3), dtype=np.float32)
    for row, positive in enumerate(positives):
        baseline[row, positive] = 0.5
        baseline[row, (positive + 1) % 3] = 1.0
        candidate[row, positive] = 2.0

    report = compare_candidate_set_to_baseline(
        candidate,
        baseline,
        positive_indices=positives,
    )

    assert report["candidate"]["full"] == pytest.approx(1.0)
    assert report["baseline"]["full"] == pytest.approx(0.5)
    assert report["delta_vs_baseline"]["full"] == pytest.approx(0.5)
    assert report["protocol"] == "comparison_only_no_blend"


def test_mean_max_relative_context_is_computed_inside_jittor_graph() -> None:
    values = np.asarray(
        [[[1.0, 4.0], [3.0, 2.0]]],
        dtype=np.float32,
    )
    expected = np.asarray(
        [
            [
                [1.0, 4.0, -1.0, 1.0, -2.0, 0.0],
                [3.0, 2.0, 1.0, -1.0, 0.0, -2.0],
            ]
        ],
        dtype=np.float32,
    )

    actual = np.asarray(
        candidate_relative_features(
            jt.array(values, dtype=jt.float32),
            mode="mean_max",
        ).numpy(),
        dtype=np.float32,
    )

    np.testing.assert_array_equal(actual, expected)


def test_pointwise_residual_remains_a_valid_fallback_when_interaction_is_zero() -> None:
    jt.flags.use_cuda = 0
    jt.set_seed(64)
    model = CandidateSetTransformer(
        CandidateSetTransformerConfig(
            input_dim=3,
            model_dim=8,
            heads=2,
            layers=1,
            dropout=0.0,
            relative_context="mean_max",
            pointwise_residual_dim=4,
        )
    )
    model.eval()
    features = jt.array(
        np.random.default_rng(64).normal(
            size=(2, 5, 3)
        ).astype(np.float32),
        dtype=jt.float32,
    )
    model.interaction_scale.assign(jt.zeros_like(model.interaction_scale))

    with jt.no_grad():
        expected = np.asarray(
            model.pointwise_scores(features).numpy(),
            dtype=np.float32,
        )
        actual = np.asarray(model(features).numpy(), dtype=np.float32)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_candidate_set_ensemble_checkpoint_round_trips_fixed_probability_blend(
    tmp_path,
) -> None:
    jt.flags.use_cuda = 0
    features = np.asarray(
        [
            [[2.0, 0.0], [-1.0, 0.1], [-1.0, -0.1]],
            [[-1.0, 0.1], [2.0, 0.0], [-1.0, -0.1]],
            [[-1.0, 0.1], [-1.0, -0.1], [2.0, 0.0]],
        ],
        dtype=np.float32,
    )
    positives = np.asarray([0, 1, 2], dtype=np.int32)
    experts = []
    for seed, residual_dim in ((65, 0), (66, 4)):
        experts.append(
            fit_candidate_set_transformer(
                features,
                positives,
                features,
                positives,
                model_config=CandidateSetTransformerConfig(
                    input_dim=2,
                    model_dim=4,
                    heads=2,
                    layers=1,
                    dropout=0.0,
                    relative_context="mean_max",
                    pointwise_residual_dim=residual_dim,
                ),
                training_config=CandidateSetTrainingConfig(
                    epochs=1,
                    batch_size=3,
                    learning_rate=0.01,
                    seed=seed,
                ),
                feature_names=("signal", "noise"),
                feature_provenance=(
                    "numpy_deterministic",
                    "numpy_deterministic",
                ),
                verbose=False,
            )
        )
    weights = (0.6, 0.4)
    checkpoint_path = tmp_path / "candidate-set-ensemble.npz"

    save_candidate_set_ensemble_checkpoint(
        checkpoint_path,
        tuple(experts),
        weights=weights,
    )
    restored = load_candidate_set_ensemble_checkpoint(checkpoint_path)
    probabilities = predict_candidate_set_ensemble_probabilities(
        restored,
        features,
        batch_size=3,
    )
    expected = np.zeros_like(probabilities)
    for weight, (model, result) in zip(weights, experts, strict=True):
        logits = predict_candidate_set_logits(
            model,
            features,
            mean=result.mean,
            std=result.std,
            batch_size=3,
        )
        shifted = logits - logits.max(axis=1, keepdims=True)
        expert_probabilities = np.exp(shifted)
        expert_probabilities /= expert_probabilities.sum(
            axis=1,
            keepdims=True,
        )
        expected += weight * expert_probabilities

    assert restored.weights == weights
    np.testing.assert_allclose(
        probabilities,
        expected,
        rtol=0.0,
        atol=1e-7,
    )
    checkpoint_bytes = checkpoint_path.read_bytes().lower()
    assert b"lightgbm" not in checkpoint_bytes
    assert b"sklearn" not in checkpoint_bytes
