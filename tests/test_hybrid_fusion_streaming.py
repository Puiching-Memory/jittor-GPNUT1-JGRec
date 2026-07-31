import jittor as jt
import numpy as np

from jgrec.rankers.hybrid.fusion import (
    FusionConfig,
    _feature_normalizer,
    _feature_normalizer_streaming,
    _metrics_from_model,
    _metrics_from_model_streaming,
    _set_jittor_seed_from_rng,
    build_fusion_from_state,
    fit_fusion_mlp,
    fit_fusion_mlp_streaming,
    predict_logits,
)
from jgrec.rankers.hybrid.setwise import setwise_context_features


def _features() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    train = rng.normal(size=(12, 5, 6)).astype(np.float32)
    val = rng.normal(size=(7, 5, 6)).astype(np.float32)
    train[:, 0, 0] += 2.0
    val[:, 0, 0] += 2.0
    return train, val


def test_streaming_normalizer_matches_in_memory_normalizer():
    train, _ = _features()
    feature_indices = (0, 1, 2, 3)

    expected_mean, expected_std = _feature_normalizer(train, feature_indices)
    actual_mean, actual_std = _feature_normalizer_streaming(train, feature_indices, batch_size=3)

    np.testing.assert_allclose(actual_mean, expected_mean, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual_std, expected_std, rtol=1e-6, atol=1e-6)


def test_streaming_metrics_match_in_memory_metrics():
    train, val = _features()
    feature_indices = (0, 1, 2)
    config = FusionConfig(epochs=1, batch_size=3, hidden_dim=8)
    model, result = fit_fusion_mlp(
        train_features=train,
        val_features=val,
        config=config,
        rng=np.random.default_rng(8),
        verbose=False,
        feature_indices=feature_indices,
        candidate_name="test",
    )

    expected = _metrics_from_model(model, val, result.mean, result.std, batch_size=3, feature_indices=feature_indices)
    actual = _metrics_from_model_streaming(
        model,
        val,
        result.mean,
        result.std,
        batch_size=3,
        feature_indices=feature_indices,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_streaming_fusion_keeps_contract_for_memmap_inputs(tmp_path):
    train, val = _features()
    train_path = tmp_path / "train.dat"
    val_path = tmp_path / "val.dat"
    train_store = np.memmap(train_path, mode="w+", dtype=np.float32, shape=train.shape)
    val_store = np.memmap(val_path, mode="w+", dtype=np.float32, shape=val.shape)
    train_store[:] = train
    val_store[:] = val
    train_store.flush()
    val_store.flush()

    model, result = fit_fusion_mlp_streaming(
        train_features=train_store,
        val_features=val_store,
        config=FusionConfig(epochs=1, batch_size=4, hidden_dim=8),
        rng=np.random.default_rng(4),
        verbose=False,
        feature_indices=(0, 1, 2, 3),
        candidate_name="stream",
    )

    assert result.candidate_name == "stream"
    assert result.feature_indices == (0, 1, 2, 3)
    assert result.mean.shape == (4,)
    assert result.std.shape == (4,)
    assert np.isfinite(result.best_val_ap)
    assert np.isfinite(result.best_val_mrr)
    assert model is not None


def test_base_fusion_context_uses_raw_mean_and_max_relative_channels():
    train, val = _features()
    feature_indices = (0, 2, 4)

    model, result = fit_fusion_mlp_streaming(
        train_features=train,
        val_features=val,
        config=FusionConfig(
            epochs=1,
            batch_size=4,
            hidden_dim=8,
            context_transform_version=1,
        ),
        rng=np.random.default_rng(14),
        verbose=False,
        feature_indices=feature_indices,
        candidate_name="base-context",
    )

    expected = setwise_context_features(train[..., feature_indices])
    expected_flat = expected.reshape((-1, expected.shape[-1]))
    assert result.feature_indices == feature_indices
    assert result.mean.shape == (len(feature_indices) * 3,)
    assert result.std.shape == (len(feature_indices) * 3,)
    np.testing.assert_allclose(
        result.mean,
        expected_flat.mean(axis=0),
        rtol=1e-6,
        atol=1e-6,
    )
    assert model.linear1.weight.shape[1] == len(feature_indices) * 3


def test_predict_logits_auto_contextualizes_raw_selected_features():
    train, val = _features()
    feature_indices = (0, 2)
    model, result = fit_fusion_mlp_streaming(
        train_features=train,
        val_features=val,
        config=FusionConfig(
            epochs=1,
            batch_size=4,
            hidden_dim=8,
            context_transform_version=1,
        ),
        rng=np.random.default_rng(15),
        verbose=False,
        feature_indices=feature_indices,
        candidate_name="base-context-predict",
    )
    raw_selected = val[..., feature_indices]
    explicit_context = setwise_context_features(raw_selected)

    actual = predict_logits(
        model,
        raw_selected,
        result.mean,
        result.std,
    )
    expected = predict_logits(
        model,
        explicit_context,
        result.mean,
        result.std,
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_build_fusion_from_state_uses_stored_context_input_width():
    train, val = _features()
    feature_indices = (0, 2)
    model, result = fit_fusion_mlp_streaming(
        train_features=train,
        val_features=val,
        config=FusionConfig(
            epochs=1,
            batch_size=4,
            hidden_dim=8,
            context_transform_version=1,
        ),
        rng=np.random.default_rng(16),
        verbose=False,
        feature_indices=feature_indices,
        candidate_name="base-context-restore",
    )

    restored = build_fusion_from_state(
        input_dim=len(feature_indices),
        hidden_dim=8,
        state=result.state,
    )

    assert restored.linear1.weight.shape == model.linear1.weight.shape
    np.testing.assert_allclose(
        predict_logits(
            restored,
            val[..., feature_indices],
            result.mean,
            result.std,
        ),
        predict_logits(
            model,
            val[..., feature_indices],
            result.mean,
            result.std,
        ),
        rtol=0.0,
        atol=0.0,
    )


def test_fusion_initialization_is_isolated_from_prior_jittor_rng_consumption():
    train, val = _features()
    config = FusionConfig(epochs=0, batch_size=4, hidden_dim=8)

    model_a, result_a = fit_fusion_mlp(
        train_features=train,
        val_features=val,
        config=config,
        rng=np.random.default_rng(123),
        verbose=False,
        feature_indices=(0, 1, 2, 3),
        candidate_name="seed-a",
    )
    _ = jt.rand((128,))
    model_b, result_b = fit_fusion_mlp(
        train_features=train,
        val_features=val,
        config=config,
        rng=np.random.default_rng(123),
        verbose=False,
        feature_indices=(0, 1, 2, 3),
        candidate_name="seed-b",
    )

    assert result_a.best_val_ap == result_b.best_val_ap
    assert result_a.best_val_mrr == result_b.best_val_mrr
    for left, right in zip(model_a.parameters(), model_b.parameters(), strict=True):
        np.testing.assert_allclose(np.asarray(left.numpy()), np.asarray(right.numpy()))


def test_jittor_seed_derivation_does_not_advance_numpy_rng():
    rng = np.random.default_rng(321)
    expected = np.random.default_rng(321)

    _set_jittor_seed_from_rng(rng)

    np.testing.assert_array_equal(
        rng.integers(0, 1_000_000, size=8),
        expected.integers(0, 1_000_000, size=8),
    )
