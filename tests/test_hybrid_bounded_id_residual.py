import jittor as jt
import numpy as np
import pytest

from jgrec.rankers.hybrid.bounded_id_residual import (
    BoundedIDResidual,
    BoundedIDResidualConfig,
    BoundedIDResidualTrainingConfig,
    bounded_id_residual_scores,
    fit_bounded_id_residual_fixed,
    load_bounded_id_residual_checkpoint,
    predict_bounded_id_residual_logits,
    save_bounded_id_residual_checkpoint,
)


@pytest.fixture(autouse=True)
def _cpu_mode():
    original = int(jt.flags.use_cuda)
    jt.flags.use_cuda = 0
    yield
    jt.flags.use_cuda = original


def _base_and_candidates():
    base = np.array(
        [
            [0.1, 0.3, -0.2, 0.7],
            [2.0, 2.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )
    candidates = np.array(
        [[1, 2, 3, 4], [4, 3, 2, 1]],
        dtype=np.int32,
    )
    return base, candidates


def test_bounded_scores_never_exceed_absolute_cap():
    base = np.array(
        [
            [100.0, -100.0, 50.0, -50.0],
            [2.0, 2.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )
    raw_residual = np.array(
        [[1e6, -1e6, 3e5, -4e5], [-1e6, 1e6, 2e6, -2e6]],
        dtype=np.float32,
    )
    cap = 0.05

    scores = bounded_id_residual_scores(
        jt.array(base, dtype=jt.float32),
        jt.array(raw_residual, dtype=jt.float32),
        cap=cap,
    ).numpy()

    assert np.all(np.abs(scores - base) <= cap + 5e-6)


def test_zero_cap_and_zero_initialized_head_exactly_reproduce_base():
    base, candidates = _base_and_candidates()
    raw = jt.array(np.ones_like(base) * 100.0, dtype=jt.float32)
    zero_cap = bounded_id_residual_scores(
        jt.array(base, dtype=jt.float32),
        raw,
        cap=0.0,
    ).numpy()
    model = BoundedIDResidual(
        BoundedIDResidualConfig(
            num_items=8,
            embedding_dim=4,
            cap=0.10,
            dropout=0.0,
        )
    )
    model.eval()
    initial = model(
        jt.array(base, dtype=jt.float32),
        jt.array(candidates, dtype=jt.int32),
    ).numpy()

    np.testing.assert_array_equal(zero_cap, base)
    np.testing.assert_array_equal(initial, base)


def test_config_rejects_cap_above_absolute_experiment_ceiling():
    with pytest.raises(ValueError, match=r"at most 0\.10"):
        BoundedIDResidualConfig(
            num_items=8,
            embedding_dim=4,
            cap=0.1001,
            dropout=0.0,
        )


def test_candidate_permutation_only_permutes_bounded_scores():
    base, candidates = _base_and_candidates()
    model = BoundedIDResidual(
        BoundedIDResidualConfig(
            num_items=8,
            embedding_dim=4,
            cap=0.10,
            dropout=0.0,
        )
    )
    model.output.weight.assign(
        jt.ones_like(model.output.weight) * 5.0
    )
    permutation = np.array([2, 0, 3, 1], dtype=np.int32)
    model.eval()

    scores = model(
        jt.array(base, dtype=jt.float32),
        jt.array(candidates, dtype=jt.int32),
    ).numpy()
    permuted = model(
        jt.array(base[:, permutation], dtype=jt.float32),
        jt.array(candidates[:, permutation], dtype=jt.int32),
    ).numpy()

    np.testing.assert_allclose(
        permuted,
        scores[:, permutation],
        rtol=1e-6,
        atol=1e-6,
    )


def test_fixed_training_checkpoint_reloads_identical_bounded_scores(tmp_path):
    rng = np.random.default_rng(12)
    base = rng.normal(size=(16, 5)).astype(np.float32)
    candidates = np.tile(
        np.arange(1, 6, dtype=np.int32),
        (16, 1),
    )
    positives = np.zeros(16, dtype=np.int32)
    config = BoundedIDResidualConfig(
        num_items=8,
        embedding_dim=8,
        cap=0.05,
        dropout=0.0,
    )
    training = BoundedIDResidualTrainingConfig(
        epochs=1,
        batch_size=4,
        learning_rate=0.01,
        weight_decay=0.001,
        seed=12,
    )

    model, result = fit_bounded_id_residual_fixed(
        base,
        candidates,
        positives,
        model_config=config,
        training_config=training,
        verbose=False,
    )
    before = predict_bounded_id_residual_logits(
        model,
        base,
        candidates,
        batch_size=4,
    )
    checkpoint = tmp_path / "bounded-id-residual.npz"
    save_bounded_id_residual_checkpoint(checkpoint, model, result)
    loaded, loaded_result = load_bounded_id_residual_checkpoint(
        checkpoint
    )
    after = predict_bounded_id_residual_logits(
        loaded,
        base,
        candidates,
        batch_size=4,
    )
    assert not np.array_equal(before, base)
    assert np.all(np.abs(before - base) <= config.cap + 1e-6)
    np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-6)
    assert loaded_result.trainable_frameworks == ("jittor",)
    assert loaded_result.non_jittor_trainable_models == ()
