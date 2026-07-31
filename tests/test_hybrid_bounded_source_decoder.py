import jittor as jt
import numpy as np
import pytest

from jgrec.rankers.hybrid.bounded_source_decoder import (
    BoundedSourceDecoder,
    BoundedSourceDecoderConfig,
    BoundedSourceDecoderTrainingConfig,
    bounded_source_residual_scores,
    fit_bounded_source_decoder_fixed,
    load_bounded_source_decoder_checkpoint,
    predict_bounded_source_decoder_logits,
    save_bounded_source_decoder_checkpoint,
    support_shrinkage,
)
from jgrec.rankers.hybrid.source_sequence_cache import SourceSequenceRows


@pytest.fixture(autouse=True)
def _cpu_mode():
    original = int(jt.flags.use_cuda)
    jt.flags.use_cuda = 0
    yield
    jt.flags.use_cuda = original


def _inputs(rows: int = 2):
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
    support = np.array(
        [[1.0, 10.0, 100.0, 1000.0], [1000.0, 100.0, 10.0, 1.0]],
        dtype=np.float32,
    )
    sequences = SourceSequenceRows(
        items=np.array([[1, 3, 0], [4, 2, 1]], dtype=np.int32),
        time_buckets=np.array([[1, 2, 0], [2, 3, 4]], dtype=np.int32),
        lengths=np.array([2, 3], dtype=np.int32),
    )
    return (
        base[:rows],
        candidates[:rows],
        support[:rows],
        SourceSequenceRows(
            items=sequences.items[:rows],
            time_buckets=sequences.time_buckets[:rows],
            lengths=sequences.lengths[:rows],
        ),
    )


def _config(cap: float = 0.05):
    return BoundedSourceDecoderConfig(
        num_items=8,
        embedding_dim=8,
        heads=2,
        source_max_length=3,
        time_bucket_count=16,
        cap=cap,
        support_tau=10.0,
        dropout=0.0,
    )


def test_empty_history_exactly_reproduces_frozen_base_even_with_nonzero_head():
    base, candidates, support, sequences = _inputs()
    empty = SourceSequenceRows(
        items=sequences.items,
        time_buckets=sequences.time_buckets,
        lengths=np.zeros_like(sequences.lengths),
    )
    model = BoundedSourceDecoder(_config())
    model.residual_head.weight.assign(
        jt.ones_like(model.residual_head.weight) * 10.0
    )
    if model.residual_head.bias is not None:
        model.residual_head.bias.assign(
            jt.ones_like(model.residual_head.bias) * 10.0
        )
    model.eval()

    scores = model(
        jt.array(base, dtype=jt.float32),
        jt.array(candidates, dtype=jt.int32),
        jt.array(empty.items, dtype=jt.int32),
        jt.array(empty.time_buckets, dtype=jt.int32),
        jt.array(empty.lengths, dtype=jt.int32),
        jt.array(support, dtype=jt.float32),
    ).numpy()

    np.testing.assert_array_equal(scores, base)


def test_residual_projection_is_row_centered_and_never_exceeds_cap():
    base = np.array(
        [[100.0, -100.0, 50.0, -50.0], [2.0, 2.0, 2.0, 2.0]],
        dtype=np.float32,
    )
    raw = np.array(
        [[1e6, -1e6, 3e5, -4e5], [-1e6, 1e6, 2e6, -2e6]],
        dtype=np.float32,
    )
    shrinkage = np.array(
        [[0.1, 1.0, 0.5, 0.8], [1.0, 1.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    has_history = np.array([1.0, 0.0], dtype=np.float32)
    cap = 0.05

    scores = bounded_source_residual_scores(
        jt.array(base, dtype=jt.float32),
        jt.array(raw, dtype=jt.float32),
        jt.array(shrinkage, dtype=jt.float32),
        jt.array(has_history, dtype=jt.float32),
        cap=cap,
    ).numpy()
    residual = scores - base

    assert np.all(np.abs(residual) <= cap + 5e-6)
    np.testing.assert_allclose(
        np.mean(residual, axis=1),
        np.zeros(2),
        rtol=0.0,
        atol=5e-6,
    )
    np.testing.assert_array_equal(scores[1], base[1])


def test_support_shrinkage_is_zero_for_unseen_and_monotonic():
    support = jt.array(
        np.array([[0.0, 1.0, 10.0, 1000.0]], dtype=np.float32)
    )

    values = support_shrinkage(support, tau=10.0).numpy()[0]

    assert values[0] == 0.0
    assert 0.0 < values[1] < values[2] < values[3] < 1.0


def test_candidate_permutation_only_permutes_bounded_decoder_scores():
    base, candidates, support, sequences = _inputs()
    model = BoundedSourceDecoder(_config())
    model.residual_head.weight.assign(
        jt.ones_like(model.residual_head.weight)
    )
    permutation = np.array([2, 0, 3, 1], dtype=np.int32)
    model.eval()

    scores = model(
        jt.array(base, dtype=jt.float32),
        jt.array(candidates, dtype=jt.int32),
        jt.array(sequences.items, dtype=jt.int32),
        jt.array(sequences.time_buckets, dtype=jt.int32),
        jt.array(sequences.lengths, dtype=jt.int32),
        jt.array(support, dtype=jt.float32),
    ).numpy()
    permuted = model(
        jt.array(base[:, permutation], dtype=jt.float32),
        jt.array(candidates[:, permutation], dtype=jt.int32),
        jt.array(sequences.items, dtype=jt.int32),
        jt.array(sequences.time_buckets, dtype=jt.int32),
        jt.array(sequences.lengths, dtype=jt.int32),
        jt.array(support[:, permutation], dtype=jt.float32),
    ).numpy()

    np.testing.assert_allclose(
        permuted,
        scores[:, permutation],
        rtol=1e-6,
        atol=1e-6,
    )


def test_training_checkpoint_reloads_identical_bounded_scores(tmp_path):
    rng = np.random.default_rng(12)
    rows = 16
    width = 5
    base = rng.normal(size=(rows, width)).astype(np.float32)
    candidates = np.tile(
        np.arange(1, width + 1, dtype=np.int32),
        (rows, 1),
    )
    support = np.tile(
        np.array([100, 50, 20, 10, 5], dtype=np.float32),
        (rows, 1),
    )
    sequences = SourceSequenceRows(
        items=np.tile(
            np.array([1, 3, 5], dtype=np.int32),
            (rows, 1),
        ),
        time_buckets=np.tile(
            np.array([1, 2, 3], dtype=np.int32),
            (rows, 1),
        ),
        lengths=np.full(rows, 3, dtype=np.int32),
    )
    config = BoundedSourceDecoderConfig(
        num_items=8,
        embedding_dim=8,
        heads=2,
        source_max_length=3,
        time_bucket_count=16,
        cap=0.05,
        support_tau=10.0,
        dropout=0.0,
    )
    training = BoundedSourceDecoderTrainingConfig(
        epochs=1,
        batch_size=4,
        learning_rate=0.01,
        weight_decay=0.01,
        seed=12,
    )
    model, result = fit_bounded_source_decoder_fixed(
        base,
        candidates,
        sequences,
        support,
        np.zeros(rows, dtype=np.int32),
        model_config=config,
        training_config=training,
        verbose=False,
    )
    before = predict_bounded_source_decoder_logits(
        model,
        base,
        candidates,
        sequences,
        support,
        batch_size=4,
    )
    checkpoint = tmp_path / "bounded-source-decoder.npz"
    save_bounded_source_decoder_checkpoint(checkpoint, model, result)
    loaded, loaded_result = load_bounded_source_decoder_checkpoint(checkpoint)
    after = predict_bounded_source_decoder_logits(
        loaded,
        base,
        candidates,
        sequences,
        support,
        batch_size=4,
    )

    assert np.max(np.abs(before - base)) <= config.cap + 1e-6
    np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-6)
    assert loaded_result.trainable_frameworks == ("jittor",)
    assert loaded_result.non_jittor_trainable_models == ()
