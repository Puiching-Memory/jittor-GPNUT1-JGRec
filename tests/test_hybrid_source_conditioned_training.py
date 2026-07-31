import jittor as jt
import numpy as np
import pytest

from jgrec.rankers.hybrid.source_conditioned_cst import (
    abcd_model_config,
)
from jgrec.rankers.hybrid.source_conditioned_training import (
    SourceConditionedTrainingConfig,
    fit_source_conditioned_cst,
    load_source_conditioned_checkpoint,
    predict_source_conditioned_logits,
    save_source_conditioned_checkpoint,
)
from jgrec.rankers.hybrid.source_sequence_cache import SourceSequenceRows


@pytest.fixture(autouse=True)
def _cpu_mode():
    original = int(jt.flags.use_cuda)
    jt.flags.use_cuda = 0
    yield
    jt.flags.use_cuda = original


def test_fit_predict_and_reload_source_conditioned_checkpoint(tmp_path):
    rng = np.random.default_rng(9)
    features = rng.normal(size=(12, 5, 3)).astype(np.float32)
    candidates = np.tile(
        np.arange(1, 6, dtype=np.int32),
        (12, 1),
    )
    sequences = SourceSequenceRows(
        items=np.tile(
            np.array([[1, 2, 0, 0]], dtype=np.int32),
            (12, 1),
        ),
        time_buckets=np.tile(
            np.array([[1, 2, 0, 0]], dtype=np.int32),
            (12, 1),
        ),
        lengths=np.full(12, 2, dtype=np.int32),
    )
    positives = np.zeros(12, dtype=np.int32)
    model_config = abcd_model_config(
        "D",
        input_dim=3,
        num_items=8,
        model_dim=8,
        heads=2,
        candidate_layers=1,
        source_layers=1,
        source_max_length=4,
        dropout=0.0,
    )
    training_config = SourceConditionedTrainingConfig(
        epochs=1,
        batch_size=4,
        learning_rate=0.001,
        seed=9,
    )

    model, result = fit_source_conditioned_cst(
        features[:8],
        candidates[:8],
        SourceSequenceRows(
            items=sequences.items[:8],
            time_buckets=sequences.time_buckets[:8],
            lengths=sequences.lengths[:8],
        ),
        positives[:8],
        features[8:],
        candidates[8:],
        SourceSequenceRows(
            items=sequences.items[8:],
            time_buckets=sequences.time_buckets[8:],
            lengths=sequences.lengths[8:],
        ),
        positives[8:],
        model_config=model_config,
        training_config=training_config,
        verbose=False,
    )
    before = predict_source_conditioned_logits(
        model,
        features,
        candidates,
        sequences,
        mean=result.mean,
        std=result.std,
        batch_size=4,
    )

    checkpoint = tmp_path / "source-conditioned.npz"
    save_source_conditioned_checkpoint(checkpoint, model, result)
    loaded_model, loaded_result = load_source_conditioned_checkpoint(
        checkpoint
    )
    after = predict_source_conditioned_logits(
        loaded_model,
        features,
        candidates,
        sequences,
        mean=loaded_result.mean,
        std=loaded_result.std,
        batch_size=4,
    )

    assert before.shape == (12, 5)
    assert np.all(np.isfinite(before))
    np.testing.assert_allclose(after, before, rtol=1e-5, atol=1e-5)
    assert result.trainable_frameworks == ("jittor",)
    assert result.non_jittor_trainable_models == ()
    assert result.best_epoch == 1
