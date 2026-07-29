import jittor as jt
import numpy as np
import pytest

from jgrec.rankers.hybrid.candidate_set_transformer import (
    CandidateSetTransformer,
    CandidateSetTransformerConfig,
    _load_state,
    _snapshot_state,
)
from jgrec.rankers.hybrid.source_conditioned_cst import (
    SourceConditionedCandidateSetTransformer,
    SourceConditionedCSTConfig,
    abcd_model_config,
)


@pytest.fixture(autouse=True)
def _cpu_mode():
    original = int(jt.flags.use_cuda)
    jt.flags.use_cuda = 0
    yield
    jt.flags.use_cuda = original


def _inputs():
    rng = np.random.default_rng(5)
    features = jt.array(
        rng.normal(size=(2, 5, 3)).astype(np.float32),
        dtype=jt.float32,
    )
    candidate_ids = jt.array(
        np.array(
            [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]],
            dtype=np.int32,
        ),
        dtype=jt.int32,
    )
    source_items = jt.array(
        np.array([[1, 6, 0, 0], [4, 7, 8, 0]], dtype=np.int32),
        dtype=jt.int32,
    )
    source_time_buckets = jt.array(
        np.array([[2, 3, 0, 0], [1, 2, 4, 0]], dtype=np.int32),
        dtype=jt.int32,
    )
    source_lengths = jt.array(
        np.array([2, 3], dtype=np.int32),
        dtype=jt.int32,
    )
    return (
        features,
        candidate_ids,
        source_items,
        source_time_buckets,
        source_lengths,
    )


def _model(variant: str):
    jt.set_seed(17)
    model = SourceConditionedCandidateSetTransformer(
        abcd_model_config(
            variant,
            input_dim=3,
            num_items=16,
            model_dim=8,
            heads=2,
            candidate_layers=1,
            source_layers=1,
            source_max_length=4,
            dropout=0.0,
        )
    )
    model.eval()
    return model


def test_abcd_flags_are_the_only_architecture_switches():
    expected = {
        "A": (False, False, True),
        "B": (True, False, True),
        "C": (True, True, False),
        "D": (True, True, True),
    }

    for variant, flags in expected.items():
        config = abcd_model_config(
            variant,
            input_dim=3,
            num_items=16,
        )
        assert (
            config.use_candidate_ids,
            config.use_source_sequence,
            config.use_candidate_self_attention,
        ) == flags


def test_variant_a_is_exactly_the_existing_raw_feature_cst():
    old_config = CandidateSetTransformerConfig(
        input_dim=3,
        model_dim=8,
        heads=2,
        layers=1,
        dropout=0.0,
        feedforward_multiplier=2,
        relative_context="mean_max",
    )
    jt.set_seed(23)
    old = CandidateSetTransformer(old_config)
    jt.set_seed(23)
    new = SourceConditionedCandidateSetTransformer(
        abcd_model_config(
            "A",
            input_dim=3,
            num_items=16,
            model_dim=8,
            heads=2,
            candidate_layers=1,
            source_layers=1,
            source_max_length=4,
            dropout=0.0,
        )
    )
    _load_state(new, _snapshot_state(old))
    old.eval()
    new.eval()
    features = _inputs()[0]

    np.testing.assert_allclose(
        new(features).numpy(),
        old(features).numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("variant", ["A", "B", "C", "D"])
def test_candidate_permutation_only_permutes_scores(variant):
    model = _model(variant)
    inputs = _inputs()
    permutation = np.array([2, 0, 4, 1, 3], dtype=np.int32)

    scores = model(*inputs).numpy()
    permuted = model(
        inputs[0][:, permutation, :],
        inputs[1][:, permutation],
        *inputs[2:],
    ).numpy()

    np.testing.assert_allclose(
        permuted,
        scores[:, permutation],
        rtol=2e-5,
        atol=2e-5,
    )


def test_sequence_inputs_are_ignored_without_sequence_decoder():
    inputs = _inputs()
    changed_items = inputs[2] + 1
    for variant in ("A", "B"):
        model = _model(variant)
        baseline = model(*inputs).numpy()
        changed = model(
            inputs[0],
            inputs[1],
            changed_items,
            inputs[3],
            inputs[4],
        ).numpy()
        np.testing.assert_array_equal(changed, baseline)


@pytest.mark.parametrize("variant", ["C", "D"])
def test_sequence_decoder_changes_scores_when_history_changes(variant):
    inputs = _inputs()
    model = _model(variant)

    baseline = model(*inputs).numpy()
    changed = model(
        inputs[0],
        inputs[1],
        inputs[2] + 1,
        inputs[3],
        inputs[4],
    ).numpy()

    assert not np.allclose(changed, baseline)


def test_padding_tokens_do_not_affect_source_conditioned_scores():
    inputs = _inputs()
    model = _model("D")
    changed_padding = np.asarray(inputs[2].numpy(), dtype=np.int32)
    changed_padding[0, 2:] = [14, 15]
    changed_padding[1, 3] = 13

    baseline = model(*inputs).numpy()
    changed = model(
        inputs[0],
        inputs[1],
        jt.array(changed_padding, dtype=jt.int32),
        inputs[3],
        inputs[4],
    ).numpy()

    np.testing.assert_allclose(changed, baseline, rtol=2e-5, atol=2e-5)


def test_candidate_and_source_paths_share_one_item_embedding():
    model = _model("D")

    embedding_modules = [
        module
        for name, module in model.named_modules()
        if "item_embedding" in name
    ]

    assert len(embedding_modules) == 1
    assert model.candidate_item_embedding is model.source_item_embedding


def test_sequence_variant_requires_all_sequence_tensors():
    model = _model("C")
    features, candidate_ids, *_ = _inputs()

    with pytest.raises(ValueError, match="source sequence"):
        model(features, candidate_ids)


def test_config_rejects_unknown_variant():
    with pytest.raises(ValueError, match="variant"):
        SourceConditionedCSTConfig(
            input_dim=3,
            num_items=16,
            variant="E",
        )
