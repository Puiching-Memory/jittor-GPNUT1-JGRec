import jittor as jt
import numpy as np
import pytest

from jgrec.rankers.hybrid.confidence_routed_topk_id import (
    ConfidenceRouterConfig,
    ConfidenceRouterTrainingConfig,
    SparseRoutingConfig,
    TopKIDCorrection,
    TopKIDCorrectionConfig,
    TopKIDCorrectionTrainingConfig,
    confidence_router_features,
    fit_confidence_router,
    fit_topk_id_correction_fixed,
    hard_confidence_route,
    load_confidence_router_checkpoint,
    load_topk_id_correction_checkpoint,
    predict_topk_id_correction,
    save_confidence_router_checkpoint,
    save_topk_id_correction_checkpoint,
    topk_bounded_correction_scores,
)


@pytest.fixture(autouse=True)
def _cpu_mode():
    original = int(jt.flags.use_cuda)
    jt.flags.use_cuda = 0
    yield
    jt.flags.use_cuda = original


def _base_raw_mask():
    base = np.array(
        [
            [0.8, 0.7, 0.4, 0.1, -0.2],
            [0.2, 0.1, 0.0, -0.1, -0.2],
        ],
        dtype=np.float32,
    )
    raw = np.array(
        [
            [-100.0, 100.0, 80.0, -90.0, 50.0],
            [100.0, -100.0, 60.0, 40.0, -80.0],
        ],
        dtype=np.float32,
    )
    mask = np.array(
        [
            [True, True, True, False, False],
            [True, True, True, False, False],
        ]
    )
    return base, raw, mask


def test_topk_correction_never_changes_candidates_outside_mask_or_exceeds_cap():
    base, raw, mask = _base_raw_mask()
    cap = 0.10

    proposed = topk_bounded_correction_scores(
        jt.array(base, dtype=jt.float32),
        jt.array(raw, dtype=jt.float32),
        jt.array(mask.astype(np.float32), dtype=jt.float32),
        cap=cap,
    ).numpy()

    np.testing.assert_array_equal(proposed[~mask], base[~mask])
    assert np.all(np.abs(proposed - base) <= cap + 1e-6)
    assert np.any(proposed[mask] != base[mask])


def test_hard_route_preserves_unrouted_rows_and_never_exceeds_row_budget():
    base = np.arange(40, dtype=np.float32).reshape(8, 5)
    proposed = base.copy()
    proposed[:, :2] += np.array([0.05, -0.05], dtype=np.float32)
    probabilities = np.array(
        [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1],
        dtype=np.float32,
    )

    routed = hard_confidence_route(
        base,
        proposed,
        probabilities,
        config=SparseRoutingConfig(
            maximum_route_fraction=0.25,
            minimum_probability=0.5,
        ),
    )

    assert routed.route_mask.tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert int(routed.route_mask.sum()) <= 2
    np.testing.assert_array_equal(
        routed.scores[~routed.route_mask],
        base[~routed.route_mask],
    )


def test_topk_proposal_and_router_features_follow_candidate_permutation():
    base, raw, mask = _base_raw_mask()
    candidates = np.array(
        [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]],
        dtype=np.int32,
    )
    support = np.array([0, 12, 8, 4, 2, 1], dtype=np.int64)
    permutation = np.array([2, 0, 4, 1, 3], dtype=np.int32)
    proposed = topk_bounded_correction_scores(
        jt.array(base, dtype=jt.float32),
        jt.array(raw, dtype=jt.float32),
        jt.array(mask.astype(np.float32), dtype=jt.float32),
        cap=0.10,
    ).numpy()
    features, names = confidence_router_features(
        base,
        proposed,
        candidates,
        support,
        mask,
    )
    permuted_proposed = topk_bounded_correction_scores(
        jt.array(base[:, permutation], dtype=jt.float32),
        jt.array(raw[:, permutation], dtype=jt.float32),
        jt.array(
            mask[:, permutation].astype(np.float32),
            dtype=jt.float32,
        ),
        cap=0.10,
    ).numpy()
    permuted_features, permuted_names = confidence_router_features(
        base[:, permutation],
        permuted_proposed,
        candidates[:, permutation],
        support,
        mask[:, permutation],
    )

    np.testing.assert_allclose(
        permuted_proposed,
        proposed[:, permutation],
        rtol=1e-6,
        atol=1e-6,
    )
    assert permuted_names == names
    np.testing.assert_allclose(
        permuted_features,
        features,
        rtol=1e-6,
        atol=1e-6,
    )


def test_zero_initialized_topk_id_model_exactly_reproduces_base():
    base, _, mask = _base_raw_mask()
    candidates = np.array(
        [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]],
        dtype=np.int32,
    )
    model = TopKIDCorrection(
        TopKIDCorrectionConfig(
            num_items=8,
            embedding_dim=4,
            cap=0.10,
            dropout=0.0,
        )
    )
    model.eval()

    scores = model(
        jt.array(base, dtype=jt.float32),
        jt.array(candidates, dtype=jt.int32),
        jt.array(mask.astype(np.float32), dtype=jt.float32),
    ).numpy()

    np.testing.assert_array_equal(scores, base)


def test_topk_correction_training_checkpoint_preserves_sparse_scores(tmp_path):
    rng = np.random.default_rng(23)
    base = rng.normal(size=(16, 5)).astype(np.float32)
    candidates = np.tile(
        np.arange(1, 6, dtype=np.int32),
        (16, 1),
    )
    mask = np.zeros_like(base, dtype=bool)
    top_indices = np.argsort(-base, axis=1, kind="stable")[:, :3]
    np.put_along_axis(mask, top_indices, True, axis=1)
    model, result = fit_topk_id_correction_fixed(
        base,
        candidates,
        mask,
        np.zeros(16, dtype=np.int32),
        model_config=TopKIDCorrectionConfig(
            num_items=8,
            embedding_dim=8,
            cap=0.10,
            dropout=0.0,
        ),
        training_config=TopKIDCorrectionTrainingConfig(
            epochs=1,
            batch_size=4,
            learning_rate=0.01,
            weight_decay=0.001,
            seed=23,
        ),
        verbose=False,
    )
    before = predict_topk_id_correction(
        model,
        base,
        candidates,
        mask,
        batch_size=4,
    )
    path = tmp_path / "correction.npz"
    save_topk_id_correction_checkpoint(path, model, result)
    loaded, loaded_result = load_topk_id_correction_checkpoint(path)
    after = predict_topk_id_correction(
        loaded,
        base,
        candidates,
        mask,
        batch_size=4,
    )

    np.testing.assert_array_equal(before[~mask], base[~mask])
    assert np.all(np.abs(before - base) <= 0.10 + 1e-6)
    np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-6)
    assert loaded_result.trainable_frameworks == ("jittor",)
    assert loaded_result.non_jittor_trainable_models == ()


def test_router_checkpoint_reloads_identical_probabilities(tmp_path):
    rng = np.random.default_rng(19)
    features = rng.normal(size=(32, 11)).astype(np.float32)
    labels = np.array([0, 1] * 16, dtype=np.float32)
    model, result = fit_confidence_router(
        features,
        labels,
        model_config=ConfidenceRouterConfig(
            input_dim=features.shape[1],
            hidden_dim=8,
            dropout=0.0,
        ),
        training_config=ConfidenceRouterTrainingConfig(
            epochs=2,
            batch_size=8,
            learning_rate=0.01,
            weight_decay=0.001,
            seed=19,
        ),
        verbose=False,
    )
    before = result.predict(model, features, batch_size=8)
    path = tmp_path / "router.npz"
    save_confidence_router_checkpoint(path, model, result)
    loaded, loaded_result = load_confidence_router_checkpoint(path)
    after = loaded_result.predict(loaded, features, batch_size=8)

    np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-6)
    assert loaded_result.trainable_frameworks == ("jittor",)
    assert loaded_result.non_jittor_trainable_models == ()


def test_router_with_no_positive_supervision_safely_routes_nothing():
    rng = np.random.default_rng(29)
    features = rng.normal(size=(16, 11)).astype(np.float32)
    labels = np.zeros(16, dtype=np.float32)

    model, result = fit_confidence_router(
        features,
        labels,
        model_config=ConfidenceRouterConfig(
            input_dim=features.shape[1],
            hidden_dim=8,
            dropout=0.0,
        ),
        training_config=ConfidenceRouterTrainingConfig(
            epochs=2,
            batch_size=8,
            learning_rate=0.01,
            weight_decay=0.001,
            seed=29,
        ),
        verbose=False,
    )
    probabilities = result.predict(model, features, batch_size=8)

    assert np.all(probabilities < 0.5)
    assert result.positive_rows == 0
