from __future__ import annotations

import pickle
import sys
import types
from dataclasses import replace

import numpy as np
import pytest

from jgrec.core.types import TestQueryArray
from jgrec.rankers.hybrid.cooccur_lift_checkpoint import (
    CausalLiftFeatureStore,
    CooccurLiftAuxiliaryState,
    fingerprint_queries,
    install_cooccur_lift_auxiliary_state,
    predict_cooccur_lift_auxiliary_probabilities,
    validate_online_promotion_receipt,
)


def _queries() -> TestQueryArray:
    return TestQueryArray(
        src=np.asarray([9, 7, 9], dtype=np.int32),
        time=np.asarray([100, 110, 120], dtype=np.int64),
        candidates=np.asarray(
            [
                [31, 32, 33],
                [41, 42, 43],
                [51, 52, 53],
            ],
            dtype=np.int32,
        ),
    )


def test_lift_store_lookup_is_independent_of_query_order() -> None:
    queries = _queries()
    lift = np.arange(18, dtype=np.float32).reshape(3, 3, 2)
    store = CausalLiftFeatureStore.from_queries(queries, lift)
    order = np.asarray([2, 0, 1])

    restored = store.lookup(queries[order])

    np.testing.assert_array_equal(restored, lift[order])
    assert store.query_fingerprints.shape == (3, 32)
    assert store.lift_features.dtype == np.float32


def test_lift_store_rejects_unknown_or_malformed_queries() -> None:
    queries = _queries()
    lift = np.zeros((3, 3, 2), dtype=np.float32)
    store = CausalLiftFeatureStore.from_queries(queries, lift)
    unknown = TestQueryArray(
        src=np.asarray([99], dtype=np.int32),
        time=np.asarray([100], dtype=np.int64),
        candidates=np.asarray([[31, 32, 33]], dtype=np.int32),
    )

    with pytest.raises(KeyError, match="not present"):
        store.lookup(unknown)
    with pytest.raises(ValueError, match="lift feature shape"):
        CausalLiftFeatureStore.from_queries(
            queries,
            np.zeros((3, 3, 3), dtype=np.float32),
        )


def test_query_fingerprint_binds_src_time_and_candidate_order() -> None:
    queries = _queries()
    baseline = fingerprint_queries(queries)
    changed_src = TestQueryArray(
        src=queries.src + 1,
        time=queries.time,
        candidates=queries.candidates,
    )
    changed_time = TestQueryArray(
        src=queries.src,
        time=queries.time + 1,
        candidates=queries.candidates,
    )
    reversed_candidates = TestQueryArray(
        src=queries.src,
        time=queries.time,
        candidates=queries.candidates[:, ::-1],
    )

    assert not np.any(np.all(baseline == fingerprint_queries(changed_src), axis=1))
    assert not np.any(np.all(baseline == fingerprint_queries(changed_time), axis=1))
    assert not np.any(
        np.all(
            baseline == fingerprint_queries(reversed_candidates),
            axis=1,
        )
    )


def test_install_auxiliary_state_only_adds_locked_checkpoint_field() -> None:
    queries = _queries()
    source = {
        "feature_names": tuple(
            "gnn_short" if index == 59 else f"feature_{index}"
            for index in range(63)
        ),
        "encoder": {"protected": np.asarray([1, 2, 3])},
        "cooccur_lift_auxiliary_state": None,
    }
    source_bytes = pickle.dumps(
        source,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    state = CooccurLiftAuxiliaryState(
        integration_id="cooccur_lift_aux_expert_v1",
        weight=0.5,
        hidden_dim=4,
        gnn_short_column=59,
        model_state={"weight": np.ones((4, 1), dtype=np.float32)},
        mean=np.zeros(195, dtype=np.float32),
        std=np.ones(195, dtype=np.float32),
        feature_indices=tuple(range(195)),
        feature_store=CausalLiftFeatureStore.from_queries(
            queries,
            np.zeros((3, 3, 2), dtype=np.float32),
        ),
        provenance={"selection_lock_sha256": "lock-sha"},
    )

    candidate = install_cooccur_lift_auxiliary_state(source, state)

    assert set(candidate) == {
        *source,
        "cooccur_lift_auxiliary_state",
    }
    assert pickle.dumps(
        source,
        protocol=pickle.HIGHEST_PROTOCOL,
    ) == source_bytes
    assert candidate["encoder"] is source["encoder"]
    assert candidate["cooccur_lift_auxiliary_state"] is state


def test_online_promotion_receipt_rejects_score_or_authority_drift() -> None:
    receipt = {
        "schema_version": 1,
        "status": "online_score_passed_before_checkpoint_wiring",
        "integration_id": "cooccur_lift_aux_expert_v1",
        "online_score": 1.357529740346302,
        "promotion_threshold": 1.3557002251184347,
        "delta_online_minus_threshold": 0.0018295152278673,
        "threshold_comparison": "strictly_greater",
        "candidate_zip_sha256": "a" * 64,
        "candidate_report_sha256": "b" * 64,
        "selection_lock_sha256": "c" * 64,
        "external_report_sha256": "d" * 64,
        "source_checkpoint_sha256": "e" * 64,
        "auxiliary_model_sha256": "f" * 64,
        "selected_weight": 0.5,
        "checkpoint_wiring_authorized": True,
        "double_replay_required": True,
        "weight_rescan_authorized": False,
        "formula_change_authorized": False,
        "model_retraining_authorized": False,
    }

    validate_online_promotion_receipt(receipt)
    for key, value in (
        ("online_score", receipt["promotion_threshold"]),
        ("checkpoint_wiring_authorized", False),
        ("selected_weight", 0.4),
        ("formula_change_authorized", True),
    ):
        drifted = dict(receipt)
        drifted[key] = value
        with pytest.raises(ValueError):
            validate_online_promotion_receipt(drifted)


def test_auxiliary_state_rejects_weight_or_context_drift() -> None:
    queries = _queries()
    state = CooccurLiftAuxiliaryState(
        integration_id="cooccur_lift_aux_expert_v1",
        weight=0.5,
        hidden_dim=4,
        gnn_short_column=59,
        model_state={"weight": np.ones((4, 1), dtype=np.float32)},
        mean=np.zeros(195, dtype=np.float32),
        std=np.ones(195, dtype=np.float32),
        feature_indices=tuple(range(195)),
        feature_store=CausalLiftFeatureStore.from_queries(
            queries,
            np.zeros((3, 3, 2), dtype=np.float32),
        ),
        provenance={"selection_lock_sha256": "lock-sha"},
    )

    with pytest.raises(ValueError, match="weight"):
        replace(state, weight=0.4)
    with pytest.raises(ValueError, match="195"):
        replace(state, feature_indices=tuple(range(194)))


def test_auxiliary_prediction_is_bracketed_by_jittor_cache_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jgrec.rankers.hybrid import (  # noqa: PLC0415
        cooccur_lift_checkpoint as checkpoint_module,
    )

    queries = _queries()
    state = CooccurLiftAuxiliaryState(
        integration_id="cooccur_lift_aux_expert_v1",
        weight=0.5,
        hidden_dim=4,
        gnn_short_column=1,
        model_state={"weight": np.ones((4, 1), dtype=np.float32)},
        mean=np.zeros(195, dtype=np.float32),
        std=np.ones(195, dtype=np.float32),
        feature_indices=tuple(range(195)),
        feature_store=CausalLiftFeatureStore.from_queries(
            queries,
            np.zeros((3, 3, 2), dtype=np.float32),
        ),
        provenance={"selection_lock_sha256": "lock-sha"},
    )
    events: list[str] = []

    def fake_predict_logits(*args: object, **kwargs: object) -> np.ndarray:
        events.append("predict")
        return np.zeros((3, 3), dtype=np.float32)

    fusion_module = types.ModuleType("jgrec.rankers.hybrid.fusion")
    fusion_module.predict_logits = fake_predict_logits  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "jgrec.rankers.hybrid.fusion",
        fusion_module,
    )
    monkeypatch.setattr(
        checkpoint_module,
        "_synchronize_and_clean_jittor",
        lambda: events.append("clean"),
        raising=False,
    )

    probabilities = predict_cooccur_lift_auxiliary_probabilities(
        state,
        object(),
        np.zeros((3, 3, 63), dtype=np.float32),
        queries,
    )

    assert events == ["clean", "predict", "clean"]
    np.testing.assert_allclose(probabilities, np.full((3, 3), 1.0 / 3.0))


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Jittor checkpoint runtime is verified on Linux",
)
def test_hybrid_snapshot_hydrates_and_blends_cooccur_lift_auxiliary() -> None:
    import jittor as jt  # noqa: PLC0415

    from jgrec.contest_checkpoint import get_model_state  # noqa: PLC0415
    from jgrec.core.types import InteractionTable  # noqa: PLC0415
    from jgrec.rankers.hybrid.config import TrainingConfig  # noqa: PLC0415
    from jgrec.rankers.hybrid.cooccur_lift import (  # noqa: PLC0415
        CooccurLiftAugmentedView,
    )
    from jgrec.rankers.hybrid.fusion import (  # noqa: PLC0415
        FusionMLP,
        FusionResult,
        predict_logits,
    )
    from jgrec.rankers.hybrid.ranker import (  # noqa: PLC0415
        TemporalHybridRanker,
    )
    from jgrec.rankers.hybrid.setwise import (  # noqa: PLC0415
        SetwiseFeatureView,
    )

    jt.flags.use_cuda = 0
    interactions = InteractionTable.from_array(
        np.asarray(
            [
                [src, 10 + (event % 5), event + 1]
                for event, src in enumerate((1, 2, 3, 4) * 30)
            ],
            dtype=np.int32,
        )
    )
    config = TrainingConfig(
        val_ratio=0.2,
        context_ratio=0.5,
        max_train_events=8,
        max_val_events=8,
        num_negatives=2,
        epochs=1,
        train_batch_size=8,
        auto_strategy_enabled=False,
        candidate_prior_enabled=False,
        target_window_enabled=False,
        structure_enabled=False,
        source_profile_enabled=False,
        two_tower_enabled=False,
        gnn_enabled=False,
        seq_enabled=False,
        encoder_state_cache_enabled=False,
        verbose=False,
    )
    queries = TestQueryArray(
        src=np.asarray([1, 2], dtype=np.int32),
        time=np.asarray([121, 121], dtype=np.int64),
        candidates=np.asarray(
            [[10, 11, 12], [12, 13, 14]],
            dtype=np.int32,
        ),
    )
    ranker = TemporalHybridRanker(recent_window=4)
    ranker.fit(interactions, config)
    raw_features = ranker.encoder.features_for_queries(queries)
    baseline_model = FusionMLP(
        input_dim=raw_features.shape[-1],
        hidden_dim=4,
    )
    baseline_result = FusionResult(
        best_val_ap=0.0,
        best_val_mrr=0.0,
        state=get_model_state(baseline_model),
        mean=np.zeros(raw_features.shape[-1], dtype=np.float32),
        std=np.ones(raw_features.shape[-1], dtype=np.float32),
        feature_indices=tuple(range(raw_features.shape[-1])),
        candidate_name="cooccur_lift_checkpoint_baseline",
    )
    ranker.fusion = baseline_model
    ranker.fusion_result = baseline_result
    ranker._fusion_hidden_dim = 4
    ranker.lgbm_result = None
    ranker.setwise_fusion = None
    ranker.setwise_fusion_result = None
    baseline = ranker.predict_batch(queries)
    lift = np.asarray(
        [
            [[1.0, 0.5], [0.0, 0.0], [-1.0, -0.5]],
            [[0.0, 1.0], [1.0, 0.0], [-1.0, -1.0]],
        ],
        dtype=np.float32,
    )
    gnn_short_column = ranker.feature_names.index("gnn_short")
    augmented = CooccurLiftAugmentedView(
        raw_features,
        short_none_scores=raw_features[..., gnn_short_column],
        gnn_short_column=gnn_short_column,
        lift_features=lift,
    )
    setwise = np.asarray(
        SetwiseFeatureView(augmented, transform_version=1)[:],
        dtype=np.float32,
    )
    auxiliary_model = FusionMLP(input_dim=195, hidden_dim=4)
    model_state = get_model_state(auxiliary_model)
    mean = np.zeros(195, dtype=np.float32)
    std = np.ones(195, dtype=np.float32)
    logits = predict_logits(
        auxiliary_model,
        setwise,
        mean,
        std,
    )
    shifted = logits - logits.max(axis=1, keepdims=True)
    auxiliary = np.exp(shifted) / np.exp(shifted).sum(
        axis=1,
        keepdims=True,
    )
    state = CooccurLiftAuxiliaryState(
        integration_id="cooccur_lift_aux_expert_v1",
        weight=0.5,
        hidden_dim=4,
        gnn_short_column=gnn_short_column,
        model_state=model_state,
        mean=mean,
        std=std,
        feature_indices=tuple(range(195)),
        feature_store=CausalLiftFeatureStore.from_queries(queries, lift),
        provenance={"selection_lock_sha256": "lock-sha"},
    )
    candidate_snapshot = install_cooccur_lift_auxiliary_state(
        ranker.snapshot(),
        state,
    )

    restored = TemporalHybridRanker()
    restored.hydrate(candidate_snapshot)
    actual = restored.predict_batch(queries[::-1])

    np.testing.assert_allclose(
        actual,
        (baseline[::-1] + auxiliary[::-1]) / 2.0,
        rtol=0.0,
        atol=1e-7,
    )
    assert (
        restored.snapshot()["cooccur_lift_auxiliary_state"]
        is not None
    )
