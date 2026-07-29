import numpy as np

from jgrec.rankers.hybrid.disagreement_temporal_correction import (
    hybrid_consensus_signal,
    oof_disagreement_signal,
    proposal_router_features,
    row_percentile_scores,
    score_multiset_correction_audit,
    strict_temporal_support_signal,
    topk_score_multiset_proposal,
)


def test_topk_proposal_preserves_exact_row_multiset_and_outside_positions():
    base = np.array(
        [
            [0.8, 0.7, 0.4, 0.1, -0.2],
            [0.2, -0.1, 0.5, 0.3, -0.2],
        ],
        dtype=np.float32,
    )
    signal = np.array(
        [
            [0.1, 0.9, 0.8, 1.0, 0.0],
            [0.7, 1.0, 0.1, 0.9, 0.8],
        ],
        dtype=np.float32,
    )

    proposed = topk_score_multiset_proposal(base, signal, top_k=3)
    top = np.argsort(-base, axis=1, kind="stable")[:, :3]
    mask = np.zeros_like(base, dtype=bool)
    np.put_along_axis(mask, top, True, axis=1)

    np.testing.assert_array_equal(proposed[~mask], base[~mask])
    np.testing.assert_array_equal(
        np.sort(proposed, axis=1),
        np.sort(base, axis=1),
    )
    assert proposed.dtype == base.dtype
    assert np.any(proposed[mask] != base[mask])

    route_mask = np.array([True, False])
    routed = base.copy()
    routed[route_mask] = proposed[route_mask]
    audit = score_multiset_correction_audit(
        base,
        proposed,
        routed,
        route_mask,
        top_k=3,
        maximum_route_fraction=0.5,
    )

    assert audit["passed"]
    assert audit["proposal_score_multisets_exact"]
    assert audit["routed_score_multisets_exact"]
    assert audit["topk_outside_exact"]
    assert audit["unrouted_rows_exact"]


def test_oof_signal_and_topk_proposal_follow_candidate_permutation():
    experts = np.array(
        [
            [[0.9, 0.1, 0.6, -0.2], [0.1, 0.8, 0.4, 0.2]],
            [[0.7, 0.2, 0.8, -0.1], [0.2, 0.7, 0.5, 0.1]],
            [[0.8, 0.3, 0.7, 0.0], [0.3, 0.9, 0.6, 0.0]],
        ],
        dtype=np.float32,
    )
    base = experts[0]
    permutation = np.array([2, 0, 3, 1], dtype=np.int32)

    original = oof_disagreement_signal(experts)
    permuted = oof_disagreement_signal(experts[:, :, permutation])
    proposed = topk_score_multiset_proposal(
        base,
        original.candidate_scores,
        top_k=3,
    )
    permuted_proposed = topk_score_multiset_proposal(
        base[:, permutation],
        permuted.candidate_scores,
        top_k=3,
    )

    np.testing.assert_allclose(
        permuted.candidate_scores,
        original.candidate_scores[:, permutation],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        permuted.row_features,
        original.row_features,
        rtol=0.0,
        atol=1e-7,
    )
    assert permuted.feature_names == original.feature_names
    np.testing.assert_array_equal(
        permuted_proposed,
        proposed[:, permutation],
    )


def test_oof_disagreement_rewards_unanimous_alternative_and_reports_consensus():
    experts = np.array(
        [
            [[0.2, 0.9, 0.5, 0.1]],
            [[0.1, 0.8, 0.4, 0.2]],
            [[0.3, 0.7, 0.6, 0.0]],
        ],
        dtype=np.float32,
    )

    result = oof_disagreement_signal(experts)

    assert int(np.argmax(result.candidate_scores[0])) == 1
    values = dict(
        zip(
            result.feature_names,
            result.row_features[0],
            strict=True,
        )
    )
    assert values["expert_top1_vote_fraction"] == 1.0
    assert values["consensus_top1_rank_std"] == 0.0
    assert values["consensus_margin"] > 0.0


def test_strict_temporal_support_excludes_equal_and_future_events():
    history_src = np.array([1, 1, 2, 1, 1], dtype=np.int32)
    history_dst = np.array([11, 11, 12, 12, 13], dtype=np.int32)
    history_time = np.array([1, 2, 2, 4, 5], dtype=np.int64)
    query_src = np.array([1, 2], dtype=np.int32)
    candidates = np.array(
        [[11, 12, 13, 99], [11, 12, 13, 99]],
        dtype=np.int32,
    )

    with_equal_and_future = strict_temporal_support_signal(
        history_src,
        history_dst,
        history_time,
        query_src,
        candidates,
        origin_time=4,
        recent_rows=3,
    )
    pre_origin_only = strict_temporal_support_signal(
        history_src[:3],
        history_dst[:3],
        history_time[:3],
        query_src,
        candidates,
        origin_time=4,
        recent_rows=3,
    )

    np.testing.assert_array_equal(
        with_equal_and_future.candidate_scores,
        pre_origin_only.candidate_scores,
    )
    np.testing.assert_array_equal(
        with_equal_and_future.row_features,
        pre_origin_only.row_features,
    )
    assert int(np.argmax(with_equal_and_future.candidate_scores[0])) == 0
    assert int(np.argmax(with_equal_and_future.candidate_scores[1])) == 1


def test_temporal_and_hybrid_signals_follow_candidate_permutation():
    history_src = np.array([1, 1, 2, 2, 1], dtype=np.int32)
    history_dst = np.array([10, 11, 11, 12, 10], dtype=np.int32)
    history_time = np.array([1, 2, 2, 3, 3], dtype=np.int64)
    query_src = np.array([1, 2], dtype=np.int32)
    candidates = np.array(
        [[10, 11, 12, 13], [10, 11, 12, 13]],
        dtype=np.int32,
    )
    permutation = np.array([2, 0, 3, 1], dtype=np.int32)
    temporal = strict_temporal_support_signal(
        history_src,
        history_dst,
        history_time,
        query_src,
        candidates,
        origin_time=4,
        recent_rows=4,
    )
    permuted_temporal = strict_temporal_support_signal(
        history_src,
        history_dst,
        history_time,
        query_src,
        candidates[:, permutation],
        origin_time=4,
        recent_rows=4,
    )
    oof = oof_disagreement_signal(
        np.array(
            [
                [[0.8, 0.4, 0.2, 0.1], [0.1, 0.8, 0.6, 0.2]],
                [[0.7, 0.5, 0.3, 0.0], [0.2, 0.7, 0.5, 0.1]],
                [[0.9, 0.3, 0.1, -0.1], [0.3, 0.9, 0.4, 0.0]],
            ],
            dtype=np.float32,
        )
    )
    hybrid = hybrid_consensus_signal(oof, temporal)
    permuted_hybrid = hybrid_consensus_signal(
        oof_disagreement_signal(
            np.array(
                [
                    [[0.8, 0.4, 0.2, 0.1], [0.1, 0.8, 0.6, 0.2]],
                    [[0.7, 0.5, 0.3, 0.0], [0.2, 0.7, 0.5, 0.1]],
                    [[0.9, 0.3, 0.1, -0.1], [0.3, 0.9, 0.4, 0.0]],
                ],
                dtype=np.float32,
            )[:, :, permutation]
        ),
        permuted_temporal,
    )

    np.testing.assert_array_equal(
        permuted_temporal.candidate_scores,
        temporal.candidate_scores[:, permutation],
    )
    np.testing.assert_allclose(
        permuted_temporal.row_features,
        temporal.row_features,
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        permuted_hybrid.candidate_scores,
        hybrid.candidate_scores[:, permutation],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        permuted_hybrid.row_features,
        hybrid.row_features,
        rtol=0.0,
        atol=1e-7,
    )


def test_row_percentile_scores_are_monotonic_and_bounded():
    values = np.array(
        [[3.0, 1.0, 4.0, 2.0], [-2.0, 2.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    percentiles = row_percentile_scores(values)

    assert percentiles.dtype == np.float32
    assert float(percentiles.min()) == 0.0
    assert float(percentiles.max()) == 1.0
    np.testing.assert_array_equal(
        np.argmax(percentiles, axis=1),
        np.argmax(values, axis=1),
    )


def test_router_features_are_label_free_and_candidate_permutation_invariant():
    experts = np.array(
        [
            [[0.9, 0.1, 0.6, -0.2], [0.1, 0.8, 0.4, 0.2]],
            [[0.7, 0.2, 0.8, -0.1], [0.2, 0.7, 0.5, 0.1]],
            [[0.8, 0.3, 0.7, 0.0], [0.3, 0.9, 0.6, 0.0]],
        ],
        dtype=np.float32,
    )
    base = experts[0]
    signal = oof_disagreement_signal(experts)
    proposed = topk_score_multiset_proposal(
        base,
        signal.candidate_scores,
        top_k=3,
    )
    features, names = proposal_router_features(
        base,
        proposed,
        signal,
        top_k=3,
    )
    permutation = np.array([2, 0, 3, 1], dtype=np.int32)
    permuted_signal = oof_disagreement_signal(experts[:, :, permutation])
    permuted_proposed = topk_score_multiset_proposal(
        base[:, permutation],
        permuted_signal.candidate_scores,
        top_k=3,
    )
    permuted_features, permuted_names = proposal_router_features(
        base[:, permutation],
        permuted_proposed,
        permuted_signal,
        top_k=3,
    )

    assert names == permuted_names
    assert features.shape[0] == base.shape[0]
    assert np.isfinite(features).all()
    np.testing.assert_allclose(
        permuted_features,
        features,
        rtol=0.0,
        atol=1e-7,
    )
