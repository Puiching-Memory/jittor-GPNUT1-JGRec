from __future__ import annotations

import pytest

from jgrec.feature_mask_validation import build_feature_mask_candidates


def test_feature_mask_candidates_cover_full_and_every_exact_group_complement():
    feature_names = (
        "stat_a",
        "stat_b",
        "prior_a",
        "target_a",
        "target_b",
        "gnn_a",
    )
    groups = {
        "stats": ("stat_a", "stat_b"),
        "prior": ("prior_a",),
        "target": ("target_a", "target_b"),
        "gnn": ("gnn_a",),
    }
    shared_config = {
        "context_transform_version": 1,
        "epochs": 15,
        "final_integration": {"mlp_weight": 0.3},
    }

    candidates = build_feature_mask_candidates(
        feature_names=feature_names,
        feature_groups=groups,
        shared_config=shared_config,
    )

    assert [candidate.candidate_id for candidate in candidates] == [
        "full_enabled",
        "loo_without_stats",
        "loo_without_prior",
        "loo_without_target",
        "loo_without_gnn",
    ]
    assert candidates[0].feature_indices == tuple(range(len(feature_names)))
    for candidate, removed_group in zip(
        candidates[1:],
        groups,
        strict=True,
    ):
        removed = set(groups[removed_group])
        expected = tuple(
            index
            for index, name in enumerate(feature_names)
            if name not in removed
        )
        assert candidate.removed_group == removed_group
        assert candidate.feature_indices == expected
        assert candidate.config["feature_indices"] == list(expected)
        assert candidate.config["shared"] == shared_config
        assert len(candidate.config_sha256) == 64
    assert len(
        {candidate.feature_indices for candidate in candidates}
    ) == len(candidates)
    assert [candidate.tie_break_priority for candidate in candidates] == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_feature_mask_candidates_reject_incomplete_or_duplicate_group_schema():
    with pytest.raises(
        ValueError,
        match="exactly once",
    ):
        build_feature_mask_candidates(
            feature_names=("a", "b"),
            feature_groups={"first": ("a",), "second": ("a",)},
            shared_config={},
        )
