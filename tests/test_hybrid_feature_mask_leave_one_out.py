from __future__ import annotations

import pytest

from jgrec.rankers.hybrid.candidate_prior import CANDIDATE_PRIOR_FEATURE_NAMES
from jgrec.rankers.hybrid.config import (
    GRAPH_WINDOW_NAMES,
    SEQUENCE_FEATURE_NAMES,
    SOURCE_PROFILE_FEATURE_NAMES,
    TARGET_WINDOW_FEATURE_NAMES,
    TWO_TOWER_FEATURE_NAMES,
    TrainingConfig,
)
from jgrec.rankers.hybrid.ranker import _feature_mask_catalog, _feature_masks
from jgrec.rankers.hybrid.stats import STAT_FEATURE_NAMES
from jgrec.rankers.hybrid.structure import STRUCTURE_FEATURE_NAMES

GROUP_SIZES = {
    "stats": len(STAT_FEATURE_NAMES),
    "prior": len(CANDIDATE_PRIOR_FEATURE_NAMES),
    "target": len(TARGET_WINDOW_FEATURE_NAMES),
    "structure": len(STRUCTURE_FEATURE_NAMES),
    "profile": len(SOURCE_PROFILE_FEATURE_NAMES),
    "tower": len(TWO_TOWER_FEATURE_NAMES),
    "gnn": len(GRAPH_WINDOW_NAMES),
    "seq": len(SEQUENCE_FEATURE_NAMES),
}
LEGACY_MASK_NAMES = [
    "stats",
    "stats_prior",
    "stats_prior_structure",
    "stats_prior_structure_tower",
    "stats_prior_structure_tower_gnn",
    "stats_prior_structure_tower_gnn_seq",
    "stats_prior_target",
    "stats_prior_target_structure",
    "stats_prior_target_structure_profile",
    "stats_prior_target_structure_profile_tower",
    "stats_prior_target_structure_profile_tower_gnn",
    "stats_prior_target_structure_profile_tower_gnn_seq",
]


def _group_indices() -> dict[str, tuple[int, ...]]:
    start = 0
    groups: dict[str, tuple[int, ...]] = {}
    for name, size in GROUP_SIZES.items():
        groups[name] = tuple(range(start, start + size))
        start += size
    return groups


def test_catalog_adds_complete_leave_one_group_out_coverage_without_duplicates():
    groups = _group_indices()
    feature_count = sum(GROUP_SIZES.values())

    catalog = _feature_mask_catalog(feature_count)
    masks = list(catalog.masks)
    full = tuple(range(feature_count))

    assert [name for name, _indices in masks[: len(LEGACY_MASK_NAMES)]] == LEGACY_MASK_NAMES
    assert catalog.enabled_groups == tuple(GROUP_SIZES)
    assert catalog.full_indices == full
    assert dict(masks)[catalog.full_candidate_name] == full
    assert len({indices for _name, indices in masks}) == len(masks)

    leave_one_out = {entry.group: entry for entry in catalog.leave_one_out}
    assert set(leave_one_out) == set(GROUP_SIZES)
    for group, group_indices in groups.items():
        expected = tuple(index for index in full if index not in set(group_indices))
        entry = leave_one_out[group]
        assert entry.alias == f"loo_without_{group}"
        assert entry.indices == expected
        assert dict(masks)[entry.candidate_name] == expected


def test_frozen_leave_one_out_alias_reuses_duplicate_mask_without_duplicate_training():
    feature_count = sum(GROUP_SIZES.values())
    catalog = _feature_mask_catalog(feature_count)
    seq_entry = next(
        entry for entry in catalog.leave_one_out if entry.group == "seq"
    )

    assert (
        seq_entry.candidate_name
        == "stats_prior_target_structure_profile_tower_gnn"
    )
    assert len({indices for _name, indices in catalog.masks}) == len(catalog.masks)

    selected = _feature_masks(
        feature_count,
        config=TrainingConfig(
            frozen_fusion_feature_candidate="loo_without_seq",
        ),
    )

    assert selected == [("loo_without_seq", seq_entry.indices)]


def test_catalog_omits_disabled_groups_and_rejects_their_leave_one_out_aliases():
    feature_count = sum(GROUP_SIZES.values())
    config = TrainingConfig(
        target_window_enabled=False,
        source_profile_enabled=False,
        gnn_enabled=False,
        seq_enabled=False,
    )

    catalog = _feature_mask_catalog(feature_count, config=config)

    assert catalog.enabled_groups == ("stats", "prior", "structure", "tower")
    assert {entry.group for entry in catalog.leave_one_out} == {
        "stats",
        "prior",
        "structure",
        "tower",
    }
    expected_full = tuple(
        index
        for group in catalog.enabled_groups
        for index in _group_indices()[group]
    )
    assert catalog.full_indices == expected_full
    assert len({indices for _name, indices in catalog.masks}) == len(catalog.masks)

    with pytest.raises(
        ValueError,
        match="frozen fusion feature candidate is unavailable",
    ):
        _feature_masks(
            feature_count,
            config=TrainingConfig(
                target_window_enabled=False,
                frozen_fusion_feature_candidate="loo_without_target",
            ),
        )
