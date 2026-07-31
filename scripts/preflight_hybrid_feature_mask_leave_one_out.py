from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

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

DEFAULT_OUTPUT = Path(
    "result/hybrid_feature_mask_leave_one_out_local_20260728/"
    "preflight-report.json"
)
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
GROUP_CONFIG_FLAGS = {
    "prior": "candidate_prior_enabled",
    "target": "target_window_enabled",
    "structure": "structure_enabled",
    "profile": "source_profile_enabled",
    "tower": "two_tower_enabled",
    "gnn": "gnn_enabled",
    "seq": "seq_enabled",
}
LEGACY_CANDIDATE_COUNT = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Hybrid leave-one-feature-group-out masks without reading "
            "training, external, or submission data."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON report path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def _group_indices() -> dict[str, tuple[int, ...]]:
    start = 0
    groups: dict[str, tuple[int, ...]] = {}
    for group, size in GROUP_SIZES.items():
        groups[group] = tuple(range(start, start + size))
        start += size
    return groups


def build_report() -> dict[str, object]:
    feature_count = sum(GROUP_SIZES.values())
    config = TrainingConfig()
    catalog = _feature_mask_catalog(feature_count, config=config)
    groups = _group_indices()
    masks = list(catalog.masks)
    mask_by_name = dict(masks)
    errors: list[str] = []

    if len({indices for _name, indices in masks}) != len(masks):
        errors.append("default candidate scan contains duplicate feature masks")
    if mask_by_name.get(catalog.full_candidate_name) != catalog.full_indices:
        errors.append("full candidate does not resolve to full enabled indices")

    leave_one_out_rows: list[dict[str, object]] = []
    for entry in catalog.leave_one_out:
        removed = set(groups[entry.group])
        expected = tuple(
            index
            for index in catalog.full_indices
            if index not in removed
        )
        selected = _feature_masks(
            feature_count,
            config=replace(
                config,
                frozen_fusion_feature_candidate=entry.alias,
            ),
        )
        exact_complement = entry.indices == expected
        frozen_alias_exact = selected == [(entry.alias, entry.indices)]
        default_candidate_exact = (
            mask_by_name.get(entry.candidate_name) == entry.indices
        )
        if not exact_complement:
            errors.append(f"{entry.alias} is not the exact group complement")
        if not frozen_alias_exact:
            errors.append(f"{entry.alias} cannot be frozen exactly")
        if not default_candidate_exact:
            errors.append(
                f"{entry.alias} has no matching unique default candidate"
            )
        leave_one_out_rows.append(
            {
                "removed_group": entry.group,
                "alias": entry.alias,
                "default_candidate_name": entry.candidate_name,
                "reuses_existing_candidate": (
                    entry.alias != entry.candidate_name
                ),
                "removed_feature_count": len(groups[entry.group]),
                "remaining_feature_count": len(entry.indices),
                "exact_full_minus_group": exact_complement,
                "frozen_alias_exact": frozen_alias_exact,
                "default_candidate_exact": default_candidate_exact,
            }
        )

    covered_groups = {entry.group for entry in catalog.leave_one_out}
    expected_groups = {
        group
        for group in catalog.enabled_groups
        if len(catalog.full_indices) > len(groups[group])
    }
    if covered_groups != expected_groups:
        errors.append(
            "leave-one-out group coverage mismatch: "
            f"expected={sorted(expected_groups)}, actual={sorted(covered_groups)}"
        )

    disabled_group_checks: list[dict[str, object]] = []
    for group, flag in GROUP_CONFIG_FLAGS.items():
        disabled_catalog = _feature_mask_catalog(
            feature_count,
            config=replace(config, **{flag: False}),
        )
        omitted = (
            group not in disabled_catalog.enabled_groups
            and group
            not in {
                entry.group
                for entry in disabled_catalog.leave_one_out
            }
        )
        if not omitted:
            errors.append(f"disabled group remains selectable: {group}")
        disabled_group_checks.append(
            {
                "group": group,
                "config_flag": flag,
                "omitted_from_leave_one_out": omitted,
            }
        )

    aliases = [entry.alias for entry in catalog.leave_one_out]
    report: dict[str, object] = {
        "schema_version": 1,
        "status": (
            "ready_for_remote_rolling"
            if not errors
            else "local_preflight_failed"
        ),
        "feature_count": feature_count,
        "feature_groups": [
            {"name": group, "feature_count": size}
            for group, size in GROUP_SIZES.items()
        ],
        "enabled_groups": list(catalog.enabled_groups),
        "legacy_candidate_count": LEGACY_CANDIDATE_COUNT,
        "unique_default_candidate_count": len(masks),
        "new_unique_candidate_count": max(
            0,
            len(masks) - LEGACY_CANDIDATE_COUNT,
        ),
        "default_candidate_names": [name for name, _indices in masks],
        "duplicate_default_masks": (
            len({indices for _name, indices in masks}) != len(masks)
        ),
        "full_candidate": {
            "name": catalog.full_candidate_name,
            "feature_count": len(catalog.full_indices),
        },
        "leave_one_out_aliases": aliases,
        "leave_one_out": leave_one_out_rows,
        "coverage_complete": covered_groups == expected_groups,
        "disabled_group_checks": disabled_group_checks,
        "formal_validation_candidates": [
            catalog.full_candidate_name,
            *aliases,
        ],
        "protocol_boundaries": {
            "training_data_read": False,
            "selection_metrics_read": False,
            "external_holdout_read": False,
            "online_score_read": False,
            "submission_package_generated": False,
            "next_gate": "rolling_origin_multi_fold",
        },
        "errors": errors,
    }
    return report


def main() -> None:
    args = parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    if report["status"] != "ready_for_remote_rolling":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
