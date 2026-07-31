from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureMaskCandidate:
    candidate_id: str
    removed_group: str | None
    feature_indices: tuple[int, ...]
    config: dict[str, Any]
    config_sha256: str
    tie_break_priority: int


def build_feature_mask_candidates(
    *,
    feature_names: Sequence[str],
    feature_groups: Mapping[str, Sequence[str]],
    shared_config: Mapping[str, Any],
) -> tuple[FeatureMaskCandidate, ...]:
    names = tuple(str(name) for name in feature_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("feature_names must be non-empty and unique")
    groups = {
        str(group): tuple(str(name) for name in members)
        for group, members in feature_groups.items()
    }
    if not groups or any(not group or not members for group, members in groups.items()):
        raise ValueError("feature groups must be named and non-empty")
    assigned = [
        name
        for members in groups.values()
        for name in members
    ]
    if (
        len(assigned) != len(names)
        or len(set(assigned)) != len(assigned)
        or set(assigned) != set(names)
    ):
        raise ValueError(
            "feature groups must assign every feature exactly once"
        )

    shared = json.loads(
        json.dumps(
            dict(shared_config),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    rows: list[tuple[str, str | None, tuple[int, ...]]] = [
        ("full_enabled", None, tuple(range(len(names))))
    ]
    for group, members in groups.items():
        removed = set(members)
        indices = tuple(
            index
            for index, name in enumerate(names)
            if name not in removed
        )
        if not indices:
            raise ValueError(
                f"removing feature group {group!r} leaves no features"
            )
        rows.append((f"loo_without_{group}", group, indices))

    if len({indices for _name, _group, indices in rows}) != len(rows):
        raise ValueError("feature-mask candidates must have unique indices")

    candidates = []
    for priority, (candidate_id, removed_group, indices) in enumerate(rows):
        config = {
            "candidate_id": candidate_id,
            "feature_indices": list(indices),
            "removed_group": removed_group,
            "shared": shared,
        }
        candidates.append(
            FeatureMaskCandidate(
                candidate_id=candidate_id,
                removed_group=removed_group,
                feature_indices=indices,
                config=config,
                config_sha256=_json_sha256(config),
                tie_break_priority=priority,
            )
        )
    return tuple(candidates)


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
