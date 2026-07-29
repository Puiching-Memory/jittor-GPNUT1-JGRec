from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from jgrec.core.types import InteractionTable
from jgrec.rankers.common.temporal_index import TemporalInteractionIndex
from jgrec.rankers.hybrid.cooccur_lift import (
    COOCCUR_LIFT_FEATURE_NAMES,
    CooccurLiftAugmentedView,
    FrozenCooccurLiftConfigError,
    cooccur_lift_scores,
    load_frozen_cooccur_lift_config,
    training_seed,
)
from jgrec.rankers.hybrid.cooccur_lift_native import (
    materialize_compact_cooccur_lift,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN_CONFIG_PATH = ROOT / "docs" / "experiments" / "cooccur-lift-aux-expert-v1.frozen.json"


def _micro_index() -> TemporalInteractionIndex:
    events = [
        (99, 10, 1),
        (99, 20, 2),
        (1, 10, 1),
        (1, 30, 3),
        (2, 20, 2),
        (2, 30, 4),
        (3, 30, 5),
        (6, 30, 8),
        (4, 10, 1),
        (4, 40, 3),
        (5, 20, 7),
        (5, 40, 8),
        (7, 10, 9),
        (7, 40, 10),
    ]
    src_times: dict[int, list[int]] = defaultdict(list)
    src_dsts: dict[int, list[int]] = defaultdict(list)
    dst_times: dict[int, list[int]] = defaultdict(list)
    dst_srcs: dict[int, list[int]] = defaultdict(list)
    pair_times: dict[tuple[int, int], list[int]] = defaultdict(list)
    for src, dst, event_time in sorted(events, key=lambda row: (row[0], row[2])):
        src_times[src].append(event_time)
        src_dsts[src].append(dst)
        dst_times[dst].append(event_time)
        dst_srcs[dst].append(src)
        pair_times[(src, dst)].append(event_time)
    for dst in dst_times:
        order = np.argsort(dst_times[dst], kind="stable")
        dst_times[dst] = [dst_times[dst][index] for index in order]
        dst_srcs[dst] = [dst_srcs[dst][index] for index in order]

    index = TemporalInteractionIndex()
    index.fit_grouped(
        src_times=src_times,
        src_dsts=src_dsts,
        dst_times=dst_times,
        dst_srcs=dst_srcs,
        pair_times=pair_times,
        max_time=10,
        total_edges=len(events),
        build_transitions=False,
        build_cooccurs=True,
        cooccur_history_limit=256,
        future_only_transition_cooccur=False,
        cooccur_time_decay_ratio=0.0,
    )
    return index


def _frozen_payload() -> dict:
    return json.loads(FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))


def test_full_lift_matches_hand_count_and_strictly_excludes_query_time() -> None:
    scores = cooccur_lift_scores(
        _micro_index(),
        src=99,
        candidates=np.array([30, 40]),
        query_time=10,
        short_window=3,
    )

    assert COOCCUR_LIFT_FEATURE_NAMES == (
        "cooccur_lift_full",
        "cooccur_lift_short",
    )
    np.testing.assert_allclose(
        scores[:, 0],
        [np.log1p(2) - np.log1p(4), np.log1p(2) - np.log1p(2)],
        atol=1e-7,
    )


def test_short_lift_excludes_events_at_or_before_window_start() -> None:
    scores = cooccur_lift_scores(
        _micro_index(),
        src=99,
        candidates=np.array([30, 40]),
        query_time=10,
        short_window=3,
    )

    np.testing.assert_allclose(
        scores[:, 1],
        [np.log1p(0) - np.log1p(1), np.log1p(1) - np.log1p(1)],
        atol=1e-7,
    )


def test_gapped_availability_collapses_short_but_preserves_full_history() -> None:
    scores = cooccur_lift_scores(
        _micro_index(),
        src=99,
        candidates=np.array([30, 40]),
        query_time=10,
        availability_time=7,
        short_window=3,
    )

    np.testing.assert_allclose(scores[:, 1], 0.0, atol=1e-7)
    assert np.any(np.abs(scores[:, 0]) > 0.0)


def test_equal_cooccurrence_count_gives_long_tail_candidate_higher_lift() -> None:
    scores = cooccur_lift_scores(
        _micro_index(),
        src=99,
        candidates=np.array([30, 40]),
        query_time=10,
        short_window=3,
    )

    assert scores[1, 0] > scores[0, 0]


def test_native_compact_materializer_matches_reference_index(tmp_path: Path) -> None:
    rows = np.asarray(
        [
            (99, 10, 1),
            (1, 10, 1),
            (4, 10, 1),
            (99, 20, 2),
            (2, 20, 2),
            (1, 30, 3),
            (4, 40, 3),
            (2, 30, 4),
            (3, 30, 5),
            (5, 20, 7),
            (6, 30, 8),
            (5, 40, 8),
            (7, 10, 9),
            (7, 40, 10),
        ],
        dtype=np.int32,
    )
    interactions = InteractionTable.from_array(rows).sort_by_time()
    index = TemporalInteractionIndex()
    index.fit(
        interactions,
        build_transitions=False,
        build_cooccurs=True,
        cooccur_history_limit=256,
        future_only_transition_cooccur=False,
        cooccur_time_decay_ratio=0.0,
    )
    query_sources = np.asarray([99, 99, 99], dtype=np.int32)
    query_candidates = np.asarray(
        [[20, 30, 40], [30, 40, 10], [30, 40, 20]],
        dtype=np.int32,
    )
    query_destinations = query_candidates[:, 0]
    query_times = np.asarray([2, 8, 10], dtype=np.int32)
    lift_path = tmp_path / "lift.npy"
    popularity_path = tmp_path / "popularity.npy"

    contract = materialize_compact_cooccur_lift(
        interactions=interactions,
        sources=query_sources,
        candidates=query_candidates,
        destinations=query_destinations,
        event_time=query_times,
        short_window=3,
        lift_path=lift_path,
        positive_popularity_path=popularity_path,
        progress_path=tmp_path / "progress.json",
        work_dir=tmp_path,
    )

    actual = np.load(lift_path, allow_pickle=False)
    expected = np.stack(
        [
            cooccur_lift_scores(
                index,
                int(src),
                candidates,
                int(query_time),
                short_window=3,
            )
            for src, candidates, query_time in zip(
                query_sources,
                query_candidates,
                query_times,
                strict=True,
            )
        ]
    )
    expected_popularity = np.asarray(
        [
            index.destination_view(int(dst), int(query_time)).cutoff
            for dst, query_time in zip(
                query_destinations,
                query_times,
                strict=True,
            )
        ],
        dtype=np.int32,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    np.testing.assert_array_equal(
        np.load(popularity_path, allow_pickle=False),
        expected_popularity,
    )
    assert contract["history_limit"] == 64
    assert contract["cooccur_history_limit"] == 256
    assert contract["cooccur_time_decay_score_reused"] is False


def test_native_materializer_separates_observation_and_availability_time(
    tmp_path: Path,
) -> None:
    rows = np.asarray(
        [
            (99, 10, 1),
            (1, 10, 1),
            (4, 10, 1),
            (99, 20, 2),
            (2, 20, 2),
            (1, 30, 3),
            (4, 40, 3),
            (2, 30, 4),
            (3, 30, 5),
            (5, 20, 7),
            (6, 30, 8),
            (5, 40, 8),
            (7, 10, 9),
            (7, 40, 10),
        ],
        dtype=np.int32,
    )
    interactions = InteractionTable.from_array(rows).sort_by_time()
    index = TemporalInteractionIndex()
    index.fit(
        interactions,
        build_transitions=False,
        build_cooccurs=True,
        cooccur_history_limit=256,
        future_only_transition_cooccur=False,
        cooccur_time_decay_ratio=0.0,
    )
    query_sources = np.asarray([99, 99], dtype=np.int32)
    query_candidates = np.asarray(
        [[30, 40], [30, 40]],
        dtype=np.int32,
    )
    query_destinations = query_candidates[:, 0]
    query_times = np.asarray([10, 10], dtype=np.int32)
    availability_times = np.asarray([7, 10], dtype=np.int32)
    lift_path = tmp_path / "lift.npy"
    popularity_path = tmp_path / "popularity.npy"

    contract = materialize_compact_cooccur_lift(
        interactions=interactions,
        sources=query_sources,
        candidates=query_candidates,
        destinations=query_destinations,
        event_time=query_times,
        availability_time=availability_times,
        short_window=3,
        lift_path=lift_path,
        positive_popularity_path=popularity_path,
        progress_path=tmp_path / "progress.json",
        work_dir=tmp_path,
    )

    expected = np.stack(
        [
            cooccur_lift_scores(
                index,
                int(src),
                candidates,
                int(query_time),
                availability_time=int(availability_time),
                short_window=3,
            )
            for src, candidates, query_time, availability_time in zip(
                query_sources,
                query_candidates,
                query_times,
                availability_times,
                strict=True,
            )
        ]
    )
    np.testing.assert_allclose(
        np.load(lift_path, allow_pickle=False),
        expected,
        atol=1e-7,
    )
    np.testing.assert_allclose(expected[0, :, 1], 0.0, atol=1e-7)
    assert contract["separate_availability_time"] is True
    assert contract["collapsed_short_rows"] == 1


def test_augmented_view_overlays_gnn_then_appends_two_lift_columns() -> None:
    base = np.arange(2 * 3 * 63, dtype=np.float32).reshape(2, 3, 63)
    short_none = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        dtype=np.float32,
    )
    lift = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ],
        dtype=np.float32,
    )

    view = CooccurLiftAugmentedView(
        base,
        short_none_scores=short_none,
        gnn_short_column=59,
        lift_features=lift,
    )
    actual = view[:]

    assert view.shape == (2, 3, 65)
    assert view.ndim == 3
    assert view.size == 2 * 3 * 65
    np.testing.assert_array_equal(actual[..., :59], base[..., :59])
    np.testing.assert_array_equal(actual[..., 60:63], base[..., 60:63])
    np.testing.assert_array_equal(actual[..., 59], short_none)
    np.testing.assert_array_equal(actual[..., 63:65], lift)


def test_frozen_config_exposes_exact_precommitted_contract() -> None:
    config = load_frozen_cooccur_lift_config(FROZEN_CONFIG_PATH)

    assert config.integration_id == "cooccur_lift_aux_expert_v1"
    assert config.weights == (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
    assert config.fold_boundaries == (
        (0, 79909, 118816),
        (0, 118816, 159804),
        (0, 159804, 200000),
    )
    assert config.history_limit == 64
    assert config.short_window_ratio == 0.05
    assert config.appended_feature_indices == (63, 64)
    assert config.context_feature_indices == tuple(range(195))
    assert training_seed(config, 0) == 30073
    assert training_seed(config, 1) == 31082
    assert training_seed(config, 2) == 32091


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda p: p.update(status="draft"), "status"),
        (lambda p: p.update(weights=[0.05, 0.1]), "weights"),
        (
            lambda p: p["feature_formula"].update(cooccur_lift_full="log1p(N_full)"),
            "feature_formula",
        ),
        (lambda p: p.update(short_window_ratio=0.1), "short_window_ratio"),
        (
            lambda p: p["index_build"].update(cooccur_time_decay_ratio=0.05),
            "index_build",
        ),
        (
            lambda p: p.update(appended_feature_indices=[62, 63]),
            "appended_feature_indices",
        ),
        (
            lambda p: p.update(
                prohibited=[value for value in p["prohibited"] if value != "reusing cooccur_time_decay_score"]
            ),
            "cooccur_time_decay_score",
        ),
    ],
)
def test_frozen_config_rejects_any_experiment_drift(mutate, match: str) -> None:
    payload = copy.deepcopy(_frozen_payload())
    mutate(payload)

    with pytest.raises(FrozenCooccurLiftConfigError, match=match):
        load_frozen_cooccur_lift_config(payload)
