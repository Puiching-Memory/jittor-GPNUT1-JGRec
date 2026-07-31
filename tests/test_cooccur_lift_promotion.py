from __future__ import annotations

import numpy as np

from jgrec.cooccur_lift_promotion import TieSafeServiceComparison


def test_tie_safe_service_tracks_numeric_equivalence_separately() -> None:
    accepted = np.asarray(
        [
            [0.70, 0.20, 0.10],
            [0.60, 0.25, 0.15],
        ],
        dtype=np.float64,
    )
    served = accepted + np.asarray(
        [
            [1e-8, -1e-8, 0.0],
            [0.0, 1e-8, -1e-8],
        ],
        dtype=np.float64,
    )

    comparison = TieSafeServiceComparison(
        numeric_tolerance=5e-7,
        diagnostic_top_ks=(1, 3),
    )
    comparison.update(accepted, served)
    report = comparison.finalize()

    assert report["status"] == "passed"
    assert report["tie_safe_service_equivalent"] is True
    assert report["raw_numeric_equivalent_at_service_boundary"] is True
    assert report["top1_disagreements"] == 0
    assert report["served_rows_with_exact_ties"] == 0


def test_tie_safe_service_can_pass_when_old_combined_numeric_gate_fails() -> None:
    accepted = np.asarray([[0.70, 0.20, 0.10]], dtype=np.float64)
    served = np.asarray([[0.701, 0.199, 0.10]], dtype=np.float64)

    comparison = TieSafeServiceComparison(
        numeric_tolerance=5e-7,
        diagnostic_top_ks=(1, 3),
    )
    comparison.update(accepted, served)
    report = comparison.finalize()

    assert report["status"] == "passed"
    assert report["tie_safe_service_equivalent"] is True
    assert report["raw_numeric_equivalent_at_service_boundary"] is False
    assert report["rows_above_numeric_tolerance"] == 1
    assert report["top1_disagreements"] == 0


def test_tie_safe_service_rejects_top1_change() -> None:
    comparison = TieSafeServiceComparison(
        numeric_tolerance=5e-7,
        diagnostic_top_ks=(1, 3),
    )
    comparison.update(
        np.asarray([[0.60, 0.40, 0.00]], dtype=np.float64),
        np.asarray([[0.39, 0.61, 0.00]], dtype=np.float64),
    )

    report = comparison.finalize()

    assert report["status"] == "failed"
    assert report["tie_safe_service_equivalent"] is False
    assert report["top1_disagreements"] == 1


def test_tie_safe_service_rejects_exact_served_ties() -> None:
    comparison = TieSafeServiceComparison(
        numeric_tolerance=5e-7,
        diagnostic_top_ks=(1, 3),
    )
    comparison.update(
        np.asarray([[0.70, 0.20, 0.10]], dtype=np.float64),
        np.asarray([[0.70, 0.15, 0.15]], dtype=np.float64),
    )

    report = comparison.finalize()

    assert report["status"] == "failed"
    assert report["tie_safe_service_equivalent"] is False
    assert report["served_rows_with_exact_ties"] == 1
    assert report["served_duplicate_adjacencies"] == 1


def test_tie_safe_service_reports_topk_residuals_without_selecting_on_them() -> None:
    comparison = TieSafeServiceComparison(
        numeric_tolerance=5e-7,
        diagnostic_top_ks=(1, 2),
    )
    comparison.update(
        np.asarray([[0.70, 0.20, 0.10]], dtype=np.float64),
        np.asarray([[0.70, 0.09, 0.21]], dtype=np.float64),
    )

    report = comparison.finalize()

    assert report["status"] == "passed"
    assert report["diagnostic_topk_set_disagreements"] == {
        "1": 0,
        "2": 1,
    }
    assert report["diagnostics_participate_in_gate"] is False
