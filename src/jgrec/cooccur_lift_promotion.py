from __future__ import annotations

from collections.abc import Iterable

import numpy as np


class TieSafeServiceComparison:
    """Accumulate the amended service-equivalence audit in bounded memory.

    Numeric deltas at this boundary remain diagnostics because the accepted
    package and standard service replay apply serialization and tie handling
    in a different order. The service gate itself requires deterministic,
    strictly ordered output with the same Top-1 decision.
    """

    def __init__(
        self,
        *,
        numeric_tolerance: float,
        diagnostic_top_ks: Iterable[int] = (1, 3, 10),
    ) -> None:
        tolerance = float(numeric_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("numeric_tolerance must be finite and non-negative")
        top_ks = tuple(int(value) for value in diagnostic_top_ks)
        if not top_ks or any(value <= 0 for value in top_ks):
            raise ValueError("diagnostic_top_ks must contain positive integers")
        if len(set(top_ks)) != len(top_ks):
            raise ValueError("diagnostic_top_ks must not contain duplicates")

        self.numeric_tolerance = tolerance
        self.diagnostic_top_ks = top_ks
        self.rows = 0
        self.values_compared = 0
        self.maximum_absolute_error = 0.0
        self.absolute_error_sum = 0.0
        self.rows_above_numeric_tolerance = 0
        self.values_above_numeric_tolerance = 0
        self.top1_disagreements = 0
        self.accepted_rows_with_exact_ties = 0
        self.accepted_duplicate_adjacencies = 0
        self.served_rows_with_exact_ties = 0
        self.served_duplicate_adjacencies = 0
        self.strict_full_order_disagreements = 0
        self.maximum_inverted_accepted_gap = 0.0
        self.maximum_inverted_served_gap = 0.0
        self.rows_with_inversion_over_numeric_tolerance = 0
        self.topk_set_disagreements = dict.fromkeys(top_ks, 0)
        self.topk_prefix_order_disagreements = dict.fromkeys(top_ks, 0)
        self._candidate_count: int | None = None

    def update(
        self,
        accepted_scores: np.ndarray,
        served_scores: np.ndarray,
    ) -> None:
        accepted = self._validated_block(accepted_scores, label="accepted")
        served = self._validated_block(served_scores, label="served")
        if accepted.shape != served.shape:
            raise ValueError("accepted and served score shapes differ")
        candidate_count = int(accepted.shape[1])
        if self._candidate_count is None:
            self._candidate_count = candidate_count
            if any(value > candidate_count for value in self.diagnostic_top_ks):
                raise ValueError("diagnostic Top-K exceeds candidate count")
        elif self._candidate_count != candidate_count:
            raise ValueError("candidate count changed between comparison blocks")

        errors = np.abs(accepted - served)
        over_tolerance = errors > self.numeric_tolerance
        self.rows += int(accepted.shape[0])
        self.values_compared += int(errors.size)
        self.maximum_absolute_error = max(
            self.maximum_absolute_error,
            float(np.max(errors)),
        )
        self.absolute_error_sum += float(np.sum(errors))
        self.rows_above_numeric_tolerance += int(
            np.count_nonzero(np.any(over_tolerance, axis=1))
        )
        self.values_above_numeric_tolerance += int(
            np.count_nonzero(over_tolerance)
        )
        self.top1_disagreements += int(
            np.count_nonzero(
                np.argmax(accepted, axis=1) != np.argmax(served, axis=1)
            )
        )

        accepted_order = np.argsort(
            -accepted,
            axis=1,
            kind="stable",
        )
        served_order = np.argsort(
            -served,
            axis=1,
            kind="stable",
        )
        self.strict_full_order_disagreements += int(
            np.count_nonzero(np.any(accepted_order != served_order, axis=1))
        )
        for top_k in self.diagnostic_top_ks:
            accepted_prefix = accepted_order[:, :top_k]
            served_prefix = served_order[:, :top_k]
            self.topk_prefix_order_disagreements[top_k] += int(
                np.count_nonzero(
                    np.any(accepted_prefix != served_prefix, axis=1)
                )
            )
            self.topk_set_disagreements[top_k] += int(
                np.count_nonzero(
                    np.any(
                        np.sort(accepted_prefix, axis=1)
                        != np.sort(served_prefix, axis=1),
                        axis=1,
                    )
                )
            )

        (
            accepted_tie_rows,
            accepted_duplicate_adjacencies,
        ) = self._tie_counts(accepted)
        served_tie_rows, served_duplicate_adjacencies = self._tie_counts(
            served
        )
        self.accepted_rows_with_exact_ties += accepted_tie_rows
        self.accepted_duplicate_adjacencies += (
            accepted_duplicate_adjacencies
        )
        self.served_rows_with_exact_ties += served_tie_rows
        self.served_duplicate_adjacencies += served_duplicate_adjacencies

        accepted_in_served_order = np.take_along_axis(
            accepted,
            served_order,
            axis=1,
        )
        accepted_inversion = self._maximum_inversion_gap(
            accepted_in_served_order
        )
        served_in_accepted_order = np.take_along_axis(
            served,
            accepted_order,
            axis=1,
        )
        served_inversion = self._maximum_inversion_gap(
            served_in_accepted_order
        )
        self.maximum_inverted_accepted_gap = max(
            self.maximum_inverted_accepted_gap,
            float(np.max(accepted_inversion)),
        )
        self.maximum_inverted_served_gap = max(
            self.maximum_inverted_served_gap,
            float(np.max(served_inversion)),
        )
        self.rows_with_inversion_over_numeric_tolerance += int(
            np.count_nonzero(accepted_inversion > self.numeric_tolerance)
        )

    def finalize(self) -> dict[str, object]:
        if self.rows == 0:
            raise ValueError("cannot finalize an empty service comparison")
        tie_safe_equivalent = (
            self.top1_disagreements == 0
            and self.served_rows_with_exact_ties == 0
        )
        raw_numeric_at_service_boundary = (
            self.rows_above_numeric_tolerance == 0
        )
        return {
            "status": "passed" if tie_safe_equivalent else "failed",
            "equivalence_contract": (
                "byte_determinism_checked_separately; served_scores_tie_free; "
                "top1_exact_against_online_accepted_package"
            ),
            "tie_safe_service_equivalent": tie_safe_equivalent,
            "raw_numeric_equivalent_at_service_boundary": (
                raw_numeric_at_service_boundary
            ),
            "direct_numeric_comparison_participates_in_gate": False,
            "diagnostics_participate_in_gate": False,
            "rows": self.rows,
            "candidate_count": self._candidate_count,
            "values_compared": self.values_compared,
            "numeric_tolerance": self.numeric_tolerance,
            "maximum_absolute_error": self.maximum_absolute_error,
            "mean_absolute_error": (
                self.absolute_error_sum / self.values_compared
            ),
            "rows_above_numeric_tolerance": (
                self.rows_above_numeric_tolerance
            ),
            "values_above_numeric_tolerance": (
                self.values_above_numeric_tolerance
            ),
            "top1_disagreements": self.top1_disagreements,
            "accepted_rows_with_exact_ties": (
                self.accepted_rows_with_exact_ties
            ),
            "accepted_duplicate_adjacencies": (
                self.accepted_duplicate_adjacencies
            ),
            "served_rows_with_exact_ties": (
                self.served_rows_with_exact_ties
            ),
            "served_duplicate_adjacencies": (
                self.served_duplicate_adjacencies
            ),
            "strict_full_order_disagreements": (
                self.strict_full_order_disagreements
            ),
            "diagnostic_topk_set_disagreements": {
                str(key): value
                for key, value in self.topk_set_disagreements.items()
            },
            "diagnostic_topk_prefix_order_disagreements": {
                str(key): value
                for key, value in self.topk_prefix_order_disagreements.items()
            },
            "maximum_inverted_accepted_gap": (
                self.maximum_inverted_accepted_gap
            ),
            "maximum_inverted_served_gap": (
                self.maximum_inverted_served_gap
            ),
            "rows_with_inversion_over_numeric_tolerance": (
                self.rows_with_inversion_over_numeric_tolerance
            ),
        }

    @staticmethod
    def _validated_block(scores: np.ndarray, *, label: str) -> np.ndarray:
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim == 1:
            values = values[np.newaxis, :]
        if (
            values.ndim != 2
            or values.shape[0] == 0
            or values.shape[1] == 0
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(
                f"{label} scores must be a non-empty finite matrix"
            )
        return values

    @staticmethod
    def _tie_counts(scores: np.ndarray) -> tuple[int, int]:
        ordered = np.sort(scores, axis=1)
        duplicates = ordered[:, 1:] == ordered[:, :-1]
        return (
            int(np.count_nonzero(np.any(duplicates, axis=1))),
            int(np.count_nonzero(duplicates)),
        )

    @staticmethod
    def _maximum_inversion_gap(scores_in_other_order: np.ndarray) -> np.ndarray:
        minimum_prefix = np.minimum.accumulate(
            scores_in_other_order,
            axis=1,
        )
        return np.max(scores_in_other_order - minimum_prefix, axis=1)
