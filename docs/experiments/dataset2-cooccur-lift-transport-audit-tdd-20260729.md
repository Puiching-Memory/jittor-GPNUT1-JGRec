# Hai TDD: Dataset2 Cooccur-Lift Transport Audit

## Target Behavior

Compute zero-label, read-only summaries for short-lift collapse, auxiliary
probability confidence/entropy, destination-popularity transport, and stable
top-1 movement without loading labels or changing experiment artifacts.

## RED

- **Test added**: `tests/test_cooccur_lift_transport_audit.py`
- **Behavior asserted**: exact-zero cells and rows are distinct; probability
  summaries expose row maximum and normalized entropy; popularity uses one
  dense training-count reference; top-1 ties keep candidate order.
- **Command**:
  `uv run --group dev pytest tests/test_cooccur_lift_transport_audit.py -q`
- **Observed failure**:
  `ModuleNotFoundError: No module named 'jgrec.cooccur_lift_transport_audit'`
- **Failure is correct because**: the audit implementation did not exist. An
  earlier invalid import through the non-package `scripts` directory was
  discarded and is not counted as RED evidence.

## GREEN

- **Minimal implementation**:
  `src/jgrec/cooccur_lift_transport_audit.py` with mmap/chunked summaries and a
  zero-label CLI.
- **Command**:
  `uv run --group dev pytest tests/test_cooccur_lift_transport_audit.py -q`
- **Observed pass**: `4 passed`.

## RED: full-history boundary correction

- **Test added**:
  `test_time_support_separates_frozen_origin_from_full_train_end`
- **Behavior asserted**: the frozen auxiliary origin and complete
  `train.csv` end must have separately named short-window support rates.
- **Command**:
  `uv run --group dev pytest tests/test_cooccur_lift_transport_audit.py -q`
- **Observed failure**:
  `ImportError: cannot import name 'time_support_summary'`.
- **Failure is correct because**: the first implementation exposed only one
  ambiguous training boundary and could overstate the complete-history gap.

## GREEN

- **Minimal implementation**: added the explicit frozen-origin and
  full-train-end gaps and support rates.
- **Command**:
  `uv run --group dev pytest tests/test_cooccur_lift_transport_audit.py -q`
- **Observed pass**: `5 passed`.

## REFACTOR

- **Refactor done**: yes.
- **Change**: removed invalid-divide warnings from Jensen-Shannon calculation
  by evaluating logarithms only on positive-mass buckets.
- **Command after refactor**:
  `uv run ruff check src/jgrec/cooccur_lift_transport_audit.py tests/test_cooccur_lift_transport_audit.py && uv run --group dev pytest tests/test_cooccur_lift_transport_audit.py -q`
- **Observed result**: `All checks passed`; `5 passed`.

## RED: full-versus-short first-layer attribution

- **Test added**:
  `test_first_layer_attribution_separates_full_and_short_channels`.
- **Behavior asserted**: independently zeroing the full or short signal plus
  its raw, centered, and max-difference context channels reports the exact
  trained first-layer pre-activation intervention.
- **Command**:
  `uv run --group dev pytest tests/test_cooccur_lift_transport_audit.py -q`.
- **Observed failure**:
  `ImportError: cannot import name 'first_layer_lift_intervention_summary'`.
- **Failure is correct because**: the audit could report transport statistics
  but could not yet distinguish which lift signal remained mechanically
  active in the auxiliary head.

## GREEN

- **Minimal implementation**: added chunked first-layer interventions using
  the frozen auxiliary model's `linear1.weight` and feature standardization;
  the CLI now records strict-external and test summaries for both signals.
- **Command**:
  `uv run --group dev pytest tests/test_cooccur_lift_transport_audit.py -q`.
- **Observed pass**: `6 passed`.

## REFACTOR: attribution audit

- **Refactor done**: yes.
- **Change**: replaced a flagged constant-value dictionary comprehension with
  `dict.fromkeys`, without changing the intervention calculation.
- **Command after refactor**:
  `uv run ruff check src/jgrec/cooccur_lift_transport_audit.py tests/test_cooccur_lift_transport_audit.py`
  followed by
  `uv run --group dev pytest tests/test_cooccur_lift_transport_audit.py -q`.
- **Observed result**: `All checks passed`; `6 passed`.

## Next Behavior

Done. Any model or weight response to these findings requires a separately
frozen experiment and fresh-fold evidence.
