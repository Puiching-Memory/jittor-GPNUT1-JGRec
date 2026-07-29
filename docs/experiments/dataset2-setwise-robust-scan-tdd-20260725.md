# Hai TDD: Dataset2 Setwise Robust Weight Scan

## Target Behavior

Generate an exact inclusive `0.750..0.900` grid at `0.005`, scan all 31
weights using only the first two chronological slices for selection, and keep
the selected weight unchanged when only the forward slice is modified.

## RED

- **Test added**:
  `test_refined_weight_scan_uses_31_point_grid_and_ignores_forward_rows`
- **Behavior asserted**: The grid includes `0.750`, `0.800`, and `0.900`,
  contains 31 weights, and forward-only prediction changes cannot alter the
  selected weight or selection MRR.
- **Command**:
  `uv run --no-sync pytest -q tests/test_hybrid_fusion_analysis.py`
- **Observed failure**: Test collection failed with
  `ImportError: cannot import name 'inclusive_weight_grid'`.
- **Failure is correct because**: The configurable inclusive grid and refined
  scan behavior did not yet exist; the failure was the missing target API.

## GREEN

- **Minimal implementation**: Added `inclusive_weight_grid` and an optional
  `primary_weights` argument to the existing prefix-only blend scanner. The
  old `0.80..1.00` behavior remains the default.
- **Command**:
  `uv run --no-sync pytest -q tests/test_hybrid_fusion_analysis.py`
- **Observed pass**: `11 passed`.

## REFACTOR

- **Refactor done**: yes
- **Change**: Parameterized the existing server scan script with start, stop,
  step, reference weight, and report name instead of duplicating the cached
  prediction pipeline. The robustness gate compares against reconstructed
  `0.80` metrics.
- **Command after refactor**:
  `uv run --no-sync ruff check ... && uv run --no-sync pytest -q
  tests/test_hybrid_fusion_analysis.py tests/test_core_cuda.py`
- **Observed result**: Local Ruff passed and `12 passed`; server Ruff passed
  and `11 passed`.

## Next Behavior

Done. The 31-point server scan selected `0.80`, rejected a new package, and
preserved the current online champion.
