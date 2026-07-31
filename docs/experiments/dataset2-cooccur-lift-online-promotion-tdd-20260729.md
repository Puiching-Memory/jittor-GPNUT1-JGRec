# Hai TDD: Dataset2 Cooccur Lift Promotion Equivalence Split

## Target Behavior

Represent raw-model equivalence and tie-safe service equivalence as separate
contracts. A service replay may pass despite numeric differences introduced
by the accepted package's eight-decimal serialization boundary, but it must
be deterministic, tie-free, and Top-1 exact. Any Top-1 change or exact tie in
served output must fail.

## RED

- **Test added**:
  `tests/test_cooccur_lift_promotion.py`
- **Behavior asserted**:
  - Raw numeric status at the service boundary is reported separately.
  - Numeric drift alone does not silently fail the amended service contract.
  - Top-1 drift fails.
  - Exact served ties fail.
  - Top-K residuals are diagnostics and do not participate in the gate.
- **Command**:

  ```powershell
  $env:PYTHONPATH='src'
  uv run --no-project --with pytest --with numpy python -m pytest `
    --override-ini addopts='' tests/test_cooccur_lift_promotion.py -q
  ```

- **Observed failure**:
  collection failed with
  `ModuleNotFoundError: No module named 'jgrec.cooccur_lift_promotion'`.
- **Failure is correct because**:
  the amended equivalence domain API did not yet exist.

## GREEN

- **Minimal implementation**:
  added `TieSafeServiceComparison`, a bounded-memory accumulator that keeps
  numeric deltas, exact ties, Top-1, Top-K sets, order changes, and inversion
  gaps separate. Its pass rule uses only the amended tie-safe service
  contract.
- **Command**:

  ```powershell
  $env:PYTHONPATH='src'
  uv run --no-project --with pytest --with numpy python -m pytest `
    --override-ini addopts='' tests/test_cooccur_lift_promotion.py -q
  ```

- **Observed pass**: `5 passed in 0.09s`.

## REFACTOR

- **Refactor done**: yes
- **Change**:
  - Reused the accumulator in the canonical double-replay script.
  - Added a read-only finalizer that verifies frozen hashes and existing
    evidence, then atomically writes replay report, promoted manifest, and
    canonical status.
  - Kept the old failed status and pre-wiring receipt immutable.
- **Command after refactor**:

  ```powershell
  uv run --no-project --with ruff ruff check `
    src/jgrec/cooccur_lift_promotion.py `
    scripts/finalize_dataset2_cooccur_lift_promotion.py `
    scripts/replay_dataset2_cooccur_lift_checkpoint.py `
    tests/test_cooccur_lift_promotion.py
  uv run --no-project --with numpy python -m compileall -q `
    src/jgrec/cooccur_lift_promotion.py `
    scripts/finalize_dataset2_cooccur_lift_promotion.py `
    scripts/replay_dataset2_cooccur_lift_checkpoint.py `
    tests/test_cooccur_lift_promotion.py
  ```

- **Observed result**: Ruff reported `All checks passed`; compileall exited
  successfully.

## Next Behavior

Done. The remote finalizer consumed only existing artifacts and published
`accepted/promoted`.
