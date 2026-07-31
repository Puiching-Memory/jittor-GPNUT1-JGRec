# Hai TDD: Dataset2 New-Link Diagnosis and Cached Growth Features

## Target Behavior

Label validation positives as repeat/new using only a fixed historical prefix, compute exact champion error concentration by segment and time slice, and derive two leakage-safe growth features from the existing cache without mutating it.

## RED 1: Historical Pair and Error-Concentration Contract

- **Test added**: `tests/test_hybrid_new_link_diagnostics.py`
- **Behavior asserted**: Duplicate historical pairs are recognized; unseen query pairs remain new; exact reciprocal ranks, Top-1 errors, regret shares, and temporal difficulty gates are computed without mutating inputs.
- **Command**: `uv run --no-sync pytest tests/test_hybrid_new_link_diagnostics.py -q`
- **Observed failure**: Collection failed with `ModuleNotFoundError: jgrec.rankers.hybrid.new_link_diagnostics`.
- **Failure is correct because**: No reusable fixed-prefix pair labeler or new-link error gate existed.

## GREEN 1

- **Minimal implementation**: Added vectorized source-target pair keys, immutable segment/slice reports, exact positive-at-zero rank accounting, and the frozen concentration gate.
- **Command**: `uv run --no-sync pytest tests/test_hybrid_new_link_diagnostics.py tests/test_hybrid_oof_hard_negatives.py -q`
- **Observed pass**: 13 focused tests passed locally and on the Linux server.

## RED 2: Cached Growth Derivation

- **Test added**: `tests/test_hybrid_new_link_features.py`
- **Behavior asserted**: Append the fixed short/long popularity ratio and source-activity cross by feature name, preserve every original value, return float32, and reject missing/misaligned inputs.
- **Command**: `uv run --no-sync pytest tests/test_hybrid_new_link_features.py -q`
- **Observed failure**: Collection failed with `ModuleNotFoundError: jgrec.rankers.hybrid.new_link_features`.
- **Failure is correct because**: The cache had raw activity/window fields but no explicit new-link growth derivation boundary.

## GREEN 2

- **Minimal implementation**: Added `append_new_link_growth_features` with exactly two frozen formulas and no model, cache, or inference side effects.
- **Command**: `uv run --no-sync pytest tests/test_hybrid_new_link_features.py tests/test_hybrid_new_link_diagnostics.py tests/test_hybrid_oof_hard_negatives.py -q`
- **Observed pass**: 15 focused tests passed locally; the related six tests passed again on the Linux server before training.

## REFACTOR

- **Refactor done**: yes
- **Change**: Kept diagnostic/accounting logic pure NumPy and independent of Jittor; reused the existing OOF contiguous-slice and temporal-gate contracts; addressed Ruff's tuple-shape recommendation before the final GREEN run.
- **Command after refactor**: `uv run --no-sync ruff check src/jgrec/rankers/hybrid/new_link_diagnostics.py src/jgrec/rankers/hybrid/new_link_features.py tests/test_hybrid_new_link_diagnostics.py tests/test_hybrid_new_link_features.py scripts/diagnose_dataset2_new_link_errors.py scripts/evaluate_dataset2_new_link_growth_features.py`
- **Observed result**: Ruff passed locally and on the server. Exact champion alignment passed, while the frozen growth-feature candidate regressed in all three intervals and was correctly rejected without inference/package work.

## Next Behavior

Done for cached growth transforms. A time-decayed two-hop feature is a separate, expensive hypothesis that requires its own goal, temporal-state tests, cache migration proof, and stop rule.
