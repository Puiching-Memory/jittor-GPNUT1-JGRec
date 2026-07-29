# Hai TDD: Dataset2 Joint Train/Validation Cache Provenance

## Target Behavior

A reranker training run accepts a Dataset2 train/validation cache pair only
when both complete reports prove they were emitted by the same Python process
and bind the exact same training-feature artifact.

## RED

- **Test added**:
  `test_joint_cache_reports_require_one_process_and_exact_train_hash_binding`
  in `tests/test_hybrid_full100_training.py`.
- **Behavior asserted**:
  Equal build ID, PID, report roles, and training-feature SHA are required;
  PID or SHA mismatches are rejected.
- **Command**:
  `uv run --no-sync pytest tests/test_hybrid_full100_training.py -q`
- **Observed failure**:
  Test collection failed because `validate_joint_cache_reports` did not exist.
- **Failure is correct because**:
  The previous trainer trusted a replay boolean and could not distinguish two
  independent encoder processes from one joint build.

## GREEN

- **Minimal implementation**:
  Added `validate_joint_cache_reports`, required it in the LightGBM/Setwise
  trainer, and added one-process train/validation output support to the
  full-100 cache builder. Both reports carry one generated build ID and PID;
  validation binds the published training-feature SHA.
- **Command**:
  `uv run --no-sync pytest tests/test_hybrid_full100_training.py -q`
- **Observed pass**:
  9 tests passed locally and on the Linux server.

## REFACTOR

- **Refactor done**: yes
- **Change**:
  Kept provenance validation in the reusable full-100 module and lifecycle,
  memmap, and atomic-publication logic in the experiment builder. The old
  training-only CLI remains available when joint arguments are omitted.
- **Command after refactor**:
  `uv run --no-sync ruff check
  src/jgrec/rankers/hybrid/full100_training.py
  scripts/build_dataset2_full100_train_cache.py
  scripts/train_dataset2_matched_lgbm_setwise.py
  tests/test_hybrid_full100_training.py`
- **Observed result**:
  Ruff passed locally and on the server; Python compilation and server shell
  syntax checks passed.

## Next Behavior

Production proof is pending: both joint caches, LightGBM and Setwise
full/three-slice MRR, and conditional package generation.
