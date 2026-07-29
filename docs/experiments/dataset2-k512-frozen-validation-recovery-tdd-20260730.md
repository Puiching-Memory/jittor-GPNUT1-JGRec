# Hai TDD: Dataset2 K512 Frozen-Validation Recovery

## Target Behavior
A validation-only recovery path computes features for an explicitly supplied
frozen candidate matrix, rejects any row/positive mismatch, and never
resamples validation negatives.

## RED
- **Test added**:
  `test_frozen_candidate_queries_preserve_exact_matrix_without_rng` and
  `test_joint_cache_reports_accept_strict_frozen_query_recovery`.
- **Behavior asserted**: Caller-supplied candidates are copied unchanged,
  positive-at-zero is enforced, and a separate recovery process is accepted
  only with explicit no-sampling/exact-sidecar evidence.
- **Command**:
  `.venv\Scripts\python.exe -m pytest tests/test_hybrid_full100_training.py -q`
- **Observed failure**: Collection failed with
  `cannot import name 'build_frozen_candidate_queries'`.
- **Failure is correct because**: The frozen-query API did not exist; the
  failure was not caused by syntax or test-environment setup.

## GREEN
- **Minimal implementation**: Added frozen-candidate query construction,
  exact validation-sidecar alignment, and strict cross-process recovery
  lineage that rejects candidate sampling.
- **Command**:
  `.venv\Scripts\python.exe -m pytest tests/test_hybrid_full100_training.py -q`
- **Observed pass**: 14 passed.

## REFACTOR
- **Refactor done**: Yes.
- **Change**: Centralized joint-cache recovery validation, added a
  validation-only K512 rebuild command, reused it in near materialization and
  contract freezing, and added an audited controller-marker rebind.
- **Command after refactor**:
  `.venv\Scripts\python.exe -m pytest tests/test_hybrid_full100_training.py tests/test_cooccur_lift_automatic_pipeline.py -q`
  and Ruff over all changed Python files.
- **Observed result**: 19 passed; Ruff passed.

## Next Behavior
Run the remote validation-only rebuild and require all five frozen sidecar
hashes plus the K512 finite feature contract before controller resume.
