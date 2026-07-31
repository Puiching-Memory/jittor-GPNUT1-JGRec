# Hai TDD: Common-Validation Blend Scan

## Target Behavior

Given two candidate-probability tensors whose first candidate is positive, report full/early/late MRR and scan all 101 reference weights from `0.00` through `1.00`, preferring the reference model deterministically when full-split MRR ties.

## RED

- **Test added**: `tests/test_hybrid_fusion_analysis.py`
- **Behavior asserted**: Temporal-half MRR uses the repository's candidate-zero-positive rank contract; fine blend search covers 101 points and picks the largest reference weight on an MRR plateau.
- **Command**: `uv run --no-sync pytest tests/test_hybrid_fusion_analysis.py -q`
- **Observed failure**: Test collection failed with `ModuleNotFoundError: No module named 'jgrec.rankers.hybrid.fusion_analysis'`.
- **Failure is correct because**: The requested analysis boundary did not exist yet; the failure was the absence of the target behavior rather than a malformed fixture or environment issue.

## GREEN

- **Minimal implementation**: Added `ranking_mrr_slices()` and `scan_probability_blend()` with candidate-zero ranking, deterministic temporal halves, a 101-point integer-derived grid, and reference-preferring tie handling.
- **Command**: `uv run --no-sync pytest tests/test_hybrid_fusion_analysis.py -q`
- **Observed pass**: `2 passed`.

## REFACTOR

- **Refactor done**: yes
- **Change**: Kept checkpoint/Jittor/filesystem work in operator scripts and the metric/selection logic in a small pure module; reused existing checkpoint, fusion prediction, submission validation, and ZIP APIs instead of duplicating model or artifact contracts.
- **Command after refactor**: `uv run --no-sync pytest tests/test_hybrid_fusion_analysis.py tests/test_hybrid_fusion_lgbm.py tests/test_hybrid_checkpoint.py tests/test_submission.py -q`
- **Observed result**: `16 passed, 4 skipped`; both operator scripts also passed `py_compile`.

## Next Behavior

Done for this experiment. Server integration reproduced the new checkpoint's logged Dataset1 and Dataset2 component MRRs on the exact cached validation tensors, and the control artifact passed remote/local hash verification.
