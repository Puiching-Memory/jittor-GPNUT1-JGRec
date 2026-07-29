# Hai TDD: Dataset2 three-seed Setwise rank ensemble

## Target Behavior

Given three aligned per-query score matrices, convert each model's scores to
query-local rank percentiles and return their deterministic uniform average.
Reject misaligned or non-finite inputs.

## RED

- **Test added**:
  `test_uniform_rank_average_uses_equal_per_model_query_local_ranks` and
  `test_uniform_rank_average_rejects_misaligned_or_non_finite_models` in
  `tests/test_hybrid_fusion_analysis.py`.
- **Behavior asserted**: Equal model contribution independent of raw score
  scale, with shape and finite-value validation.
- **Command**:
  `uv run --no-sync pytest tests/test_hybrid_fusion_analysis.py -q`
  on the server.
- **Observed failure**: Test collection failed because
  `uniform_rank_average` could not be imported.
- **Failure is correct because**: The rank-ensemble behavior did not yet exist.
  A preceding local Windows `pymetis` build failure was classified as an
  environment failure and was not counted as RED evidence.

## GREEN

- **Minimal implementation**: Added `uniform_rank_average` to
  `src/jgrec/rankers/hybrid/fusion_analysis.py`. It validates at least two
  finite, aligned 2-D matrices, converts each row to higher-is-better
  percentiles, and takes an equal arithmetic mean.
- **Command**:
  `uv run --no-sync pytest tests/test_hybrid_fusion_analysis.py -q`
- **Observed pass**: `13 passed`.

## REFACTOR

- **Refactor done**: no
- **Change**: The pure helper was already isolated and required no further
  structural change.
- **Command after refactor**:
  `uv run --no-sync ruff check src/jgrec/rankers/hybrid/fusion_analysis.py tests/test_hybrid_fusion_analysis.py`
- **Observed result**: All checks passed.

## Next Behavior

Done. Runtime validation trained seed 17 and 41 with the frozen full-100 cache,
reused the hash-verified seed-60 champion model, and evaluated the fixed
three-model rank average.

The first final-evaluation attempt correctly stopped because it referenced the
old 0.07-Setwise training baseline. A read-only evaluator was then run against
the explicit 0.80-Setwise champion report without retraining or changing any
prediction. The corrected gate rejected the ensemble:

- ensemble full MRR: `0.5452611238`;
- champion full MRR: `0.5469178184`;
- full delta: `-0.0016566947`;
- slice deltas: `-0.0036291057`, `-0.0024193925`, `+0.0010788245`;
- package generated: no.
