# Hai TDD: Dataset2 OOF Hard-Negative LambdaRank

## Target Behavior

Score every cached training query with a miner that did not fit that query, preserve the positive at candidate zero, retain a fixed number of highest-scoring negatives with stable tie handling, and continue only if a fixed candidate improves every chronological validation slice and full MRR by at least 0.002.

## RED 1: OOF and Selection Contract

- **Test added**: `tests/test_hybrid_oof_hard_negatives.py`
- **Behavior asserted**: Three contiguous held folds cover every row once; fitting ranges exclude held rows; top-K keeps candidate zero and resolves ties by original position; feature input is not mutated; invalid partitions/counts fail.
- **Command**: `uv run --no-sync pytest tests/test_hybrid_oof_hard_negatives.py -q`
- **Observed failure**: Import failed with `ModuleNotFoundError: jgrec.rankers.hybrid.oof_hard_negatives`.
- **Failure is correct because**: No OOF partition or hard-negative selection implementation existed.

## GREEN 1

- **Minimal implementation**: Added immutable contiguous-fold descriptors, stable negative-position ranking, and positive-preserving feature gathering.
- **Command**: `uv run --no-sync pytest tests/test_hybrid_oof_hard_negatives.py -q`
- **Observed pass**: Eight contract tests passed after correcting the adjacent-boundary iteration discovered by the first GREEN attempt.

## RED 2: Temporal Acceptance Gate

- **Test added**: `test_temporal_gate_requires_every_slice_and_minimum_full_delta`
- **Behavior asserted**: Acceptance requires strict improvement in every time slice and a full MRR delta of at least 0.002.
- **Command**: `uv run --no-sync pytest tests/test_hybrid_oof_hard_negatives.py::test_temporal_gate_requires_every_slice_and_minimum_full_delta -q`
- **Observed failure**: Import failed because `passes_temporal_mrr_gate` did not exist.
- **Failure is correct because**: The predeclared stop rule was not represented in executable code.

## GREEN 2

- **Minimal implementation**: Added validated finite-input temporal gating with strict per-slice improvement and minimum full-delta enforcement.
- **Command**: `uv run --no-sync pytest tests/test_hybrid_oof_hard_negatives.py -q`
- **Observed pass**: Nine focused tests passed locally and on the Linux server.

## REFACTOR

- **Refactor done**: yes
- **Change**: Replaced manual adjacent-boundary zipping with `itertools.pairwise`; isolated mining contracts from the operational experiment; kept the server script fixed to one configuration and wrote its config before validation predictions.
- **Command after refactor**: `uv run --no-sync pytest tests/test_hybrid_oof_hard_negatives.py tests/test_hybrid_fusion_lgbm.py tests/test_dataset2_lgbm_tuning.py -q` and `uv run --no-sync ruff check ...`
- **Observed result**: 16 related tests passed and Ruff passed. The server scored every training row exactly once out of fold and correctly rejected the candidate after all three validation slices regressed.

## Next Behavior

Done for this hypothesis. The stop rule forbids a strict full-candidate cache rebuild or submission package after the mechanism gate failed.
