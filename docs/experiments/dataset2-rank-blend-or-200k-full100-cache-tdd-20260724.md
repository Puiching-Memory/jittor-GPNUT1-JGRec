# TDD Evidence: Dataset2 Rank Blend and Recent-200k Full-100 Cache

## Target Behavior

- Compare the champion and every safe existing Dataset2 alternate on the exact
  same `20,000 × 100 × 63` validation tensor.
- Scan champion weights from `0.00` through `1.00` at `0.01` intervals for
  probability, row-z-score, and rank-percentile blending.
- Select without the final chronological slice; accept only full MRR
  `>= champion + 0.001` with no regression in any of three slices.
- If rejected, select exactly the latest 200,000 supervised rows before
  validation and build one `200,000 × 100 × 63` cache with aligned candidate,
  src, dst, time, and sorted-row-index sidecars.

## RED 1 — Missing Alignment-Aware Scan Primitives

- Test surface: `tests/test_hybrid_fusion_analysis.py`.
- Initial failure: imports for the three-slice MRR and prefix-selected `0.01`
  scan did not exist.
- Why this was the right failure: the old helper could not enforce the frozen
  three-slice or untouched-final-slice protocol.

## GREEN 1

- Added tested three-slice MRR, finite/shape checks, three score
  normalizations, and 101-point prefix-selected scanning.
- Initial server scan exposed a suspicious `+0.017` rank-percentile gain.

## RED 2 — Tie-Induced Positive-Column Leakage

- Added `test_ranking_mrr_three_slices_uses_tie_neutral_average_rank`.
- It failed because discrete rank-percentile blends produced ties and the old
  strict-greater rank calculation gave the positive at fixed column zero the
  best possible tie position.
- This was a real metric defect, not a changed expectation.

## GREEN 2

- MRR now uses the tie-neutral average rank
  `1 + greater_count + 0.5 × equal_count`.
- Corrected scan rejected the blend branch:
  - best full delta: `+0.0008869814`, with slice 0 regression;
  - best all-slices-non-decreasing delta: `+0.0005275346`.

## RED 3 — Recent Window and Shared Cache Contract

- Added tests importing `select_recent_events` and
  `validate_full100_cache_arrays`; collection failed because neither existed.
- The new tests require an exact tail window, reject an undersized pool, and
  reject any feature/candidate/sidecar row mismatch.

## GREEN 3

- Added exact recent-tail selection with sorted row indices.
- Generalized the full-100 builder with explicit `--train-rows` and
  `--train-selection recent`.
- Added atomic row-index sidecar output and post-build mmap contract
  validation.
- Added a detached launcher with overwrite protection, PID, and log files.

## Refactor Decision

- Kept recent selection and cache-shape validation in
  `full100_training.py`, where both LightGBM and future setwise consumers can
  reuse the contract.
- Kept orchestration, memmap writes, hashing, and server launch outside model
  code.
- No broader sampling or checkpoint behavior was changed.

## Verification

```text
uv run --no-sync pytest \
  tests/test_hybrid_full100_training.py \
  tests/test_hybrid_fusion_analysis.py -q
14 passed

uv run --no-sync ruff check \
  src/jgrec/rankers/hybrid/full100_training.py \
  scripts/build_dataset2_full100_train_cache.py \
  tests/test_hybrid_full100_training.py
All checks passed!
```

The same commands passed on the server. Final cache acceptance remains pending
until the background build completes and its manifest hashes and mmap reload
checks pass.
