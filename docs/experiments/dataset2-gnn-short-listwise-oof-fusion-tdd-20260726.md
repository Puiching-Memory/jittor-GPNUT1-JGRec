# Dataset2 gnn_short OOF fusion — TDD evidence

## Target Behavior

Build chronological, leakage-free OOF `gnn_short` features for Dataset2 so
fusion training and validation use the same listwise GNN representation.
Every scored OOF row must be produced by a model trained only on earlier rows.

## RED

Added
`test_expanding_oof_folds_cover_post_burn_in_rows_without_future_leakage`
to `tests/test_hybrid_gnn_listwise.py`.

The first server run failed during collection because
`expanding_oof_folds` did not exist in
`jgrec.rankers.hybrid.gnn_listwise`. This was the expected failure: the
temporal OOF contract had not yet been implemented.

## GREEN

Implemented `TemporalOOFFold` and `expanding_oof_folds` in
`src/jgrec/rankers/hybrid/gnn_listwise.py`.

Verification:

```text
uv run pytest tests/test_hybrid_gnn_listwise.py -q
5 passed

uv run ruff check src/jgrec/rankers/hybrid/gnn_listwise.py \
  scripts/build_dataset2_gnn_short_listwise_oof.py \
  scripts/evaluate_dataset2_gnn_short_listwise_fusion.py \
  tests/test_hybrid_gnn_listwise.py
All checks passed
```

## REFACTOR

Kept fold boundaries in an immutable dataclass and centralized fold
construction in one pure helper. The build script consumes that contract
instead of duplicating boundary arithmetic.

## Next Behavior

Completed the runtime contract:

- burn-in rows: `[0, 25000)`, excluded from OOF fusion training;
- seven expanding folds of 25,000 rows;
- each fold trains on `[0, score_start)` and scores only its future interval;
- OOF coverage: `[25000, 200000)`;
- OOF cache shape: `[175000, 100, 63]`;
- only feature column 59 (`gnn_short`) was replaced;
- all other feature columns remained byte-equivalent;
- `leakage_free: true` in the saved build report.

## Experiment Result

The aligned OOF representation improved the retrained Setwise expert to
MRR `0.5429646129`, but the final 0.80 Setwise fusion reached only
`0.5443614783`, below the champion `0.5469178184`.

All three temporal slices declined, so the hard gate rejected the candidate.
No submission package was generated.
