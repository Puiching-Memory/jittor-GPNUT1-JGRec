# Hai TDD: Dataset2 Two-Tower Listwise 200k

## Target Behavior

Two-Tower can optionally train each positive-at-column-zero candidate group with
softmax cross-entropy, sample a tower-specific number of negatives from the
test-candidate frequency distribution, and select its best epoch by complete-candidate
MRR while preserving all legacy defaults.

## RED

- **Tests added**:
  - positive-at-column-zero group-softmax loss matches a NumPy reference;
  - full-candidate MRR is converted to a minimizing early-stop signal;
  - tower-specific negatives/objective/metric do not change fusion negatives;
  - tower batches actually draw from the public test-candidate distribution.
- **Behavior asserted**: listwise loss, MRR stop signal, tower-specific config
  wiring, and DatasetProfile sampling propagation.
- **Command**:
  `uv run --no-sync pytest tests/test_hybrid_two_tower.py::<four focused tests> -q`
  on the Linux server.
- **Observed failure**: the first pair failed with missing helper imports; the
  second pair failed on an unknown `two_tower_num_negatives` constructor field
  and an unknown `dataset_profile` context argument.
- **Failure is correct because**: none of the requested contracts existed in
  production code before the tests.

## GREEN

- **Minimal implementation**:
  - added optional `listwise` objective and `mrr` early-stop metric;
  - added tower-specific negative count and test-candidate ratio;
  - wired DatasetProfile candidate IDs/frequencies into Two-Tower sampling;
  - retained legacy `bce`/`loss` defaults and 64-dimensional architecture.
- **Command**:
  `uv run --no-sync pytest tests/test_hybrid_two_tower.py
  tests/test_hybrid_checkpoint.py tests/test_hybrid_negatives.py
  tests/test_cli.py -q`.
- **Observed pass**: 45 passed on the Linux CUDA server; 22 non-Jittor
  CLI/config tests also passed locally.

## REFACTOR

- **Refactor done**: yes.
- **Change**: precompute the deterministic candidate groups once because the
  per-event negative seeds were already fixed across epochs; use weighted
  replacement plus duplicate rejection to preserve sequential weighted
  without-replacement semantics without scanning the full vocabulary per row.
- **Command after refactor**:
  `uv run --no-sync pytest tests/test_hybrid_two_tower.py
  tests/test_hybrid_negatives.py -q` and focused Ruff checks.
- **Observed result**: 18 passed on Linux; Ruff reported `All checks passed`.

## Next Behavior

The formal detached Dataset2 run now has to finish the frozen 20,000 x 100
full/per-slice MRR gate. A pass authorizes a separate full-reranker integration
experiment; it does not itself authorize a submission package.
