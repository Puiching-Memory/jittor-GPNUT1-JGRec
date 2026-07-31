# Hai TDD: Isolated gnn_short Validation Cache Replacement

## Target Behavior

Given a `[queries, candidates, features]` cache and replacement GNN scores, publish a new cache in which exactly one named feature column changes and every other value remains identical.

## RED

- **Test added**: `test_replace_gnn_feature_column_preserves_every_other_value`
- **Behavior asserted**: The selected column equals replacement scores, all other columns and the source array are unchanged, and the operation reports its exact shape/column contract.
- **Command**: `uv run --no-sync pytest -q tests/test_hybrid_gnn_listwise.py`
- **Observed failure**: `ImportError: cannot import name 'replace_feature_column'`
- **Failure is correct because**: The cache replacement API did not exist; the failure was not caused by syntax, fixtures, or the server environment.

## GREEN

- **Minimal implementation**: Added batched, non-overwriting `.npy` publication with shape/column validation and a second-pass unchanged-column equality check.
- **Command**: `uv run --no-sync pytest -q tests/test_hybrid_gnn_listwise.py`
- **Observed pass**: `4 passed`

## REFACTOR

- **Refactor done**: no
- **Change**: No refactor was needed; the helper remains isolated from the existing cache builders and champion prediction path.
- **Command after refactor**: `uv run --no-sync ruff check src/jgrec/rankers/hybrid/gnn_listwise.py tests/test_hybrid_gnn_listwise.py`
- **Observed result**: `All checks passed!`

## Next Behavior

Done. Runtime verification additionally confirmed a `(20000, 100, 63)` cache with only feature column 59 (`gnn_short`) replaced and all other columns equal.
