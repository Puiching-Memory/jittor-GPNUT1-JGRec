# Hai TDD: Dataset2 GNN Marginal-Contribution Ablation

## Target Behavior

Safely perturb only the GNN feature channels without modifying source caches,
then train an exact no-GNN Setwise control that removes the corresponding nine
context channels and preserves every other training setting.

## RED

- **Test added**: `tests/test_feature_ablation.py`
- **Behavior asserted**: Mean-neutralization and selected-column replacement
  copy their input arrays; candidate permutation changes only selected columns;
  excluding three source features removes all three derived context copies.
- **Command**: `uv run --no-sync pytest -q tests/test_feature_ablation.py`
- **Observed failure**:
  1. `ModuleNotFoundError` for the missing ablation module.
  2. `ImportError` for missing candidate-axis permutation after the first
     diagnostic exposed an invalid cross-query permutation.
  3. `ImportError` for the missing retained-context index mapper.
- **Failure is correct because**: Each failure corresponded to a missing
  behavior required for safe and correctly aligned GNN ablation.

## GREEN

- **Minimal implementation**:
  - Added immutable neutralization and replacement helpers.
  - Added within-query candidate-axis permutation with complete-permutation
    validation.
  - Added deterministic retained-context index construction.
  - Added cached perturbation and no-GNN control scripts.
- **Command**:
  `uv run --no-sync pytest -q tests/test_feature_ablation.py`
- **Observed pass**: `4 passed` locally and on the server.

## REFACTOR

- **Refactor done**: yes
- **Change**: Isolated reusable array-safety and context-index logic in
  `feature_ablation.py`; experiment scripts own model loading, metrics, hashes,
  and artifact reports. The invalid v1 cross-query permutation report was
  preserved but superseded by v2 rather than overwritten.
- **Command after refactor**:
  `uv run --no-sync ruff check ...` and targeted pytest commands.
- **Observed result**: Ruff passed locally and remotely; broader local targeted
  suite reported `15 passed`.

## Next Behavior

GNN contribution is confirmed. The next separate goal should compare
short/recent-window edge weighting or candidate-aligned graph training while
keeping the current architecture fixed.
