# Hai TDD: Dataset2 GNN Capacity Experiment Controls

## Target Behavior

Select only the requested graph experiment variant and override only
`epochs`/`max_train_edges`, leaving the source `GraphTowerConfig` unchanged so
the runner can compare `short_none 50/40k` with `short_none 50/200k`.

## RED

- **Test added**:
  `tests/test_hybrid_gnn_capacity_experiment.py::test_resolve_gnn_capacity_experiment_changes_only_requested_capacity`
- **Behavior asserted**: selecting `short_none` returns only that variant,
  changes 40k edges to 200k, preserves all other graph fields, and does not
  mutate the baseline config.
- **Command**:
  `uv run --no-sync pytest -q tests/test_hybrid_gnn_capacity_experiment.py`
- **Observed failure**:
  `ModuleNotFoundError: No module named 'jgrec.rankers.hybrid.gnn_experiment'`
- **Failure is correct because**: the capacity experiment resolver did not
  exist; test collection reached exactly the missing behavior boundary.

## GREEN

- **Minimal implementation**: added
  `resolve_gnn_capacity_experiment()` plus the shared variant registry, then
  exposed `--variant`, `--graph-epochs`, and `--graph-max-train-edges` in the
  existing targeted-GNN evaluator.
- **Command**:
  `uv run --no-sync pytest -q tests/test_hybrid_gnn_capacity_experiment.py`
- **Observed pass**: `1 passed`.

## REFACTOR

- **Refactor done**: yes
- **Change**: reused the existing targeted-GNN scoring/fusion pipeline, kept
  its default five-variant behavior compatible, and generalized report
  generation for a one-variant capacity run. Added a non-overwriting detached
  launcher for the exact 50/200k experiment.
- **Command after refactor**:
  `uv run --no-sync pytest -q tests/test_hybrid_gnn_capacity_experiment.py tests/test_hybrid_gnn_window_config.py`
  followed by
  `uv run --no-sync ruff check src/jgrec/rankers/hybrid/gnn_experiment.py scripts/evaluate_dataset2_targeted_gnn_edges.py tests/test_hybrid_gnn_capacity_experiment.py`
  and
  `bash -n scripts/run_dataset2_gnn_short_capacity_200k_20260727.sh`.
- **Observed result**: `2 passed`; `All checks passed!`; shell syntax check
  exited 0.

## Next Behavior

Done for capacity scanning. The 200k candidate failed its metric gate, so no
production default change or repeat run is authorized.

