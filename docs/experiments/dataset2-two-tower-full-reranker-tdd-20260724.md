# Hai TDD: Dataset2 Two-Tower Full Reranker Integration

## Target Behavior

A Dataset2-only candidate build can use tower-specific 200k/99/listwise/MRR
settings while fusion remains train31/val99, and its Dataset2 checkpoint state can
replace the champion Dataset2 state without changing champion Dataset1.

## RED

- **Test added**:
  `test_compose_checkpoint_keeps_champion_dataset1_and_replaces_dataset2_from_partial`.
- **Behavior asserted**: a Dataset2-only partial checkpoint can replace Dataset2
  while champion Dataset1 and base metadata remain unchanged.
- **Command**:
  `uv run --no-sync pytest tests/test_contest_checkpoint.py::test_compose_checkpoint_keeps_champion_dataset1_and_replaces_dataset2_from_partial -q`.
- **Observed failure**: test collection failed with
  `ImportError: cannot import name 'compose_checkpoint_datasets'`.
- **Failure is correct because**: the repository had no safe composition API;
  manually copying a partial dataset record was the missing behavior.

## GREEN

- **Minimal implementation**: added `compose_checkpoint_datasets`, which validates
  model/dataset metadata, loads only the named replacement states, streams states
  through the atomic writer, and aborts the temporary output on failure.
- **Command**:
  `uv run --no-sync pytest tests/test_contest_checkpoint.py -q`.
- **Observed pass**: 7 passed locally.

## REFACTOR

- **Refactor done**: yes.
- **Change**: normalized replacement names and retained non-reserved champion
  metadata plus an explicit `composed_replacements` record.
- **Command after refactor**:
  `uv run --no-sync pytest tests/test_contest_checkpoint.py
  tests/test_hybrid_two_tower.py tests/test_hybrid_checkpoint.py
  tests/test_cli.py -q`.
- **Observed result**: 46 passed on the Linux CUDA server; focused Ruff checks passed.

## Next Behavior

Complete the Dataset2-only build, then validate the full ensemble and checkpoint
composition gate.
