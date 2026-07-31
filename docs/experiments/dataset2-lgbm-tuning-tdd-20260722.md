# Hai TDD: Dataset2-Only LightGBM Tuning and Checkpoint Overlay

## Target Behavior

Keep a chronological pseudo-B slice outside hyperparameter selection, choose the Dataset2 LightGBM configuration deterministically from tune-only metrics, reject candidates that do not improve both time slices, and allow only a passed report to replace the Dataset2 LightGBM checkpoint state.

## RED

- **Test added**: `tests/test_dataset2_lgbm_tuning.py`
- **Behavior asserted**: Chronological split isolation, tune-only winner selection, bounded deterministic grid, two-slice robustness gate, and passed-report-only checkpoint replacement.
- **Command**: `uv run --no-sync pytest tests/test_dataset2_lgbm_tuning.py -q`
- **Observed failure**: Initial collection failed with `ModuleNotFoundError: jgrec.rankers.hybrid.lgbm_tuning`; the checkpoint-overlay slice later failed with `ImportError: cannot import name 'apply_tuned_lgbm_result'`.
- **Failure is correct because**: The public tuning and gate APIs did not exist yet; both failures directly identified the missing target behavior.

## GREEN

- **Minimal implementation**: Added immutable tune-trial records, chronological slice construction, deterministic tune-only selection, a fixed 12-point Dataset2 grid, a strict two-slice gate, and guarded replacement of only `lgbm_result` while preserving feature indices and the rest of the checkpoint state.
- **Command**: `uv run --no-sync pytest tests/test_dataset2_lgbm_tuning.py -q`
- **Observed pass**: `5 passed` locally and `5 passed` on the server.

## REFACTOR

- **Refactor done**: yes
- **Change**: Kept pure selection/gate/checkpoint-replacement behavior in `lgbm_tuning.py`; isolated cache loading, LightGBM execution, reporting, checkpoint writing, and submission packaging in operator scripts.
- **Command after refactor**: `uv run --no-sync pytest tests/test_dataset2_lgbm_tuning.py tests/test_hybrid_checkpoint.py -q`
- **Observed result**: `11 passed, 4 skipped`; Ruff and compileall also passed for the changed surfaces.

## Integration RED/GREEN

- **RED**: The first server launch stopped before trial 1 with `Cannot change feature_pre_filter after constructed Dataset handle`.
- **Why it was correct**: The reusable LightGBM Dataset had been constructed before the grid's invariant `feature_pre_filter=False` parameter was applied.
- **GREEN**: Dataset construction now fixes `feature_pre_filter=False` up front; the new `_v2` run completed all 12 trials, froze the winner before reading pseudo-B, and passed the robustness gate.

## Next Behavior

Done. The remaining quality evidence is the user's external leaderboard submission; no additional offline tuning should use the pseudo-B slice.
