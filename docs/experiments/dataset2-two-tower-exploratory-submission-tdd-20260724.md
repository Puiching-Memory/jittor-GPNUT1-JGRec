# Hai TDD: Explicit Exploratory Dataset2 Packaging

## Target Behavior

Keep rejected Dataset2 tuning reports blocked by default, while allowing an explicit, auditable override to hydrate a user-requested exploratory candidate without changing the selected feature contract.

## RED

- **Test added**: `test_apply_tuned_lgbm_result_allows_explicit_exploratory_override`
- **Behavior asserted**: A rejected report can be applied only when `allow_rejected_report=True`, preserving feature indices and applying the frozen model, score, name, and blend weight.
- **Command**: `uv run --no-sync pytest tests/test_dataset2_lgbm_tuning.py::test_apply_tuned_lgbm_result_allows_explicit_exploratory_override -q`
- **Observed failure**: `TypeError: apply_tuned_lgbm_result() got an unexpected keyword argument 'allow_rejected_report'`
- **Failure is correct because**: The production API did not yet expose an explicit exploratory override.

## GREEN

- **Minimal implementation**: Added the default-false `allow_rejected_report` argument and the CLI-only `--allow-rejected-tuning` switch; recorded `exploratory_override` in the candidate report.
- **Command**: `uv run --no-sync pytest tests/test_dataset2_lgbm_tuning.py -q`
- **Observed pass**: 6 tests passed locally.

## REFACTOR

- **Refactor done**: no
- **Change**: No refactor needed; the gate remains a single default-safe condition.
- **Command after refactor**: `uv run --no-sync pytest tests/test_dataset2_lgbm_tuning.py tests/test_contest_checkpoint.py -q && uv run --no-sync ruff check src/jgrec/rankers/hybrid/lgbm_tuning.py scripts/build_dataset2_lgbm_tuned_candidate.py tests/test_dataset2_lgbm_tuning.py`
- **Observed result**: 13 tests passed on the server and Ruff passed.

## Next Behavior

Done. The generated ZIP passed server and local archive integrity checks, and its local SHA-256 matched the server.
