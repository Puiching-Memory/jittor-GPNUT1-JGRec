# Goal Document: Decouple Hybrid Train and Validation Negatives

## Go / No-Go

- **Judgment**: Go
- **Reason**: The hybrid feature and fusion pipeline already supports different candidate widths for train and validation; the missing behavior is configuration and routing.

## Target Outcome

Hybrid runs can independently choose training and validation negative counts, while every existing command that only sets `--num-negatives` behaves exactly as before.

## Goal Definition

- **Type**: technical and quality
- **Boundary**: Add hybrid CLI/config overrides, route the resolved training count to train feature construction and Two-Tower training, and route the resolved validation count only to validation feature construction.
- **Non-goals**:
  - Do not change negative-sampling distributions.
  - Do not change fusion losses or LightGBM parameters.
  - Do not change temporal-graph semantics.
- **Deferred work**:
  - Fixed validation-candidate manifests and feature caching.
  - Full dataset2 experiments using the new flags.
- **Verification rule**: Automated tests must observe different train/validation candidate widths and unchanged legacy fallback behavior.
- **Evidence source**: Focused pytest failures and passes, then the broader hybrid/CLI test suite.
- **Pass criteria**: Train and validation builders receive their independently resolved counts; legacy `num_negatives=N` resolves both to `N`; invalid explicit counts fail clearly.
- **Confidence note**: Tests exercise the stable config and feature-builder boundaries without requiring a full Jittor training run.
- **Judgment owner**: Automated tests.

## Current State

- `TrainingConfig.num_negatives` controls Two-Tower training, fusion train candidates, and fusion validation candidates.
- `_learn_fusion()` calls the same `_build_supervised_features()` path for train and validation.
- Candidate width is derived from `config.num_negatives + 1` inside the supervised feature builder.
- MLP and LightGBM consume train and validation arrays independently and do not require equal candidate widths.

## Priority Rationale

- Protect legacy behavior first because existing champion commands and checkpoints use `num_negatives`.
- Prove routing at the feature-builder boundary before changing implementation.
- Keep the change isolated so later metric-alignment work has a trustworthy validation protocol.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
| --- | --- | --- | --- |
| New CLI flags are `--train-num-negatives` and `--val-num-negatives` | confirmed | Gives explicit user-facing control | Implement in `CLIConfig` |
| Omitted overrides inherit `--num-negatives` | confirmed | Preserves existing commands | Cover with regression test |
| Two-Tower uses the resolved training count | confirmed | Keeps all training-only sampling aligned | Cover in config test |
| Temporal-graph ignores the new hybrid overrides | confirmed | Avoids scope expansion | Cover in CLI config test |

## Phases

### Phase 1: Configuration Contract

- **Purpose**: Define fallback, override, and validation behavior before production changes.
- **Entry condition**: Existing hybrid and CLI tests are discoverable.
- **Phase rules**:
  - Add tests before implementation.
  - Test public dataclass/config construction rather than internal assignments.
- **Todos**:
  - [x] Add RED tests for legacy fallback and independent overrides.
    - **Surface**: `tests/test_cli.py`, hybrid config tests.
    - **Proof**: Focused pytest fails because the fields/resolvers do not exist.
    - **Depends on**: None.
  - [x] Implement optional overrides and resolved-count methods.
    - **Surface**: CLI and hybrid config.
    - **Proof**: Configuration tests pass.
    - **Depends on**: RED evidence.
- **Exit proof**: CLI and config tests pass for fallback, overrides, and temporal-graph isolation.
- **Stop condition**: Stop if Tyro cannot represent optional integer overrides without breaking CLI parsing.

### Phase 2: Train/Validation Routing

- **Purpose**: Ensure the two counts affect the correct candidate builders.
- **Entry condition**: Phase 1 is green.
- **Phase rules**:
  - Do not duplicate feature-building logic.
  - Preserve existing call behavior when no explicit count is supplied.
- **Todos**:
  - [x] Add a RED test that captures train and validation feature widths from `_learn_fusion()`.
    - **Surface**: Hybrid negative/supervised-feature tests.
    - **Proof**: Test fails because both calls currently use one count.
    - **Depends on**: Phase 1.
  - [x] Route train and validation counts through the feature builder.
    - **Surface**: Hybrid ranker and supervised feature builder.
    - **Proof**: Focused tests pass with different widths.
    - **Depends on**: RED evidence.
- **Exit proof**: Tests observe train width `train_negatives + 1` and validation width `val_negatives + 1`.
- **Stop condition**: Stop if either fusion backend assumes equal candidate widths.

### Phase 3: Compatibility and Documentation

- **Purpose**: Verify the old workflow and document the new one.
- **Entry condition**: Phases 1 and 2 are green.
- **Phase rules**:
  - No unrelated cleanup.
  - Existing `--num-negatives` examples remain valid.
- **Todos**:
  - [x] Run focused and broader tests.
    - **Surface**: CLI and hybrid test suites.
    - **Proof**: All selected pytest commands pass.
    - **Depends on**: Phase 2.
  - [x] Document fallback and example usage.
    - **Surface**: Submission/experiment documentation.
    - **Proof**: New flags and compatibility behavior are stated together.
    - **Depends on**: Green implementation.
- **Exit proof**: Tests pass and documentation includes `--train-num-negatives 31 --val-num-negatives 99`.
- **Stop condition**: Stop on a backward-compatibility regression in existing tests.

## Dry-Run Findings

- Train and validation feature tensors may have different second dimensions because fusion functions flatten them independently.
- Two-Tower currently reads the shared count and must use the resolved training value.
- Existing helper tests monkeypatch `_build_supervised_features()` with its current positional signature, so routing should preserve compatibility or update those tests deliberately.
- Old checkpoint objects may not contain newly added attributes; resolver methods should tolerate missing override fields.

## Final Validation

- Focused RED tests failed for the intended missing CLI fields and shared train/validation routing.
- `uv run --no-sync pytest tests/test_cli.py tests/test_hybrid_negatives.py tests/test_hybrid_supervised_features.py tests/test_hybrid_checkpoint.py -q`: `41 passed, 4 skipped`.
- `uv run --no-sync jgrec-build --help` exposes all three negative-count flags.
- `uv run --no-sync ruff check ...`: passed.
- `uv run --no-sync python -m compileall -q src tests`: passed.

## First Execution Step

Add configuration-contract tests that fail because train and validation negative overrides are not yet available.
