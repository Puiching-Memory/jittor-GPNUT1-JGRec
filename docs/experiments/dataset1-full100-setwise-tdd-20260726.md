# Hai TDD: Dataset1 Full-100 Setwise Protocol

## Target Behavior

Choose one Dataset1 Setwise training scale and one Setwise/LightGBM blend
weight using only the first two chronological validation slices, while binding
same-process train/validation cache reports to the same dataset.

## RED

- **Test added**:
  `tests/test_hybrid_fusion_analysis.py::test_setwise_model_and_weight_selection_ignores_forward_rows`
- **Behavior asserted**: Selection compares the recent-100k and recent-200k
  Setwise candidates over a fixed weight grid, applies deterministic tie
  breaking, and remains unchanged when all forward-held model scores change.
- **Command**:
  `uv run --no-sync pytest tests/test_hybrid_fusion_analysis.py::test_setwise_model_and_weight_selection_ignores_forward_rows -q`
- **Observed failure**:
  `ImportError: cannot import name 'select_setwise_model_blend_on_prefix'`
- **Failure is correct because**: The multi-model prefix selector did not
  exist; collection reached the intended missing public behavior.

## GREEN

- **Minimal implementation**: Added
  `select_setwise_model_blend_on_prefix()` and
  `SetwiseModelBlendSelection`. The selector scans only rows before
  `selection_stop`, prefers higher Setwise weight on equal selection MRR, and
  applies an explicit model priority last.
- **Command**:
  `uv run --no-sync pytest tests/test_hybrid_fusion_analysis.py::test_setwise_model_and_weight_selection_ignores_forward_rows -q`
- **Observed pass**: `1 passed in 2.23s`.

## REFACTOR

- **Refactor done**: yes
- **Change**: Removed full/three-slice metrics from the selection result and
  stopped calling the existing full-array scanner. The regression now places
  `NaN` values in all forward-held model rows and still passes, proving that
  selection does not validate or score them.
- **Command after refactor**:
  `uv run --no-sync pytest tests/test_hybrid_fusion_analysis.py::test_setwise_model_and_weight_selection_ignores_forward_rows -q`
- **Observed result**: `1 passed in 2.23s`.

## Next Behavior

Bind joint cache reports to the same dataset name.

---

## Target Behavior

Reject a same-process cache pair when the training and validation reports name
different datasets, while retaining compatibility with legacy reports that
both omit the dataset field.

## RED

- **Test added**:
  `tests/test_hybrid_full100_training.py::test_joint_cache_reports_reject_different_dataset_names`
- **Behavior asserted**: A Dataset1 train report cannot be paired with a
  Dataset2 validation report even if build ID, PID, and feature hash match.
- **Command**:
  `uv run --no-sync pytest tests/test_hybrid_full100_training.py::test_joint_cache_reports_reject_different_dataset_names -q`
- **Observed failure**: `AssertionError: ValueError not raised`.
- **Failure is correct because**: The existing provenance validator checked
  process and hash identity but had no dataset identity contract.

## GREEN

- **Minimal implementation**: Added dataset-name agreement validation whenever
  either joint report declares `dataset_name`; legacy pairs with both fields
  absent remain valid.
- **Command**:
  `uv run --no-sync pytest tests/test_hybrid_full100_training.py::test_joint_cache_reports_reject_different_dataset_names -q`
- **Observed pass**: `1 passed in 2.23s`.

## REFACTOR

- **Refactor done**: yes
- **Change**: Parameterized the existing same-process full-100 builder by
  dataset, made historical replay inputs optional for a fresh joint build,
  added a Dataset1 wrapper, and added dual-scale training/evaluation plus
  conditional Dataset1 packaging adapters. The evaluation protocol was aligned
  exactly with the persistable Dataset2 winner: original MLP/LightGBM blend is
  the gate baseline, while Setwise blends with the champion LightGBM expert.
- **Command after refactor**:
  `uv run --no-sync pytest tests/test_hybrid_full100_training.py tests/test_hybrid_fusion_analysis.py tests/test_hybrid_checkpoint.py -q`
  and focused Ruff/`py_compile` checks over all changed modules and scripts.
- **Observed result**: `30 passed, 4 skipped`; Ruff reported
  `All checks passed!`; all four production scripts compiled.

## Next Behavior

Make the exact Dataset1 200k train window fit without changing Dataset2's
already-valid context boundary.

---

## Target Behavior

Keep the configured context boundary when it already leaves enough supervised
rows; otherwise move it earlier by only the number of rows needed for the
frozen exact recent-200k window, while retaining at least one context row and
never entering validation.

## RED

- **Test added**:
  `tests/test_hybrid_full100_training.py::test_training_context_backs_off_only_enough_to_fit_requested_recent_rows`
- **Behavior asserted**: A large Dataset2-like split stays at its configured
  75% context boundary; Dataset1's `train_end=587221` resolves to
  `context_end=387221`; a request consuming the entire pre-validation region
  is rejected.
- **Command**:
  `uv run --no-sync pytest tests/test_hybrid_full100_training.py -q`
- **Observed failure**:
  `ImportError: cannot import name 'resolve_training_context_end'`.
- **Failure is correct because**: The builder previously used the configured
  ratio unconditionally, and production preflight proved that leaves only
  146,806 Dataset1 rows.

## GREEN

- **Minimal implementation**: Added `resolve_training_context_end()` and used
  it in the shared full-100 builder. The function returns
  `min(configured_context_end, train_end-requested_train_rows)` after validating
  that context and validation remain non-empty.
- **Command**:
  `uv run --no-sync pytest tests/test_hybrid_full100_training.py -q`
- **Observed pass**: `11 passed in 2.49s`.

## REFACTOR

- **Refactor done**: yes
- **Change**: Cache reports now record `configured_context_end`,
  `context_end`, and `context_backoff_rows`, making the Dataset1-only boundary
  adjustment auditable. Dataset2 behavior remains unchanged when its
  configured pool already contains the requested rows.
- **Verification**: Focused Ruff reported `All checks passed!`.

## Next Behavior

Sync the green change to Linux, rerun the remote suite, and build the
same-process 200k/20k cache pair.

---

## Target Behavior

Treat a sparse row key greater than the final stored CSR key as a missing row
instead of indexing past the row-key array during batched source-profile
feature generation.

## RED

- **Test added**:
  `tests/test_hybrid_source_profile.py::test_sparse_batch_counts_treat_keys_after_last_row_as_missing`
- **Behavior asserted**: Keys present in the map return their stored counts,
  while a key beyond the largest row key returns zero.
- **Command**:
  `uv run --no-sync pytest tests/test_hybrid_source_profile.py::test_sparse_batch_counts_treat_keys_after_last_row_as_missing -q`
- **Observed failure**:
  `IndexError: index 2 is out of bounds for axis 0 with size 2`.
- **Failure is correct because**: `np.searchsorted()` can return
  `len(row_keys)`, but `batch_get_counts()` dereferenced all indices before
  applying the bounds mask. Production reproduced the same fault after 20,480
  valid cache rows.

## GREEN

- **Minimal implementation**: Compute the in-bounds mask first, then compare
  row keys only at valid positions, matching the safe lookup pattern already
  used by `sum_rows_arrays()`.
- **Focused command**:
  `uv run --no-sync pytest tests/test_hybrid_source_profile.py::test_sparse_batch_counts_treat_keys_after_last_row_as_missing -q`
- **Observed pass**: `1 passed in 2.48s`.
- **Linux regression command**:
  `.deps/uv/bin/uv run --no-sync pytest tests/test_hybrid_source_profile.py tests/test_hybrid_structure.py tests/test_hybrid_checkpoint.py -q`
- **Linux result**: `47 passed`; Ruff reported `All checks passed!`.

## REFACTOR

- **Refactor done**: no
- **Reason**: The five-line mask-order fix is already the smallest
  behavior-preserving repair. Failed `.part` arrays were removed, while the
  failed log and progress report were retained as evidence before a clean
  same-seed restart.

## Next Behavior

The clean same-process build completed and the frozen Setwise experiment ran
to its gate. Recent-100k at weight `1.00` was locked before forward evaluation,
but full delta `+0.0014274006` missed the `+0.002` threshold and slice 0
regressed `-0.0000677395`. The report therefore records `status=rejected`,
`gate_passed=false`, and `package_authorized=false`; no package was built.
