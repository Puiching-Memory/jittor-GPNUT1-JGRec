# Hai TDD: Dataset2 Full-Candidate Time-Decayed Two-Hop Feature

## Target Behavior

Append one causal `cooccur_time_decay_score` column without changing the original 63 columns, keep the champion MLP on 63 columns, let only Dataset2 LightGBM consume all 64 columns, preserve legacy checkpoints, and enforce the exact full-100 validation gate before packaging.

## RED 1: Compact Causal Aggregate

- **Test added**: `test_time_decayed_cooccurrence_matches_direct_events_after_compaction_and_hydrate`
- **Behavior asserted**: The compact anchor-normalized sparse aggregate equals direct event-time decay before compaction, in future-only mode, and after snapshot/hydrate.
- **Command**: `uv run --no-sync pytest tests/test_hybrid_structure.py -q`
- **Observed failure**: The structure config rejected the new decay fields and no time-decay feature path existed.
- **Failure is correct because**: The temporal index could only retain integer co-occurrence counts, not the time-weighted aggregate required by the feature.

## GREEN 1

- **Minimal implementation**: Added `SparseFloatMap`, anchor/tau metadata, causal co-occurrence decay construction, scoring, copying, compaction, and hydrate support.
- **Command**: `uv run --no-sync pytest tests/test_hybrid_structure.py -q`
- **Observed pass**: Direct, future-only, and hydrated decay values agree within `1e-6`.

## RED 2: Stable Feature Schema and Independent Experts

- **Tests added**: Encoder-cache tests for an appended column and `test_hybrid_fusion_lgbm.py` coverage for independent expert masks.
- **Behavior asserted**: Disabled configs remain exactly 63-dimensional; enabled configs append column 64; MLP selects `0..62`; LightGBM may select `0..63`.
- **Commands**: `uv run --no-sync pytest tests/test_hybrid_encoder_cache.py tests/test_hybrid_fusion_lgbm.py -q`
- **Observed failures**: The enabled encoder still emitted 63 columns, and inference reused the MLP feature selection for LightGBM.
- **Failure is correct because**: Without both contracts, the feature would either shift champion inputs or never reach LightGBM.

## GREEN 2

- **Minimal implementation**: Appended the opt-in feature after all original columns and added independent MLP/LightGBM selection in prediction.
- **Command**: `uv run --no-sync pytest tests/test_hybrid_encoder_cache.py tests/test_hybrid_fusion_lgbm.py -q`
- **Observed pass**: Old schema remains 63 columns; enabled schema is 64; each expert receives its declared input dimension.

## RED 3: Gate Semantics

- **Test added**: `test_temporal_gate_allows_unchanged_slice_when_other_slices_supply_full_gain`
- **Behavior asserted**: A chronological slice may remain equal to baseline because the production rule is “must not decrease,” while full MRR still must improve by at least `0.002`.
- **Command**: `uv run --no-sync pytest tests/test_hybrid_oof_hard_negatives.py -q`
- **Observed failure**: The gate required every slice to be strictly greater than baseline.
- **Failure is correct because**: Strict improvement was stronger than the user-approved non-regression contract.

## GREEN 3

- **Minimal implementation**: Changed slice comparison from `>` to `>=`; retained the full-delta threshold and finite-input validation.
- **Command**: `uv run --no-sync pytest tests/test_hybrid_oof_hard_negatives.py -q`
- **Observed pass**: Improvement, regression, insufficient-full-gain, and equal-slice cases all behave as specified.

## RED 4: Legacy Checkpoint and Decay-Only Packaging State

- **Tests added**: `test_temporal_index_shallow_copy_accepts_legacy_state_without_decay_fields` and `test_time_decay_map_can_build_without_raw_cooccurrence_counts`.
- **Behavior asserted**: A pre-feature checkpoint can hydrate with missing decay attributes, and packaging can build only the float decay map without duplicating the raw integer co-occurrence map.
- **Command**: `uv run --no-sync pytest tests/test_hybrid_structure.py::test_temporal_index_shallow_copy_accepts_legacy_state_without_decay_fields tests/test_hybrid_structure.py::test_time_decay_map_can_build_without_raw_cooccurrence_counts -q`
- **Observed failures**: Legacy shallow-copy raised `AttributeError`; decay construction was incorrectly coupled to `build_cooccurs=True`.
- **Failure is correct because**: Both failures would block loading the champion or exceed the intended packaging memory budget.

## GREEN 4

- **Minimal implementation**: Added legacy defaults during shallow-copy/scoring and decoupled float decay-map construction from integer co-occurrence counts.
- **Command**: Same focused command on Windows and Linux.
- **Observed pass**: Both tests pass on Windows and the server Linux environment.

## REFACTOR

- **Refactor done**: yes
- **Change**: Kept the original 63-feature cache immutable, stored the new train/validation column separately, froze experiment configuration before recovery/training, and split conditional packaging into its own gate-checked script.
- **Command after refactor**: `uv run --no-sync pytest tests/test_hybrid_oof_hard_negatives.py tests/test_hybrid_structure.py tests/test_hybrid_encoder_cache.py tests/test_hybrid_fusion_lgbm.py -q` and Ruff on all touched production files.
- **Observed result**: 41 focused tests passed locally and on Linux before the production run; subsequent legacy/decay-only tests also passed on both platforms.

## Production Evidence

- **Cache alignment**: Passed for 50,000 x 32 train and 20,000 x 100 validation tensors; no seen candidate remained unresolved. Conservative maximum decay was used for 351 train and 566 validation negative positions, within the 800/1,000 hard limits.
- **Exact baseline check**: Reproduced `0.5428303297309955` before candidate scoring.
- **Gate**: Full blend MRR delta `>= 0.002`; each of three contiguous slice deltas `>= 0`.
- **Observed candidate**: Full blend MRR `0.5419462615742707`, delta `-0.0008840681567248154`; slice deltas `[-0.0025513638756688994, +0.0005956229735167851, -0.0006964354245194704]`.
- **Packaging**: Correctly prohibited on gate failure; no checkpoint or result ZIP was generated.
- **Evidence**: `result/dataset2_two_hop_decay_full100_seed60_20260723/full100-report.json` and `decay-cache-report.json`.

## Next Behavior

Do not submit this feature. Retain the report as negative evidence: standalone 32-candidate proxy lift did not transfer to the complete 100-candidate blended objective.
