# Hai TDD: Dataset2 Full-100 Candidate Training Cache

## Target Behavior

A supervised feature cache may atomically persist candidate-ID matrices aligned with train/validation feature tensors, reject mismatched identities, and continue loading legacy feature-only caches.

## RED

- **Test added**: `test_cache_round_trip_loads_candidate_identity_sidecars` and `test_cache_rejects_candidate_identity_shape_mismatch_and_keeps_legacy_optional`.
- **Behavior asserted**: Candidate sidecars round-trip and malformed shapes are rejected without invalidating legacy cache behavior.
- **Command**: `uv run --no-sync pytest tests/test_hybrid_supervised_feature_cache.py::test_cache_round_trip_loads_candidate_identity_sidecars tests/test_hybrid_supervised_feature_cache.py::test_cache_rejects_candidate_identity_shape_mismatch_and_keeps_legacy_optional -q`
- **Observed failure**: `save()` rejected `train_candidates`, and `load_candidate_ids()` did not exist.
- **Failure is correct because**: The current cache stored only feature tensors and post-build RNG state.

## GREEN

- **Minimal implementation**: Added optional atomic train/validation candidate sidecars, manifest descriptors, read-only memmap loading, and pre-write shape/integer validation while retaining cache version 1.
- **Command**: Same focused command.
- **Observed pass**: `2 passed`.

## REFACTOR

- **Refactor done**: no.
- **Change**: No refactor needed beyond small validation/path helpers.
- **Command after refactor**: `uv run --no-sync pytest tests/test_hybrid_supervised_feature_cache.py -q` and Ruff on the cache module/test.
- **Observed result**: `7 passed`; Ruff passed.

## Next Behavior

The bounded replay also requires pure candidate-width and comparison contracts before
starting the server build.

## RED: Replay, Candidate, and Gate Contracts

- **Tests added**: candidate position/uniqueness validation, replay mismatch reporting,
  and the frozen full/per-slice MRR gate.
- **Behavior asserted**: every group has the positive in column zero and unique IDs;
  replay mismatches are measurable; packaging is authorized only for a full MRR delta
  of at least `+0.002` with no chronological-slice regression.
- **Observed failures**: the full-100 helper module did not exist, followed by a missing
  `passes_full100_gate` import.
- **Failure is correct because**: the expensive pipeline previously had no independent,
  testable contracts for these decisions.

## GREEN: Replay, Candidate, and Gate Contracts

- **Minimal implementation**: added `validate_candidate_matrix`,
  `replay_feature_report`, and `passes_full100_gate`.
- **Command**: `uv run --no-sync pytest tests/test_hybrid_full100_training.py -q`.
- **Observed pass**: `3 passed`; combined cache/full-100 suite `10 passed`.
- **Static proof**: Ruff and Python compilation pass for the cache builder and evaluator.

## Next Behavior

Run the bounded replay on the server. Only a matching replay may start the
`50,000 x 100 x 63` memmap build and frozen LightGBM evaluation.

## Replay Result and Matched-Control RED/GREEN

- **Replay result**: candidate contracts passed, but learned feature values did not
  reproduce the old combined-run cache; the original direct route stopped.
- **Protocol change**: one newly fitted encoder owns one 100-candidate cache; each
  group's first 32 positions are the exact nested control view.
- **RED test**:
  `test_matched_control_gate_also_requires_full100_to_beat_control` failed because
  `passes_matched_control_gate` did not exist.
- **GREEN implementation**: the gate composes the original champion `+0.002` and
  all-slice checks with a nonnegative full-MRR comparison against the matched
  32-candidate control.
- **Observed proof**: full helper suite `4 passed`; combined helper/cache suite
  `11 passed`; Ruff and Python compilation pass for both matched build/evaluation
  scripts.
