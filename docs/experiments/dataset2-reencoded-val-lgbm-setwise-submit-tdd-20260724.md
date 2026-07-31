# Hai TDD: Dataset2 Matched Validation, Setwise Reranker, and Checkpoint Inference

## Target Behavior

Reconstruct the current Dataset2 feature pipeline, prove it matches the recent
200k cache, generate a chronological full-100 validation cache at the
`train_end` snapshot, train a genuinely set-aware listwise reranker with
full-candidate-MRR early stopping, and preserve its predictions through
checkpoint snapshot/hydrate.

## RED 1

- **Test added**:
  `test_matched_cache_replay_requires_exact_candidates_and_close_features` and
  `test_sample_chronological_events_preserves_sorted_global_row_identity`.
- **Behavior asserted**:
  replay requires candidate-by-candidate identity plus close feature values;
  sampled validation rows retain strictly increasing global identities.
- **Command**:
  `uv run --no-sync pytest tests/test_hybrid_full100_training.py -q`
- **Observed failure**:
  import failed because `matched_cache_replay_report` and
  `sample_chronological_events` did not exist.
- **Failure is correct because**:
  the old cache code could validate positive position and shape, but could not
  prove two encoder runs produced the same candidate-feature rows.

## GREEN 1

- **Minimal implementation**:
  added exact candidate mismatch reporting, tolerant feature replay reporting,
  and chronological sampling with global row sidecars.
- **Command**:
  `uv run --no-sync pytest tests/test_hybrid_full100_training.py -q`
- **Observed pass**:
  8 tests passed locally and on the Linux server; Ruff passed.

## REFACTOR 1

- **Refactor done**: yes
- **Change**:
  kept reusable replay and sampling contracts in `full100_training.py`; kept
  encoder orchestration and memmap writes in a task-specific script.
- **Command after refactor**:
  same pytest and Ruff commands.
- **Observed result**:
  all passed.

## RED 2

- **Test added**:
  `test_setwise_context_features_add_row_relative_mean_and_max_channels` and
  `test_streaming_listwise_trainer_early_stops_on_full_candidate_mrr`.
- **Behavior asserted**:
  each candidate receives raw, relative-row-mean, and relative-row-max
  channels; training restores the best epoch and stops after MRR patience.
- **Command**:
  `uv run --no-sync pytest tests/test_hybrid_fusion_listwise.py -q`
  on the Linux server.
- **Observed failure**:
  import failed because the Setwise transform and streaming listwise trainer
  did not exist.
- **Failure is correct because**:
  the previous listwise MLP evaluated only after a fixed number of epochs and
  had no candidate-set context channels.

## GREEN 2

- **Minimal implementation**:
  added a lazy `SetwiseFeatureView` over memmaps and a listwise trainer that
  evaluates full-candidate MRR each epoch, snapshots the best state, and early
  stops.
- **Command**:
  server pytest for `tests/test_hybrid_fusion_listwise.py` plus Ruff.
- **Observed pass**:
  8 tests passed; Ruff passed.

## REFACTOR 2

- **Refactor done**: yes
- **Change**:
  the context transform is a lazy view, so the 5 GB source cache is not
  duplicated into a 15 GB transformed cache.
- **Command after refactor**:
  same server pytest and Ruff commands.
- **Observed result**:
  all passed.

## RED 3

- **Test added**:
  extended `test_hybrid_snapshot_round_trips_predictions` with a Setwise model,
  direct Setwise prediction equivalence, and snapshot/hydrate persistence.
- **Behavior asserted**:
  the hybrid ranker uses Setwise scores when present and a reloaded checkpoint
  returns identical probabilities.
- **Command**:
  `uv run --no-sync pytest
  tests/test_hybrid_checkpoint.py::test_hybrid_snapshot_round_trips_predictions
  -q`
- **Observed failure**:
  the ranker returned the legacy MLP probabilities instead of direct Setwise
  probabilities.
- **Failure is correct because**:
  checkpoint state and prediction routing had no Setwise fields or branch.

## GREEN 3

- **Minimal implementation**:
  added optional Setwise model/result/hidden-dimension state to
  `TemporalHybridRanker`, snapshot/hydrate support, Setwise context prediction,
  and fixed-weight blending with the existing LightGBM expert.
- **Command**:
  the targeted server checkpoint test and Ruff.
- **Observed pass**:
  1 checkpoint round-trip test passed; Ruff passed.

## REFACTOR 3

- **Refactor done**: no
- **Change**:
  no further refactor; the optional branch is isolated and legacy checkpoints
  default to no Setwise model.
- **Command after refactor**:
  not needed beyond the GREEN command.
- **Observed result**:
  legacy behavior remains the default.

## Next Behavior

Production evidence is pending: bounded 4096-row cache replay, matched
validation cache completion, LightGBM and Setwise full/three-slice MRR, and
conditional package generation.
