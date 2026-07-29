# TDD Evidence: Dataset2 short_none 50/40k Checkpoint Integration

## Target Behavior

Install a Setwise fusion head trained against the verified Dataset2
`short_none / 50 epochs / 40000 max_train_edges` scores into a new contest
checkpoint while preserving the existing encoder and every non-fusion
Dataset2 field. The output must inherit the current Dataset1 champion, load
through the standard checkpoint path, and produce repeatable full Dataset2
predictions.

## RED

- The focused test initially failed during collection because
  `jgrec.rankers.hybrid.gnn_short_checkpoint` did not exist.
- The first real checkpoint build rejected the actual source schema because
  legacy checkpoints omit absent Setwise fields instead of storing them as
  `None`; this exposed the need to distinguish allowlisted additions from
  forbidden schema changes.
- The next real build reached the streaming state audit and failed on NumPy
  protocol-5 `pickle.PickleBuffer`; this exposed a memory-safe hashing path
  that had only been exercised with small Python objects.

## GREEN

- Added `install_gnn_short_setwise_fusion()` with hard guards for:
  - 50 graph epochs;
  - 40,000 maximum train edges;
  - unweighted `gnn_short`;
  - the 63-to-189 Setwise context schema;
  - a bounded blend weight;
  - a source LightGBM expert.
- The helper returns a new state mapping and changes only the Setwise state,
  Setwise result, Setwise hidden dimension, and `lgbm_result.mlp_weight`.
- The production state audit now accepts only those explicit additions or
  replacements and streams protocol-5 pickle buffers into SHA-256 without
  materializing multi-gigabyte serialized objects.
- The persisted head passed the offline gate at full MRR `0.5485470649` with
  positive deltas on all three chronological slices.
- The resulting 5,009,231,092-byte checkpoint passed standard loader/hydrate;
  its encoder SHA-256 stayed
  `e5201f3009f9ed999f2b8c4fbbcf586974a65600e2339cafc4b22c061060ee3b`.

## Refactor Decision

The integration policy lives in a small ranker module so experiment scripts
cannot silently install a 200k-edge or weighted-window artifact. Expensive
artifact construction, state auditing, and replay verification remain in
scripts because they are release workflow concerns rather than prediction
runtime behavior. The encoder is deliberately not rebuilt.

## Verification Evidence

Focused contract test:

```text
24 passed in 9.29s
```

Static checks:

```text
ruff check: All checks passed
py_compile: passed
streaming NumPy pickle hash: passed
```

Checkpoint integration:

```text
status: complete
standard_hydrate_passed: true
changed_top_level_keys:
  - lgbm_result
  - setwise_fusion_result
  - setwise_fusion_state
  - setwise_hidden_dim
encoder_hash_stable: true
output_checkpoint_sha256:
  0b0a846cb6c7f5b5403a75a04ed12707340f024dee99a27cded3258bb108a7aa
```

Full replay:

```text
standard_load_replays: 2
dataset2_rows: 153420
byte_identical: true
replay_a_sha256:
  b5544d15fac4bd6d5737c5e7e30d5d413553d1e11109d5ee23acb7b18513cc3a
replay_b_sha256:
  b5544d15fac4bd6d5737c5e7e30d5d413553d1e11109d5ee23acb7b18513cc3a
```

Deliverables:

```text
checkpoint_zstd_test: passed
checkpoint_zstd_sha256:
  0efd6ad0ca70b0fba5743cc3c4fdbdd6e312f56dea331bf32c57830aeeaa9cef
candidate_zip_sha256:
  104f68dc82aed862600be3328f779d80e04746283c0ec75193a3582266438193
submission_authorized: false
```
