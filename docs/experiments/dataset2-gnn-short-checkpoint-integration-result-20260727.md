# Dataset2 short_none 50/40k Checkpoint Integration Result

## Decision

**Integrated.** Stop increasing graph edges. The winning
`short_none / 50 epochs / 40000 max_train_edges` signal now has a persisted
Setwise fusion head, an independent dual-dataset checkpoint, two successful
standard-load replays, and a non-submitted candidate package.

## Offline Result

The persisted head uses `0.80 Setwise + 0.20 LightGBM`.

| Metric | Original baseline | Persisted 50/40k | Delta |
|---|---:|---:|---:|
| Full MRR | 0.5469178184 | 0.5485470649 | +0.0016292464 |
| Slice 0 | 0.5863014322 | 0.5882028774 | +0.0019014452 |
| Slice 1 | 0.5482466914 | 0.5493313411 | +0.0010846497 |
| Slice 2 | 0.5061992242 | 0.5081009094 | +0.0019016851 |

It also exceeds the later conservative-window Dataset2 candidate on all four
offline metrics:

| Metric | Conservative α=0.30 | Persisted 50/40k | Delta |
|---|---:|---:|---:|
| Full MRR | 0.5478966506 | 0.5485470649 | +0.0006504143 |
| Slice 0 | 0.5866881339 | 0.5882028774 | +0.0015147436 |
| Slice 1 | 0.5490138054 | 0.5493313411 | +0.0003175358 |
| Slice 2 | 0.5079820256 | 0.5081009094 | +0.0001188838 |

The first exploratory process did not save its Setwise weights. Retraining
used the exact preserved graph-score arrays:

- train score SHA-256:
  `8bd4fb9af4da319765a313667359a8a2b1fd91bdd4903cf4b2bca2dbc17f2ead`;
- validation score SHA-256:
  `ba64aa07e98d72662b266110df3e1853f01b6311a20339b0e14819fc9e3690e7`.

The persisted trajectory is independently gated; it is not mislabeled as a
bitwise recovery of the discarded in-memory head.

## Checkpoint Contract

The source Dataset2 service encoder was already the desired 50/40k
unweighted graph encoder, so it was not rebuilt.

Only these Dataset2 top-level fields changed:

- `lgbm_result` — model text unchanged, `mlp_weight` changed to `0.80`;
- `setwise_fusion_state`;
- `setwise_fusion_result`;
- `setwise_hidden_dim`.

Encoder SHA-256 before and after:

```text
e5201f3009f9ed999f2b8c4fbbcf586974a65600e2339cafc4b22c061060ee3b
```

The checkpoint inherits Dataset1 from the current `gamma=0.5` champion and
passes the standard loader/hydrate path.

## Replay and Artifacts

- Two independent full Dataset2 replays: 153,420 rows each.
- Replay CSVs are byte-identical.
- Dataset2 CSV SHA-256:
  `b5544d15fac4bd6d5737c5e7e30d5d413553d1e11109d5ee23acb7b18513cc3a`.
- Checkpoint SHA-256:
  `0b0a846cb6c7f5b5403a75a04ed12707340f024dee99a27cded3258bb108a7aa`.
- Compressed checkpoint SHA-256:
  `0efd6ad0ca70b0fba5743cc3c4fdbdd6e312f56dea331bf32c57830aeeaa9cef`.
- Candidate ZIP SHA-256:
  `104f68dc82aed862600be3328f779d80e04746283c0ec75193a3582266438193`.
- Candidate package explicitly records `submission_authorized: false`.

Remote checkpoint:

```text
/home/edu/workspace/jittor-GPNUT1-JGRec/checkpoints/
d1_time_ramp_g050_d2_gnn_short_none_e50_edges40000_setwise_w080_seed60_20260727.pkl
```

Remote compressed checkpoint:

```text
/home/edu/workspace/jittor-GPNUT1-JGRec/checkpoints/
d1_time_ramp_g050_d2_gnn_short_none_e50_edges40000_setwise_w080_seed60_20260727.pkl.zst
```

The candidate ZIP, fusion model, evaluation report, checkpoint audit, and
replay report are also synchronized into the local workspace under `result/`.

## Verification

```text
ruff check: passed
pytest: 24 passed
standard hydrate: passed
zstd integrity test: passed
two full standard-load replays: byte-identical
```

No online submission was made.
