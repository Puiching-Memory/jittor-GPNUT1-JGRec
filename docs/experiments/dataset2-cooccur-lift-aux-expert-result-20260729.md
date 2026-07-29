# Dataset2 Cooccur Lift Auxiliary Expert v1 — Result

## Verdict

`cooccur_lift_aux_expert_v1` passed every preregistered rolling hard gate
and the single external evaluation. The locked weight is `0.50`. This
authorizes packaging an online candidate, but not promoting or rewiring the
production checkpoint before an online score is returned.

The result supports the registered mechanism: the diagnostic-only causal
destination-popularity Q1 segment improved strongly on all three rolling
folds. That segment was not an input to weight selection.

## Frozen execution

- Frozen config SHA-256:
  `55561f8c52d0be1b97fbc9a80742e11995fa1cbe3a1b836f475a2d85fde34c78`
- Weights: `0.05, 0.10, 0.20, 0.30, 0.40, 0.50`
- Fold seeds: `30073`, `31082`, `32091`
- Full-origin external/test seed: `33100`
- Fold boundaries:
  - train `[0, 79909)`, score `[79909, 118816)`
  - train `[0, 118816)`, score `[118816, 159804)`
  - train `[0, 159804)`, score `[159804, 200000)`
- Auxiliary head: one 195-column Setwise head per fold, four epochs
- Formula: `(1 - w) * fold_champion + w * auxiliary`
- No formula, window, salt, epoch, or weight change was made after metrics.
- `cooccur_time_decay_score` remained disabled and was not reused.

## Exact memory-safe materialization

Two direct Python-index attempts exceeded the 32 GB server envelope before
producing any candidate metric. They were terminated at about 29.4 GB RSS.
The final exact native backend uses symmetric triangular `uint8` count
arrays plus overflow maps and a streaming short-window queue.

The native backend was checked against a micro
`TemporalInteractionIndex`, then materialized the shared
`(200000, 100, 2)` float32 matrix in 18.42 seconds at about 13.1 GB peak
RSS. Its SHA-256 is:

```text
c4ca979de5c56de772edd87c0c9792868f3e22cb47e333ade4c2ca8fe7ff64f4
```

No failed memory attempt reached fold training, selector metrics, or the
external holdout.

## Rolling result

All six frozen weights were eligible under the unchanged selector. The
selector locked the largest preregistered eligible weight, `0.50`.

| Panel | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| Pooled MRR | 0.426997720 | 0.514384713 | +0.087386992 |
| Pooled Hit@1 | — | — | +0.071670650 |
| Pooled Hit@3 | — | — | +0.122448810 |
| Pooled Hit@10 | — | — | +0.105470020 |
| Pooled mean rank | — | — | -2.043958300 |

Fold MRR deltas were `+0.090299195`, `+0.087118175`, and
`+0.084842292`; fold NDCG@10 deltas were `+0.099841122`,
`+0.099472467`, and `+0.093920964`. Pooled movements were 50,820
improved, 48,890 unchanged, and 20,381 worsened.

## Diagnostic-only Q1 result

The stable lower quartile of causal positive-destination popularity
contained 30,023 rolling queries. At locked weight `0.50`:

| Panel | Baseline MRR | Candidate MRR | Delta |
|---|---:|---:|---:|
| Fold 0, 9,727 rows | 0.159635354 | 0.375373762 | +0.215738408 |
| Fold 1, 10,247 rows | 0.154971938 | 0.349652106 | +0.194680169 |
| Fold 2, 10,049 rows | 0.161447359 | 0.353016502 | +0.191569143 |
| Pooled | 0.158650203 | 0.359111632 | +0.200461429 |

This diagnostic agrees with the hypothesis but did not select the weight
or relax any global gate.

## One-shot external result

Nineteen validation rows shared the exact training maximum timestamp and
were excluded by the existing strict temporal boundary. The one-shot
external panel therefore contained 19,981 rows and was opened only after
the selection lock existed.

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| MRR | 0.548360090 | 0.584030858 | +0.035670768 |
| Hit@1 | 0.370902357 | 0.404984735 | +0.034082378 |
| Hit@3 | 0.662179070 | 0.707622241 | +0.045443171 |
| Hit@10 | 0.905510235 | 0.931585006 | +0.026074771 |
| NDCG@10 | 0.630064128 | 0.664929207 | +0.034865079 |
| Mean rank | 4.283769581 | 3.633802112 | -0.649967469 |

Movements were 5,407 improved, 11,379 unchanged, and 3,195 worsened.
All seven external gates passed. No leaderboard or adjacent-weight rescan
was authorized or performed.

## Online candidate

The exact test auxiliary probability matrix completed in 1,089.31 seconds:

```text
shape: 153420 x 100
maximum row-sum error: 1.1102230246251565e-15
probability SHA-256:
da1c18fd6d3c85f95d8aa0987da5777ebefc28da1060bb5b7645bcdd4c12b707
```

The package uses the locked formula
`0.50 * champion + 0.50 * auxiliary`. Its audit confirms:

- ZIP members are exactly `dataset1.csv` and `dataset2.csv`;
- Dataset1 has 61,051 rows and is byte-identical to the champion member;
- Dataset2 has 153,420 rows and 100 columns;
- maximum persisted CSV rounding error is
  `4.999999865529237e-09`;
- production checkpoint modification and promotion authorization are both
  false.

Artifact:

```text
result/d1_time_ramp_g050_d2_cooccur_lift_aux_w050_20260729/result.zip
SHA-256:
7ebfeb7ea29d8dcd03a43a7433a43ddc8de0e24245d93115ecbf8ebd17ef50eb
```

## Evidence

Remote root:

```text
/home/edu/workspace/jittor-GPNUT1-JGRec/result/
dataset2_cooccur_lift_aux_expert_v1_20260728_compact_retry2/
```

SHA-256:

```text
e3a45685d5f1e74aa98966453ca853ec79ff30f72c575f652e0e0673fa3d67b5  run contract
52ca9c13e521956e502b2221033fd9b774ded3a14ffcde00c7f53684fd1fd459  rolling manifest
018ab7984dd553b165718d1e9f60051ad48a4dd5e2d61e7cd6301267c8068540  selection report
cf3e933e1edb3b96330330c4de4f50d5c87a6094dc8661c72a5768852f2f9839  selection lock
6e40f56c0e6bae8065bc4848318f87755ebd3ee4699717c70fb310aaffca6a65  external open receipt
29cc12163f563bdf26851f7603d3ff92d3ceb26458e7a7eff596812c10b93aa6  external report
da1c18fd6d3c85f95d8aa0987da5777ebefc28da1060bb5b7645bcdd4c12b707  test auxiliary probabilities
7ebfeb7ea29d8dcd03a43a7433a43ddc8de0e24245d93115ecbf8ebd17ef50eb  online candidate ZIP
```

## Stop rule

The experiment family is complete at the accepted package stage. Do not
rescan weights, change the lift formula/window, reuse the rejected
time-decay switch, or change the salt. Formal checkpoint wiring and double
replay require a separate online score above the current promotion
threshold.
