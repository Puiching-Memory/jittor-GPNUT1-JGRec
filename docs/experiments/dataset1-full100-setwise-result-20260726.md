# Dataset1 Full-100 Setwise Result

## Verdict

**Rejected by the frozen offline gate; no checkpoint or submission package was
generated.**

The experiment found a positive but insufficient and temporally uneven signal:
full-validation MRR improved by `+0.0014274006`, below the predeclared
`+0.002` minimum, while the earliest slice regressed by
`-0.0000677395`.

## Locked Selection

| Field | Result |
|---|---:|
| Selection rows | `[0:13334]` |
| Forward-held rows | `[13334:20000]` |
| Selected scale | recent 100k |
| Selected Setwise weight | `1.00` |
| Models tested | 2 |
| Weights tested per model | 20 |
| Selection MRR | `0.7949377455` |
| Forward rows used for selection | no |

The 100k model stopped at epoch 6 with its best state from epoch 4. The 200k
model stopped at epoch 9 with its best state from epoch 7.

## Gate Metrics

| Window | Champion MRR | Candidate MRR | Delta |
|---|---:|---:|---:|
| Full 20k | `0.7894189977` | `0.7908463983` | `+0.0014274006` |
| Slice 0 | `0.7936535922` | `0.7935858527` | `-0.0000677395` |
| Slice 1 | `0.7957883902` | `0.7962896384` | `+0.0005012482` |
| Slice 2, unseen gate | `0.7788134199` | `0.7826624763` | `+0.0038490563` |

Gate decisions:

- Full delta at least `+0.002`: **failed**
- Every chronological slice non-decreasing: **failed**
- Forward-held slice non-decreasing: passed
- Package authorized: **no**

## Cache and Provenance Proof

- Train tensor: `200000 x 100 x 63`
- Validation tensor: `20000 x 100 x 63`
- Exact train window: sorted rows `[387221:587221]`
- Configured context boundary: `440415`
- Actual context boundary: `387221`
- Minimal context backoff: `53194`
- Same build ID:
  `6c6ddd8647d442c598c9ae854adc8cd7`
- Same Python PID for train and validation: `603645`
- Train feature SHA-256:
  `a8f4b5d71dedd1b5aa89a9f0c40e1501afc882929d036ddaaa73076e5be6a6ef`
- Validation feature SHA-256:
  `43f32c3430eec82180314f889cdfe94b9a8fdb9dc3fd338f7b22f3fa44ad6906`
- Source cache hashes before and after Setwise training: unchanged
- Frozen Dataset2 SHA-256 after the experiment:
  `d7e0d574789aa6b507e592fffb5da054839aba663f63ed58134ea2d951b1ae1e`

## Production Repairs

Two pre-result failures were repaired without changing the frozen model search:

1. Dataset1's configured supervised pool had only 146,806 rows. A tested
   boundary resolver now moves context earlier only enough to fit exact 200k;
   Dataset2 behavior remains unchanged when its pool is already sufficient.
2. `SparseCountMap.batch_get_counts()` dereferenced a `searchsorted()` sentinel
   before applying its bounds mask. A focused RED reproduced the production
   `IndexError`; the minimal mask-order fix passed 47 related Linux tests and
   the clean build then completed.

Failure logs and progress evidence are retained under
`artifacts/experiments/dataset1_full100_setwise_seed60_20260726/cache/`.

## Verification

- Local protocol/checkpoint suite: `31 passed, 4 skipped`
- Local sparse sentinel regression: `1 passed`
- Linux protocol preflight: `35 passed`
- Linux sparse/source-profile regression suite: `47 passed`
- Ruff: passed
- Production scripts: `py_compile` passed
- Downloaded train-report, validation-report, selection-report, selected-model,
  and validation-prediction hashes: all match the frozen evaluation report

## Interpretation

Setwise is useful for Dataset1, but its gain concentrates late in time. The
frozen selector chose pure Setwise on the first two slices, yet that choice
slightly harms slice 0 and strongly helps the unseen final slice. The next
highest-value follow-up is therefore a **predeclared time/recency-ramped blend
between the champion and this 100k Setwise expert**, selected only on slices 0
and 1 and gated unchanged on slice 2. It can reuse the finished caches and
models; no 200k feature rebuild is needed.

Do not weaken the current gate or package the rejected pure-Setwise model.
