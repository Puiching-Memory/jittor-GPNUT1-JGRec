# Dataset2 bounded source-sequence decoder result

## Verdict

**The catastrophic overfitting path is fixed, but the decoder does not improve
the current champion. No submission was generated.**

The old end-to-end D model scored `0.47447143` externally. The new bounded
decoder scored `0.54429985`, nearly recovering the frozen CST score
`0.54439158`. Its remaining delta versus the frozen base is only
`-0.00009172`, instead of the old D model's roughly `-0.06992`.

## Structural fix

The new model enforces:

- frozen CST logits with no gradients;
- no standalone candidate-ID logit or hidden-state branch;
- candidate IDs are used only as queries into source history;
- shared candidate/source item embeddings;
- source time and position embeddings;
- frequency shrinkage from pre-origin item support;
- exact row-centered residuals;
- hard caps `0.02`, `0.05`, or `0.10`;
- exact base fallback for empty histories.

All trainable components are Jittor modules.

## Fold0/Fold1 selection

| Cap | Fold0 delta | Fold1 delta | Mean |
|---:|---:|---:|---:|
| 0.02 | +0.00071814 | +0.00095396 | +0.00083605 |
| 0.05 | +0.00179794 | +0.00211559 | +0.00195676 |
| 0.10 | +0.00370269 | +0.00396685 | +0.00383477 |

All caps passed the fixed selection rule. Cap `0.10` was locked before Fold2.

## Unseen Fold2

Cap `0.10` passed the gate:

- Fold2 delta: `+0.00297166`;
- three-fold mean delta: `+0.00354707`;
- maximum residual: `0.10000002`;
- maximum absolute row mean: `3.32e-7`;
- 1,305 empty-history rows reproduced the base exactly.

All activity groups remained positive:

| Activity group | Delta |
|---|---:|
| Q1 | +0.00114951 |
| Q2 | +0.00311886 |
| Q3 | +0.00321377 |
| Q4 | +0.00440449 |

This confirms that bounding removes the large activity-dependent collapse seen
in the unconstrained model.

## External validation

| Metric | MRR |
|---|---:|
| Bounded source decoder | 0.54429985 |
| Frozen CST | 0.54439158 |
| Current champion | 0.54789665 |

- delta versus frozen CST: `-0.00009172`;
- delta versus champion: `-0.00359680`;
- time-slice deltas versus frozen CST:
  - slice 0: `-0.00022996`;
  - slice 1: `+0.00039620`;
  - slice 2: `-0.00044143`.

The external residual audit passed:

- maximum residual: `0.10000002`;
- maximum absolute row mean: `3.19e-7`;
- 875 empty-history rows reproduced the base exactly;
- all scores were finite.

## What was solved

The old D model let `candidate_id_scale` grow to `1.407` and fell
`-0.073425` behind the champion. The new model cannot create that failure:

- relative to the frozen CST, the external damage fell from about `-0.06992`
  to `-0.00009172`;
- more than 99.8% of the catastrophic base-relative regression was removed;
- the remaining champion gap is essentially the frozen CST's existing gap,
  not a newly created decoder failure.

## What remains

Bounding solves overfitting but does not make source history a sufficiently
strong standalone improvement. The next useful step is not a larger cap.
Source history should become a sparse specialist:

- train OOF correction labels from the bounded decoder;
- route only stable short-history/repeat cases;
- require agreement on long-gap folds;
- keep the same hard cap and exact fallback.

The external result from this run must not be used to scan caps or blends.

## Compliance and artifacts

- `trainable_frameworks = ["jittor"]`
- `non_jittor_trainable_models = []`
- `submission_generated = false`
- online champion unchanged: `1.3545839690981516`

Machine-readable evidence:

- `artifacts/dataset2_bounded_source_decoder_20260727/evaluation-report.json`;
- `artifacts/dataset2_bounded_source_decoder_20260727/gate-report.json`;
- `artifacts/dataset2_bounded_source_decoder_20260727/external-evaluation-report.json`;
- `artifacts/dataset2_bounded_source_decoder_20260727/smoke-report.json`.
