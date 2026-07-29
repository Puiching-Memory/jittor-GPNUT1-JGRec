# Dataset2 OOF / temporal signal correction result

## Verdict

**Implementation complete; model candidate rejected; no submission.**

Strict temporal support looked exceptionally strong on all three short-horizon
rolling folds, but reversed sharply on the long-horizon external validation.
The online champion remains unchanged at `1.3545839690981516`.

## Frozen candidates

All candidates used frozen-base top-10 score reassignment and a pure-Jittor
router capped at 5% of rows.

| Candidate | Fold0 delta | Fold1 delta | Mean | Selection |
|---|---:|---:|---:|---|
| OOF disagreement | -0.00011623 | +0.00042115 | +0.00015246 | Rejected: Fold0 negative |
| Strict temporal support | +0.01577671 | +0.01124613 | +0.01351142 | Selected |
| Hybrid consensus | +0.00625587 | +0.00504468 | +0.00565028 | Eligible, not selected |

The OOF disagreement signal therefore supplied diversity, but not a stable
correction direction by itself. The temporal signal dominated the fixed
selection protocol.

## Unseen Fold2 gate

Strict temporal support passed the unseen temporal gate:

- Fold2 delta: `+0.01458585`;
- three-fold mean delta: `+0.01386956`;
- routed rows: `2009 / 40196` (`4.998%`);
- all exact score-multiset/scope audits passed.

Activity-quartile deltas were all positive:

| Activity group | Delta |
|---|---:|
| Q1 | +0.00988466 |
| Q2 | +0.01743623 |
| Q3 | +0.02079035 |
| Q4 | +0.01023216 |

This is materially more stable across activity levels than the earlier static
ID correction.

## External validation

The frozen gate allowed one external evaluation. It rejected the candidate:

| Metric | MRR |
|---|---:|
| Sparse temporal correction | 0.53635013 |
| Frozen CST base | 0.54439158 |
| Current champion | 0.54789665 |

- delta versus frozen CST: `-0.00804145`;
- delta versus champion: `-0.01154652`;
- every external time slice was negative;
- routed rows: `1000 / 20000` (`5%`);
- all exact score-multiset/scope audits still passed.

No checkpoint, score file, or submission package replaced the champion.

## New finding: temporal horizon mismatch

The read-only diagnostic shows that this is not fixed by routing fewer
high-confidence rows:

- full strict-temporal proposal:
  - Fold0 `+0.02035299`;
  - Fold1 `+0.00853259`;
  - Fold2 `+0.02457574`;
  - external `-0.08516557`;
- even the top-probability external `0.1%` route was negative:
  `-0.00030946`.

The time horizons explain the reversal:

- rolling score windows span about 31, 34, and 36 days;
- external validation starts at the final train timestamp but extends another
  468 days.

The static pre-origin source-to-candidate support is therefore a strong
**short-term repeat expert**, but becomes stale over the external horizon. The
router learned a real short-horizon pattern; it did not learn a horizon-stable
one.

## What this rules out

- Static ID was not the only problem: replacing it with a semantically stronger
  temporal signal still fails under long horizon shift.
- Reducing the 5% route fraction alone is not enough: the most confident 0.1%
  is already negative externally.
- OOF expert disagreement alone is too weak and fold-unstable.
- The fixed equal OOF/temporal hybrid attenuates the temporal gain but does not
  beat it on short rolling folds.

## Next experiment implied by the evidence

Do not tune another route threshold. Rebuild the validation protocol with
**long-gap rolling origins matching the external 468-day horizon**, then make
temporal support explicitly query-age-aware:

- decay source-candidate support from `last_hit` to each query timestamp;
- zero or abstain when all support is older than the learned/frozen horizon;
- route only when short-window and long-gap folds agree;
- keep the same score-multiset and 5% hard bounds.

That experiment must be treated as a new protocol. The external result in this
run has already been observed and cannot be reused as a blind tuning target.

## Reproducibility and compliance

- Trainable frameworks: `["jittor"]`.
- Non-Jittor trainable models: `[]`.
- NumPy is used only for deterministic signals, score reassignment, metrics,
  and audits.
- No sklearn, LightGBM, learned ID embedding, or free score residual is used.
- Submission generated: `false`.

Machine-readable evidence:

- `artifacts/dataset2_disagreement_temporal_correction_20260727/evaluation-report.json`;
- `artifacts/dataset2_disagreement_temporal_correction_20260727/external-route-diagnostic.json`;
- `artifacts/dataset2_disagreement_temporal_correction_20260727/smoke-report.json`.
