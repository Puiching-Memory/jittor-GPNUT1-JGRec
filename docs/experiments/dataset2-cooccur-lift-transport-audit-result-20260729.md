# Dataset2 Cooccur-Lift Transport Audit Result

## Verdict

The surviving online mechanism is mainly `lift_full`; `lift_short` remains
active on some test rows but is materially attenuated. Audit A confirms a
time-support mismatch, Audit B does not support a marginal
candidate-popularity sampling mismatch, and Audit C shows that the auxiliary
head's realized first-layer lift energy changes from `63.82% full / 36.18%
short` on strict external rows to `76.54% full / 23.46% short` on test rows.
The external `+0.03567` is therefore not a credible online effect-size
estimate, chiefly because it never exercises the post-history short-window
state present online.

This is a read-only, zero-label diagnostic. It did not change a model, weight,
formula, selection lock, checkpoint, or package.

## Audit A: short-window support

The short window is `17,038,080` seconds (`197.2` days).

| Statistic | Strict external | Test |
|---|---:|---:|
| Exact-zero candidate cells | 74.1023% | 89.8562% |
| Rows with all 100 candidates exactly zero | 0.0000% | 39.9720% |
| Median per-row zero fraction | 74% | 90% |
| P75 per-row zero fraction | 78% | 100% |
| Constant rows | 0.0000% | 39.9720% |

The complete `train.csv` ends at time `1296259200`. Test queries start one day
later and span gaps of `1` to `349` days:

- `39.8312%` of test queries have `t - w` strictly after the complete
  `train.csv` end;
- this closely matches the `39.9720%` all-zero-row rate;
- no strict external row has `t - w` after the complete train end, because
  external rows are historical rows within that full interaction timeline.

The frozen auxiliary origin (`1255824000`) and complete `train.csv` end
(`1296259200`) are intentionally reported separately. Conflating them would
overstate the complete-history gap.

## Audit B1: auxiliary output transport

| Statistic | Strict external | Test | Test minus external |
|---|---:|---:|---:|
| Mean row maximum probability | 0.412887 | 0.438719 | +0.025832 |
| Median row maximum probability | 0.381723 | 0.405374 | +0.023651 |
| P90 row maximum probability | 0.657229 | 0.719508 | +0.062279 |
| Mean normalized entropy | 0.423230 | 0.408503 | -0.014727 |

- Row-maximum KS distance: `0.06435`.
- Normalized-entropy KS distance: `0.07943`.
- Standardized mean shifts are `+0.145` for row maximum and `-0.128` for
  entropy.

This is a moderate confidence shift, not a CST-scale catastrophic signature.
The external and test rows are not paired, so the CST paired maximum
probability difference of `0.283086` is not directly comparable.

## Audit B2: candidate popularity transport

Popularity is the full `train.csv` destination event count, applied under the
same definition to both candidate pools.

| Comparison | JS divergence (nats) | Total variation | Unseen-rate delta | Mean log1p-pop delta |
|---|---:|---:|---:|---:|
| Validation all → test all | 0.0000121 | 0.00230 | -0.00222 | +0.01091 |
| Validation negatives → test all | 0.0002597 | 0.00939 | -0.00766 | +0.04848 |

The marginal popularity distributions are extremely close. Test candidates are
slightly more, not less, popular under this reference. This audit therefore
does not support the proposed “local negatives are much more popular than
test candidates” failure mechanism. It cannot rule out higher-order or
source-conditional sampler differences.

## Top-1 movement

- External baseline → integrated candidate: `4,377 / 19,981 = 21.9058%`.
- Online champion package → cooccur-lift package:
  `25,731 / 153,420 = 16.7716%`.
- The online package is below the requested `20%` high-risk threshold.

The lower online movement than external movement is consistent with transport
attenuation; it is not evidence of a candidate-popularity distribution shift.

## Audit C: full-versus-short mechanistic attribution

This counterfactual sets each lift signal and its three setwise context
channels (raw, row-centered, and row-max-difference) to zero, then measures the
exact change in the trained auxiliary head's first-layer pre-activation. It is
a zero-label mechanism diagnostic, not a decomposition of hidden leaderboard
points; later ReLU layers and relevance labels are not evaluated.

| Statistic | Strict external | Test |
|---|---:|---:|
| Full share of separate first-layer energy | 63.8221% | 76.5385% |
| Short share of separate first-layer energy | 36.1779% | 23.4615% |
| Mean full intervention row RMS | 0.698249 | 0.661601 |
| Mean short intervention row RMS | 0.526538 | 0.277622 |
| Exactly-zero short intervention rows | 0.0000% | 39.9720% |
| Full/short first-layer energy ratio | 1.764 | 3.262 |

- The short energy share falls by `12.72` percentage points, or `35.15%`
  relative, from external to test.
- Mean short intervention RMS falls by `47.27%`; mean full intervention RMS
  falls by only `5.25%`.
- The full/short intervention cosine is small on both pools (`0.0963`
  external, `0.0914` test), so the share change is not an artifact of two
  nearly collinear interventions.

The defensible attribution is therefore: the online `+0.0018295` is mainly
carried by `lift_full`, with a minority contribution from short-window
residuals on the roughly `60%` of test rows that have not fully collapsed.
The exact split of the leaderboard gain is not identifiable without test
labels or a separately submitted full-only counterfactual, so this audit does
not assign point values to the two signals.

The external-to-online effect ratio is `19.50x`
(`0.0356708 / 0.0018295`). The evidence explains its direction:

1. strict external has no all-zero short rows, while test has `39.97%`;
2. the realized short-channel RMS nearly halves and its energy share drops
   from `36.18%` to `23.46%`;
3. top-1 movement falls by `23.44%` relative (`21.91%` to `16.77%`);
4. marginal candidate popularity is almost invariant, so it is not the
   observed cause of the attenuation.

These facts do not numerically account for every factor in `19.50x`.
External MRR delta and the leaderboard score delta need not share a metric
scale, later nonlinearities can suppress or amplify either intervention, and
hidden relevance conditional on changed rows is unavailable. Higher-order or
source-conditional candidate-pool differences also remain possible even
though the marginal popularity explanation is rejected.

## Decision implication

The two risks do not combine:

1. **Time-span mismatch is real and material.** Downgrade confidence in the
   magnitude of the external gain; external rows never exercise the
   post-history all-zero state seen in roughly 40% of test rows.
2. **Marginal candidate-popularity mismatch is not observed.** Do not attribute
   attenuation to the organizer candidate sampler on this evidence.
3. **No protocol action is authorized by this diagnostic.** It does not select
   a new weight, formula, model, or threshold.

Canonical machine-readable report:
`result/dataset2_cooccur_lift_transport_audit_20260729/audit-report-v3.json`,
SHA-256
`2cbb3231d911245c494dac8237df8655e6bea508a95555077a802d273e8979aa`.
