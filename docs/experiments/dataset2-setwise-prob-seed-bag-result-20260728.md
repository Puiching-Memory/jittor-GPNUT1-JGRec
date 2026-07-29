# Dataset2 Setwise Probability Seed Bag v1 — Result

## Verdict

`setwise_prob_seed_bag_v1` is **rejected and closed**.

All six frozen weights failed the existing three-fold hard gate. No
selection lock was created, the external holdout was not opened, and no
package was generated. The experiment must not be continued by replacing
salts, changing epochs, or scanning adjacent weights.

## Frozen execution

- Weights: `0.05, 0.10, 0.20, 0.30, 0.40, 0.50`
- Seed salts: `10007, 20011`
- Derived fold seeds:
  - fold-0: `10067, 20071`
  - fold-1: `11076, 21080`
  - fold-2: `12085, 22089`
- Fold boundaries:
  - train `[0, 79909)`, score `[79909, 118816)`
  - train `[0, 118816)`, score `[118816, 159804)`
  - train `[0, 159804)`, score `[159804, 200000)`
- Training: four epochs per head, six heads total
- Auxiliary: arithmetic mean of the two new-seed probability matrices
- Candidate: `(1 - w) * fold_champion + w * auxiliary`
- Producer elapsed time: `247.58s`

The producer wrote six model artifacts and six per-seed probability
matrices. It did not compute a ranking decision panel; the existing robust
selector did that once after all three folds were complete.

## Rolling result

| Weight | Pooled MRR delta | Fold MRR deltas | Worst fold | Decision |
|---:|---:|---|---:|---|
| 0.05 | +0.000093797 | -0.000009918 / +0.000198833 / +0.000087079 | -0.000009918 | reject |
| 0.10 | +0.000006088 | -0.000112822 / +0.000027968 / +0.000098873 | -0.000112822 | reject |
| 0.20 | -0.000008834 | -0.000266800 / +0.000064226 / +0.000166362 | -0.000266800 | reject |
| 0.30 | +0.000046324 | -0.000267493 / +0.000181384 / +0.000212356 | -0.000267493 | reject |
| 0.40 | -0.000032522 | -0.000306509 / +0.000210275 / -0.000014902 | -0.000306509 | reject |
| 0.50 | -0.000296047 | -0.000433363 / -0.000067948 / -0.000395729 | -0.000433363 | reject |

The smallest weight showed the hypothesized `+0.000x` pooled effect, but
fold-0 MRR decreased. The hard gate intentionally treats that instability
as disqualifying. At higher weights the fold-0 loss grows, and additional
NDCG/Hit/mean-rank gates fail.

## Isolation audit

- Pipeline stage: `rolling_rejected_closed_no_external`
- Pipeline exit: `2` (selector rejection)
- Selection status: `rejected`
- Selected weight: `null`
- `external_holdout_read`: `false`
- Selection lock count: `0`
- External receipt count: `0`
- Package ZIP count: `0`

## Evidence

Remote root:

```text
/home/edu/workspace/jittor-GPNUT1-JGRec/result/
dataset2_setwise_prob_seed_bag_v1_20260728/
```

SHA-256:

```text
989ddbb050c42538365401d5458701ed93404e03c5e167d2b7402c9be5d7ce7b  frozen config
2a22a356701fcad7e742f9376a5cac9b89f6f4d359fac28e500cc5206fb10610  run contract
8f8d85fe78f7ed09627f8f05243ea450de0fcea5557698939da41165a002c6aa  rolling manifest
0bc5e119dd6bf01178a836c1521afdccbe8169e46d1b20d0b11cea7de5649c8a  selection report
c5646a52704ea6563dc4051f039d2c3fb998f7ebd4361c078204d56153832731  training report
```

## Closed-family rule

This run answers the probability-averaging question: it produces a very
small pooled gain at low weight, but not a cross-fold-stable gain. Per the
precommitment, this candidate family is closed. Any future work must start
as a materially new hypothesis on new folds, not as a rescan of this run.

