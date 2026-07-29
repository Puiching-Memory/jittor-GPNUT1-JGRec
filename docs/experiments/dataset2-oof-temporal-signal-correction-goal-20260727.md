# Dataset2 OOF disagreement / strict temporal signal correction goal

## 1. Goal

Replace the confidence-routed top-k corrector's static ID correction signal with
two leakage-safe signals and one fixed consensus:

1. multi-model OOF disagreement;
2. strict origin-frozen source-to-candidate temporal support;
3. a fixed OOF/temporal consensus.

The experiment succeeds only if a pure-Jittor confidence router can use one of
these signals to sparsely correct the frozen Dataset2 base ranking without
changing the base score multiset, while passing rolling-origin selection,
unseen-time gating, and the final external-validation gate.

## 2. Why this experiment

The previous top-k ID corrector found a small opportunity on the first two
rolling folds, but the opportunity decayed to almost zero on the unseen third
fold. Static item identity therefore does not provide a stable correction
direction.

This experiment targets two more defensible sources of information:

- disagreement among independently trained OOF Jittor experts, which can expose
  candidates for which the current base model is uncertain;
- source-specific candidate support observed strictly before the scoring
  origin, which can expose repeat/recent behavior without looking into the
  scored period.

## 3. Verifiable target

Produce three fixed candidates:

- `oof-disagreement-top10-route05`;
- `strict-temporal-support-top10-route05`;
- `hybrid-consensus-top10-route05`.

For every candidate:

- the proposal may reorder only the frozen-base top 10 candidates;
- it must reuse the exact frozen-base score multiset for every row;
- the hard router may change at most 5% of rows;
- all trainable routing parameters must be implemented with `jt.nn.Module`;
- no item/source ID embedding, LightGBM, or sklearn may affect final ranking;
- the temporal signal may use only events with `event_time < origin_time`;
- OOF router features and labels may use only expert logits generated for the
  next, unseen time segment.

## 4. Current state

- Online champion score: `1.3545839690981516`.
- Frozen external champion validation MRR:
  `0.5478966505694405`.
- Existing OOF expert library:
  `cst_main`, `cst_residual`, and `setwise_mlp`.
- Existing rolling-origin fold manifest and frozen CST base logits are
  available locally and remotely.
- The previous confidence-routed ID experiment failed the frozen selection
  rule; its Fold2 routed corrections were rank-neutral.

## 5. Boundaries

### In scope

- deterministic NumPy feature/signal construction;
- pure-Jittor confidence-router training;
- rolling-origin OOF selection and final unseen-time gating;
- exact ranking and leakage audits;
- external validation only after the temporal gate passes;
- submission packaging only if the external gate passes.

### Out of scope

- retraining the expert library;
- learning free score residuals;
- changing Dataset1;
- tuning thresholds or signal weights on external validation;
- using candidate IDs as learned parameters;
- using future/equal-time events in temporal support;
- using sklearn or LightGBM in training or inference.

## 6. Fixed experimental protocol

### 6.1 OOF disagreement signal

For each expert, transform logits into stable within-row percentile ranks.
Build a consensus candidate score from the mean expert percentile with a fixed
disagreement penalty. Expose only label-free row descriptors such as:

- consensus advantage of the proposed top1 over the frozen-base top1;
- number/fraction of experts voting for the proposed top1;
- expert-rank variance;
- consensus top1 margin;
- whether the proposed top1 differs from the base top1.

### 6.2 Strict temporal-support signal

At each origin, build source-to-candidate support using only history events
whose timestamp is strictly less than the origin timestamp. Candidate support
combines fixed transforms of:

- source-candidate occurrence count;
- source-candidate last-hit recency;
- recent global candidate support from a fixed trailing history window.

No statistic may be updated by the router holdout or scored time segment.

### 6.3 Hybrid consensus

Convert both signals to within-row percentile scores and take a fixed equal
average. The blend weight is not tuned on external validation.

### 6.4 Proposal and routing

For each signal:

1. select the frozen-base top 10;
2. order those candidate positions by the signal;
3. assign the original top-10 scores, in descending order, to that signal
   order;
4. keep every top-10-external position exactly unchanged;
5. train a small pure-Jittor confidence router on the last 20k rows before the
   fold origin;
6. route only rows with probability at least `0.5`, capped at the highest
   probability 5% of rows.

The router target is whether the deterministic proposal improves reciprocal
rank over the frozen base.

## 7. Phase route

### Phase A: contracts and RED tests

- lock score-multiset and top-k scope invariants;
- lock candidate-permutation behavior;
- lock OOF consensus behavior;
- lock strict exclusion of equal/future temporal events.

Proof: focused tests fail because the new module does not yet exist.

### Phase B: deterministic signals and proposal

- implement OOF disagreement signal;
- implement vectorized strict temporal support;
- implement the fixed hybrid;
- implement exact score-multiset top-k proposal;
- make the Phase A tests pass.

Proof: focused unit tests pass locally and remotely.

### Phase C: rolling-origin runner

- load existing fold caches and OOF expert logits;
- build router holdout and scored-fold features without leakage;
- train the existing pure-Jittor confidence router;
- save metrics, route audits, time/activity slices, and checkpoints.

Proof: a small smoke run completes and all audit counters are zero.

### Phase D: frozen selection and unseen-time gate

Selection on Fold0 and Fold1:

- each fold delta must be non-negative;
- mean delta must be at least `+0.0001`.

Unseen-time gate on Fold2:

- Fold2 delta must be non-negative;
- three-fold mean delta must be at least `+0.0001`;
- worst activity-quartile delta must be at least `-0.0005`.

Proof: machine-readable report records every predicate before any external
validation is read.

### Phase E: external validation and submission decision

Only the temporal-gate winner may be evaluated externally. It must:

- beat the frozen external champion by at least `+0.0002`;
- have non-negative delta in every external time slice;
- pass all exact-scope and score-multiset audits.

Only then may a submission package be produced.

## 8. Per-phase rules

- Do not inspect external metrics before Phase D passes.
- Do not tune thresholds, top-k, blend weights, or router architecture after
  seeing Fold2 or external results.
- Treat a zero-opportunity router as a valid negative result, not as a reason
  to loosen the route cap.
- Save enough metadata to reproduce history boundaries and OOF row ranges.
- Abort on any NaN/Inf, row-count mismatch, origin mismatch, or invariant
  violation.

## 9. Todo with proof

- [ ] Add RED tests for proposal invariants.
  Proof: import/behavior failure recorded.
- [ ] Add RED tests for OOF disagreement behavior.
  Proof: import/behavior failure recorded.
- [ ] Add RED test proving equal/future events are excluded.
  Proof: test fixture changes only pre-origin support.
- [ ] Implement deterministic signal module.
  Proof: focused tests green.
- [ ] Implement experiment runner and launcher.
  Proof: CLI help and smoke run.
- [ ] Run three rolling-origin candidates.
  Proof: fold metrics and router checkpoints.
- [ ] Apply frozen selection and Fold2 gate.
  Proof: `evaluation-report.json`.
- [ ] Conditionally evaluate external validation.
  Proof: report states evaluated/skipped and why.
- [ ] Document the TDD trace and final result.
  Proof: experiment TDD and result documents.

## 10. Dry run

1. Fold0 router history ends before its router holdout origin.
2. Its OOF expert rows come from experts trained only on an earlier segment.
3. Temporal support filters history again with strict `< origin_time`.
4. A deterministic top-10 proposal is constructed with the base score
   multiset.
5. The Jittor router learns only whether that proposal helps on the temporal
   holdout.
6. The hard route changes no more than 5% of Fold0 score rows.
7. The same frozen procedure runs on Fold1.
8. Only a Fold0/Fold1-selected candidate is opened on unseen Fold2.
9. Only a Fold2-gated winner is opened on external validation.

## 11. Go / No-Go judgment

**Go** for the bounded experiment above.

The first falsifier is simple: if no signal passes Fold0/Fold1 selection, stop
without reading external validation. If a signal passes selection but loses on
Fold2, reject it and do not retune against Fold2. If it passes temporal gating
but misses the external margin or time-slice rule, keep the online champion and
produce no submission.
