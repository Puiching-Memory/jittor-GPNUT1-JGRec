# Goal Document: Dataset2 Setwise three-seed rank ensemble

## Go / No-Go

- **Judgment**: Go
- **Reason**: The full-100 feature cache and seed-60 champion already exist, so
  seeds 17 and 41 can be trained without rebuilding features. The experiment
  has a fixed blend and an explicit rejection gate.

## Target Outcome

Train otherwise-identical Dataset2 Setwise models with seeds 17, 41, and 60,
combine their per-query candidate ranks with a uniform average, and determine
whether this fixed ensemble improves the current `1.3530197201` champion
robustly enough to package.

## Goal Definition

- **Type**: learning and delivery
- **Boundary**: Dataset2 only; identical full-100 training/validation features,
  rows, model hyperparameters, and early-stopping rule; seeds 17/41/60; uniform
  rank average.
- **Non-goals**:
  - Searching seed weights or segment-specific weights.
  - Changing feature definitions, candidates, Setwise architecture, or GNN.
  - Rebuilding the full-100 feature cache.
- **Deferred work**:
  - Dataset1 optimization.
  - Additional seeds or model families.
- **Verification rule**: Compare the ensemble and champion on the same full
  validation set and the same three chronological slices.
- **Evidence source**: frozen evaluation report plus package manifest when
  accepted.
- **Pass criteria**: full validation MRR delta at least `+0.001`, with no
  decline on any of the three slices.
- **Confidence note**: This is a paired validation comparison and protects
  temporal consistency, but remains a leaderboard proxy.
- **Judgment owner**: the fixed metric gate in the experiment runner.

## Current State

- The online champion package scores `1.3530197200911278`.
- Its Dataset2 Setwise model uses seed 60 and full-100 validation.
- Cached training and validation features are available on the server.
- Recent leakage-free `gnn_short` OOF work did not beat the champion, so graph
  changes are excluded from this experiment.

## Priority Rationale

- Reuse the strongest proven model family and existing cache.
- Fix aggregation before training so validation cannot drive blend weights.
- Train only missing seeds 17 and 41 if the seed-60 artifact passes contract
  checks.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| Seed-60 champion artifact matches the frozen feature schema and hyperparameters | assumed | Required for a fair three-seed ensemble | Verify hashes/config before reuse |
| Candidate order is identical for all predictions | confirmed by cache contract | Required for rank averaging | Assert shapes and identity sidecars |
| Uniform mean rank is fixed before validation | confirmed | Prevents weight-search overfit | Encode as a tested helper |

## Phases

### Phase 1: Freeze rank-ensemble behavior

- **Purpose**: Make the aggregation and gate mechanically testable.
- **Entry condition**: Existing Setwise prediction and validation utilities are
  available.
- **Phase rules**:
  - Write a failing behavior test first.
  - Rank direction and tie behavior must be explicit.
- **Todos**:
  - [ ] Test uniform per-query rank averaging across three score matrices.
    - **Surface**: tests and Setwise/ranking utility
    - **Proof**: focused RED then GREEN test
    - **Depends on**: none
  - [ ] Test the fixed full/slice acceptance gate.
    - **Surface**: tests and evaluation utility
    - **Proof**: candidate below `+0.001` or with any declining slice is rejected
    - **Depends on**: aggregation contract
- **Exit proof**: focused tests and Ruff pass.
- **Stop condition**: Existing prediction artifacts cannot be aligned by the
  same candidate identities.

### Phase 2: Train and evaluate

- **Purpose**: Produce three aligned prediction matrices and a fixed ensemble.
- **Entry condition**: Phase 1 is green and server cache contracts pass.
- **Phase rules**:
  - All non-seed settings remain identical.
  - No validation-driven weight or segment search.
  - Reuse seed 60 only after contract verification; otherwise retrain it.
- **Todos**:
  - [ ] Train seeds 17 and 41 and obtain seed-60 predictions.
    - **Surface**: server artifacts
    - **Proof**: three model manifests and prediction hashes
    - **Depends on**: cache preflight
  - [ ] Compute uniform rank-average validation predictions.
    - **Surface**: evaluation report
    - **Proof**: full and three-slice MRR
    - **Depends on**: three aligned predictions
- **Exit proof**: frozen report includes per-seed, ensemble, champion, deltas,
  and gate decision.
- **Stop condition**: Any training run changes data, features, hyperparameters,
  or validation identities.

### Phase 3: Conditional packaging

- **Purpose**: Create a submission only for a robustly accepted ensemble.
- **Entry condition**: full delta is at least `+0.001` and every slice is
  non-decreasing.
- **Phase rules**:
  - Dataset1 remains byte-identical to the online champion.
  - Package exactly the fixed three-seed rank ensemble.
- **Todos**:
  - [ ] Integrate Dataset2 test predictions and build `result.zip`.
    - **Surface**: checkpoint/package
    - **Proof**: manifest, hashes, and ZIP contents
    - **Depends on**: metric gate pass
- **Exit proof**: verified local package and report, or explicit rejected status
  with no package.
- **Stop condition**: Metric gate fails.

## Dry-Run Findings

- Only two new training runs should be necessary if seed 60 is contract-compatible.
- Rank averaging requires scores to be converted to a consistent
  higher-is-better rank representation before averaging.
- Packaging must remain a separate conditional phase to avoid producing a
  tempting but rejected submission.

## Final Validation

Run focused tests and Ruff, train/evaluate on the server, then require:

```text
ensemble_full_mrr - champion_full_mrr >= 0.001
ensemble_slice_i_mrr >= champion_slice_i_mrr  for i in 0,1,2
```

## First Execution Step

Add a failing test for deterministic uniform rank averaging of three aligned
per-query score matrices.
