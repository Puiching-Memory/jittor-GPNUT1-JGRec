# Goal Document: Dataset2 Cached-Feature Listwise MLP

## Go / No-Go

- **Judgment**: Go
- **Reason**: Dataset2 already has a verified 50,000-query training feature cache and a 20,000-query full-candidate validation cache. Replacing pointwise BCE with query-level softmax cross-entropy directly matches the ranking task and can be tested without regenerating tower features.

## Target Outcome

Determine whether a freshly trained listwise MLP improves the current champion's fixed `MLP 0.07 + LightGBM 0.93` Dataset2 ranking on every chronological validation interval. Build a candidate checkpoint and submission package only if the frozen offline gate passes.

## Goal Definition

- **Type**: technical, learning, quality, and delivery
- **Boundary**: Dataset2 only; existing supervised-feature cache; unchanged MLP architecture and selected feature columns; five fixed epochs; batch size 512; learning rate 0.001; seed 60; candidate zero as the positive class; unchanged champion Dataset2 LightGBM and fixed MLP weight 0.07; three chronological validation slices.
- **Non-goals**:
  - Change Dataset1, towers, candidate generation, feature definitions, LightGBM, or the submission schema.
  - Sweep epochs, learning rates, hidden sizes, feature subsets, losses, or blend weights after reading validation metrics.
  - Use A/B labels or leaderboard feedback for training or selection.
  - Early-stop or select an epoch on the reused validation set.
- **Deferred work**:
  - ListMLE, pairwise losses, LambdaLoss/NDCG weighting, or distillation.
  - Blend-weight calibration on a separate untouched time interval.
- **Verification rule**: Train exactly once from a seed-60 fresh initialization for five complete epochs with query-level softmax cross-entropy. Read validation labels only after training is complete. Compare the fixed 0.07 listwise-MLP blend with the exact champion blend over all 20,000 validation queries and three equal chronological slices.
- **Evidence source**: RED/GREEN loss tests; frozen training configuration written before evaluation; epoch loss trace; final pure-MLP metrics; fixed-blend full and per-slice MRR; checkpoint/package hashes if accepted.
- **Pass criteria**: The fixed blend must strictly improve all three chronological slices and improve full validation MRR by at least `+0.002`. Scores and weights must remain finite, Dataset1 output must remain byte-identical, and the checkpoint must load both datasets.
- **Confidence note**: The same validation set has informed previous experiments, so a small local gain is not strong evidence. The `+0.002` and all-slice rules are deliberate protection against another segment-gate style online regression.
- **Judgment owner**: Tests own loss correctness; the frozen temporal gate owns offline continuation; the user owns any leaderboard submission.

## Current State

- Online champion: `1.3426473547970703`.
- Champion Dataset2 fusion is approximately `MLP 0.07 + LightGBM 0.93`; LightGBM is already LambdaRank.
- Current MLP training minimizes independent candidate-level sigmoid BCE even though exactly one candidate per query is positive.
- Cached Dataset2 train tensor: `50,000 x 32 x 63` (one positive plus 31 negatives).
- Cached Dataset2 validation tensor: `20,000 x 100 x 63` (one positive plus 99 negatives).
- Positive-at-index-zero is a supervised-cache construction contract.
- Previous online experiments showed that reused-holdout improvements below a few thousandths are not reliably leaderboard-aligned.

## Priority Rationale

- Listwise loss is a high-ROI objective correction because it reuses the expensive tower-feature cache.
- A fixed training recipe isolates the causal change from BCE to listwise softmax cross-entropy.
- Keeping the champion LightGBM and 0.07 blend unchanged limits package risk and avoids fitting another degree of freedom to the reused holdout.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| Each cached query has exactly one positive at index zero | confirmed | Defines the listwise class target | Unit test and cache contract |
| Train and validation may have different candidate counts | confirmed | Loss must operate on arbitrary group width | Unit test with multiple widths |
| Fresh five-epoch training at 0.001 is fixed before metrics | confirmed design | Prevents validation-driven epoch selection | Frozen report |
| Champion feature indices and hidden dimension are reusable | assumed, checkpoint-verifiable | Keeps architecture unchanged | Server preflight |
| Fixed 0.07 weight transfers to listwise logits | assumed | Cleanest single-variable test but may underweight the new model | Report pure MLP and fixed blend; do not scan |

## Phases

### Phase 1: Listwise Loss Contract

- **Purpose**: Prove that the training objective is query-level softmax cross-entropy with candidate zero as the target.
- **Entry condition**: Goal document is complete.
- **Phase rules**:
  - Write a failing test before production code.
  - Exercise the actual Jittor loss helper used by training.
  - Cover row-wise shift invariance, numerical reference values, and arbitrary candidate widths.
- **Todos**:
  - [x] Add RED tests for the listwise positive-class loss.
    - **Surface**: `tests/test_hybrid_fusion_listwise.py`.
    - **Proof**: Import or behavior fails because the helper is absent.
    - **Depends on**: none.
  - [x] Implement the minimal loss helper and fixed-epoch trainer.
    - **Surface**: `src/jgrec/rankers/hybrid/fusion.py`.
    - **Proof**: Focused tests pass and the trainer has no validation-based selection path.
    - **Depends on**: RED test.
- **Exit proof**: Focused tests and Ruff pass.
- **Stop condition**: Stop if Jittor cannot stably compute the grouped softmax loss on cached shapes.

### Phase 2: Fixed Cached Training

- **Purpose**: Produce one listwise MLP without regenerating supervised features.
- **Entry condition**: Phase 1 is green; cache and champion checkpoint contracts match.
- **Phase rules**:
  - Seed 60, fresh initialization, five epochs, batch 512, learning rate 0.001.
  - Use the champion's feature indices and hidden dimension.
  - Write the configuration artifact before validation scoring.
- **Todos**:
  - [x] Train on all 50,000 cached Dataset2 groups.
    - **Surface**: server experiment script, model state, and loss trace.
    - **Proof**: Exactly five epochs complete with finite losses and a deterministic config hash.
    - **Depends on**: Phase 1.
  - [x] Score the pure listwise MLP after training.
    - **Surface**: experiment report.
    - **Proof**: Full and per-slice MRR/AP, with no epoch selection.
    - **Depends on**: trained state.
- **Exit proof**: One immutable candidate state and post-training metrics exist.
- **Stop condition**: Reject on non-finite loss, shape mismatch, or incomplete cache coverage.

### Phase 3: Frozen Temporal Gate

- **Purpose**: Decide whether the listwise MLP helps the actual champion blend robustly.
- **Entry condition**: Phase 2 completes without configuration changes.
- **Phase rules**:
  - Reuse the exact champion LightGBM scores.
  - Use MLP weight 0.07 only; do not run a weight scan.
  - Compare three chronological intervals plus all validation rows.
- **Todos**:
  - [x] Evaluate champion and listwise candidate with the same scorer.
    - **Surface**: frozen JSON report.
    - **Proof**: Baseline/candidate/delta for pure MLP and fixed blend by slice and full set.
    - **Depends on**: Phase 2.
  - [x] Apply the predeclared acceptance rule.
    - **Surface**: report decision.
    - **Proof**: Boolean gate and explicit failure reasons.
    - **Depends on**: metrics.
- **Exit proof**: Continue only if every slice improves and full fixed-blend delta is at least `+0.002`.
- **Stop condition**: No checkpoint or ZIP if the gate fails.

### Phase 4: Guarded Candidate Package

- **Purpose**: Replace only the champion's Dataset2 MLP state and make a traceable local ZIP.
- **Entry condition**: Phase 3 passes.
- **Phase rules**:
  - Derive from the `1.3426473547970703` champion checkpoint.
  - Preserve Dataset1 checkpoint state and CSV byte-for-byte.
  - Never overwrite champion artifacts.
- **Todos**:
  - [ ] Overlay and round-trip the Dataset2 MLP state.
    - **Surface**: new checkpoint.
    - **Proof**: Both datasets load and Dataset2 logits match the accepted report.
    - **Depends on**: Phase 3.
  - [ ] Infer, validate, hash, and copy the package locally.
    - **Surface**: new result directory and ZIP.
    - **Proof**: Expected CSV rows/members, byte-identical Dataset1 CSV, ZIP integrity, and matching remote/local SHA-256.
    - **Depends on**: checkpoint.
- **Exit proof**: One local, traceable ZIP is ready for the user to submit.
- **Stop condition**: Stop on any checkpoint, CSV, ZIP, or hash mismatch.

## Dry-Run Findings

- Unlike pointwise BCE, listwise softmax forces all candidates in a query to compete for one unit of probability mass and directly rewards raising the positive relative to its negatives.
- Training has 32 candidates while validation has 100; a query-level softmax supports both without architecture changes.
- The MLP contributes only 0.07 to the champion blend, so large pure-MLP changes may become small ensemble changes. This is a real falsifier, not a reason to tune the weight on the reused validation set.
- Validation-based early stopping would turn five epoch states into five hidden trials; the fixed-epoch recipe removes that leakage path.

## Final Validation

- Focused RED/GREEN tests and Ruff.
- Server cache/checkpoint preflight and frozen configuration.
- Exactly five training epochs with finite losses.
- Full and three-slice MRR against the exact champion.
- If accepted: checkpoint round-trip, CSV/ZIP validation, byte-identical Dataset1, and local/remote SHA-256 match.

## First Execution Step

Add a failing test that compares the Jittor listwise loss with a NumPy log-sum-exp reference and verifies invariance to adding a constant to all candidates in a query.

## Execution Result

- **Status**: Rejected at the frozen temporal gate; Phase 4 was intentionally not started.
- Linux RED failed with `ImportError` because `_listwise_positive_loss` did not exist. GREEN passed six focused tests, and the related checkpoint suite brought the server total to 15 passing tests.
- The fixed seed-60 model trained all five epochs in 28.86 seconds. Loss decreased monotonically from `2.07730` to `1.49207`; validation was evaluated only after epoch five.
- Pure listwise MLP full MRR improved from `0.52858632` to `0.53527022`, delta `+0.00668390`. Its three chronological slices all improved.
- The champion's fixed 0.07 blend improved only from `0.54283033` to `0.54303712`, delta `+0.00020679`.
- Fixed-blend slice deltas were `+0.00027884`, `-0.00018679`, and `+0.00052837`. The middle interval regressed and the full delta missed the predeclared `+0.002` threshold.
- No checkpoint overlay or submission ZIP was produced. The trained MLP and report were copied locally to `result/dataset2_listwise_mlp_seed60_20260723/`; the model SHA-256 matched the report.
- **Judgment**: Listwise training improves the MLP expert, but the current global 0.07 probability blend suppresses nearly all of the gain. The next hypothesis should be leakage-controlled fusion calibration, not more listwise epochs.
