# Goal Document: Dataset2 OOF Hard-Negative LambdaRank

## Go / No-Go

- **Judgment**: Go
- **Reason**: Dataset2 already has a 50,000-query training feature cache and a 20,000-query 100-candidate validation cache. The existing expert is already LambdaRank, so OOF negative selection can be isolated and tested cheaply before rebuilding a strict train-only 100-candidate cache.

## Target Outcome

Determine whether selecting high-scoring false candidates with out-of-fold Dataset2 miners materially improves the current champion's fixed `MLP 0.07 + LightGBM 0.93` ranking across later time intervals, without using A/B labels. Build a B-safe candidate only after both a cheap mechanism test and a strict train-only candidate-pool test pass.

## Goal Definition

- **Type**: technical, learning, quality, and delivery
- **Boundary**: Dataset2 only; cached supervised features; three disjoint OOF miners; fixed top-16 negative retention; existing `learning_rate=0.03` LambdaRank configuration and 308 boosting rounds; fixed MLP weight `0.07`; three chronological validation slices; guarded checkpoint/package construction.
- **Non-goals**:
  - Read or infer A/B positive labels.
  - Tune from leaderboard feedback.
  - Change Dataset1, MLP, towers, feature definitions, or candidate generation in the mechanism test.
  - Sweep negative counts, model parameters, or blend weights after validation is read.
- **Deferred work**:
  - Strict `test_candidate_negative_ratio=0` and 99-negative Dataset2 feature-cache rebuild, unless the cheap mechanism gate passes.
  - New Dataset2 temporal or structural features.
- **Verification rule**: Each training query is scored only by a miner whose fitting rows exclude that query's contiguous OOF fold. Keep the positive plus the 16 highest-scoring negatives. Train one fixed LambdaRank model for 308 rounds. Compare its fixed 0.07 MLP blend with the current champion on three chronological validation slices.
- **Evidence source**: RED/GREEN unit tests; OOF coverage/disjointness report; fixed configuration artifact written before validation scoring; per-slice and full-candidate MRR; candidate identities/counts; model/checkpoint/CSV/ZIP hashes when applicable.
- **Pass criteria**: All three chronological validation slices strictly improve over the champion, mean full validation improvement is at least `+0.002 MRR`, no query is mined in-sample, and no non-finite score or duplicate candidate is introduced. Phase 2 additionally requires the strict train-only 99-negative cache to pass the same gate.
- **Confidence note**: Phase 1 reuses the champion cache, whose negatives were sampled from the public unlabeled test-candidate distribution; it proves the mining mechanism but is not the final no-test-distribution model. Phase 2 is required for the strongest B-safety claim.
- **Judgment owner**: Tests own implementation correctness; the fixed temporal gate owns offline continuation; the user owns the single final leaderboard submission.

## Current State

- Online champion: `1.3426473547970703`.
- Current Dataset2 LightGBM already uses `objective=lambdarank`, full-candidate MRR early stopping, `learning_rate=0.03`, best iteration 308, and MLP weight `0.07`.
- Cached Dataset2 train tensor: `50,000 x 32 x 63` (one positive plus 31 negatives).
- Cached Dataset2 validation tensor: `20,000 x 100 x 63`.
- Positive candidate index zero is a supervised-cache construction contract, not a test-file assumption.
- Segment-aware fusion improved the reused holdout but regressed online, so validation selection must be narrower and the acceptance margin larger.

## Priority Rationale

- Prove hard-negative concentration on the existing cache before paying for expensive feature regeneration.
- Keep model parameters and blend weight fixed so the only causal variable is OOF negative selection.
- Require strict train-only candidate generation before treating a passing mechanism as B-safe.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| Current LightGBM is already LambdaRank | confirmed | Avoids a fake objective-change experiment | Existing implementation and checkpoint report |
| Three contiguous OOF folds cover all 50,000 rows exactly once | confirmed design | Prevents in-sample mining | Unit-test fold contract |
| Top-16 is fixed before real metrics | confirmed design | Prevents negative-count search overfit | Record in frozen config |
| Existing cache uses public unlabeled test-candidate sampling | confirmed | Phase 1 is mechanism-only | Phase 2 sets ratio to zero |
| Fixed 308 rounds and 0.07 blend transfer to mined data | assumed | Keeps validation untouched by tuning | Reject if temporal gate fails |

## Phases

### Phase 1: OOF Mining Contract

- **Purpose**: Prove exact fold exclusion, stable hard-negative ordering, candidate-zero preservation, and fixed group shapes.
- **Entry condition**: Goal document is complete.
- **Phase rules**:
  - Write RED tests before production code.
  - No test-file reader or leaderboard input may appear in the mining API.
  - Ties are resolved by original candidate position.
- **Todos**:
  - [x] Test and implement three contiguous OOF folds.
    - **Surface**: hard-negative module and tests.
    - **Proof**: Held folds are disjoint, cover every row once, and never overlap their fitting indices.
    - **Depends on**: none.
  - [x] Test and implement positive-preserving top-K negative selection.
    - **Surface**: hard-negative module and tests.
    - **Proof**: Candidate zero remains first; exactly K unique negative positions follow in stable score order.
    - **Depends on**: fold contract.
- **Exit proof**: Focused tests and lint pass.
- **Stop condition**: Stop if the cache does not preserve query groups or positive-at-zero construction.

### Phase 2: Cheap Mechanism Gate

- **Purpose**: Test whether OOF-mined negatives improve the existing cached problem.
- **Entry condition**: Phase 1 is green; checkpoint/cache contracts match.
- **Phase rules**:
  - Use exactly three contiguous folds, top-16, 308 rounds, and MLP weight 0.07.
  - Write the configuration artifact before validation predictions.
  - Report the current champion and a same-cache unmined reference.
- **Todos**:
  - [x] Train three miners and materialize the OOF-mined training tensor.
    - **Surface**: server experiment script and report.
    - **Proof**: Per-fold train/held ranges, row counts, selected-score quantiles, and checksum.
    - **Depends on**: Phase 1.
  - [x] Train the fixed final LambdaRank and score three validation intervals.
    - **Surface**: frozen report and model artifact.
    - **Proof**: Per-slice/full MRR deltas versus champion with no post-result grid expansion.
    - **Depends on**: OOF tensor.
- **Exit proof**: All slices improve and full delta is at least `+0.002`, or the direction is rejected.
- **Stop condition**: Do not rebuild features if this gate fails.

### Phase 3: Strict Train-Only Candidate Pool

- **Purpose**: Remove reliance on test-candidate sampling before a B candidate exists.
- **Entry condition**: Phase 2 passes.
- **Phase rules**:
  - Rebuild Dataset2 training features with 99 negatives and `test_candidate_negative_ratio=0`.
  - Keep the Phase 2 mining/model configuration frozen.
  - Validation remains chronological and no leaderboard feedback enters selection.
- **Todos**:
  - [ ] Build and verify the strict 100-candidate training cache.
    - **Surface**: server cache manifest and logs.
    - **Proof**: Shape, ratio-zero configuration, train-only candidate audit, and cache hash.
    - **Depends on**: Phase 2 pass.
  - [ ] Repeat the fixed OOF experiment and temporal gate.
    - **Surface**: strict report/model.
    - **Proof**: Same acceptance rule as Phase 2.
    - **Depends on**: strict cache.
- **Exit proof**: Strict experiment passes without configuration changes.
- **Stop condition**: Reject on any slice regression or mean delta below `+0.002`.

### Phase 4: Guarded Candidate

- **Purpose**: Package only a validated Dataset2 replacement while retaining champion Dataset1.
- **Entry condition**: Phase 3 passes.
- **Phase rules**:
  - Derive from the `1.3426473547970703` champion checkpoint.
  - Dataset1 CSV remains byte-identical to the champion.
  - Never overwrite champion artifacts.
- **Todos**:
  - [ ] Patch and round-trip the Dataset2 LightGBM state.
    - **Surface**: contest checkpoint.
    - **Proof**: Both datasets load and the new model identity matches the strict report.
    - **Depends on**: Phase 3.
  - [ ] Infer Dataset2, validate ZIP, hash, and download.
    - **Surface**: server/local result.
    - **Proof**: Expected rows, two CSV members, matching hashes.
    - **Depends on**: checkpoint.
- **Exit proof**: One local traceable ZIP is ready for the user's single submission.
- **Stop condition**: No package if either offline gate fails.

## Dry-Run Findings

- Merely changing to LambdaRank would do nothing because the champion already uses it.
- The current 31-negative cache is enough to test whether OOF concentration helps, but not enough to claim strict train-only/full-candidate training.
- Rebuilding 99-negative Dataset2 features is expensive because previous Dataset2 encoding consumed hours; it belongs after the cheap gate.
- Fixed rounds and blend weight deliberately trade some possible local gain for a cleaner causal test and lower B overfit risk.

## Final Validation

- Focused RED/GREEN tests and Ruff.
- Server OOF audit and frozen configuration.
- Three chronological MRR deltas against the exact champion.
- If continued: ratio-zero cache audit, checkpoint round-trip, CSV validation, ZIP integrity, and local/remote SHA-256 match.

## First Execution Step

Add a failing test that requires three contiguous OOF folds to cover every training row exactly once while excluding held rows from each miner's fitting indices.

## Execution Result

- **Status**: Rejected at the cheap mechanism gate; Phases 3 and 4 were intentionally not started.
- Three contiguous OOF miners covered all 50,000 training rows exactly once (`coverage_min=coverage_max=1`) and retained the positive plus 16 highest-scoring negatives per query.
- Fixed candidate configuration: Dataset2 `learning_rate=0.03` LambdaRank, 308 rounds, MLP weight `0.07`; no validation early stopping, parameter search, negative-count sweep, or blend scan.
- Champion validation blend MRR: `0.5428303297`.
- OOF hard-negative blend MRR: `0.5419859603`, delta `-0.0008443695`.
- Chronological slice deltas: `-0.0016931175`, `-0.0007442052`, and `-0.0000956734`; all three intervals regressed.
- Because the predeclared gate failed decisively, no strict 99-negative ratio-zero cache rebuild, checkpoint overlay, submission inference, or leaderboard submission was performed.
- Local report: `result/dataset2_oof_hardneg_top16_seed60_20260723/oof-hard-negative-report.json`.
- **Judgment**: The existing 31-negative pool already contains sufficiently hard examples; discarding half of it loses useful coverage and does not justify a more expensive full-candidate rebuild under this hypothesis.
