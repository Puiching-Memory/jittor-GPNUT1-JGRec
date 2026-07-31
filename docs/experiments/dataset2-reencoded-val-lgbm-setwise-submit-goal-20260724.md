# Goal Document: Dataset2 Re-Encoded Validation, Dual Rerankers, Conditional Submission

## Go / No-Go

- **Judgment**: Go
- **Reason**: The shared recent-200k/full-100 training cache is complete, but its
  encoder does not match the historical champion validation cache. Rebuilding
  validation with a reproducibly identical encoder is the prerequisite for a
  trustworthy LightGBM versus Setwise comparison.

## Target Outcome

Reproduce the encoder pipeline that generated the recent 200,000-row Dataset2
training cache, then use the matching `train_end` temporal snapshot to build a
leakage-free full-100 validation cache, train LightGBM and Setwise rerankers on
the shared training cache,
compare both against the champion on three chronological validation slices,
and generate one submission package only if a candidate passes the frozen
gate.

## Goal Definition

- **Type**: technical / learning / delivery
- **Boundary**:
  - Dataset2 reranking only; Dataset1 remains the current champion.
  - Training cache:
    `dataset2_recent200k_full100_seed60_20260724`.
  - Full 100-candidate training and validation groups.
  - One LightGBM candidate and one Setwise neural reranker candidate.
- **Non-goals**:
  - No leaderboard-driven weight tuning.
  - No new graph tower or Two-Tower architecture change.
  - No Dataset1 retraining.
  - No submission package when the offline gate fails.
- **Deferred work**:
  - Broader hyperparameter sweeps.
  - Ensembling LightGBM and Setwise unless one first proves standalone value.
- **Verification rule**:
  - Before validation generation, replay the saved candidates and features for
    a bounded prefix of the 200k cache with the reconstructed `context_end`
    encoder.
  - Build validation features with the same configuration and recorded RNG
    continuation, but with the standard `train_end` temporal snapshot so every
    validation query sees all permitted past interactions.
  - Validation queries come only from the chronological validation partition;
    model selection never uses test labels or leaderboard feedback.
  - Champion and both candidates score the exact same validation tensor.
- **Evidence source**:
  RED/GREEN tests, bounded replay report, validation cache manifest and hashes,
  full and three-slice MRR report, model artifacts, and conditional package
  manifest.
- **Pass criteria**:
  - Encoder replay has zero candidate-ID mismatches and feature values match
    within `rtol=2e-5`, `atol=2e-6`.
  - Candidate full MRR improves by at least `+0.002` over the champion on the
    rebuilt validation cache.
  - All three chronological slice MRRs are no lower than the champion.
  - A package is generated only for the best candidate satisfying both metric
    conditions.
- **Confidence note**:
  The rebuilt validation remains an offline proxy, but same-encoder and
  same-candidate evaluation removes the known train/validation feature
  mismatch that invalidated the historical comparison.
- **Judgment owner**:
  Automated replay and MRR gates decide whether packaging is allowed.

## Current State

- The recent cache is complete at `200,000 × 100 × 63`.
- Candidate, src, dst, time, and sorted-row-index sidecars pass the cache
  contract and have SHA-256 hashes.
- Selected interaction rows are `[1,722,091, 1,922,091)`.
- The historical champion validation cache has shape `20,000 × 100 × 63`, but
  its bounded encoder replay was rejected and cannot be used as the matched
  validation tensor for this experiment.
- The encoder object used by the 200k build was not persisted, so deterministic
  reconstruction must be proven against the completed training cache before
  proceeding.

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| Rebuild Dataset2 validation features | rewrite | First prove the reconstructed encoder matches the completed training cache. |
| Train LightGBM and Setwise | keep | They consume the same cache and test complementary ranking objectives. |
| Compare three time periods | keep | Prevents a gain concentrated in one period. |
| Generate a package after training | rewrite | Packaging is conditional on the frozen `+0.002` and no-slice-regression gate. |

## Drift Diagnosis

- **Goal drift**: Additional towers do not resolve the current encoder mismatch.
- **Phase drift**: Training before replay proof would produce untrustworthy MRR.
- **Validation drift**: Comparing against the historical validation tensor
  preserves the known feature-distribution mismatch.
- **Compatibility drift**: Both rerankers must consume the exact same candidate
  rows, schema, and validation tensor.
- **Cleanup drift**: Existing checkpoints, packages, and caches remain
  untouched.

## Priority Rationale

- Encoder replay is the cheapest decisive test and can stop an invalid
  experiment before two model fits.
- A single validated cache pair makes the LightGBM/Setwise comparison fair and
  reusable.
- The offline gate remains frozen before any model score is observed.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| Seed-60 reconstruction reproduces the recent cache encoder pipeline | unresolved | Blocks all trustworthy evaluation | Prove with bounded feature replay |
| Validation negative sampling can use an independent frozen RNG | assumed | Avoids coupling selection to training RNG consumption | Record seed and candidate hashes |
| `+0.002` and no slice regression is the submission gate | frozen | Controls package generation | Automated metric gate |
| Dataset1 remains from the champion checkpoint | confirmed | Limits blast radius | Compose only Dataset2 state on pass |

## Phases

### Phase 1: Same-Encoder Replay and Validation Cache

- **Purpose**: Establish a matched train/validation feature space.
- **Entry condition**: Complete 200k cache and build report are available.
- **Phase rules**:
  - Do not train a reranker until bounded replay passes.
  - Validation events must be chronological post-training events.
  - Store candidate and query sidecars plus hashes.
- **Todos**:
  - [ ] Add deterministic replay/validation cache contracts through RED/GREEN.
    - **Surface**: full-100 utilities, validation builder, tests.
    - **Proof**: tests reject mismatched candidates, features, and temporal rows.
    - **Depends on**: none.
  - [ ] Reconstruct the encoder and replay a bounded prefix.
    - **Surface**: server report.
    - **Proof**: candidate equality and close feature report.
    - **Depends on**: cache contracts.
  - [ ] Build the full-100 Dataset2 validation cache.
    - **Surface**: memmaps, sidecars, manifest.
    - **Proof**: `20,000 × 100 × 63`, finite scan, hashes, mmap reload.
    - **Depends on**: replay pass.
- **Exit proof**: Same-encoder replay and validation cache manifests both pass.
- **Stop condition**: Reconstructed training-prefix features do not match the
  completed cache.

### Phase 2: Matched LightGBM and Setwise Training

- **Purpose**: Train two reranking objectives on identical supervised groups.
- **Entry condition**: Phase 1 passes.
- **Phase rules**:
  - Shared train and validation candidate tensors.
  - Early stopping and checkpoint selection use full-candidate validation MRR.
  - No parameter search after viewing the final slice.
- **Todos**:
  - [ ] Train the frozen LightGBM configuration.
    - **Surface**: model text and training report.
    - **Proof**: deterministic model artifact and validation predictions.
    - **Depends on**: Phase 1.
  - [ ] Train the frozen Setwise reranker configuration.
    - **Surface**: model state and training report.
    - **Proof**: best-epoch artifact selected by full-candidate validation MRR.
    - **Depends on**: Phase 1.
- **Exit proof**: Both model artifacts score the exact same validation cache.
- **Stop condition**: Non-finite loss/scores, candidate misalignment, or memory
  exceeds server headroom.

### Phase 3: Three-Slice Gate and Conditional Package

- **Purpose**: Decide whether any candidate deserves a submission.
- **Entry condition**: Both reranker reports are complete.
- **Phase rules**:
  - Champion baseline is rescored on the rebuilt validation cache.
  - The package contains current champion Dataset1 unchanged.
  - No package when no candidate passes.
- **Todos**:
  - [ ] Compute champion and candidate full/three-slice MRR.
    - **Surface**: frozen evaluation report.
    - **Proof**: exact deltas and gate booleans.
    - **Depends on**: Phase 2.
  - [ ] Generate one package for the best passing candidate.
    - **Surface**: checkpoint, `result.zip`, hashes.
    - **Proof**: reload and submission schema validation.
    - **Depends on**: metric gate pass.
- **Exit proof**: Either a validated package exists or a rejection report
  proves why none was generated.
- **Stop condition**: Any gate ambiguity or inability to reproduce inference
  features.

## Dry-Run Findings

- The completed 200k cache cannot by itself prove effect; it has no matched
  validation tensor.
- Reusing the `context_end` encoder object directly for validation would omit
  the supervised-training time window; the standard `train_end` snapshot is
  required to avoid that temporal-context error.
- Reusing the old champion validation cache would repeat the known replay
  mismatch and make the MRR comparison unreliable.
- Packaging requires a reproducible inference encoder, not just a trained
  reranker; this must be captured in the candidate checkpoint.
- LightGBM and Setwise can share feature/candidate memmaps, but neural training
  must stream batches rather than materialize the 5 GB tensor in RAM.

## Final Validation

- Same-encoder bounded replay passes.
- Validation cache shape, sidecars, finite values, hashes, and reload pass.
- Champion, LightGBM, and Setwise full/three-slice MRR are recorded.
- Best candidate satisfies full delta `>= +0.002` and all slice deltas `>= 0`.
- Conditional checkpoint reloads and `result.zip` passes submission schema
  validation.

## First Execution Step

Write a failing test for exact candidate/feature replay and chronological
validation-row identity before implementing the validation cache builder.

## Execution Update — 2026-07-24

- Replay/validation builder PID: `551772`.
- Reranker training supervisor PID: `553416`; waits for a passing validation
  cache before starting.
- Conditional package supervisor PID: `554556`; waits for training and refuses
  packaging unless the evaluation report authorizes a winner.
- Setwise implementation:
  lazy raw/relative-mean/relative-max context channels, group-softmax loss,
  full-100 MRR early stopping, and checkpoint round-trip inference.
- Frozen model settings:
  - LightGBM: one `lr=0.03` LambdaRank configuration, at most 800 rounds,
    patience 60.
  - Setwise: hidden dimension 32, at most 10 epochs, patience 2, batch size
    256, learning rate `0.001`.
- Frozen gate remains full MRR `>= champion + 0.002` and no regression in any
  of three chronological slices.
