# Goal Document: Dataset2 Joint Cache Build and Dual Rerankers

## Go / No-Go

- **Judgment**: Go
- **Reason**: The previous replay proved that candidates are reproducible but a
  separately reconstructed encoder is not. Rebuilding training and validation
  caches inside one process removes that invalid boundary.

## Target Outcome

In one server process, create the recent-200k/full-100 Dataset2 training cache
and the chronological 20k/full-100 validation cache from one live ranker and
encoder-state cache, then train LightGBM and Setwise on that cache pair and
apply the frozen three-period submission gate.

## Goal Definition

- **Type**: technical / operational / delivery
- **Boundary**: Dataset2 only; 200,000 recent training groups; 20,000
  chronological validation groups; 100 candidates; 63 features; LightGBM and
  Setwise rerankers; conditional package generation and copy-back.
- **Non-goals**:
  - Dataset1 retraining.
  - New towers or feature families.
  - Leaderboard-driven tuning.
  - Overwriting any existing cache, checkpoint, or package.
- **Deferred work**:
  - Hyperparameter sweeps after this fixed comparison.
- **Verification rule**: Both cache reports must carry the same unique joint
  build ID and process ID, the validation report must bind the exact training
  feature hash, and both rerankers must score the same validation tensor.
- **Evidence source**: RED/GREEN tests, joint build reports and hashes, server
  process/log, full and three-slice MRR report, and optional package manifest.
- **Pass criteria**:
  - Joint cache pair is complete and passes shape/hash/temporal contracts.
  - Candidate full MRR is at least champion `+0.002`.
  - None of the three chronological slices regresses.
  - A package is generated only for the best passing candidate.
- **Confidence note**: Same-process construction removes the observed learned
  feature replay mismatch. Offline MRR remains a proxy for the leaderboard.
- **Judgment owner**: Automated cache-pair and MRR gates.

## Current State

- The old recent-200k cache is complete but cannot be paired with a separately
  reconstructed validation encoder.
- The prior matched-validation report was rejected with candidate IDs equal but
  feature values unequal.
- LightGBM, Setwise, evaluation, and conditional packaging code already exists.
- The worktree contains user changes and prior experiment files; they must be
  preserved.

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| Replay the old 200k cache in a new process | remove | It already failed and cannot prove a shared learned state. |
| Build only validation in a second process | replace | Train and validation caches must be produced by one live session. |
| Train LightGBM and Setwise | keep | Both consume the same newly bound cache pair. |
| Three-period `+0.002` gate | keep | It remains the frozen anti-overfit criterion. |
| Conditional packaging | keep | No invalid or losing model should become a submission. |

## Drift Diagnosis

- **Goal drift**: Retrying old-cache replay does not create comparable features.
- **Phase drift**: Training cannot start before the joint-cache contract passes.
- **Validation drift**: File existence alone is insufficient; reports must prove
  one process and exact hash binding.
- **Compatibility drift**: Existing standalone builders remain available; the
  new path is additive.
- **Cleanup drift**: No old artifacts will be removed.

## Priority Rationale

- The shared encoder lifecycle is the root correctness condition and therefore
  precedes expensive reranker training.
- A unique build ID plus process ID makes the same-process claim machine
  checkable instead of relying on filenames.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| Server has enough GPU, RAM, and disk headroom | unresolved | Blocks production build if false | Preflight before launch |
| One live encoder-state cache is sufficient to bind context-end and train-end encoders | assumed | Core experiment validity | Joint builder and report contract |
| Frozen model/gate settings remain unchanged | confirmed | Prevents post-hoc tuning | Existing training script |
| A failing gate generates no package | confirmed | Protects submission budget | Existing conditional supervisor |

## Phases

### Phase 1: Joint-cache contract

- **Purpose**: Make same-process provenance enforceable.
- **Entry condition**: Existing cache validation utilities and tests pass.
- **Phase rules**:
  - RED before production changes.
  - Do not weaken shape, hash, temporal, or no-overwrite checks.
- **Todos**:
  - [ ] Add a report-pair test that rejects different build/process IDs.
    - **Surface**: full-100 utilities and tests.
    - **Proof**: targeted RED then GREEN pytest.
    - **Depends on**: none.
  - [ ] Require the trainer to validate the joint pair.
    - **Surface**: matched reranker trainer.
    - **Proof**: focused regression tests and Ruff.
    - **Depends on**: report-pair contract.
- **Exit proof**: Tests prove that two independent cache runs cannot pass as a
  joint pair.
- **Stop condition**: The contract would require weakening an existing gate.

### Phase 2: One-process cache construction

- **Purpose**: Produce comparable train and validation tensors.
- **Entry condition**: Phase 1 is green.
- **Phase rules**:
  - Use one ranker, encoder-state cache, RNG progression, PID, and build ID.
  - Publish only atomic artifacts and never overwrite existing outputs.
  - Validation rows are chronological and strictly after `train_end`.
- **Todos**:
  - [ ] Extend the full-100 builder with optional joint validation outputs.
    - **Surface**: cache builder and launch script.
    - **Proof**: CLI/static tests plus server manifests.
    - **Depends on**: Phase 1.
  - [ ] Run preflight and launch the unique 20260725 experiment.
    - **Surface**: server process, cache, logs.
    - **Proof**: complete reports, hashes, shapes, and matching joint IDs.
    - **Depends on**: joint builder.
- **Exit proof**: Complete `200000x100x63` train and `20000x100x63` validation
  caches with one joint provenance token.
- **Stop condition**: OOM, insufficient disk, non-finite data, temporal
  mismatch, or provenance mismatch.

### Phase 3: Rerankers and conditional delivery

- **Purpose**: Determine whether either reranker merits submission.
- **Entry condition**: Phase 2 passes.
- **Phase rules**:
  - One fixed LightGBM configuration and one fixed Setwise configuration.
  - Full-candidate MRR early stopping.
  - No package unless full delta and all slice gates pass.
- **Todos**:
  - [ ] Train and score both rerankers.
    - **Surface**: model artifacts and evaluation report.
    - **Proof**: baseline/candidate full and slice MRR values.
    - **Depends on**: Phase 2.
  - [ ] Conditionally build, verify, and copy back the best package.
    - **Surface**: checkpoint/result.zip and hashes.
    - **Proof**: reload/schema checks and local artifact.
    - **Depends on**: passing metric gate.
- **Exit proof**: A verified local package exists, or a rejection report records
  why no package was authorized.
- **Stop condition**: Ambiguous winner, failed reload, or failed submission
  schema validation.

## Dry-Run Findings

- Reusing the old train cache cannot satisfy same-process provenance.
- The train encoder may be released after its features are published, but the
  ranker and encoder-state cache must remain alive until the train-end
  validation encoder has been created.
- A new unique prefix is required because existing artifacts are protected.
- The trainer must reject legacy matched reports for this new path even if
  their shapes happen to match.

## Final Validation

- Targeted tests and Ruff pass locally and on the server.
- Joint reports share build ID/PID and bind exact artifact hashes.
- Both rerankers finish against the same validation tensor.
- Metric gate determines packaging without manual override.

## First Execution Step

Add and run a failing unit test for the joint train/validation report-pair
contract.
