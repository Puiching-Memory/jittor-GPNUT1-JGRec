# Goal Document: Dataset2 Two-Tower Exploratory Submission

## Go / No-Go

- **Judgment**: Go
- **Reason**: The user explicitly accepts the offline-score risk and wants one leaderboard probe. Structural safety and Dataset1 preservation remain mandatory.

## Target Outcome

Create one locally available, structurally valid submission ZIP that keeps the current champion Dataset1 prediction unchanged and replaces only Dataset2 with predictions from the newly trained Two-Tower reranker plus its pseudo-B-selected LightGBM fusion.

## Goal Definition

- **Type**: delivery / learning
- **Boundary**: Dataset2 prediction regeneration, Dataset1/Dataset2 CSV composition, ZIP validation, checkpoint validation, and local copy.
- **Non-goals**:
  - Automatically submitting to the leaderboard.
  - Claiming the candidate is better offline.
  - Retraining any Dataset1 model.
- **Deferred work**:
  - Further tower or feature experiments.
- **Verification rule**: Dataset1 CSV is byte-identical to the `1.3426473547970703` champion; Dataset2 is produced by the frozen candidate MLP and pseudo-B-selected LightGBM (`minchild50`, iteration 231, MLP weight 0.19); the combined checkpoint reloads both datasets; the ZIP passes row, column, archive-member, and prediction-finiteness checks.
- **Evidence source**: automated tests, checkpoint reload, hashes, CSV/ZIP validation, and candidate report.
- **Pass criteria**: all structural and provenance checks pass and the final ZIP is copied under the local `result/` tree.
- **Confidence note**: This proves submit-readiness, not expected leaderboard improvement. Offline full MRR is `0.54157761`, below the champion `0.54283033`.
- **Judgment owner**: automated validation for artifact safety; the user for the leaderboard experiment.

## Current State

- The full Dataset2 reranker run completed successfully.
- A pseudo-B-selected Dataset2 LightGBM model exists with MLP weight `0.19`.
- The candidate missed the original offline acceptance gate, but the user explicitly requested an exploratory package because local MRR is not perfectly calibrated to the leaderboard.
- The current candidate checkpoint is Dataset2-only and the tuned LightGBM model has not yet been written into it.

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---------------|----------|--------|
| Reject candidate when offline delta is below `+0.002` | rewrite | Keep the negative result in provenance, but permit one user-authorized leaderboard probe. |
| Do not generate a package | remove | Explicitly overridden by the user. |
| Preserve champion Dataset1 | keep | Isolates the experiment to Dataset2. |
| Do not auto-submit | keep | The user asked for a package, not an external submission. |
| Validate checkpoint and ZIP | keep | Offline risk does not justify artifact-integrity risk. |

## Drift Diagnosis

- **Goal drift**: No new model research is allowed during packaging.
- **Phase drift**: Prediction must follow checkpoint construction so provenance is exact.
- **Validation drift**: A ZIP existing is insufficient; its contents and source hashes must be checked.
- **Compatibility drift**: The combined checkpoint must continue loading both datasets through the public checkpoint API.
- **Cleanup drift**: Existing unrelated worktree changes remain untouched.

## Priority Rationale

- First prove the tuned Dataset2 state can be represented and reloaded.
- Then generate predictions from that exact state and compose the smallest isolated leaderboard probe.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|------|--------|--------|-------------------|
| Champion Dataset1 CSV is the intended fixed half | confirmed | Prevents Dataset1 regression from contaminating the probe | Verify byte hash before packaging |
| Pseudo-B winner is the intended Dataset2 fusion | confirmed | Uses the strongest non-leaky candidate selected before pseudo-B evaluation | Freeze `minchild50`, iteration 231, weight 0.19 |
| Offline regression is acceptable for one submission | confirmed by user | Overrides only the score gate | Record the regression in the candidate report |

## Phases

### Phase 1: Freeze and hydrate the candidate

- **Purpose**: Create an exact, reloadable Dataset2 tuned state.
- **Entry condition**: Full reranker checkpoint and tuning report exist.
- **Phase rules**:
  - Change only Dataset2 LightGBM model text, metadata, and blend weight.
  - Do not alter Dataset1 state.
- **Todos**:
  - [x] Add a tested helper that replaces a checkpoint Dataset2 LightGBM result from a frozen model.
    - **Surface**: checkpoint/candidate construction code and tests.
    - **Proof**: RED then GREEN test plus checkpoint reload.
    - **Depends on**: none.
- **Exit proof**: Reloaded Dataset2 state reports the frozen model and weight `0.19`.
- **Stop condition**: Model schema or feature indices are incompatible.

### Phase 2: Predict and compose

- **Purpose**: Produce the isolated Dataset2 leaderboard probe.
- **Entry condition**: Phase 1 checkpoint reload passes.
- **Phase rules**:
  - Reuse the completed encoder state; do not retrain.
  - Dataset1 prediction must be byte-identical to the champion.
- **Todos**:
  - [x] Generate Dataset2 CSV from the tuned candidate checkpoint.
    - **Surface**: server prediction artifact.
    - **Proof**: expected rows, columns, finite scores, and source checkpoint hash.
    - **Depends on**: Phase 1.
  - [x] Combine champion Dataset1 and candidate Dataset2 into `result.zip`.
    - **Surface**: submission artifact.
    - **Proof**: archive membership and CSV validation.
    - **Depends on**: candidate CSV.
- **Exit proof**: One complete, validated two-dataset ZIP exists.
- **Stop condition**: Prediction requires refitting, or Dataset1 differs from champion.

### Phase 3: Copy and report

- **Purpose**: Make the experiment available to the user locally.
- **Entry condition**: ZIP and combined checkpoint pass validation.
- **Phase rules**:
  - Do not submit automatically.
  - Include hashes and the known offline regression.
- **Todos**:
  - [x] Copy ZIP and candidate reports to the local result directory; retain the 4.98 GB checkpoint on the server.
    - **Surface**: local artifacts.
    - **Proof**: ZIP size and SHA-256 match the server artifact; checkpoint reload and hash were verified on the server.
    - **Depends on**: Phase 2.
- **Exit proof**: Local files match server hashes.
- **Stop condition**: Any transfer or hash mismatch.

## Dry-Run Findings

- The raw full-reranker `result.zip` is Dataset2-only and cannot be submitted directly.
- The tuned LightGBM model was created after the full-reranker prediction, so the existing Dataset2 CSV must not be mislabeled as the tuned candidate.
- A combined checkpoint is required to make prediction provenance explicit and reproducible.

## Final Validation

- Reload both datasets from the combined checkpoint.
- Validate exact Dataset1 hash, Dataset2 row/schema/finite values, ZIP members, and local/server SHA-256 equality.

## First Execution Step

Write a failing test for replacing only Dataset2's LightGBM model and blend weight while preserving every other checkpoint field.

## Outcome

- Candidate ZIP: `result/d1_champion_d2_twotower200k_exploratory_seed60_20260724/result.zip`
- ZIP bytes: `65,380,052`
- ZIP SHA-256: `f0b637fab7ff65dfc64b6b1d8175a475cf3e329864776ed547b7687e8fedede7`
- Dataset1 remained byte-identical to the online champion.
- Dataset2 used the pseudo-B-selected `minchild50` LightGBM at iteration 231 with MLP weight `0.19`.
- The checkpoint reloaded both datasets on the server and has SHA-256 `b46a5514ebfd9e0e5b4ec11b4c8e2d1e8e1ab15e65a8b6b940e5bd2ad7732caa`.
- The package is explicitly exploratory: the tuning report did not pass the offline gate.
