# Goal Document: Dataset2 Setwise High-Weight Scan

## Go / No-Go

- **Judgment**: Go
- **Reason**: The already-trained pure Setwise expert beats the champion on the
  full validation set and all three chronological slices; the rejected result
  came from retaining the champion's unrelated `0.07` MLP weight.

## Target Outcome

Using the completed joint validation cache and trained Setwise artifact, scan
Setwise weights from `0.80` through `1.00`, choose a weight without consulting
the final chronological slice, verify it on that held-forward slice, and
generate a reload-verified submission package only if the frozen gate passes.

## Goal Definition

- **Type**: technical / learning / delivery
- **Boundary**: Dataset2 only; existing `20,000 x 100 x 63` validation cache;
  existing champion LightGBM and Setwise models; weights `0.80..1.00` in `0.01`
  increments; Dataset1 remains unchanged.
- **Non-goals**:
  - No feature-cache rebuild.
  - No Setwise or tower retraining.
  - No leaderboard-driven weight selection.
  - No use of validation slice 2 to choose the weight.
- **Deferred work**:
  - Recency-weighted LightGBM.
  - OOF learned gates.
- **Verification rule**: Select the highest slices-0/1 mean MRR, with ties
  broken toward the larger Setwise weight; then evaluate the locked weight on
  slice 2 and the full set.
- **Evidence source**: RED/GREEN tests, complete scan report, exact per-weight
  full/three-slice MRR, checkpoint reload, and submission schema validation.
- **Pass criteria**:
  - Scan contains all 21 weights including `1.00`.
  - Locked candidate improves full MRR by at least `+0.002`.
  - No chronological slice is below the champion.
  - Checkpoint persists the selected weight and reloads to identical inference.
- **Confidence note**: Forward-held slice 2 reduces weight-selection overfit,
  but offline-to-leaderboard calibration remains imperfect.
- **Judgment owner**: Automated scan, metric, reload, and schema gates.

## Current State

- Champion fixed blend MRR is `0.495890`.
- Pure Setwise MRR is `0.546274`, with slice deltas
  `(+0.084631, +0.065558, +0.000954)`.
- The previous pipeline evaluated only `0.07 Setwise + 0.93 LightGBM`, which
  scored `0.494417` and was correctly rejected.
- Existing candidate packaging hard-codes `0.07`; it cannot yet represent the
  selected high Setwise weight.

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| Rebuild caches | remove | Required tensors already exist and are matched. |
| Retrain Setwise | remove | Pure expert already passes the observed metric gate. |
| Fixed `0.07` Setwise blend | replace | The value belonged to the old MLP/LGBM blend and masks Setwise. |
| Three-slice gate | keep | It protects the future-like slice. |
| Conditional package | keep | Delivery still requires reload and schema proof. |

## Drift Diagnosis

- **Goal drift**: More training does not test the identified weight mismatch.
- **Phase drift**: Packaging before weight persistence is tested could produce
  a package different from the evaluated candidate.
- **Validation drift**: Choosing weight on all three slices would leak the
  forward slice into selection.
- **Compatibility drift**: Legacy `0.07` packages must continue to load.
- **Cleanup drift**: Existing models, reports, caches, and packages remain
  untouched.

## Priority Rationale

- Re-scoring cached predictions is minutes rather than another eight-hour
  feature build.
- Checkpoint weight persistence is the highest-risk correctness boundary and
  is tested before server packaging.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| Setwise and champion LightGBM predictions can be reconstructed exactly | confirmed | Enables cached scan | Existing model artifacts and validation cache |
| Slices 0/1 may select the weight; slice 2 is forward holdout | frozen | Controls overfit | Scan implementation |
| Tie break favors higher Setwise weight | frozen | Avoids unnecessary dependence on weak LightGBM | Scan implementation |
| Selected package is worth one leaderboard submission | assumed | Final online proof | User submits copied package |

## Phases

### Phase 1: Scan and persistence contracts

- **Purpose**: Lock selection and checkpoint behavior before implementation.
- **Entry condition**: Existing validation report and models are complete.
- **Phase rules**:
  - RED before production code.
  - Selection cannot read slice-2 MRR.
- **Todos**:
  - [x] Test the inclusive weight grid and forward-slice-independent selector.
    - **Surface**: reusable fusion analysis and tests.
    - **Proof**: targeted RED then GREEN.
    - **Depends on**: none.
  - [x] Test that a selected Setwise weight survives checkpoint hydration.
    - **Surface**: candidate builder/checkpoint test.
    - **Proof**: prediction equivalence after reload.
    - **Depends on**: selector contract.
- **Exit proof**: Tests fail on omitted `1.00`, slice-2 selection leakage, or
  fallback to `0.07`.
- **Stop condition**: Existing inference cannot represent a Setwise weight.

### Phase 2: Cached server scan

- **Purpose**: Select and validate one high Setwise weight.
- **Entry condition**: Phase 1 is green locally and on the server.
- **Phase rules**:
  - Reuse exact joint validation cache and model hashes.
  - Record every weight and metric.
  - Do not alter the grid after scores are observed.
- **Todos**:
  - [x] Reconstruct champion LightGBM and Setwise probabilities.
    - **Surface**: scan script.
    - **Proof**: pure expert MRR matches the prior report.
    - **Depends on**: Phase 1.
  - [x] Scan, lock on slices 0/1, and run the frozen gate.
    - **Surface**: scan report.
    - **Proof**: selected weight, full/slice deltas, gate booleans.
    - **Depends on**: reconstructed predictions.
- **Exit proof**: Complete deterministic scan report.
- **Stop condition**: Model hash mismatch, prediction mismatch, non-finite
  scores, or failed gate.

### Phase 3: Conditional package and copy-back

- **Purpose**: Produce the exact evaluated candidate for user submission.
- **Entry condition**: Phase 2 passes.
- **Phase rules**:
  - Preserve champion Dataset1.
  - Persist the selected Setwise weight exactly.
  - Verify reload and submission schema before copy-back.
- **Todos**:
  - [x] Build the Dataset2 Setwise checkpoint and result package.
    - **Surface**: checkpoint and `result.zip`.
    - **Proof**: builder report, model hashes, reload prediction equality.
    - **Depends on**: passing scan.
  - [x] Copy the verified submission artifact to the local workspace.
    - **Surface**: local `result.zip`; server checkpoint and local reports.
    - **Proof**: matching `result.zip` SHA-256; checkpoint hash verified on
      the server.
    - **Depends on**: verified package.
- **Exit proof**: Local submit-ready package and checkpoint match server hashes.
- **Stop condition**: Any metric, reload, schema, or hash check fails.

## Dry-Run Findings

- Pure Setwise already satisfies the observed full/three-slice gate, so the
  scan is a robustness check rather than a search for any passing point.
- The existing evaluation report is rejected and cannot authorize packaging;
  the scan must write a new report with its own frozen selection protocol.
- The candidate builder must consume the selected weight from that report,
  not a CLI default or the champion's historical weight.

## Final Validation

- Focused tests and Ruff pass locally and on the server.
- Selected weight is chosen only from slices 0/1.
- Full and all three slice gates pass.
- Reloaded checkpoint uses the exact selected weight.
- Local `result.zip` hash matches the server. The 5.01 GB checkpoint remains
  on the server with its stored weight and hash verified; copying it is not
  required to submit the candidate.

## Final Outcome

- Selected Setwise weight: `0.80`.
- Selection used only chronological slices 0 and 1; slice 2 remained a
  forward holdout.
- Full validation MRR: `0.5469178184`, delta `+0.0510277246` versus the
  champion reference.
- Slice deltas: `(+0.0852139861, +0.0654423689, +0.0024195278)`.
- Submission SHA-256:
  `6b8fdf96d3fbded938865b644fdf103cfcb67f7df38e4915e4f62aba9d8cab26`.
- Checkpoint SHA-256:
  `333f0df1465a30268c87ac3945f6fd1356743a27bdd735552b7680bfb0877e89`.

## First Execution Step

Write a failing test for the inclusive `0.80..1.00` grid and a selector whose
answer is unchanged when only slice-2 metrics are modified.
