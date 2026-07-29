# Goal Document: Dataset2 Bug-Fixed v1 External Safety Submission

## Go / No-Go

- **Judgment**: Complete; external safety gate passed.
- **Reason**: The deterministic full-data refit, exact replay, package
  validation, local transfer, and user-performed manual submission all
  completed. The reported score `1.3577315048069973` is slightly above the
  prior V1 score and is retained only as a safety signal.

## Target Outcome

Produce one reproducible submission package from the bug-fixed v1 training
path, prove its local packaging and replay contracts, and submit exactly that
package once to the configured competition endpoint.

## Goal Definition

- **Type**: operational / quality / delivery.
- **Boundary**: identify the corrected v1 implementation, train or locate its
  full-data checkpoint, generate the standard Dataset1+Dataset2 submission,
  verify the package, and perform one explicitly authorized external submit.
- **Non-goals**:
  - Do not train or submit either v2 candidate.
  - Do not relax the `2e-5` replay tolerance.
  - Do not rescan v1-family weights.
  - Do not use the external result as an effect-size estimate.
- **Deferred work**:
  - Root-cause analysis and repair of the `fold-0` replay mismatch.
  - Resuming the preregistered v2 duel.
- **Verification rule**: the submitted file SHA-256 must equal the locally
  audited package SHA-256; the full-data checkpoint must replay deterministically;
  the submission response must identify the accepted job/file.
- **Evidence source**: checkpoint/config hashes, replay report, package report,
  submission file hash, and external submission receipt.
- **Pass criteria**:
  - The asset is a full-data v1 checkpoint, not a rolling-fold model.
  - Dataset1 remains the frozen champion output.
  - Dataset2 uses the corrected v1 path with the frozen `0.50` blend.
  - Schema, row count, candidate count, finiteness, normalization, and artifact
    hashes pass.
  - Exactly one package is submitted.
- **Confidence note**: local replay and byte hashes prove identity of the
  delivered artifact. The external score is retained only as a safety signal
  and receives the frozen 19.5x interpretation discount.
- **Judgment owner**: automated preflight gates for the artifact; the
  competition receipt for delivery.

## Current State

- The `0.10732766809698313` mismatch was traced to execution-level CUDA
  training nondeterminism; replay tolerance was not widened.
- A deterministic CPU full-data V1 refit replayed exactly with maximum
  probability error `0.0`; test scoring remained on CUDA.
- The audited package was copied to the shared workspace with SHA-256
  `b90960c3427f70e2745bcb381289fca4625c208ebfaefb43ecdbc7a7387ff2f0`.
- The user manually uploaded that package and reported score
  `1.3577315048069973`.
- V2 remained unopened and V1-family weights remained frozen.

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---------------|----------|--------|
| Diagnose `0.10733` immediately | defer | The user reprioritized a bug-fixed v1 safety submission. |
| Resume v2 duel | remove from this goal | It remains blocked by the replay gate. |
| Submit in-memory fold-0 replay | reject | It is incomplete and not a full-data candidate. |
| Reuse existing v1 package blindly | reject | It may predate the remembered bug fix. |
| Preserve replay and hash gates | keep | They prevent submitting the wrong model. |

## Drift Diagnosis

- **Goal drift**: treating the failed fold replay as a submit-ready model would
  substitute a diagnostic artifact for the requested full-data v1.
- **Phase drift**: packaging cannot precede exact bug-fix/checkpoint identity.
- **Validation drift**: a successful CSV write is insufficient without replay,
  schema, and hash evidence.
- **Compatibility drift**: historical and corrected v1 paths cannot coexist
  under the same label without explicit hashes.
- **Cleanup drift**: no unrelated refactor belongs in this submission.

## Priority Rationale

- Establishing artifact identity comes first because an external submission is
  irreversible and limited.
- A dry-run package provides the cheapest proof before any external action.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|------|--------|--------|-------------------|
| The corrected v1 code/checkpoint exists in current workspace history | resolved by full refit | Historical reuse was rejected; a SHA-bound deterministic full-data refit was created | Complete |
| One external submission is authorized | confirmed | Allows delivery after all local gates pass | User instruction in this conversation |
| External score is a safety gate only | confirmed | Prevents effect-size reinterpretation | Frozen protocol |
| Competition submit command is already configured | not required | The user manually uploaded the audited local package | Complete |

## Phases

### Phase 1: Establish Bug-Fixed v1 Identity

- **Purpose**: identify the exact corrected implementation and determine
  whether a complete checkpoint already exists.
- **Entry condition**: far-horizon job is terminal and no v2/external job is
  running.
- **Phase rules**:
  - Read-only inspection only.
  - Reject rolling-fold or partial outputs.
  - Require file hashes and recorded configuration.
- **Todos**:
  - [x] Audit historical v1 checkpoints, replay reports, promotion records, and
    code changes associated with the remembered bug fix.
    - **Surface**: experiment docs, result manifests, git diff/history, remote artifacts.
    - **Proof**: one provenance table naming exact paths and SHA-256 values.
    - **Depends on**: none.
  - [x] Decide reuse versus full refit.
    - **Surface**: checkpoint contract.
    - **Proof**: complete checkpoint satisfies corrected-code and full-data requirements.
    - **Depends on**: provenance audit.
- **Exit proof**: one unambiguous bug-fixed full-data v1 training contract.
- **Stop condition**: multiple plausible artifacts cannot be distinguished by
  recorded evidence.

### Phase 2: Build and Verify the Package

- **Purpose**: produce the exact candidate file without external delivery.
- **Entry condition**: Phase 1 identifies or creates a valid full-data
  checkpoint.
- **Phase rules**:
  - Keep the frozen `0.50` v1 blend.
  - No v2 candidate or weight scan.
  - No external request.
- **Todos**:
  - [x] Replay the full-data checkpoint with the corrected implementation.
    - **Surface**: replay report and Dataset2 probabilities.
    - **Proof**: deterministic replay gates pass.
    - **Depends on**: Phase 1.
  - [x] Compose the standard submission and validate all rows.
    - **Surface**: package CSV and manifest.
    - **Proof**: schema/count/finiteness/normalization checks and SHA-256.
    - **Depends on**: replay.
- **Exit proof**: one locally audited package with immutable hash.
- **Stop condition**: any replay, identity, schema, or numerical check fails.

### Phase 3: Submit Once

- **Purpose**: deliver the audited artifact as an external safety check.
- **Entry condition**: Phase 2 passes and the configured submit mechanism is
  available.
- **Phase rules**:
  - Submit exactly the audited SHA-256 once.
  - Record the receipt without copying secrets.
  - Do not reinterpret the score as an effect size.
- **Todos**:
  - [x] Execute the configured competition submission.
    - **Surface**: manual competition-page upload by the user.
    - **Proof**: user-reported score bound in the local result record.
    - **Depends on**: Phase 2.
  - [x] Record delivery status and interpretation rule.
    - **Surface**: experiment result document.
    - **Proof**: score, file hash, recorded date, and `safety_gate_only=true`.
    - **Depends on**: accepted submission.
- **Exit proof**: accepted external receipt matches the audited file hash.
- **Stop condition**: endpoint, account, dataset, or package identity is
  ambiguous.

## Dry-Run Findings

- The failed near-fold replay cannot be packaged directly because it stopped
  before creating a complete full-data candidate.
- The word "new v1" is not an artifact identifier; exact provenance must be
  resolved before an irreversible submission.
- The existing v2 watcher is already terminal and external remained unopened,
  so it cannot race this workflow.

## Final Validation

- [x] Full-data checkpoint provenance and SHA-256.
- [x] Deterministic replay report.
- [x] Submission package validation and SHA-256.
- [x] One user-performed submission result recorded for the audited package.
- [x] Result record states that external is a safety gate only.

## First Execution Step

Completed: the provenance audit selected a new deterministic full-data refit
instead of reusing an ambiguous historical artifact.
