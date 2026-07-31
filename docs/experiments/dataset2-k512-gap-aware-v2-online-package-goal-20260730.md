# Goal Document: Dataset2 K=512 gap-aware v2 online package

## Go / No-Go

- **Judgment**: Complete
- **Reason**: The current K=512 run completed the frozen near + gapped duel,
  the one-time external safety gate authorized packaging, and the generated
  ZIP passed remote and local lineage/member-hash validation.

## Target Outcome

Produce one locally available, validated `result.zip` whose Dataset2 member is
the frozen `cooccur_lift_gap_aware_v2` blend over the current run's bugfixed-v1
baseline, with weight `0.5`, and whose Dataset1 member is carried forward
unchanged from the standard champion package.

## Goal Definition

- **Type**: delivery
- **Boundary**: Verify the current run, materialize test-only successor scores,
  build the two-member submission ZIP, validate it, hash it, and copy it local.
- **Non-goals**:
  - Retraining any model or rebuilding training/validation features.
  - Rescanning weights or features.
  - Interpreting external metrics as effect sizes.
  - Uploading the package to the competition service.
- **Deferred work**:
  - Source-conditioned/candidate joint-structure drift diagnosis.
- **Verification rule**: Every input must match the current selection lock and
  external authorization; both CSV members must pass schema/shape/finite-value
  validation; the downloaded ZIP hash must equal the remote ZIP hash.
- **Evidence source**: Frozen JSON contracts, SHA-256 hashes, materialization
  report, package report, submission validator, and remote/local hash equality.
- **Pass criteria**: One package is generated with current-run lineage, selected
  weight `0.5`, no rescan, valid Dataset1 and Dataset2 members, and an identical
  local copy.
- **Confidence note**: Hash-bound lineage and deterministic validation prove
  package identity and structural validity; leaderboard quality remains the
  user's manual submission result.
- **Judgment owner**: Frozen contracts and validators own package correctness;
  the user owns upload and leaderboard interpretation.

## Current State

- Run
  `result/dataset2_k512_cooccur_lift_successor_v2_rerun_20260729` is
  `complete_external_accepted`.
- `cooccur_lift_gap_aware_v2` is selected and all seven external safety gates
  passed.
- The existing preregistered online-package contract is historical and binds a
  different selection lock and V1 package, so it cannot authorize this run.
- The current run contains a newly trained bugfixed-v1 full-origin model used by
  the accepted external comparison.

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| Historical online-package contract | rewrite | It binds a different selection lock and baseline artifact. |
| Historical finish shell script | reuse workflow only | Its sequencing is useful, but its fixed hashes are stale for this run. |
| Weight `0.5` and no-rescan rule | keep | These are frozen by the accepted duel and external gate. |
| Current run's full-origin V1 | keep | This is the baseline actually used by the accepted external comparison. |

## Drift Diagnosis

- **Goal drift**: Reusing the historical V1 ZIP would produce a package that was
  not the evaluated current-run comparison.
- **Phase drift**: Materialization must not begin until all current identities
  and the Dataset1 carry-forward source are frozen.
- **Validation drift**: File existence alone is insufficient; hashes, manifests,
  member shapes, and local transfer equality are required.
- **Compatibility drift**: No fallback to the historical contract is allowed.
- **Cleanup drift**: No unrelated refactor belongs in this delivery.

## Priority Rationale

- Resolve lineage first because a valid-looking ZIP with the wrong V1 baseline
  would be an unrecoverable experimental interpretation error.
- Materialize once only after the contract is frozen, then package and download.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| Current V1 full-origin model is the package baseline | confirmed | Preserves the evaluated comparison | Verify report/model hashes against current contracts |
| Selected weight is exactly `0.5` | confirmed | Prevents post-gate tuning | Enforce in package contract |
| Dataset1 is unchanged | confirmed | Keeps this a Dataset2-only candidate | Freeze exact source member hash |
| Current V1 test baseline package/materialization exists or is reproducible | assumed | Required to blend successor scores | Discover and verify before inference |

## Phases

### Phase 1: Freeze current-run package lineage

- **Purpose**: Make every package input explicit and hash-bound.
- **Entry condition**: Current external report says package authorized.
- **Phase rules**:
  - Read-only inspection only.
  - Stop on any hash, selected candidate, gate, or baseline mismatch.
- **Todos**:
  - [x] Verify selection lock, external receipt/report, V1 training report/model,
    candidate model, source checkpoint, and data inputs.
    - **Surface**: Current run artifacts and source inputs.
    - **Proof**: Exact SHA-256 inventory and cross-reference validation.
    - **Depends on**: none.
  - [x] Freeze Dataset1 carry-forward and current V1 Dataset2 package inputs.
    - **Surface**: Package execution contract.
    - **Proof**: Member hashes and expected row/column counts.
    - **Depends on**: Current-run identity verification.
- **Exit proof**: A new immutable execution contract validates without inference.
- **Stop condition**: Any required artifact is absent or belongs to a different
  run.

### Phase 2: Materialize and package once

- **Purpose**: Generate the exact selected online scores and submission archive.
- **Entry condition**: Phase 1 contract validation passes.
- **Phase rules**:
  - Use the current K=512 full-origin structures and frozen weight `0.5`.
  - No training, feature/weight rescan, or tolerance relaxation.
  - Write to a new output directory only.
- **Todos**:
  - [x] Materialize test successor probabilities and validate support-collapse
    statistics and binding hashes.
    - **Surface**: Test materialization directory.
    - **Proof**: Materialization report and score-array hashes.
    - **Depends on**: Phase 1.
  - [x] Build and validate the standard two-member ZIP.
    - **Surface**: Submission directory.
    - **Proof**: Package report, CSV validation, and ZIP SHA-256.
    - **Depends on**: Test materialization.
- **Exit proof**: The remote ZIP and package report pass all frozen checks.
- **Stop condition**: Materialization binding, collapse count, member validation,
  or hash verification fails.

### Phase 3: Copy and verify locally

- **Purpose**: Deliver the package for manual upload.
- **Entry condition**: Remote package validation passes.
- **Phase rules**:
  - Do not submit externally.
  - Do not overwrite an unrelated local artifact.
- **Todos**:
  - [x] Download ZIP and provenance reports.
    - **Surface**: Local result directory.
    - **Proof**: Remote/local SHA-256 equality.
    - **Depends on**: Phase 2.
- **Exit proof**: Local paths and matching hashes are recorded.
- **Stop condition**: Transfer or hash equality fails.

## Dry-Run Findings

- The historical package contract cannot be reused because its selection-lock
  hash is stale.
- The only unresolved prerequisite is whether the current V1 online baseline
  package already exists or must be deterministically regenerated; Phase 1
  resolves this before any inference.
- No circular dependency exists: lineage freeze precedes materialization,
  packaging precedes transfer.

## Final Validation

- Complete. Both ZIP members and all contract-bound hashes passed; downloaded
  `result.zip` SHA-256 equals the verified remote SHA-256
  `a7f1a6522b977a70584aca3e4388f27f7e448ddaa792f4180edfe729448913b3`.

## First Execution Step

Completed: inventoried the current run and existing V1/Dataset1 inputs, froze
and verified their hashes, generated both staging and final packages, and
copied the validated artifact locally.
