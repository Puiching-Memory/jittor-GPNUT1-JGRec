# Goal Document: Dataset2 K512 Frozen-Validation Recovery

## Go / No-Go
- **Judgment**: Go
- **Reason**: The 200k K512 training cache is complete and exactly aligned
  with the frozen training queries. Only the 20k validation negatives differ,
  so a validation-only atomic rebuild can recover the frozen comparison
  without repeating the training cache.

## Target Outcome
Recompute the K512 validation features against the exact frozen 20260725
validation candidate matrix, prove all query sidecars are byte-identical to
the frozen alignment, then resume the automatic pipeline from
`materialize_near_lift`.

## Goal Definition
- **Type**: technical / operational / quality
- **Boundary**: Add a frozen-candidate validation rebuild path, rebuild only
  the 20k validation cache, validate it, and resume the existing automatic
  run.
- **Non-goals**:
  - Do not rebuild or alter the completed 200k training cache.
  - Do not change K, rows, features, folds, seeds, model heads, precision,
    tolerances, or the external-gate policy.
  - Do not copy a frozen candidate sidecar over features computed for another
    candidate matrix.
- **Deferred work**:
  - General migration of every historical cache builder to isolated RNG
    streams.
- **Verification rule**: The rebuilt validation feature tensor must be
  computed from the frozen candidates; candidates, src, dst, time, and row
  indices must match the frozen reference exactly; the positive must remain
  at column zero; reports and hashes must describe the new artifacts.
- **Evidence source**: Unit tests, focused cache-contract tests, remote
  SHA-256/array equality checks, materialization report, and pipeline status.
- **Pass criteria**: Exact frozen query alignment passes, K512 validation
  shape is `(20000, 100, 63)`, all values are finite, `materialize_near_lift`
  exits zero, and the controller advances beyond that stage.
- **Confidence note**: Recomputing features from the frozen query tensor
  preserves the comparison population directly; it does not depend on
  replaying a stochastic encoder's RNG consumption.
- **Judgment owner**: Exact artifact checks and the automatic controller.

## Current State
- The K512 training cache is complete and matches the frozen training query
  alignment.
- The existing K512 validation cache has correct rows and positives but a
  newly sampled negative matrix.
- The automatic controller stopped at `materialize_near_lift`; no external
  stage was opened.
- The server's SSH banner has been intermittent, so local implementation and
  tests must complete before remote recovery begins.

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| Rebuild the full joint cache | remove | The 200k train cache is already valid |
| Reuse current validation features and replace candidates | remove | Features would describe different candidate IDs |
| Restore historical RNG trajectory | replace | Frozen candidates are the direct contract and avoid stochastic replay |
| Resume the whole controller from its failed status | rewrite | Resume must start only after atomic validation-cache replacement |

## Drift Diagnosis
- **Goal drift**: Re-running 200k work does not prove frozen validation
  alignment.
- **Phase drift**: Recovery and resume must be separated by an exact artifact
  gate.
- **Validation drift**: Shape and positive-column checks alone are
  insufficient; every query sidecar must match.
- **Compatibility drift**: The existing report schema must remain consumable
  by `materialize_near_lift`.
- **Cleanup drift**: The failed cache is archived, not deleted or silently
  overwritten.

## Priority Rationale
- First prove that feature computation can accept a caller-supplied frozen
  query matrix without invoking negative sampling.
- Only then perform the expensive remote encoder fit.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| Frozen 20260725 validation sidecars remain on the server | confirmed | Required recovery input | Revalidate hashes before execution |
| Completed 200k train cache remains valid | confirmed | Avoids a full rebuild | Recheck report and query alignment |
| Validation encoder refit may be numerically nondeterministic | confirmed | Must not affect candidate identity | Use a separate encoder RNG and frozen queries |
| Failed controller supports safe stage resume | assumed | Determines restart mechanism | Inspect marker/status logic before launch |

## Phases

### Phase 1: Freeze recovery behavior with tests
- **Purpose**: Prove supplied candidates are preserved and mismatched query
  sidecars are rejected.
- **Entry condition**: Recovery boundary documented.
- **Phase rules**:
  - Tests precede production implementation.
  - No tolerance-based comparisons.
- **Todos**:
  - [x] Add RED tests for frozen candidate query construction and mismatch
    rejection.
    - **Surface**: validation cache recovery API.
    - **Proof**: focused pytest fails because the API is missing.
    - **Depends on**: none.
- **Exit proof**: Focused tests pass after minimal implementation.
- **Stop condition**: Any design permits candidates to be generated
  implicitly.

### Phase 2: Atomic validation-only rebuild
- **Purpose**: Generate K512 validation features for the frozen query tensor.
- **Entry condition**: Phase 1 is green.
- **Phase rules**:
  - Write to new temporary artifacts.
  - Preserve the failed cache until all new hashes pass.
  - Do not touch training artifacts.
- **Todos**:
  - [x] Implement a validation-only recovery command.
    - **Surface**: script and cache-contract helper.
    - **Proof**: unit tests, Ruff, dry run.
    - **Depends on**: Phase 1.
  - [ ] Run the recovery remotely and validate all six artifacts.
    - **Surface**: K512 validation cache and report.
    - **Proof**: exact array equality and finite feature report.
    - **Depends on**: implementation.
- **Exit proof**: Rebuilt cache passes `materialize_near_lift` preconditions.
- **Stop condition**: Any query-sidecar mismatch, non-finite feature, shape
  drift, or artifact hash mismatch.

### Phase 3: Resume the automatic pipeline
- **Purpose**: Continue the frozen stage order from
  `materialize_near_lift`.
- **Entry condition**: Phase 2 exact gate passes.
- **Phase rules**:
  - External remains unavailable until near+gapped selection authorizes it.
  - Existing completed stage markers are preserved.
- **Todos**:
  - [ ] Resume and confirm `materialize_near_lift` completes.
    - **Surface**: controller status, stage marker, near-assets report.
    - **Proof**: zero exit and next stage observed.
    - **Depends on**: Phase 2.
- **Exit proof**: Controller advances beyond `materialize_near_lift`.
- **Stop condition**: Any frozen contract or stage-order validation fails.

## Dry-Run Findings
- Replacing only `.val-candidates.npy` would corrupt feature-to-candidate
  correspondence and is forbidden.
- Replaying the old post-encoder RNG state is brittle and unnecessary because
  the frozen candidate tensor is available.
- The recovery needs an archive-and-promote step so a failed rebuild cannot
  damage the completed artifacts.

## Final Validation
- Focused RED/GREEN pytest and related pipeline/cache tests.
- Ruff on all changed Python files.
- Remote exact equality for candidates/src/dst/time/row indices.
- Remote K512 shape, finite-value, report-hash, and positive-column checks.
- Successful `materialize_near_lift` and controller advancement.

## First Execution Step
Add failing tests for constructing validation queries from caller-supplied
frozen candidates without consuming the encoder RNG.
