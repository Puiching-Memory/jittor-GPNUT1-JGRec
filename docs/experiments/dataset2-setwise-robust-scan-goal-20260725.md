# Goal Document: Dataset2 Setwise Robust Weight Scan

## Go / No-Go

- **Judgment**: Go
- **Reason**: The online score `1.3530197201` validates the high-Setwise
  direction, while the previous `0.80..1.00` scan selected its lower boundary.
  Expanding below `0.80` is the cheapest unresolved experiment.

## Target Outcome

Reuse the existing validation cache and trained experts to scan Setwise weights
from `0.750` through `0.900` at `0.005`, identify a time-stable optimum without
using the final chronological slice for selection, and authorize a new package
only when it robustly improves the current `0.80` online champion candidate.

## Goal Definition

- **Type**: learning / delivery
- **Boundary**: Dataset2 cached predictions; champion LightGBM and Setwise
  experts; 31 weights; Dataset1 unchanged.
- **Non-goals**:
  - No cache or encoder rebuild during scanning.
  - No model retraining.
  - No leaderboard-driven selection among the new weights.
- **Deferred work**:
  - New features, new towers, and Setwise retraining.
- **Verification rule**: Select using chronological slices 0 and 1 only, then
  compare the locked weight against `0.80` on full validation and slice 2.
- **Evidence source**: RED/GREEN tests, complete per-weight scan report, exact
  cache/model hashes, and conditional package report.
- **Pass criteria**:
  - The grid contains exactly 31 weights including `0.750`, `0.800`, and
    `0.900`.
  - Changing slice-2 predictions cannot change the selected weight.
  - A new package is authorized only if full MRR improves by at least
    `0.0002` over `0.80` and none of the three slice MRRs declines.
  - Otherwise `0.80` remains champion and no expensive package is generated.
- **Confidence note**: The forward holdout limits local weight overfit, but
  offline-to-leaderboard calibration remains imperfect.
- **Judgment owner**: Frozen scan and authorization metrics.

## Current State

- Online champion candidate: `0.80 Setwise + 0.20 LightGBM`, score
  `1.3530197201`.
- Its offline full MRR is `0.5469178184`; slice-2 MRR is `0.5061992242`.
- Pure Setwise `1.00` is slightly worse, so the unresolved region is below
  `0.80`.
- Existing cached tensors and model artifacts are sufficient for this scan.

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| `0.80..1.00` grid | rewrite | The optimum landed on the lower boundary. |
| Slices 0/1 selection | keep | Prevents forward-holdout leakage. |
| Slice 2 gate vs old champion | strengthen | Compare against the proven `0.80` candidate. |
| Immediate package generation | make conditional | Avoid spending compute/submission budget on noise. |

## Drift Diagnosis

- **Goal drift**: Retraining models would not answer whether the optimum lies
  just below `0.80`.
- **Phase drift**: Packaging must follow, not precede, the robustness gate.
- **Validation drift**: The online `0.80` score is evidence for the direction,
  not permission to select new weights from the leaderboard.
- **Compatibility drift**: The existing `0.80` package and checkpoint remain
  untouched.
- **Cleanup drift**: No unrelated refactor is included.

## Priority Rationale

- Cached rescoring takes seconds and directly resolves a boundary optimum.
- A `0.005` grid is fine enough for blend weights without pretending that
  noisier precision is meaningful.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| Cached predictions reconstruct the prior `0.80` metrics exactly | confirmed | Makes comparisons valid | Hash and metric assertions |
| `0.005` is the refinement step | frozen | Produces 31 candidates | Grid contract test |
| Full `+0.0002` and non-declining slice 2 authorize packaging | frozen | Controls noisy submissions | Scan gate |

## Phases

### Phase 1: Scan contract

- **Purpose**: Lock the refined grid and leakage boundary.
- **Entry condition**: Prior scan implementation and artifacts exist.
- **Phase rules**:
  - RED before production changes.
  - Slice 2 cannot affect selection.
- **Todos**:
  - [x] Add a failing test for the 31-point grid and slice-2 independence.
    - **Surface**: fusion-analysis tests.
    - **Proof**: targeted RED.
    - **Depends on**: none.
- **Exit proof**: Test fails for the missing configurable robust scan.
- **Stop condition**: Prior cached metrics cannot be reconstructed.

### Phase 2: Cached robust scan

- **Purpose**: Measure the complete region and lock one candidate.
- **Entry condition**: Phase 1 is GREEN locally and on the server.
- **Phase rules**:
  - Reuse existing cache/model hashes.
  - Record all weights before applying the gate.
- **Todos**:
  - [x] Run the 31-weight scan on the server.
    - **Surface**: JSON report.
    - **Proof**: full and three-slice MRR for every weight.
    - **Depends on**: Phase 1.
  - [x] Compare the locked weight with `0.80`.
    - **Surface**: authorization fields.
    - **Proof**: full delta and slice-2 delta.
    - **Depends on**: completed scan.
- **Exit proof**: Deterministic report with an authorized/rejected judgment.
- **Stop condition**: Any hash or prior-metric assertion fails.

### Phase 3: Conditional delivery

- **Purpose**: Create a submission only for a robust improvement.
- **Entry condition**: Phase 2 authorizes a weight other than `0.80`.
- **Phase rules**:
  - Preserve Dataset1.
  - Do not rebuild the validation cache.
- **Todos**:
  - [x] Generate and verify the selected package if authorized.
    - **Surface**: checkpoint, ZIP, candidate report.
    - **Proof**: reload, schema, and SHA-256 checks.
    - **Depends on**: passing robust gate.
- **Exit proof**: Verified local ZIP, or an evidence-backed decision to retain
  `0.80`.
- **Stop condition**: Full gain is below `0.0002` or slice 2 declines.

## Dry-Run Findings

- The old scan already includes `0.80..0.90`; only `0.750..0.795` is new, but
  rerunning the full refined grid provides one auditable report.
- Package building is orders of magnitude more expensive than cached scanning,
  so it remains conditional.

## Final Validation

- Targeted tests and Ruff pass locally and on the server.
- Report contains all 31 weights and exactly reproduces the `0.80` baseline.
- The authorization rule is applied before any package generation.

## Final Outcome

- The complete 31-point scan finished in `14.33` seconds.
- `0.80` remained the best selection weight and the best full-validation
  weight at MRR `0.5469178184`.
- `0.75` had the best forward-slice MRR (`0.5067208355`) but reduced slice 0
  by `0.0008912809` and did not improve full MRR.
- The gate rejected a new package. The online `0.80` candidate remains the
  champion, and no final encoder rebuild was started.

## First Execution Step

Add a failing test for a configurable inclusive `0.750..0.900` grid with
`0.005` spacing and selection independent of slice 2.
