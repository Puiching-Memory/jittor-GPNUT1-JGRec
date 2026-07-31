# Goal Document: Dataset2 K=512 Predict Smoke

## Go / No-Go
- **Judgment**: Go
- **Reason**: The source checkpoint is loadable and already freezes both
  `structure_predict_neighbor_limit=512` and
  `source_profile_predict_history_limit=512`; the smoke is isolated from
  training, external evaluation, and submission generation.

## Target Outcome
Measure current-server Dataset2 prediction throughput and peak host RSS for a
real K=512 checkpoint replay slice, then estimate full 153,420-row prediction
time with an explicit confidence range.

## Goal Definition
- **Type**: operational learning
- **Boundary**: Dataset2 only; existing checkpoint; first 50,000 test rows;
  production prediction batch size 2,048; hard wall-clock cap 30 minutes.
- **Non-goals**:
  - Do not retrain or modify the checkpoint.
  - Do not open external evaluation.
  - Do not generate or authorize a submission package.
- **Deferred work**:
  - Corrected v2 gapped folds and full-origin refit.
- **Verification rule**: Run the real checkpoint replay prediction path under
  `/usr/bin/time -v`; preserve CLI progress/memory logs and report elapsed
  time, rows/s, maximum RSS, and a full-row extrapolation.
- **Evidence source**: Process exit status, generated slice CSV validation,
  memory log, GNU time resource report, and source checkpoint configuration.
- **Pass criteria**: The 50,000-row slice completes within 30 minutes without
  OOM; output has 50,000 finite 100-candidate rows; peak RSS remains below
  physical memory.
- **Confidence note**: A prefix slice includes real feature construction and
  model inference but may not capture the full-run cache high-water mark;
  the estimate therefore includes a conservative range and is compared with
  the two prior full checkpoint replays.
- **Judgment owner**: Process metrics and structural output validation.

## Current State
- Dataset2 has 153,420 test rows.
- The checkpoint SHA-256 is
  `0b0a846cb6c7f5b5403a75a04ed12707340f024dee99a27cded3258bb108a7aa`.
- Its frozen Dataset2 configuration already uses K=512 for both temporal
  truncation limits.
- Two prior full replays took about 19.4–19.6 minutes, but did not capture
  process peak RSS with GNU time.

## Priority Rationale
- Measure the risky K=512 prediction path before spending compute on the
  corrected v2 rerun.
- Reuse the exact checkpoint replay path instead of a synthetic tensor
  benchmark.

## Assumptions and Open Decisions
| Item | Status | Impact | Owner / Next step |
|------|--------|--------|-------------------|
| Prefix 50k is representative enough for throughput | assumed | Full estimate may drift as caches warm/grow | Compare against prior full replays and report a range |
| Production prediction batch size remains 2,048 | confirmed | Keeps timing comparable to prior replay | CLI contract |
| K means both structure and source-profile predict limits | confirmed | Defines the tested memory/quality setting | Checkpoint inspection |

## Phases

### Phase 1: Bounded replay smoke
- **Purpose**: Obtain real throughput and RSS evidence without risking an
  unbounded full replay.
- **Entry condition**: Checkpoint K=512 and server free memory confirmed.
- **Phase rules**:
  - Hard timeout at 1,800 seconds.
  - Stop if RSS approaches 28 GiB or the process becomes unhealthy.
  - No external, training, checkpoint mutation, or packaging.
- **Todos**:
  - [ ] Replay the first 50,000 Dataset2 rows.
    - **Surface**: Existing CLI checkpoint prediction path.
    - **Proof**: Exit status, log, CSV shape, GNU time report.
    - **Depends on**: None.
- **Exit proof**: Resource and structural evidence are complete.
- **Stop condition**: OOM, non-finite output, invalid shape, or timeout.

### Phase 2: Extrapolation
- **Purpose**: Turn the slice measurement into a practical full-run estimate.
- **Entry condition**: Phase 1 completed.
- **Phase rules**:
  - Separate checkpoint-load overhead from prediction time where logs allow.
  - Report a range, not false precision.
- **Todos**:
  - [ ] Compute rows/s, seconds/1k rows, peak RSS, headroom, and projected
    153,420-row duration.
- **Exit proof**: A concise estimate with assumptions and comparison to prior
  full replays.
- **Stop condition**: Slice is too short or unstable to support extrapolation.

## Dry-Run Findings
- The checkpoint already carries K=512, so no configuration mutation is
  required.
- A 50,000-row slice is roughly one third of Dataset2 and should complete well
  inside the 30-minute cap based on prior replay evidence.
- GNU time captures process high-water RSS; the project memory log provides
  stage timestamps and endpoint RSS.

## Final Validation
- Verify exit code zero.
- Verify exactly 50,000 rows and 100 finite values per row.
- Parse elapsed time and maximum resident set size from the resource log.

## First Execution Step
Confirm current server memory/GPU state and start the bounded 50,000-row
checkpoint replay under the 30-minute timeout.
