# Goal Document: Dataset2 K=512 Full Predict

## Go / No-Go
- **Judgment**: Go
- **Reason**: The 50,000-row smoke completed successfully with no OOM or swap,
  and the exact checkpoint has three prior full-run records around 19.4–19.8
  minutes.

## Target Outcome
Run the frozen K=512 checkpoint through all 153,420 Dataset2 test rows and
preserve the complete prediction CSV, execution log, memory log, and GNU time
resource report on the server.

## Goal Definition
- **Type**: operational delivery
- **Boundary**: Dataset2 only; existing checkpoint SHA-256
  `0b0a846cb6c7f5b5403a75a04ed12707340f024dee99a27cded3258bb108a7aa`;
  K=512 for both prediction truncation limits; batch size 2,048.
- **Non-goals**:
  - Do not retrain or modify the checkpoint.
  - Do not open external evaluation.
  - Do not treat an automatically created Dataset2-only archive as a complete
    competition submission.
- **Deferred work**:
  - Corrected v2 gapped folds and full-origin refit.
  - Local download or submission packaging unless separately requested.
- **Verification rule**: Complete the full checkpoint replay with exit status
  zero and validate exactly 153,420 finite rows with 100 candidate scores each.
- **Evidence source**: Full CSV, CLI log, memory log, GNU time report, process
  exit status, and checkpoint configuration.
- **Pass criteria**: Full output is structurally valid, the process does not
  OOM or swap, and all artifacts remain saved under one uniquely named result
  directory.
- **Confidence note**: The smoke measured 12.18 GiB peak RSS on a 31 GiB
  server; bounded prediction caches and prior full replays provide strong
  evidence that full execution is safe.
- **Judgment owner**: CLI structural validation plus artifact and process
  checks.

## Current State
- Dataset2 has 153,420 test rows.
- The 50,000-row K=512 smoke completed in 14:10 wall time, with 12.18 GiB peak
  RSS and zero swap.
- Three historical full K=512 replays completed in 19:24–19:45.
- The server was idle enough for the smoke and had substantial memory headroom.

## Plan Rewrite Notes
| Existing item | Decision | Reason |
|---------------|----------|--------|
| 50,000-row bounded replay | completed | Established memory safety and measured throughput |
| Time extrapolation only | replace | User now authorizes the full replay |
| No packaging | keep | This task saves prediction artifacts but does not authorize submission |

## Drift Diagnosis
- **Goal drift**: The former learning-only goal no longer delivers the requested
  full output.
- **Phase drift**: None; execution can move directly from the completed smoke
  to the full replay.
- **Validation drift**: Completion is tied to full row/column/finite checks, not
  merely process termination.
- **Compatibility drift**: None; the frozen checkpoint configuration is reused.
- **Cleanup drift**: No unrelated code or experiment changes are included.

## Priority Rationale
- Reuse the exact smoke-proven command path and checkpoint.
- Preserve evidence in a fresh result directory so the smoke and full output
  cannot overwrite one another.

## Assumptions and Open Decisions
| Item | Status | Impact | Owner / Next step |
|------|--------|--------|-------------------|
| Server remains healthy for roughly 25 minutes | assumed | A host interruption could stop the replay | Save logs continuously and inspect final status |
| Batch size remains 2,048 | confirmed | Matches the smoke and historical runs | CLI contract |
| Dataset2-only output is not a submission package | confirmed | Prevents accidental use as a full submission | Explicit boundary |

## Phases

### Phase 1: Full checkpoint replay
- **Purpose**: Generate the complete Dataset2 prediction artifact.
- **Entry condition**: Smoke pass and source checkpoint/configuration confirmed.
- **Phase rules**:
  - Use all 153,420 rows with no `--limit-rows`.
  - Keep K=512 and batch size 2,048 unchanged.
  - Preserve stdout/stderr, memory, and GNU time evidence.
  - Do not launch external evaluation or training.
- **Todos**:
  - [ ] Run the full Dataset2 checkpoint replay.
    - **Surface**: Existing CLI prediction path.
    - **Proof**: Exit status, full CSV, logs, and resource report.
    - **Depends on**: Completed smoke.
- **Exit proof**: Process exits zero and all named artifacts exist.
- **Stop condition**: OOM, non-finite prediction, invalid shape, missing
  checkpoint, or conflicting active replay.

### Phase 2: Validate and preserve
- **Purpose**: Prove the saved output is complete and usable as a Dataset2
  prediction artifact.
- **Entry condition**: Full replay exits successfully.
- **Phase rules**:
  - Validate exact row and column counts and finite values.
  - Report the server-side artifact path and measured resource use.
- **Todos**:
  - [ ] Validate output structure and record checksums.
    - **Surface**: Result directory.
    - **Proof**: Validator output and SHA-256.
    - **Depends on**: Phase 1.
- **Exit proof**: 153,420 × 100 finite score matrix plus preserved logs and
  checksums.
- **Stop condition**: Any mismatch requires diagnosis before handoff.

## Dry-Run Findings
- The smoke removed the main memory-risk uncertainty.
- A unique full-run name prevents collision with the 50,000-row smoke.
- Historical full runs are a better timing guide than prefix-only linear
  extrapolation because full source grouping has better cache locality.

## Final Validation
- Confirm exit status zero.
- Confirm exactly 153,420 rows and 100 finite values per row.
- Record SHA-256 of the full CSV and parse elapsed time/maximum RSS.

## First Execution Step
Check for a conflicting replay process, confirm current free memory, and start
the full replay in a durable result directory.
