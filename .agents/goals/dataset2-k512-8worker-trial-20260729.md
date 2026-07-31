# Goal Document: Dataset2 K512 8-Worker Trial

## Go / No-Go
- **Judgment**: Go
- **Reason**: The 4-worker run reached the first sequential structure batch
  before any completed cache batch was written, so a controlled 4-versus-8
  trial can still select the faster exact path without invalidating a
  completed artifact.

## Target Outcome
Resume the unchanged K=512 automatic pipeline with an evidence-driven first
batch trial that selects 8 workers only when its output is byte-identical to
both sequential and 4-worker output and its measured time improves materially
over 4 workers.

## Goal Definition
- **Type**: operational / quality / performance
- **Boundary**: Only the structure/source-profile process count and its
  first-batch selection gate may change.
- **Non-goals**:
  - No change to K, rows, candidates, features, models, heads, folds, seeds,
    weights, epochs, precision, tolerances, or external policy.
  - No reuse of an incomplete cache batch.
- **Deferred work**:
  - General adaptive worker tuning for unrelated experiments.
- **Verification rule**: Sequential, 4-worker, and 8-worker first-batch
  features must have equal shapes/dtypes, exact array equality, and identical
  byte SHA-256. Eight workers must also exceed the frozen incremental speed
  threshold and retain the memory reserve.
- **Evidence source**: Unit tests, first-batch parity reports, worker PIDs,
  measured seconds, RSS/available-memory readings, and cache progress.
- **Pass criteria**: Select 8 workers only if all exactness checks pass,
  `four_worker_seconds / eight_worker_seconds >= 1.10`, at least eight worker
  PIDs are observed, and available memory remains at least 8 GiB.
- **Confidence note**: Exact feature bytes protect model inputs; the same
  downstream candidate schedule and learned-head contracts remain frozen.
- **Judgment owner**: The automatic first-batch gate selects 8 or falls back
  to 4.

## Current State
- The original controller was paused after 11 minutes while its first
  sequential structure batch was running.
- No completed cache batch or progress report exists.
- Two preallocated `.part` memmaps and the stopped run logs remain and will be
  archived, not deleted.
- The server returned to roughly 27 GiB available memory after the process
  group stopped.

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| Four-worker exact gate | keep as control arm | It is the proven production acceleration path |
| Fixed four-worker continuation | rewrite to controlled selector | User explicitly requested an 8-worker trial |
| Remaining automatic pipeline | keep unchanged | Performance trial must not alter decision quality |
| Interrupted run/parts | archive | Preserve evidence and avoid unsafe overwrite |

## Drift Diagnosis
- **Goal drift**: A blind `4 -> 8` configuration edit would test neither
  incremental speed nor fallback safety.
- **Phase drift**: Running a separate long smoke would duplicate the expensive
  encoder fit; the trial belongs in the first real batch.
- **Validation drift**: Process count alone is not success; exact bytes,
  incremental speed, and memory reserve are required.
- **Compatibility drift**: No alternate cache schema or downstream path is
  introduced.
- **Cleanup drift**: Historical artifacts are archived rather than cleaned up.

## Priority Rationale
- Preserve quality first through exact triple parity.
- Decide 4 versus 8 before any completed cache row is committed.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| Eight forked workers fit in memory | confirmed | 13.28 GiB remained at the decision point | Runtime 8 GiB reserve gate passed |
| Eight workers beat four by 10% | confirmed | 8 workers were 1.731x faster than 4 | First-batch timing gate passed |
| Candidate RNG remains unchanged | confirmed by design | Protects query alignment | Same queries scored by all arms |

## Phases

### Phase 1: Test the selector
- **Purpose**: Freeze evidence-based selection before implementation.
- **Entry condition**: Original process group stopped.
- **Phase rules**:
  - RED must fail because 8-worker selection behavior is absent.
- **Todos**:
  - [x] Add a test for exact, faster 8-worker selection and slower-trial
    fallback.
    - **Surface**: parallel structure policy.
    - **Proof**: focused RED/GREEN pytest.
    - **Depends on**: none.
- **Exit proof**: Focused tests pass.
- **Stop condition**: Selection can accept non-exact evidence.

### Phase 2: Integrate and relaunch
- **Purpose**: Run sequential/4/8 on one real first batch and continue with
  the selected pool.
- **Entry condition**: Phase 1 green.
- **Phase rules**:
  - Archive interrupted artifacts first.
  - Do not relax the existing sequential parity threshold.
- **Todos**:
  - [x] Integrate controlled 4/8 trial and memory gate.
    - **Surface**: cache builder and automatic controller.
    - **Proof**: Ruff, related tests, dry run.
    - **Depends on**: Phase 1.
  - [x] Relaunch and wait for the real first-batch decision.
    - **Surface**: remote automatic run.
    - **Proof**: parity report and observed active worker count.
    - **Depends on**: integration.
- **Exit proof**: Cache progress records selected worker count and all trial
  evidence.
- **Stop condition**: Any byte mismatch, fewer than expected worker PIDs,
  memory reserve below 8 GiB, or 8-worker speedup below 1.10.

## Dry-Run Findings
- Restart costs another roughly six-minute encoder fit but no completed cache
  work.
- Keeping both process pools alive would inflate memory; the 4-worker pool
  must close before the 8-worker pool starts.
- If 8 loses, recreating the 4-worker pool costs one batch but safely resumes
  the proven path.

## Final Validation
- Focused selector and parallel-structure tests: passed.
- Existing automatic-pipeline, successor, validation-protocol, and fusion
  regression tests: 61 passed.
- Real first-batch SHA-256 was identical for sequential, 4-worker, and
  8-worker arms:
  `a248fe58de56b13b5f75011c362c54ab5dba6e9634b825cd4d8a80d0b4759f88`.
- Four workers took 72.146 seconds; eight workers took 41.673 seconds,
  yielding a 1.731x incremental speedup.
- Eight distinct worker PIDs were observed, 13.28 GiB remained available at
  selection, and the normal second batch completed in 42.341 seconds.
- **Outcome**: 8 workers selected; the unchanged automatic pipeline
  continues with the 8-worker pool.

## First Execution Step
Add the failing selector tests before changing the cache builder.
