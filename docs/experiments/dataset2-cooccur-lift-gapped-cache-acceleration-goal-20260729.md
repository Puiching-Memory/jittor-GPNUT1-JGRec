# Goal Document: Dataset2 cooccur-lift gapped cache exact parallel acceleration

## Go / No-Go

- **Judgment**: Go with proof gates.
- **Reason**: The current formal materializer is healthy but spends about 93% of
  feature time in a single-core `structure` path. The user authorized a faster
  retry only if quality is unchanged. The current PID remains the fallback
  until code-level exactness tests pass; cutover is allowed only through a
  first-batch real-data parity and throughput gate.

## Target Outcome

Materialize the exact same 277,035-row gapped cache with multiple CPU workers,
while preserving candidate order and all 63 base-feature values exactly, keeping
the plan-v2 hash chain unchanged, and retaining enough host capacity for
OS/sshd.

## Goal Definition

- **Type**: technical / operational / quality.
- **Boundary**: parallelize only the frozen structure-feature query path and
  retain bounded per-worker caches. Candidate sampling, input rows, feature
  schema, labels, fold boundaries, model parameters, seeds, selector, and
  external policy are unchanged.
- **Non-goals**:
  - Dropping or approximating any feature.
  - Changing the two candidate definitions or v1 baseline.
  - Reading metrics during materialization.
  - Opening external.
- **Deferred work**:
  - General cross-platform parallel execution; the formal acceleration target
    is the Linux remote host with `fork`.
  - Native C++ structure materialization.
- **Verification rule**: unit RED/GREEN plus a real remote first-batch A/B in
  one encoder process.
- **Evidence source**: pytest, first-batch candidate fingerprint, NumPy exact
  equality, measured wall-clock throughput, process/resource telemetry, cache
  artifact hashes.
- **Pass criteria**:
  - sequential and parallel first-batch candidate matrices are identical by
    construction;
  - all `(rows, 100, 63)` base feature values pass `np.array_equal`;
  - parallel full-feature wall time is at least `1.5x` faster on the real first
    batch;
  - at least two worker PIDs participate;
  - `MemAvailable` remains at least `max(25% RAM, 8 GiB)` and SSH banner remains
    available;
  - final report binds the unchanged plan/checkpoint hashes and says
    `external_scores_read=false`.
- **Confidence note**: exact equality is stronger than a metric tolerance and
  prevents numerical or ranking quality drift. The speed gate is measured on
  the same queries and candidates within one process, avoiding data/setup
  confounding.
- **Judgment owner**: automated parity/speed/resource gates; the frozen
  selector remains the only candidate judgment owner.

## Current State

- Formal cache PID `62698` is healthy at 53,248/277,035 rows but uses about one
  CPU core out of 32; conservative remaining time is 5–6 hours.
- `structure` accounts for roughly 2,130 of 2,299 profiled feature seconds.
- Host load is about 1.1, GPU is idle, RSS is about 8.1 GiB, and
  `MemAvailable` is about 23 GiB.
- The current partial directory is not resumable but will be preserved as
  evidence if superseded.
- Watcher PID `64371` must be stopped before terminating the cache PID, so it
  cannot misinterpret an authorized accelerated cutover as a cache failure.

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| Sequential gapped cache PID 62698 | keep as fallback until local/remote code tests pass; then supersede without deleting | Avoids abandoning a healthy run before the fast path is proven |
| Batch candidate generation | keep | Preserves RNG sequence and candidate fingerprint |
| Structure query evaluation | rewrite as source-stable forked workers | Uses idle CPU while keeping each query's exact pure calculation |
| Per-batch cache clearing | rewrite for worker-local bounded caches | Existing 256 MiB-per-worker LRU already bounds memory; cross-batch reuse is exact |
| Downstream duel/selector watcher | replace only after accelerated PID is healthy | Prevents duplicate or premature downstream runs |
| Plan-v2 / candidate configs | keep byte-for-byte | Acceleration is infrastructure, not a candidate rescan |

## Drift Diagnosis

- **Goal drift**: adding features, changing batch candidates, or tuning model
  parameters would optimize the experiment rather than its execution and is
  forbidden.
- **Phase drift**: killing the old process before parity tests would move
  cutover risk ahead of proof.
- **Validation drift**: matching metrics is insufficient; feature arrays must
  be exactly equal.
- **Compatibility drift**: worker-count one remains the original sequential
  behavior; no alternate feature semantics are introduced.
- **Cleanup drift**: do not delete `gapped-cache-v1` or unrelated remote files.

## Priority Rationale

- Exactness helpers and Linux fork behavior are proved before any process
  termination.
- The real-data parity/speed gate happens in the first accelerated batch, so a
  bad optimization fails within minutes rather than after a full run.
- Resource and SSH gates stay above speed; using all 32 cores is not a goal if
  it compromises operability.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| Structure features are pure for a fixed historical index and query | confirmed by code; test pending | Enables row partition/reassembly | Exact parity test |
| Linux `fork` shares the large read-only index copy-on-write | assumed | Makes 4 workers memory-feasible | Remote telemetry |
| Four workers provide at least 1.5x full-feature speedup | unresolved | Determines Go/No-Go at first batch | Real-data A/B gate |
| Worker-local caches stay within headroom | assumed from bounded LRUs | Protects sshd | Check PSS/MemAvailable |
| External can remain closed | confirmed | Preserves protocol | Pipeline contract |

## Phases

### Phase 1: Exactness contract RED

- **Purpose**: make any row-order or numerical drift fail automatically.
- **Entry condition**: current sequential PID remains healthy.
- **Phase rules**:
  - Tests before implementation.
  - No remote process termination.
- **Todos**:
  - [x] Add a Linux-only synthetic parity test for source-partitioned structure
    features.
    - **Surface**: `tests/test_parallel_structure.py`
    - **Proof**: missing implementation produces the expected RED.
    - **Depends on**: none.
  - [x] Add parity/speed gate tests for exact equality and minimum speedup.
    - **Surface**: unit tests.
    - **Proof**: mismatches and sub-threshold speed are rejected.
    - **Depends on**: none.
- **Exit proof**: smallest pytest command fails only because the new exact
  parallel behavior is absent.
- **Stop condition**: behavior cannot be expressed without changing feature
  semantics.

### Phase 2: Minimal parallel GREEN

- **Purpose**: use multiple CPU processes only for structure queries.
- **Entry condition**: Phase 1 valid RED.
- **Phase rules**:
  - Stable source-to-worker partition.
  - Original row order restored before concatenation.
  - Candidate generation remains in the parent.
  - Worker caches remain byte-bounded.
- **Todos**:
  - [x] Implement the forked structure wrapper and lifecycle.
    - **Surface**: hybrid structure infrastructure.
    - **Proof**: exact synthetic parity and multiple worker PIDs.
    - **Depends on**: Phase 1.
  - [x] Add first-batch sequential-versus-parallel A/B gate to the gapped
    builder.
    - **Surface**: cache builder/report.
    - **Proof**: exact `(rows,100,63)` equality and speed report.
    - **Depends on**: wrapper.
- **Exit proof**: focused and existing cooccur/standard protocol tests pass;
  Ruff passes.
- **Stop condition**: any equality failure, deadlock, or unbounded cache.

### Phase 3: Controlled remote cutover

- **Purpose**: prove actual speed/resource behavior before committing to a full
  retry.
- **Entry condition**: local and remote focused tests GREEN; synced hashes
  match.
- **Phase rules**:
  - Stop watcher PID `64371` first.
  - Gracefully terminate PID `62698`; never delete `gapped-cache-v1`.
  - New output directory only (`gapped-cache-v2-parallel4`).
  - First batch must pass exact equality and `>=1.5x`.
- **Todos**:
  - [x] Synchronize exact file list and rerun remote tests.
    - **Surface**: remote source/test files.
    - **Proof**: SHA-256 and pytest.
    - **Depends on**: Phase 2.
  - [x] Perform controlled PID cutover and launch 4-worker retry.
    - **Surface**: remote process/result directory.
    - **Proof**: preserved v1 directory, new PID/log/status.
    - **Depends on**: synced GREEN.
  - [x] Inspect first-batch parity, speed, worker PIDs, memory, load, and SSH.
    - **Surface**: runtime telemetry.
    - **Proof**: machine-readable parity report plus `ps/free/uptime`.
    - **Depends on**: retry launch.
- **Exit proof**: accelerated process continues beyond the first batch with all
  pass criteria met.
- **Stop condition**: parity differs, speedup below 1.5x, fewer than two workers,
  memory reserve breached, or SSH banner degrades.

### Phase 4: Full accelerated cache and frozen continuation

- **Purpose**: finish the same cache faster and resume the preregistered duel.
- **Entry condition**: Phase 3 pass.
- **Phase rules**:
  - No parameter changes after the first-batch gate.
  - Verify every final artifact hash from cache-report.
  - Run duel and selector; external remains blocked.
- **Todos**:
  - [ ] Complete and verify the accelerated cache.
    - **Surface**: `gapped-cache-v2-parallel4`.
    - **Proof**: complete report, artifact hashes, external false.
    - **Depends on**: Phase 3.
  - [ ] Attach a replacement low-priority continuation watcher.
    - **Surface**: remote operational pipeline.
    - **Proof**: watcher binds the accelerated PID/path and stops after selector.
    - **Depends on**: accelerated PID.
- **Exit proof**: selector reaches selected/rejected and no external receipt
  exists.
- **Stop condition**: any frozen hash or protocol boundary changes.

## Dry-Run Findings

- Parallelizing the whole encoder would fork CUDA/Jittor work and increase
  quality/resource risk; only the pure NumPy structure tower is delegated.
- Generating candidates inside workers would change RNG ordering; candidates
  stay parent-generated.
- A standalone benchmark would rebuild the historical encoder and compete with
  the live job. The first accelerated batch instead computes sequential and
  parallel results in the same process, then proceeds only after exact parity.
- The existing watcher is bound to PID 62698 and must be replaced during
  cutover.

## Final Validation

- `uv run --no-sync pytest tests/test_parallel_structure.py
  tests/test_cooccur_lift.py tests/test_cooccur_lift_successor.py
  tests/test_standard_validation_protocol.py -q`
- Ruff on new/changed Python.
- Remote first-batch `np.array_equal`, speedup `>=1.5`, multiple worker PIDs,
  resource reserve, and 10-second SSH probe.
- Final cache/plan/checkpoint/artifact hash audit; no external artifacts.

## First Execution Step

Add the failing Linux fork parity and exact speed-gate tests while PID 62698
continues as the untouched fallback.
