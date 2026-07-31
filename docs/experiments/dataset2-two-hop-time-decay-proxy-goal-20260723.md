# Goal Document: Dataset2 Two-Hop Time-Decay Proxy

## Go / No-Go

- **Judgment**: Go
- **Reason**: Dataset2 validation positives are 100% new edges, cached activity/growth transforms regressed, and time-decayed two-hop evidence is the only proposed signal not already represented. A 2,000-query standalone proxy can test incremental temporal value before any full cache migration.

## Target Outcome

Decide whether medium-horizon time decay improves positive ranking over raw two-hop co-occurrence counts on a deterministic, chronologically representative 2,000-query Dataset2 proxy. Stop without production integration if it does not improve every time slice and full tie-neutral MRR by at least 0.01.

## Goal Definition

- **Type**: learning, technical, and quality
- **Boundary**: Dataset2 only; exact seed-60 sampled validation positives; 2,000 evenly spaced rows across the 20,000 cached-validation event identities; 31 negatives per query sampled from public unlabeled test-candidate frequencies; fixed pre-validation interaction prefix; latest 64 unique source-history targets; co-occurrence history limit 128; one fixed decay horizon `tau = 0.05 * prefix_time_span`; standalone raw-count versus decayed-sum ranking.
- **Non-goals**:
  - Modify the champion, structure tower, inference, checkpoint, submission, or existing cache.
  - Tune decay horizons, history limits, negative counts, candidate sources, or score transforms after metrics are read.
  - Claim leaderboard lift from this proxy.
  - Use A/B labels or positive positions from test files.
- **Deferred work**:
  - Production sparse decayed-cooccurrence state, cache-key migration, full-feature LightGBM training, and packaging.
- **Verification rule**: Reconstruct exact validation positives, deterministically choose 2,000 rows, sample 31 test-distribution negatives, build only the item-item pairs required by those query groups, collect co-occurrence event times from the fixed prefix, and compare tie-neutral reciprocal ranks from raw counts and decayed sums over the full proxy and three chronological slices.
- **Evidence source**: RED/GREEN unit tests for co-occurrence timing, future exclusion, recent history, tie-neutral ranking, and gate logic; frozen server configuration; pair/coverage counts; full/per-slice metrics and report.
- **Pass criteria**: At least 20% of queries have a nonzero two-hop score for some candidate; decayed tie-neutral MRR exceeds raw-count tie-neutral MRR in every chronological slice; full improvement is at least `+0.01`. All conditions are frozen before proxy scores are computed.
- **Confidence note**: This proves only that time carries incremental information conditional on the same two-hop paths. Candidate sampling and standalone scoring differ from the final ensemble, so a pass authorizes a full feature experiment rather than a submission.
- **Judgment owner**: Tests own temporal correctness; the frozen proxy gate owns whether full cache work starts.

## Current State

- Champion Dataset2 validation MRR is `0.5428303297309955` on 20,000 cached queries.
- All 20,000 validation positives are unseen `(source,target)` pairs relative to the fixed prefix.
- Existing `cooccur_score` stores raw two-hop count; source-profile features provide count/cosine and recent-source-history variants but no decay by the co-occurrence event time.
- Existing supervised encoders use future-only compact co-occurrence counts, so production time decay would require new temporal aggregate state rather than a trivial column transform.
- The cached growth-ratio experiment regressed all three slices, so no other feature change is bundled into this proxy.

## Priority Rationale

- Measure whether event time changes two-hop ordering before engineering a persistent sparse float map and rebuilding expensive features.
- Generate only pair histories needed by 2,000 query groups, limiting memory and runtime.
- Use tie-neutral ranks because standalone two-hop scores contain many zeros; candidate-zero-favored tie handling would fabricate signal.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| `0.05 * graph_span` is a reasonable medium horizon | confirmed design | Avoids a decay sweep; matches an existing structure horizon | Frozen proxy config |
| Latest 64 unique source targets preserve enough group context | confirmed design | Bounds required-pair count | Coverage report |
| 31 test-frequency negatives approximate the training competition | proxy assumption | Limits external validity | A pass still requires full-cache evaluation |
| Tie-neutral MRR is appropriate for sparse standalone scores | confirmed design | Prevents positive-at-zero leakage | Unit test |
| Required-pair collection can finish within the server memory budget | assumed | Main feasibility risk | Progress and pair-count audit |

## Phases

### Phase 1: Temporal Two-Hop Contract

- **Purpose**: Define the exact event time and ranking semantics before touching full data.
- **Entry condition**: Goal configuration and thresholds are frozen.
- **Phase rules**:
  - RED tests precede implementation.
  - A co-occurrence event is timestamped when the second previously unseen target appears in a source sequence.
  - Events at or after query time are excluded.
  - Item pairs are canonical and symmetric.
- **Todos**:
  - [x] Test and implement required-pair co-occurrence event collection.
    - **Surface**: pure NumPy/Python proxy module and tests.
    - **Proof**: Duplicate targets, history eviction, symmetry, and event times match hand calculations.
    - **Depends on**: none.
  - [x] Test and implement raw/decayed two-hop scores and tie-neutral MRR.
    - **Surface**: proxy module and tests.
    - **Proof**: Future events are excluded; decay equals an analytical reference; ties receive average rank.
    - **Depends on**: event collection.
  - [x] Test and implement the frozen continuation gate.
    - **Surface**: proxy module and tests.
    - **Proof**: Coverage, every-slice improvement, and `+0.01` full delta are all required.
    - **Depends on**: ranking metrics.
- **Exit proof**: Focused tests and Ruff pass locally and on Linux.
- **Stop condition**: Stop if causal event-time semantics cannot match the current co-occurrence construction.

### Phase 2: Frozen 2,000-Query Server Proxy

- **Purpose**: Measure incremental temporal signal at bounded cost.
- **Entry condition**: Phase 1 green; exact validation reconstruction still aligns to 20,000 rows.
- **Phase rules**:
  - Write configuration before scoring.
  - Use exactly 2,000 evenly spaced validation identities, 31 negatives, history 64, co-occurrence limit 128, and tau 0.05.
  - No parameter expansion after results.
- **Todos**:
  - [x] Build candidate groups and required canonical item pairs.
    - **Surface**: server diagnostic script.
    - **Proof**: Unique query rows, candidate uniqueness, pair count, checksums, and chronological ranges.
    - **Depends on**: Phase 1.
  - [x] Stream the prefix into required-pair event histories and score.
    - **Surface**: server report.
    - **Proof**: Runtime, coverage, nonzero rates, raw/decayed full and per-slice tie-neutral MRR.
    - **Depends on**: required pairs.
- **Exit proof**: Report declares pass/reject under the frozen gate.
- **Stop condition**: Stop on memory pressure, event/candidate alignment mismatch, coverage below 20%, any slice regression, or full delta below `+0.01`.

### Phase 3: Conditional Full Feature Goal

- **Purpose**: Open production implementation only when the proxy proves incremental time value.
- **Entry condition**: Phase 2 passes all conditions.
- **Phase rules**:
  - Create a separate goal before production edits.
  - Preserve the original raw `cooccur_score`; add one temporal aggregate only.
  - Full cache and ensemble still require the existing `+0.002` all-slice gate.
- **Todos**:
  - [x] If passed, write the production temporal-state/cache migration goal.
    - **Surface**: new goal document.
    - **Proof**: state shape, snapshot/hydrate, cache key, prediction cost, and rollback are specified.
    - **Depends on**: Phase 2 pass.
- **Exit proof**: Separate production goal exists, or proxy is rejected and no full work begins.
- **Stop condition**: No production implementation on proxy failure.

## Dry-Run Findings

- Standalone strict-`>` MRR would reward all-zero ties as rank one because the positive is stored at column zero; average tie ranks are mandatory.
- Building every item-item co-occurrence timestamp is unnecessary. Canonical required pairs from the bounded query set can filter collection during one prefix pass.
- The proxy's candidates will not be byte-identical to the cached 100 candidates, so the result is deliberately a mechanism test rather than a champion delta.
- A positive proxy result still leaves production memory design unresolved because future-only count maps currently discard timestamps.

## Final Validation

- Focused tests and Ruff locally and on Linux.
- Frozen config precedes score output.
- Exact validation-positive reconstruction and candidate/pair checksums.
- Coverage plus raw/decayed full and three-slice tie-neutral MRR.
- No checkpoint, cache, or submission mutation.

## First Execution Step

Add a failing hand-worked test for canonical required-pair co-occurrence event times, including a duplicate target and an event exactly at query time.

## Execution Result

- **Status**: Passed on the frozen server proxy.
- **Coverage**: 95.4% of queries had a nonzero two-hop score for at least one candidate; 89.9% of positives had nonzero evidence.
- **Raw-count tie-neutral MRR**: `0.5596467652965532`.
- **Time-decayed tie-neutral MRR**: `0.6005228755355890` (`+0.0408761102390358`).
- **Chronological slice deltas**: `+0.0409883699516439`, `+0.0341554149056185`, and `+0.0474944684358527`.
- **Scale/runtime**: 3,628,617 required pairs, 397,672 matched pairs, 3,475,220 co-occurrence events, 46.9 seconds.
- **Artifacts**: `result/dataset2_two_hop_decay_proxy_seed60_20260723/` contains the frozen config and report.
- **Safety**: No champion checkpoint, supervised cache, or submission package was changed.
- **Continuation**: Production work is governed by `dataset2-two-hop-time-decay-production-goal-20260723.md`.
