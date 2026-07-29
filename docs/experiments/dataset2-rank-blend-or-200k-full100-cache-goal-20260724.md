# Goal Document: Dataset2 Rank Blend, Then 200k Full-100 Cache

## Go / No-Go

- **Judgment**: Go
- **Reason**: A cheap, matched-candidate blend scan can test whether existing models are complementary. If it fails a frozen stability gate, the next highest-leverage asset is a reusable recent-200k/full-100 Dataset2 feature cache.

## Target Outcome

Either identify a stable `0.01`-grid Dataset2 rank blend against the current champion, or, if no blend passes, start and complete a reusable cache containing the latest 200,000 supervised Dataset2 queries with all 100 candidates for both LightGBM and a future setwise reranker.

## Goal Definition

- **Type**: learning / technical / delivery
- **Boundary**:
  - Dataset2 only.
  - Existing checkpoint predictions for the blend scan.
  - Exact candidate identity alignment before any score combination.
  - Probability, row-normalized-score, and rank-percentile blends on a `0.01` grid.
  - Three fixed chronological validation slices.
  - Conditional recent-200k/full-100 cache construction.
- **Non-goals**:
  - Automatically submitting any blend.
  - Using public test labels or leaderboard scores as training labels.
  - Training the setwise reranker in this goal.
  - Modifying Dataset1.
- **Deferred work**:
  - LightGBM and setwise training consume the new cache only after it is complete and validated.
- **Verification rule**:
  - A blend is evaluated only when both models score the same query rows and candidate IDs.
  - The blend winner is selected without reading the final chronological slice.
  - A stable blend must improve full validation MRR by at least `+0.001` and not decrease any of three chronological slices versus the champion.
  - If no blend passes, the cache must contain exactly `200,000 × 100 × feature_count` float32 features plus query/candidate identity sidecars and a manifest with hashes.
- **Evidence source**: RED/GREEN tests, candidate-ID hashes, frozen selection report, temporal-slice MRR, cache manifest, array shapes, and SHA-256 hashes.
- **Pass criteria**: One of:
  1. a blend passes the frozen full/slice gate; or
  2. no blend passes and the recent-200k/full-100 cache is complete and reload-valid.
- **Confidence note**: The latest offline regression and online regression agreed in direction and similar magnitude, so the local gate is useful for rejection even though it does not predict the exact leaderboard score.
- **Judgment owner**: automated alignment and metric checks; the frozen gate determines whether cache construction is skipped or required.

## Current State

- Current champion online score: `1.3426473547970703`.
- Latest Two-Tower candidate online score: `1.3409701504529212`.
- Latest Two-Tower candidate offline blend MRR: `0.5415776145`, below champion `0.5428303297`.
- Existing blend utilities scan `0.01` probability weights but report only two temporal halves and assume one feature tensor can represent both checkpoints.
- Champion and Two-Tower caches may encode different learned tower features; score mixing without candidate-ID proof would be invalid.
- The current supervised cache uses 50,000 training groups with 32 candidates; a recent-200k/full-100 shared cache does not yet exist.

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---------------|----------|--------|
| Scan champion and current candidate at `0.01` | keep | Lowest-cost complementarity test. |
| Mix outputs from separate validation caches directly | remove | Candidate identities and learned feature semantics are not proven aligned. |
| Three temporal slices | keep | Protects against one-period gains. |
| Build recent-200k/full-100 cache immediately | reorder | Only pay this cost if blend fails. |
| Train LightGBM and setwise model in the same run | split | Cache completion is a reviewable reusable boundary. |

## Drift Diagnosis

- **Goal drift**: New tower training is unrelated to the immediate blend decision.
- **Phase drift**: Alignment must precede scanning; scanning must precede cache construction.
- **Validation drift**: Best full MRR alone is not stable improvement.
- **Compatibility drift**: Cache consumers require explicit candidate/query sidecars, not inferred row order.
- **Cleanup drift**: Existing unrelated result and checkpoint artifacts remain untouched.

## Priority Rationale

- Candidate alignment is the cheapest high-risk assumption to settle.
- Blend scanning is seconds once matched predictions exist.
- The expensive cache is built only after the cheap path is conclusively rejected.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|------|--------|--------|-------------------|
| Validation query order is reproducible from seed 60 | assumed | Required to compare checkpoints | Reconstruct and hash query identities |
| Candidate sampling is reproducible or sidecars can be regenerated | unresolved | Required for valid blending | Audit sampling and current caches before scoring |
| “Stable” means full `+0.001` and no slice regression | frozen | Determines branch | Enforce in a tested gate |
| Recent 200k means the latest 200k supervised rows before validation | frozen | Avoids random/uniform drift | Record exact event-index boundaries in manifest |

## Phases

### Phase 1: Matched blend evidence

- **Purpose**: Produce aligned champion and existing-model validation predictions.
- **Entry condition**: Checkpoints, feature caches, and sampling code are available.
- **Phase rules**:
  - No blending before candidate-ID equality is proven.
  - Do not select on the final chronological slice.
  - Keep Dataset1 out of scope.
- **Todos**:
  - [x] Inventory viable Dataset2 checkpoints and their validation artifacts.
    - **Surface**: server checkpoints/results/caches.
    - **Proof**: checkpoint and cache manifest table.
    - **Depends on**: none.
  - [x] Add alignment-aware three-slice rank-blend primitives through RED/GREEN.
    - **Surface**: fusion analysis code and unit tests.
    - **Proof**: failing alignment/grid/gate tests followed by passing tests.
    - **Depends on**: none.
  - [x] Generate or reconstruct matched validation predictions.
    - **Surface**: server validation artifacts.
    - **Proof**: exact query/candidate sidecar equality and hashes.
    - **Depends on**: inventory.
- **Exit proof**: Frozen aligned score matrices exist for champion and every viable alternate.
- **Stop condition**: Candidate identity cannot be reconstructed without retraining; fall back to one explicit matched replay rather than mixing incompatible caches.

### Phase 2: Frozen `0.01` scan

- **Purpose**: Decide whether an existing-model blend is robust enough to stop.
- **Entry condition**: Phase 1 alignment proof passes.
- **Phase rules**:
  - Select weights and normalization method on the first two slices only.
  - Open the third slice once.
  - Scan reference weights `0.00..1.00` inclusive at `0.01`.
- **Todos**:
  - [x] Scan probability, row-z-score, and rank-percentile blends.
    - **Surface**: cached score matrices.
    - **Proof**: complete 101-weight report per method/model.
    - **Depends on**: Phase 1.
  - [x] Apply the full/slice stability gate.
    - **Surface**: frozen report.
    - **Proof**: full delta and three slice deltas versus champion.
    - **Depends on**: scan.
- **Exit proof**: A reproducible pass/reject report selects the next branch.
- **Stop condition**: Any non-finite score, alignment mismatch, or final-slice use during selection.

### Phase 3: Conditional recent-200k/full-100 cache

- **Purpose**: Build the shared training asset only if no blend is stable.
- **Entry condition**: Phase 2 report is rejected.
- **Phase rules**:
  - Use the latest 200,000 supervised rows before validation, not a random sample.
  - Use 99 negatives from the test-candidate distribution without test labels.
  - Store query IDs, candidate IDs, feature schema, split boundaries, RNG state, and hashes.
  - Do not train LightGBM or setwise models yet.
- **Todos**:
  - [x] Add cache identity and recent-window contracts through RED/GREEN.
    - **Surface**: supervised cache/sampling code and tests.
    - **Proof**: tests reject missing sidecars and wrong event windows.
    - **Depends on**: rejected blend report.
  - [ ] Build the Dataset2 cache on the server. (running as PID `548792`)
    - **Surface**: memmapped feature arrays and manifest.
    - **Proof**: exact shape, sidecar alignment, finite-value scan, hashes, and reload.
    - **Depends on**: cache contracts.
- **Exit proof**: Cache is complete and safe for both LightGBM and setwise consumers.
- **Stop condition**: Projected disk/memory exceeds server headroom or candidate generation uses unavailable test labels.

## Dry-Run Findings

- The existing blend helper’s two-half report is insufficient for the requested stability rule.
- One feature cache cannot be assumed valid for checkpoints whose learned tower features differ.
- Test CSVs can be blended for a submission only after a weight is selected on aligned validation data; test outputs themselves provide no MRR signal.
- Cache identity sidecars are a prerequisite for safely reusing the 200k/full-100 asset across two learners.

## Final Validation

- Phase 2: verify 101 weights per method, frozen selection protocol, exact candidate alignment, full delta `>= +0.001`, and all three slice deltas `>= 0`.
- Phase 3 if required: verify `200000 × 100 × feature_count`, recent-window boundaries, identity sidecars, finite features, manifest hashes, and mmap reload.

## First Execution Step

Inventory server checkpoints/caches and write a failing test for the three-slice alignment-aware blend gate.

## Execution Update — 2026-07-24

- Corrected tie-neutral scan report:
  `result/dataset2_existing_rank_blend_tieneutral_seed60_20260724/rank-blend-report.json`
- Scan judgment: rejected; cache construction required.
- Best full-MRR delta: `+0.0008869814`, but the first chronological slice fell by
  `-0.0001098027` and the full gain missed `+0.001`.
- Best all-slices-non-decreasing delta: `+0.0005275346`, below the frozen threshold.
- Dataset2 training pool: 480,523 rows, so the exact recent window is sorted
  interaction rows `[1,722,091, 1,922,091)`.
- Server cache build:
  `cache/supervised_features/dataset2_recent200k_full100_seed60_20260724`
- Background PID: `548792`.
