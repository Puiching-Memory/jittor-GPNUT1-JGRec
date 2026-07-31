# Goal Document: Dataset2 New-Link Feature Gate

## Go / No-Go

- **Judgment**: Go for diagnosis; feature implementation is conditional.
- **Reason**: Dataset2 is profiled as `new_link_cold`, but that dataset-level label does not prove the current champion's remaining ranking errors are disproportionately caused by new positive edges. The exact champion must be segmented before paying for another feature-cache rebuild.

## Target Outcome

Establish whether the `1.3426473547970703` champion has a stable, material new-edge error concentration on its exact cached Dataset2 validation queries. Only if that gate passes, add the smallest non-duplicative new-link feature slice and test it against the champion under a frozen protocol.

## Goal Definition

- **Type**: learning, technical, quality, and delivery
- **Boundary**: Dataset2 only; exact champion MLP/LightGBM blend; existing 20,000-query/100-candidate validation cache; positive edge history measured against the same fixed training prefix used to encode validation; three chronological validation slices; the four user-proposed feature families considered only after diagnosis.
- **Non-goals**:
  - Infer repeat/new status from the top predicted candidate or from test labels.
  - Treat the positive candidate's repeat flag as an inference-time gating feature.
  - Implement all four feature ideas if current towers already encode their information.
  - Change Dataset1, towers unrelated to the selected feature, negative sampling, blend weights, or leaderboard submission based on a diagnostic alone.
- **Deferred work**:
  - Query-level segment fusion, neural architecture changes, and new negative-mining experiments.
  - Any package construction until a feature candidate passes a frozen temporal gate.
- **Verification rule**: Reconstruct the exact sampled validation positives from seed 60 and the checkpoint training configuration. Label a positive as repeat only when `(source, positive target)` exists in the fixed pre-validation history. Score the exact champion, compute reciprocal rank and top-1 error by repeat/new segment over the full set and three chronological slices, then apply the predeclared concentration gate.
- **Evidence source**: RED/GREEN unit tests for temporal edge labeling and error accounting; cache/checkpoint/event alignment checks; exact champion predictions; full and per-slice JSON report; feature redundancy audit if the gate passes.
- **Pass criteria**: Feature work starts only when all conditions hold: both segments contain at least 100 rows in every slice; new edges are at least 50% of validation rows; new edges contribute at least 65% of total reciprocal-rank regret; repeat-edge MRR exceeds new-edge MRR by at least 0.03 overall and by at least 0.02 in every chronological slice. These thresholds are frozen before champion scoring.
- **Confidence note**: Error concentration on the reused holdout establishes mechanism and addressable mass, not leaderboard lift. A subsequent feature still needs a separate fixed comparison and cannot be justified by a tiny local gain.
- **Judgment owner**: The executable concentration gate owns entry into feature work; focused tests own diagnostic correctness; the later temporal feature gate owns candidate construction; the user owns any submission.

## Current State

- Online champion score: `1.3426473547970703`.
- Dataset2's cached validation tensor has shape `20,000 x 100 x 63`; candidate zero is the supervised positive by construction.
- Current Dataset2 champion uses a global `MLP 0.07 + LightGBM 0.93` blend.
- The existing feature set already contains `src_activity`, four target-popularity time windows, pair time decays, co-occurrence structure, and ten source-profile/co-occurrence/cosine features.
- Therefore three proposed families may be explicit transformations of existing signals rather than new information; a redundancy audit is mandatory before implementation.
- The prior listwise MLP improved its expert by `+0.006684 MRR` but only improved the fixed champion blend by `+0.000207`, demonstrating that mechanism improvements can be suppressed or miscalibrated at ensemble level.

## Priority Rationale

- Exact error segmentation is cheap relative to regenerating Dataset2 tower features.
- Prevalence alone is insufficient: a dataset with mostly new edges will naturally have mostly new-edge errors, so the gate also requires a stable per-row difficulty gap.
- Redundancy is checked before feature work because LightGBM and the MLP can already form interactions from existing raw signals.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| Cached validation rows can be reconstructed from checkpoint config and seed 60 | assumed, verifiable | Required to attach true positive-edge history labels to cached scores | Alignment preflight |
| Validation encoder uses one fixed prefix, not earlier validation events | confirmed from `_learn_fusion` | Defines repeat/new without temporal leakage | Diagnostic test and implementation |
| Candidate zero is the positive only inside supervised cache | confirmed | Supports reciprocal-rank scoring; must never be treated as a test-file label | Cache contract |
| Existing source-profile features cover group similarity | confirmed by feature definitions | Likely removes the fourth proposal from implementation | Redundancy audit |
| A new feature can be appended without changing old feature semantics | unresolved until gate passes | Determines cache/checkpoint migration scope | Phase 2 design |

## Phases

### Phase 1: Exact Champion New-Edge Diagnosis

- **Purpose**: Decide whether new positive edges are truly the champion's dominant and harder error segment.
- **Entry condition**: Goal and thresholds are frozen.
- **Phase rules**:
  - Write RED tests before diagnostic production code.
  - Repeat/new is based only on the positive edge and history strictly available to the validation encoder.
  - Report prevalence, MRR, top-1 error, and reciprocal-rank regret; do not use validation results to revise thresholds.
- **Todos**:
  - [x] Test and implement vectorized historical pair membership.
    - **Surface**: new-link diagnostic module and tests.
    - **Proof**: Repeated pairs are detected across duplicate events; future/query-only pairs remain new; inputs are not mutated.
    - **Depends on**: none.
  - [x] Test and implement segment error accounting and the frozen gate.
    - **Surface**: diagnostic module and tests.
    - **Proof**: Known ranks produce exact segment MRR/regret shares and reject unstable or undersized slices.
    - **Depends on**: pair membership.
  - [x] Reconstruct and score the exact cached Dataset2 validation rows.
    - **Surface**: server diagnostic report.
    - **Proof**: Row count/time range/config/cache identity and champion baseline MRR match the known `0.5428303297` value.
    - **Depends on**: tests green.
- **Exit proof**: Frozen JSON report declares pass or reject with every condition visible.
- **Stop condition**: Stop all feature work on alignment mismatch, missing segment support, unstable slice gaps, or failed concentration gate.

### Phase 2: Conditional Redundancy Audit and Feature Selection

- **Purpose**: Select only information not already represented in the 63 cached features.
- **Entry condition**: Phase 1 concentration gate passes.
- **Phase rules**:
  - Audit all four proposals against exact existing definitions before coding.
  - Select at most two minimal features for one experiment; do not bundle redundant transformations.
  - Freeze formulas, windows, normalization, and acceptance criteria before rebuilding cache.
- **Todos**:
  - [x] Map each proposal to existing fields and transformations.
    - **Surface**: experiment report and feature definitions.
    - **Proof**: Keep/derive/drop decision with file-level evidence for each proposal.
    - **Depends on**: Phase 1 pass.
  - [x] Freeze the smallest feature slice.
    - **Surface**: goal amendment and RED tests.
    - **Proof**: Exact leakage-safe formula and temporal boundary for each selected feature.
    - **Depends on**: redundancy audit.
- **Exit proof**: At least one non-duplicative, causally available feature survives; otherwise stop.
- **Stop condition**: Stop if every proposal is already recoverable from existing features or requires future information.

### Phase 3: Conditional Feature Implementation and Cache Rebuild

- **Purpose**: Add the selected new-link signal without changing existing feature behavior.
- **Entry condition**: Phase 2 freezes formulas and tests.
- **Phase rules**:
  - RED→GREEN for scalar, batched, snapshot/hydrate, and future-query paths.
  - Rebuild Dataset2 cache under a new content key; never overwrite the champion cache.
  - Train one fixed LightGBM/listwise configuration declared before validation scoring.
- **Todos**:
  - [x] Implement and verify selected features.
    - **Surface**: relevant tower, config names, snapshots, and tests.
    - **Proof**: Focused and regression tests; no future-event access.
    - **Depends on**: Phase 2.
  - [ ] Rebuild and validate cache.
    - **Surface**: server cache and manifest.
    - **Proof**: New key, expected shapes, finite values, old-column byte equality where applicable.
    - **Depends on**: implementation green.
- **Exit proof**: One traceable cache and one frozen candidate model exist.
- **Stop condition**: Stop on semantic regression, cache mismatch, or non-finite feature values.

### Phase 4: Conditional Temporal Evaluation and Candidate

- **Purpose**: Determine whether the feature improves the full champion, not just the new-edge slice.
- **Entry condition**: Phase 3 succeeds.
- **Phase rules**:
  - Compare against the exact champion on full validation, new/repeat segments, and three chronological slices.
  - No post-result parameter or blend scan.
  - Package only if every chronological slice improves and full MRR improves by at least `+0.002`.
- **Todos**:
  - [x] Run frozen evaluation and apply gate.
    - **Surface**: server report.
    - **Proof**: Full/per-slice/segment deltas and explicit gate result.
    - **Depends on**: Phase 3.
  - [ ] If accepted, overlay Dataset2, verify checkpoint/CSV/ZIP, and copy locally.
    - **Surface**: candidate checkpoint and result ZIP.
    - **Proof**: Dataset1 byte-identical, both states load, row counts and hashes match.
    - **Depends on**: evaluation pass.
- **Exit proof**: A validated local package exists or the direction is explicitly rejected without a package.
- **Stop condition**: No package on any slice regression or full delta below `+0.002`.

## Dry-Run Findings

- “新边错误多” can be a base-rate illusion, so error share alone cannot open the feature phase.
- Source-history target-group similarity already exists as co-occurrence, cosine, recent-cosine, and item2vec profile features.
- Target short/long popularity values already exist across four windows; an explicit growth contrast may help a linear-ish learner but is not automatically new information for LightGBM/MLP.
- Existing pair decays concern direct source-target history, while time-decayed two-hop evidence appears to be the clearest potentially new signal.
- Reconstructing validation samples must consume the same seed-60 train sampling before validation sampling; otherwise cached rows and positive labels silently misalign.

## Final Validation

- Focused RED/GREEN tests and Ruff.
- Champion/cache/event alignment reproduces baseline full MRR `0.5428303297`.
- Frozen concentration conditions reported full and by three chronological slices.
- Only if passed: redundancy report, new cache manifest, fixed full/per-slice comparison, checkpoint round-trip, CSV/ZIP integrity, and hashes.

## First Execution Step

Add a failing test for historical pair membership and for a synthetic score matrix where new-edge errors dominate overall but fail one chronological-slice difficulty condition.

## Phase 1 Result and Protocol Amendment

- The exact champion/cache/event alignment reproduced MRR `0.5428303297309955` with zero numerical delta.
- All 20,000 validation positives are new edges relative to the fixed pre-validation history. Repeat-positive rows are exactly zero in the full set and every slice.
- Consequently 100% of the 12,597 Top-1 errors and 100% of reciprocal-rank regret occur on new edges. The original comparative gate returned false because its minimum repeat-row support is intentionally unmet; it is impossible to estimate “new minus repeat” difficulty on this dataset.
- This is not treated as a comparative-gate pass. It does satisfy the user's literal prerequisite that the champion's Dataset2 errors are concentrated on new edges, so the next step is limited to a cheap cached-feature derivation test. The expensive time-decayed two-hop rebuild remains blocked until that cheap test passes.

## Conditional Redundancy Audit

| Proposed family | Existing evidence | Decision before candidate metrics |
|---|---|---|
| Source activity × target recent growth | `src_activity` and four target-window popularity shares exist, but their explicit multiplicative interaction does not | Keep one derived cross |
| Time-decayed two-hop path | `cooccur_score` is count-based; direct pair decays are not two-hop decays | Defer; genuinely new but requires structure-state/cache rebuild |
| Short-term growth relative to long-term heat | Four window shares exist; explicit short/long contrast does not | Keep one derived ratio and merge with the first family |
| Source history group vs candidate similarity | Ten co-occurrence/cosine/recent-cosine/item2vec source-profile fields already implement it | Drop as duplicate |

The frozen cheap feature slice is exactly two columns derived from the existing cache: `growth = log1p(target_pop_share_w001 / max(target_pop_share_w100, 1e-12))` and `src_activity * growth`. Dataset2 LightGBM remains the only retrained expert, with the champion's `lr003`, 308 rounds, and MLP weight 0.07 fixed before validation scoring.

## Execution Result

- **Status**: Rejected; no cache rebuild, inference integration, checkpoint overlay, or submission ZIP was performed.
- Exact champion alignment passed with full MRR `0.5428303297309955` and absolute delta zero.
- All 20,000 sampled validation positives were new edges relative to the fixed training prefix. The 12,597 Top-1 errors and all `9143.3934` reciprocal-rank regret therefore occurred on new edges; there was no repeat-positive control segment.
- The two cached derived features were trained once with fixed Dataset2 `lr003` LambdaRank for 308 rounds and MLP weight 0.07. Training took 9.70 seconds.
- Candidate fixed-blend MRR was `0.5410983356`, delta `-0.0017319941` versus the champion.
- Chronological slice deltas were `-0.0024564436`, `-0.0015005952`, and `-0.0012388695`; every interval regressed.
- LightGBM used both derived fields (`463` and `227` splits), so the rejection is not caused by unused columns. The explicit transforms added misleading/redundant signal rather than useful information.
- Reports and the rejected model were copied locally under `result/dataset2_new_link_diagnosis_champion_20260723/` and `result/dataset2_new_link_growth_lr003_seed60_20260723/`; the model SHA-256 matched its report.
- **Judgment**: Dataset2 is structurally an all-new-edge task, but explicit combinations of existing activity/popularity windows do not help. Do not implement group-similarity duplicates or package this model. Time-decayed two-hop evidence remains the only materially new proposal, but its cache rebuild is not justified by this failed cheap gate without a separate explicit decision.
