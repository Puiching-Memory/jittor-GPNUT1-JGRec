# Goal Document: Dataset2 Two-Tower 200k Listwise Experiment

## Go / No-Go

- **Judgment**: Go, staged through a standalone Two-Tower gate before any full reranker rebuild.
- **Reason**: Two-Tower is already the strongest learned representation block, while its
  current supervision is materially under-aligned with deployment: 50,000 sampled
  events, 31 negatives, pointwise sigmoid BCE, and loss-based early stopping. The
  proposed change improves its supervision without adding a tower or increasing the
  64-dimensional representation.

## Target Outcome

Implement a configurable listwise Two-Tower training path, then compare the current
Dataset2 Two-Tower against a candidate trained on 200,000 events with 99 test-distribution
negatives and complete-candidate MRR early stopping. Only if the standalone full-100
MRR improves consistently may the candidate advance to an expensive full reranker build.

## Goal Definition

- **Type**: technical, learning, quality, and delivery.
- **Boundary**: Dataset2 Two-Tower only; 64-dimensional embedding and hidden width;
  200,000 sampled training events; 99 negatives; test-candidate frequency distribution;
  group softmax/listwise loss; full-100 MRR early stopping; one matched chronological
  evaluation candidate tensor shared by baseline and candidate.
- **Non-goals**:
  - Do not add a new tower.
  - Do not increase embedding or hidden dimensions.
  - Do not change GNN, sequence, source-profile, structure, fusion, or Dataset1.
  - Do not use test labels; test candidate IDs/frequencies are inputs only.
  - Do not generate or submit a package from the standalone tower proxy.
- **Deferred work**:
  - Full Dataset2 reranker feature rebuild and fusion retraining.
  - Two-Tower architecture changes, temperature tuning, multi-seed sweeps, or dimension sweeps.
- **Verification rule**: RED/GREEN tests prove listwise loss, full-candidate MRR
  selection, tower-specific negative sampling, and backward-compatible defaults.
  A frozen server experiment compares baseline and candidate on identical 20,000
  chronological 100-candidate groups.
- **Evidence source**: focused tests, frozen-config JSON, epoch logs, model snapshots,
  full/per-slice standalone MRR report, and artifact hashes.
- **Pass criteria**:
  - Candidate uses exactly 200,000 sampled events, 100 candidates per group, listwise
    loss, MRR early stopping, and 64 dimensions.
  - Baseline and candidate score exactly the same external validation groups.
  - Candidate full standalone MRR improves by at least `+0.002`.
  - Each of three chronological validation slices is non-decreasing.
  - Only a passing candidate authorizes the later full reranker integration phase.
- **Confidence note**: Standalone tower MRR is an attribution-friendly proxy, not a
  leaderboard guarantee. A full fixed-blend validation gate remains required before
  packaging.
- **Judgment owner**: Automated standalone full/per-slice MRR gate, followed by the
  existing full-reranker gate if integration is authorized.

## Current State

- Current Two-Tower defaults are embedding/hidden dimension `64`, `50,000` maximum
  samples, `31` negatives, sigmoid BCE, and validation-loss early stopping.
- Two-Tower already accepts deterministic per-event negative seeds but does not receive
  the DatasetProfile test-candidate distribution.
- Global supervised `train_num_negatives` currently also controls Two-Tower negatives,
  so tower-specific negatives must be decoupled to keep fusion supervision frozen.
- The candidate profile already stores test candidate IDs and frequency weights without
  test labels.
- The prior full-100 LightGBM experiment showed candidate-distribution alignment alone
  is insufficient, increasing the value of improving the representation tower itself.

## Priority Rationale

- First prove the new objective and metric contracts in small tests.
- Then run a standalone matched tower comparison; it isolates Two-Tower quality without
  paying for multi-hour fusion feature generation.
- Defer full reranker integration until the tower clears a meaningful, temporally stable
  standalone gate.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|------|--------|--------|-------------------|
| Test candidate frequency is a legal transductive input | confirmed | Enables deployment-aligned negatives without labels | Reuse `DatasetProfile.test_candidate_counts` |
| 99 unique test-distribution negatives are available per group | assumed | Required for true 100-candidate groups | Validate every generated group; stop on fallback duplicates |
| 20,000 external validation groups are sufficient for the first gate | confirmed | Provides full and three-slice MRR | Freeze before either tower is trained |
| `+0.002` standalone MRR is a meaningful first-stage gain | confirmed | Prevents an expensive integration for noise-level movement | Automated gate |
| One seed is sufficient for package authorization | rejected | Standalone pass only authorizes integration | Require later full-reranker validation |

## Phases

### Phase 1: Training Contracts

- **Purpose**: Add the smallest backward-compatible controls for the proposed objective.
- **Entry condition**: Existing Two-Tower/config tests pass.
- **Phase rules**:
  - RED before production code.
  - Existing defaults remain BCE, loss early stopping, inherited negative count, and no
    tower-specific test-distribution override.
  - No architecture or dimension change.
- **Todos**:
  - [x] Add failing tests for listwise positive-at-zero loss and full-candidate MRR.
    - **Surface**: `tests/test_hybrid_two_tower.py`.
    - **Proof**: RED fails because the objective/metric helpers do not exist.
    - **Depends on**: none.
  - [x] Add failing tests for tower-specific 99 negatives, test-distribution ratio,
    objective, and early-stop metric propagation.
    - **Surface**: TrainingConfig/CLI and TwoTowerConfig tests.
    - **Proof**: RED fails on missing fields.
    - **Depends on**: none.
  - [x] Implement the minimal listwise/MRR/config path.
    - **Surface**: Two-Tower, config, ranker wiring, and CLI.
    - **Proof**: focused and broader hybrid tests pass.
    - **Depends on**: both RED slices.
- **Exit proof**: focused tests, Ruff, and Python compilation pass.
- **Stop condition**: Stop if old checkpoints cannot hydrate with default behavior.

### Phase 2: Frozen Standalone Comparison

- **Purpose**: Determine whether the supervision change improves the tower itself.
- **Entry condition**: Phase 1 is green.
- **Phase rules**:
  - Dataset2 only.
  - Freeze external validation groups before training either model.
  - Reset Jittor and NumPy seeds identically for baseline and candidate initialization.
  - Candidate differs only in max samples, negative count/distribution, objective, and
    early-stop metric.
- **Todos**:
  - [x] Reuse and hash identical 20,000 x 100 external validation candidate IDs
    from the frozen full-100 cache.
    - **Surface**: server experiment artifacts.
    - **Proof**: shape, positive-first, uniqueness, and SHA-256 report.
    - **Depends on**: Phase 1.
  - [x] Read the frozen current-baseline scores and train the proposed candidate.
    - **Surface**: Two-Tower snapshots and epoch logs.
    - **Proof**: frozen configs and model hashes.
    - **Depends on**: validation freeze.
  - [x] Score full and three chronological slices and apply the gate.
    - **Surface**: evaluation report.
    - **Proof**: deterministic pass/reject.
    - **Depends on**: both tower snapshots.
- **Exit proof**: A report declares `passed` or `rejected`; no submission package exists.
- **Stop condition**: Stop on invalid candidate groups, OOM, non-finite loss/scores, or
  any validation identity drift.

### Phase 3: Conditional Full Integration

- **Purpose**: Test whether a standalone tower gain transfers to the complete reranker.
- **Entry condition**: Phase 2 passes.
- **Phase rules**:
  - Replace only Two-Tower training behavior; keep every other tower and fusion setting frozen.
  - Reuse the established full/per-slice fixed-blend gate.
- **Todos**:
  - [ ] Rebuild Dataset2 supervised features and fusion with the candidate Two-Tower.
    - **Surface**: checkpoint/cache/evaluation artifacts.
    - **Proof**: fixed-blend full MRR `>= +0.002` and all three slices non-decreasing.
    - **Depends on**: Phase 2 pass.
  - [ ] Package only after checkpoint reload and prediction equality.
    - **Surface**: result ZIP.
    - **Proof**: Dataset1 preservation, row counts, hashes, and ZIP validation.
    - **Depends on**: full integration gate.
- **Exit proof**: Verified package on pass, or explicit rejection with no package.
- **Stop condition**: Any full-reranker regression or checkpoint mismatch.

## Dry-Run Findings

- Using global `train_num_negatives=99` would also change fusion supervision, so a
  tower-specific negative-count override is required for attribution.
- Test-distribution sampling requires wiring DatasetProfile into Two-Tower; copying test
  labels is neither required nor allowed.
- Internal MRR early stopping is not enough evidence by itself because its validation
  events participate in the sampled training pool split; a separate chronological
  20,000-group evaluation remains mandatory.
- A standalone tower gate avoids another multi-hour full feature build if the
  representation itself does not improve.

## Final Validation

- Focused RED/GREEN Two-Tower and config tests.
- Broader hybrid checkpoint, encoder, CLI, and negative-sampling regressions.
- Frozen standalone baseline/candidate report with exact candidate hashes and
  full/per-slice MRR.
- Conditional full-reranker gate only after a standalone pass.

### Standalone Result

- Finished after `1465.7` seconds; MRR early stopping selected epoch `15` and
  stopped at epoch `18`.
- Frozen baseline raw Two-Tower MRR: `0.01494646`.
- Candidate raw Two-Tower MRR: `0.46412481`.
- Full delta: `+0.44917835`.
- Candidate temporal slice MRRs: `0.46878611`, `0.46007202`, `0.46351699`;
  all three exceeded the frozen baseline slices.
- Gate status: passed. Full reranker integration is authorized; no submission
  package was generated by this standalone experiment.

## First Execution Step

Add failing tests for a positive-at-column-zero group-softmax loss and MRR-based
early-stop signal before modifying Two-Tower production code.
