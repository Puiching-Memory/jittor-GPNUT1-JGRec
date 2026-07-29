# Goal Document: Dataset2 Two-Tower Full Reranker Integration

## Go / No-Go

- **Judgment**: Go, with a matched full-reranker validation gate before packaging.
- **Reason**: The 200k/listwise Two-Tower passed its standalone 20,000 x 100
  candidate gate in all three temporal slices. The raw-tower gain is only a proxy,
  so the next necessary proof is whether it improves the complete Dataset2
  MLP/LightGBM ensemble while every other component stays frozen.

## Target Outcome

Train a complete Dataset2 reranker whose only intended representation change is
the 200k/99-negative/listwise/MRR-stopped Two-Tower, compare it with the current
Dataset2 champion on identical full-100 validation queries, and produce a
checkpoint and local submission package only if the complete reranker improves
full MRR by at least `+0.002` without decreasing any of three temporal slices.

## Goal Definition

- **Type**: technical, quality, learning, and delivery.
- **Boundary**:
  - Dataset2 only during training and validation.
  - Fusion train/validation rows, fusion negative counts, feature schema, other
    towers, seed, dimensions, and prediction limits remain at champion values.
  - Dataset1 in any final package is copied unchanged from the current champion.
- **Non-goals**:
  - Do not add another tower or increase Two-Tower dimensions.
  - Do not tune LightGBM, segment gates, or ensemble weights in the same run.
  - Do not retrain Dataset1.
  - Do not submit to the leaderboard automatically.
- **Deferred work**:
  - Additional seeds, temperature tuning, and architecture changes.
  - B-list-specific adjustment.
- **Verification rule**: checkpoint/config tests first; then a fresh Dataset2
  build and an identical-candidate full/per-slice MRR comparison; finally
  checkpoint reload and ZIP validation only on gate pass.
- **Evidence source**: RED/GREEN tests, frozen config, candidate-ID hashes,
  build logs, checkpoint state, full/per-slice MRR report, and artifact hashes.
- **Pass criteria**:
  - Dataset2 Two-Tower config is exactly 200,000 events, 99 negatives,
    test-candidate ratio 1.0, listwise loss, MRR early stopping, and 64/64 dims.
  - Fusion remains 50,000 train events with 31 negatives and 20,000 validation
    events with 99 negatives.
  - Candidate and champion are evaluated on identical 20,000 x 100 candidate IDs.
  - Full ensemble MRR delta is at least `+0.002`.
  - All three chronological slice deltas are non-negative.
  - Reloaded checkpoint predictions match the in-memory candidate.
- **Confidence note**: This matched offline gate is much stronger than the
  standalone raw-tower proxy, but leaderboard calibration remains imperfect.
- **Judgment owner**: the automated matched full/per-slice MRR gate.

## Current State

- The standalone candidate stopped at epoch 18 with best internal MRR at epoch 15.
- Its raw Two-Tower external MRR was `0.46412481`; all three slices passed.
- The current champion checkpoint is
  `d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722.pkl`.
- Existing full-100 feature caches and candidate IDs can anchor an identical
  validation protocol, but candidate features must be rebuilt because the tower changed.
- The repository already supports separate fusion train/validation negative counts
  and tower-specific negative controls.

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---------------|----------|--------|
| Standalone Two-Tower gate | keep as entry evidence | It proves the new supervision can learn, but not final reranker gain |
| Immediate submission package | reorder after full reranker gate | Raw tower MRR cannot authorize submission |
| Full reranker integration | promote to current phase | It is now the highest-value uncertainty |
| New towers or dimension sweep | defer | They would destroy attribution |

## Drift Diagnosis

- **Goal drift**: treating the standalone `+0.449` raw-tower delta as leaderboard
  gain would skip the actual target metric.
- **Phase drift**: packaging before matched full-reranker validation would make
  artifact completion look like model success.
- **Validation drift**: comparing different candidate IDs would confound feature
  quality with candidate sampling.
- **Compatibility drift**: retraining Dataset1 would add an unrelated source of score change.
- **Cleanup drift**: no unrelated code cleanup belongs in this run.

## Priority Rationale

- First prove configuration and checkpoint composition, because a multi-hour run
  with the wrong Dataset2-only boundary would be wasted.
- Then run the full Dataset2 build once with frozen champion settings.
- Package only after the matched validation gate and reload proof.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|------|--------|--------|-------------------|
| Champion fusion uses train31/val99 | confirmed from prior full-MRR run | Keeps fusion supervision frozen | Verify from checkpoint/config before launch |
| Existing cache contains candidate IDs | confirmed in current code path | Enables identity checks | Assert hashes and shapes |
| Dataset1 can be copied byte-for-byte from champion checkpoint/package | assumed | Prevents Dataset1 drift | Add checkpoint composition test |
| Offline `+0.002` is sufficient to package | confirmed | Stop rule for costly packaging | Automated gate |

## Phases

### Phase 1: Integration Contract

- **Purpose**: prove the new tower settings enter Dataset2 without changing fusion
  negatives or Dataset1 state.
- **Entry condition**: standalone gate passed.
- **Phase rules**:
  - RED before integration code or launch-script changes.
  - No model training in this phase.
  - Legacy checkpoint loading must remain green.
- **Todos**:
  - [x] Add a regression test for exact full-reranker candidate configuration.
    - **Surface**: config/CLI tests.
    - **Proof**: RED fails on any coupled fusion/tower negative setting.
    - **Depends on**: none.
  - [x] Add a failing test for composing champion Dataset1 with candidate Dataset2.
    - **Surface**: checkpoint utility/test.
    - **Proof**: output checkpoint contains unchanged Dataset1 and replaced Dataset2.
    - **Depends on**: none.
- **Exit proof**: focused tests and checkpoint regressions pass.
- **Stop condition**: stop if Dataset1 cannot be preserved exactly.

### Phase 2: Full Dataset2 Build

- **Purpose**: rebuild all Dataset2 supervised features and fusion with only the
  new Two-Tower behavior.
- **Entry condition**: Phase 1 is green and frozen config is written.
- **Phase rules**:
  - Fusion train31/val99 and all non-Two-Tower settings stay frozen.
  - Save supervised feature and candidate IDs.
  - Run detached on the server.
- **Todos**:
  - [ ] Launch the fresh Dataset2-only build.
    - **Surface**: server log, feature cache, Dataset2 checkpoint.
    - **Proof**: completed build report and artifact hashes.
    - **Depends on**: Phase 1.
- **Exit proof**: candidate Dataset2 state, cache, and validation metrics exist.
- **Stop condition**: OOM, non-finite metrics, cache identity failure, or config drift.

### Phase 3: Matched Gate and Conditional Package

- **Purpose**: determine actual final-reranker value and hand back a usable artifact.
- **Entry condition**: Phase 2 completed successfully.
- **Phase rules**:
  - Compare on identical 20,000 x 100 candidates.
  - No weight tuning after seeing validation.
  - Package only on the predeclared gate.
- **Todos**:
  - [ ] Compare champion and candidate full/per-slice ensemble MRR.
    - **Surface**: evaluation report.
    - **Proof**: full delta and three slice deltas.
    - **Depends on**: Phase 2.
  - [ ] On pass, compose champion Dataset1 plus candidate Dataset2 checkpoint,
    reload, predict, validate ZIP, and copy artifacts locally.
    - **Surface**: checkpoint and `result.zip`.
    - **Proof**: hashes, row counts, reload equality, ZIP validation.
    - **Depends on**: matched gate pass.
- **Exit proof**: verified package on pass, or explicit rejection with no package.
- **Stop condition**: full delta below `+0.002`, any slice regression, or reload mismatch.

## Dry-Run Findings

- Running `jgrec-build` for both datasets would retrain Dataset1 and break attribution;
  the build must be Dataset2-only, followed by explicit checkpoint composition.
- Global `train_num_negatives=99` would unintentionally change fusion training;
  the tower must use its dedicated 99-negative override while fusion remains at 31.
- The standalone model was trained on a context prefix and cannot simply be copied
  into the final full-data encoder; the full reranker must retrain the tower.

## Final Validation

- Focused config/CLI/checkpoint tests and broader hybrid checkpoint regressions.
- Frozen server config and build log.
- Identical candidate-ID hashes.
- Full and three-slice MRR gate.
- Conditional checkpoint reload and ZIP validation.

## First Execution Step

Add RED tests that freeze the exact full-reranker config and champion-Dataset1 /
candidate-Dataset2 checkpoint composition behavior.
