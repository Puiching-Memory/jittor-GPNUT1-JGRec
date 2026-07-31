# Goal Document: Dataset2 gnn_short Listwise Fusion Evaluation

## Go / No-Go

- **Judgment**: Go
- **Reason**: The trained `gnn_short` artifact, frozen validation cache, champion checkpoint, and existing Setwise/fusion training paths are available.

## Target Outcome

Produce a leakage-free, same-validation-set comparison between the current Dataset2 champion and fusion models retrained after replacing only the validation cache's `gnn_short` feature with scores from the new listwise GNN.

## Goal Definition

- **Type**: learning
- **Boundary**: Replace only `gnn_short` in the 20,000 × 100 Dataset2 validation cache, retrain the existing Dataset2 Setwise/fusion paths with their established training data, and compare full MRR plus three chronological slices.
- **Non-goals**:
  - Changing any other cached feature.
  - Retraining the GNN, LightGBM feature encoder, or champion.
  - Generating a submission package.
- **Deferred work**:
  - Rebuilding final-test `gnn_short` features and checkpoint integration.
- **Verification rule**: The replacement cache preserves shape, schema, row order, candidates, and every non-`gnn_short` value; evaluation uses identical validation rows for candidate and champion.
- **Evidence source**: Contract tests, cache hashes/checks, full-candidate MRR, and three time-slice MRRs.
- **Pass criteria**: Report exact candidate-versus-champion deltas. Integration is authorized only if full MRR improves by at least `+0.002` and none of the three slices declines.
- **Confidence note**: This is an offline validation proxy, not a leaderboard guarantee; same-row paired comparison makes it suitable for deciding whether to continue.
- **Judgment owner**: The metric gate declares whether final-test integration is authorized.

## Current State

- The listwise `gnn_short` model completed at best epoch 16 with standalone full-100 validation MRR `0.4659887151`.
- The frozen validation cache has shape `(20000, 100, 63)` and contains one `gnn_short` column.
- Existing Dataset2 Setwise and fusion models provide established training/evaluation conventions.
- The worktree is dirty with prior experiment work; unrelated files must remain untouched.

## Priority Rationale

- Prove cache isolation before expensive model fitting because an accidental multi-column or row-order change would invalidate every metric.
- Score the new GNN once and reuse the replacement cache for both fusion candidates.
- Keep the `+0.002`/three-slice gate consistent with prior Dataset2 experiments.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|------|--------|--------|-------------------|
| Existing validation sidecars match the GNN model ID map | confirmed by prior GNN run | Required for deterministic rescoring | Runtime contract verifies again |
| Existing Setwise training cache remains valid | assumed | Avoids rebuilding the 5 GB training cache | Preflight verifies schema/hash |
| “Fusion” means the established Setwise and current fusion/reranker comparison paths | assumed | Keeps scope compatible with prior experiments | Report each candidate separately |

## Phases

### Phase 1: Freeze replacement behavior

- **Purpose**: Make the cache mutation auditable.
- **Entry condition**: Goal document exists.
- **Phase rules**:
  - Tests precede implementation.
  - Only the indexed `gnn_short` column may differ.
- **Todos**:
  - [ ] Add a failing replacement-contract test.
    - **Surface**: test and cache helper
    - **Proof**: focused RED failure
    - **Depends on**: none
- **Exit proof**: Focused test passes after minimal implementation.
- **Stop condition**: Cache schemas or row identities do not match.

### Phase 2: Build replacement validation cache

- **Purpose**: Score the new GNN and publish a verified cache.
- **Entry condition**: Replacement contract is green.
- **Phase rules**:
  - Use the trained best-epoch model and checkpoint ID map.
  - Preserve candidates and all non-GNN columns.
- **Todos**:
  - [ ] Generate `gnn_short` scores on the frozen 20k × 100 validation groups.
    - **Surface**: server cache artifact
    - **Proof**: shape, finite-value, and unchanged-column checks
    - **Depends on**: Phase 1
- **Exit proof**: Replacement report records hashes and exact changed column.
- **Stop condition**: Non-finite scores, positive/candidate mismatch, or non-target column drift.

### Phase 3: Retrain and compare fusion

- **Purpose**: Determine whether the representation change improves the complete ranking system.
- **Entry condition**: Verified replacement cache exists.
- **Phase rules**:
  - Use identical training and validation row definitions for candidate and champion.
  - Do not package regardless of result.
- **Todos**:
  - [ ] Retrain Dataset2 Setwise and established fusion candidate.
    - **Surface**: server experiment artifacts
    - **Proof**: completed model reports
    - **Depends on**: Phase 2
  - [ ] Compare full and three-slice MRR against champion.
    - **Surface**: evaluation report
    - **Proof**: paired deltas and gate decision
    - **Depends on**: retraining
- **Exit proof**: Evaluation report contains candidate/champion metrics and integration authorization.
- **Stop condition**: Validation rows differ or model fitting fails numerically.

## Dry-Run Findings

- Training features must not be modified: only validation needs the new `gnn_short` score because the GNN training cache already contains historical champion features and retraining a fusioner on mismatched train/validation distributions is a risk. The experiment must explicitly report this limitation.
- The new GNN model file stores model weights only; scoring must reconstruct the exact champion-matched graph and ID map.

## Final Validation

Run focused tests and Ruff, then require a server report with finite full/slice MRRs, identical row/candidate hashes, and the `+0.002`/no-slice-decline gate.

## First Execution Step

Inspect the existing Setwise/fusion scripts and artifact schemas, then add the RED test for isolated feature replacement.
