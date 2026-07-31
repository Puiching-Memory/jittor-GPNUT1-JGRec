# Goal Document: Leakage-Free Dataset2 gnn_short OOF Fusion

## Go / No-Go

- **Judgment**: Go
- **Reason**: The previous validation-only replacement isolated the failure to train/validation representation mismatch; chronological candidate caches and the working listwise GNN trainer provide the prerequisites for honest OOF construction.

## Target Outcome

Build chronological out-of-fold `gnn_short` scores for Dataset2 fusion training, use the same listwise representation on final validation, retrain Setwise fusion, and compare it with the current champion on identical validation queries.

## Goal Definition

- **Type**: learning
- **Boundary**: Use the first 25,000 cached rows as burn-in; score the remaining 175,000 rows in seven expanding-window folds of at most 25,000 rows; replace only `gnn_short`; retrain Setwise and evaluate the established 0.80 Setwise/0.20 LightGBM blend.
- **Non-goals**:
  - Random K-fold or any fold trained on future events.
  - Fabricating OOF values for the burn-in rows.
  - Changing other towers, feature columns, or the champion.
  - Generating a submission package.
- **Deferred work**:
  - Final-test encoder/checkpoint integration.
  - More fold-size or burn-in sweeps.
- **Verification rule**: Every scored row must belong to exactly one fold whose training rows and graph cutoff precede the scored interval; candidate order remains positive-at-zero; only `gnn_short` changes.
- **Evidence source**: Fold-contract tests, fold manifests, cache isolation checks, full MRR, and three chronological slice MRRs.
- **Pass criteria**: OOF coverage is exactly rows `[25000, 200000)`, every fold satisfies `train_stop <= score_start`, and final fusion improves full MRR by at least `+0.002` with no slice decline.
- **Confidence note**: The first 25,000 rows are intentionally excluded rather than assigned leaky or representation-mismatched values; the resulting 175,000-row fusion set is smaller but honest.
- **Judgment owner**: Automated fold/cache contracts authorize training; the paired MRR gate authorizes any later final-test work.

## Current State

- Validation-only replacement produced fusion MRR `0.52406798` versus champion `0.54691782`, with all slices declining.
- The full 200,000-row training cache is chronological and has complete 100-candidate groups.
- The final validation listwise-GNN scores are already verified and cached.
- One full listwise GNN training run is fast on the server, but seven expanding folds plus Setwise retraining will consume material GPU/CPU time.

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---------------|----------|--------|
| Replace validation `gnn_short` only | rewrite | It created a train/validation representation mismatch |
| Keep original training cache | remove | Fusion must see the new representation during training |
| Current champion comparison | keep | It is the stable paired benchmark |
| `+0.002` and all-slices gate | keep | Prevents advancing unstable offline gains |
| Generate submission package | defer | Representation must first pass honest OOF validation |

## Drift Diagnosis

- **Goal drift**: The previous run measured distribution mismatch more than model quality.
- **Phase drift**: Validation replacement preceded construction of compatible training features.
- **Validation drift**: Standalone GNN MRR did not prove end-to-end fusion value.
- **Compatibility drift**: Old training `gnn_short` and new validation `gnn_short` shared a name but not a learned distribution.
- **Cleanup drift**: None; unrelated work remains out of scope.

## Priority Rationale

- Prove fold chronology before training because one future-trained fold would invalidate the entire experiment.
- Build and persist OOF scores before copying the 4.4GB subset cache, so a failed fold cannot publish a misleading complete cache.
- Retrain exactly one Setwise candidate before considering any hyperparameter sweep.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|------|--------|--------|-------------------|
| 25k burn-in is sufficient to initialize listwise GNN | assumed | Controls early-fold representation quality | Fold metrics will expose failure |
| 25k fold size balances temporal fidelity and cost | assumed | Produces seven models | Do not sweep until base run finishes |
| Existing final-validation listwise cache is exact | confirmed | Avoids rebuilding it | Verify hash and standalone MRR |
| Fusion training can use 175k instead of 200k rows | confirmed by design | Trades volume for leakage safety | Record row range and sidecars |

## Phases

### Phase 1: Freeze chronological OOF contract

- **Purpose**: Make leakage mechanically impossible.
- **Entry condition**: Goal document exists.
- **Phase rules**:
  - Write RED before implementation.
  - Every fold trains only on a strict prefix.
- **Todos**:
  - [ ] Add fold-plan tests for coverage, disjointness, and chronology.
    - **Surface**: `gnn_listwise` tests/helper
    - **Proof**: RED then GREEN focused test
    - **Depends on**: none
- **Exit proof**: Fold plan covers `[25000, 200000)` once and satisfies chronology.
- **Stop condition**: Any requested row requires future or self labels.

### Phase 2: Train folds and publish OOF cache

- **Purpose**: Produce honest new-representation training features.
- **Entry condition**: Fold contract is green.
- **Phase rules**:
  - Fold early stopping uses only a tail split inside that fold's training prefix.
  - The scored fold never influences training or early stopping.
  - Publish cache only after all seven folds complete.
- **Todos**:
  - [ ] Train seven expanding-window listwise GNN models and score held-out rows.
    - **Surface**: server models, score memmap, fold manifest
    - **Proof**: per-fold cutoff/metric/hash records
    - **Depends on**: Phase 1
  - [ ] Copy rows `[25000, 200000)` and replace only `gnn_short`.
    - **Surface**: OOF training cache and sidecars
    - **Proof**: changed-column and row-alignment checks
    - **Depends on**: all folds
- **Exit proof**: Complete OOF manifest and isolated 175k cache.
- **Stop condition**: Non-finite loss/scores, missing coverage, overlap, or column drift.

### Phase 3: Retrain and gate fusion

- **Purpose**: Test whether aligned representation improves the end-to-end ranker.
- **Entry condition**: Verified OOF training cache and final validation cache exist.
- **Phase rules**:
  - Use fixed seed/config and 0.80 Setwise weight.
  - Do not tune on slice 2 and do not package.
- **Todos**:
  - [ ] Retrain Setwise on OOF features and evaluate final validation.
    - **Surface**: model and evaluation report
    - **Proof**: full/slice paired metrics
    - **Depends on**: Phase 2
- **Exit proof**: Gate decision and exact deltas versus champion.
- **Stop condition**: Cache mismatch or numerical failure.

## Dry-Run Findings

- A conventional random OOF split is label-disjoint but temporally leaky, so it is rejected.
- The first fold cannot be honestly scored with the same listwise representation because it supplies the initial supervision. Those 25,000 rows must be excluded.
- Internal early stopping must be drawn from each training prefix, not from the fold being scored.

## Final Validation

Require focused tests and Ruff, a seven-fold manifest with strict temporal inequalities, an isolated 175k training cache, and a paired champion report using the unchanged 20k validation rows.

## First Execution Step

Add and run the failing expanding-window fold-plan contract test.
