# Goal Document: Same-Validation Checkpoint Blend and Dataset1 MLP Control

## Go / No-Go

- **Judgment**: Go
- **Reason**: Both checkpoints, the 100-candidate supervised feature cache, and both submission CSVs already exist on the server, so the comparison and control package can be produced without retraining any tower.

## Target Outcome

Evaluate the old `1.3402985197047874` checkpoint and the new `1.3411684997174933` checkpoint on the exact same 100-candidate validation rows, scan relevant blend weights at `0.01` resolution with temporal-slice evidence, and produce a validated submission ZIP whose Dataset1 prediction is pure new-checkpoint MLP while Dataset2 is byte-for-byte the current new champion prediction.

## Goal Definition

- **Type**: learning and delivery
- **Boundary**: Add tested offline MRR/blend analysis, run it against Dataset1 and Dataset2 caches on the server, generate Dataset1 MLP-only predictions from the new checkpoint, splice the current new Dataset2 CSV, validate, hash, and download the control ZIP.
- **Non-goals**:
  - Retraining GNN, sequence, two-tower, source-profile, MLP, or LightGBM models.
  - Changing candidate features, negative sampling, or the current champion artifact.
  - Automatically submitting to Educoder; the user owns submission.
- **Deferred work**:
  - Segment-aware gating and hard-negative mining.
  - Broader LightGBM hyperparameter sweeps.
- **Verification rule**: Pure-function tests constrain MRR, temporal slices, `0.01` grid coverage, and deterministic tie-breaking; server output must show both checkpoints evaluated on identical cache shapes; the control ZIP must pass structural and row/column validation and preserve Dataset2 bytes exactly.
- **Evidence source**: RED/GREEN pytest output, server analysis TSV/JSON, prediction logs, CSV SHA-256 hashes, ZIP validation, and final artifact SHA-256.
- **Pass criteria**: No training log is emitted; each dataset has 20,000 validation queries with 100 candidates; the scan includes 101 weights; Dataset1 MLP-only CSV is newly inferred; Dataset2 CSV hash equals the current champion Dataset2 CSV hash; the ZIP contains only `dataset1.csv` and `dataset2.csv` and is downloaded locally with matching hash.
- **Confidence note**: Common-cache validation makes model comparisons fair offline, but leaderboard score remains the final quality judge and is intentionally owned by the user's submission.
- **Judgment owner**: Tests and artifact validators declare implementation/delivery complete; the competition score declares whether the variant improves quality.

## Current State

- Old checkpoint: `checkpoints/ensemble_towers50_k512_d1d2_seed60_20260721.pkl` on the server.
- New checkpoint: `checkpoints/lgbm_fullmrr_v99_t31_towers50_k512_d1d2_seed60_20260722.pkl` on the server.
- New 100-candidate cache contains separate Dataset1 and Dataset2 train/validation arrays.
- Current blend search uses only eleven weights (`0.0` through `1.0` by `0.1`).
- Dataset1 new-checkpoint MLP MRR is `0.79425`; LightGBM MRR is `0.78540`; their selected blend ties the MLP.
- Dataset2 new-checkpoint MLP MRR is `0.52859`; LightGBM MRR is `0.54044`; their selected blend reaches `0.54089`.
- Checkpoints are about 5 GB each, so evaluation must load one dataset snapshot at a time and release it before the next load.

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---------------|----------|--------|
| Old/new checkpoint common-validation evaluation | keep | It attributes the online change on an identical proxy distribution. |
| `0.01` fine blend scan | rewrite | Include full and temporal-half MRR plus deterministic tie behavior, not only a best full-split number. |
| Dataset1 pure MLP control package | keep | Dataset1 showed no validation gain from LightGBM, making this the cleanest low-cost control. |
| Immediate broad LightGBM sweep | remove | It would confound attribution before the control is measured. |

## Drift Diagnosis

- **Goal drift**: Retraining towers or changing features would no longer isolate fusion behavior.
- **Phase drift**: Package generation must follow common-cache evaluation, not run concurrently with an unverified override path.
- **Validation drift**: A higher full-split MRR alone is insufficient; early/late temporal halves must be reported.
- **Compatibility drift**: The current champion ZIP and checkpoints remain immutable; variants use new run names.
- **Cleanup drift**: No unrelated CLI, model, or documentation cleanup belongs in this experiment.

## Priority Rationale

- Prove deterministic weight-scanning behavior before loading multi-gigabyte checkpoints.
- Evaluate checkpoints before packaging so the artifact has an interpretable hypothesis.
- Reuse the new checkpoint's final encoder for Dataset1 inference and copy the champion Dataset2 CSV unchanged.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|------|--------|--------|-------------------|
| Old and new checkpoints share the same 63-feature layout | assumed | Required for scoring the same cached tensors | Verify snapshot feature names before inference. |
| Cache manifests can be mapped to datasets from array shapes and checkpoint config | assumed | Prevents evaluating the wrong dataset | Verify interaction-derived cache key or log-recorded key. |
| Dataset2 control output must be byte-identical to the new champion CSV | confirmed | Isolates the Dataset1 change | Compare SHA-256 before packaging. |
| User will submit the final ZIP | confirmed | No platform login automation is needed | Deliver local path and hash. |

## Phases

### Phase 1: Tested Analysis Primitive

- **Purpose**: Make MRR comparison and `0.01` blend selection deterministic and independently testable.
- **Entry condition**: Target metrics and tie behavior are specified.
- **Phase rules**:
  - Write a failing test before each production behavior.
  - Pure functions only; no checkpoint, Jittor, filesystem, or network dependency.
  - Weight endpoints must be included and ties must prefer the explicitly supplied simpler/reference side.
- **Todos**:
  - [ ] Test and implement full/early/late MRR reporting.
    - **Surface**: analysis module and unit tests.
    - **Proof**: Focused pytest RED then GREEN.
    - **Depends on**: none.
  - [ ] Test and implement 101-point probability blend scanning.
    - **Surface**: analysis module and unit tests.
    - **Proof**: Known synthetic optimum and tie case.
    - **Depends on**: slice metric behavior.
- **Exit proof**: Focused test module passes and the scan reports exactly 101 candidates.
- **Stop condition**: Stop if MRR semantics differ from the repository's candidate-zero-positive contract.

### Phase 2: Common-Cache Checkpoint Evaluation

- **Purpose**: Compare old and new model components on identical 100-candidate validation rows.
- **Entry condition**: Phase 1 is green and server artifacts exist.
- **Phase rules**:
  - Do not call any model `fit()` method.
  - Load one dataset snapshot/checkpoint at a time and release large state promptly.
  - Record cache key, shape, feature names, component MRRs, stored blend MRRs, fine blend optimum, and old/new blend optimum.
- **Todos**:
  - [ ] Implement checkpoint-to-cached-feature scoring.
    - **Surface**: operator script using tested analysis primitives.
    - **Proof**: Server log contains no training epochs and reproduces new component MRRs within floating tolerance.
    - **Depends on**: Phase 1.
  - [ ] Run Dataset1 and Dataset2 comparison and persist report.
    - **Surface**: server JSON/TSV report.
    - **Proof**: Both report 20,000 rows, 100 candidates, and temporal slices.
    - **Depends on**: checkpoint scoring.
- **Exit proof**: Reproduced new metrics match the training log and old/new comparisons are recorded.
- **Stop condition**: Stop if feature layouts differ, cache identity is ambiguous, or reproduced metrics materially disagree with the run log.

### Phase 3: Dataset1 Pure-MLP Control Package

- **Purpose**: Isolate removal of the non-improving Dataset1 LightGBM component.
- **Entry condition**: Phase 2 validates the new Dataset1 MLP component and snapshot compatibility.
- **Phase rules**:
  - Use the new checkpoint encoder and MLP; do not retrain.
  - Copy Dataset2 from the `1.3411684997174933` package byte-for-byte.
  - Never overwrite either champion package.
- **Todos**:
  - [ ] Generate full Dataset1 predictions with MLP weight `1.0`.
    - **Surface**: server inference run.
    - **Proof**: 61,051 rows, 100 values per row, valid probabilities.
    - **Depends on**: Phase 2.
  - [ ] Splice Dataset2 and build the control ZIP.
    - **Surface**: new result directory.
    - **Proof**: Dataset2 SHA-256 matches champion; ZIP root contains exactly two CSV files.
    - **Depends on**: Dataset1 prediction.
  - [ ] Download and verify locally.
    - **Surface**: local `result/` artifact.
    - **Proof**: Local/remote ZIP SHA-256 equality.
    - **Depends on**: package validation.
- **Exit proof**: A locally available, validated control ZIP and analysis report are handed to the user.
- **Stop condition**: Stop before packaging if inference changes Dataset2, CSV validation fails, or an existing artifact would be overwritten.

## Dry-Run Findings

- The comparison cannot use old logged 32-candidate MRR; both checkpoints must be rescored on the new 100-candidate cache.
- A fine scan without temporal slices would overstate confidence, so early/late halves are mandatory evidence.
- A generic checkpoint rewrite would copy about 5 GB unnecessarily; inference should apply an in-memory MLP-only override instead.
- Dataset2 must be copied from the new champion ZIP/result directory, not regenerated, to preserve the control boundary.

## Final Validation

- Focused and related pytest suites pass.
- Server report reproduces new Dataset1 and Dataset2 component MRRs.
- No tower-training epoch appears in the analysis or control inference logs.
- Dataset1/Dataset2 row and column counts pass; probabilities are finite and formatted correctly.
- Dataset2 CSV hashes match; remote/local ZIP hashes match.

## First Execution Step

Add a failing unit test for full/early/late MRR and 101-point blend selection with deterministic tie-breaking.

## Execution Result

- Focused RED failed because `fusion_analysis` did not exist; GREEN passed both new tests.
- Related regression suite: `16 passed, 4 skipped`.
- Dataset1 common-cache result:
  - New MLP `0.79424685`; new stored ensemble `0.79425192`; old stored ensemble `0.79456965`.
  - Best stored-output blend: new weight `0.37`, MRR `0.79509294`; early `0.80475153`, late `0.78543435`.
- Dataset2 common-cache result:
  - New stored ensemble `0.54088842`; old stored ensemble `0.53897133`.
  - Best stored-output blend: new weight `0.65`, MRR `0.54234892`; early `0.57229904`, late `0.51239881`.
- Dataset1 pure-MLP control ZIP:
  - `result/d1_pure_mlp_d2_fullmrr_champion_seed60_20260722/result.zip`
  - `60,167,300` bytes.
  - SHA-256 `f644613bac546307d08c107f2f22b7562a801bb096db7b9d851001187c36484e`.
  - Dataset2 was copied byte-for-byte from the `1.3411684997174933` champion package.
