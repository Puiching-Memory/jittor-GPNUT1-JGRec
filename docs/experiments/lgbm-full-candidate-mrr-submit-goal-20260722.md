# Goal Document: LightGBM Full-Candidate MRR Early Stop and Submission

## Go / No-Go

- **Judgment**: Go
- **Reason**: The current LightGBM fusion already receives complete per-query validation candidate groups, but early stopping is driven by LightGBM MAP; replacing that metric is isolated, testable, and aligned with the competition MRR objective.

## Target Outcome

LightGBM fusion early-stops on MRR computed over each complete validation candidate row, then a seed-60 dataset1+dataset2 package is built on the server, validated, archived locally, and submitted to the configured competition endpoint with the returned submission result recorded.

## Goal Definition

- **Type**: technical, learning, operational, and delivery
- **Boundary**: Implement grouped full-candidate MRR evaluation for LightGBM, retain the AP path, test it, run one controlled server experiment with the existing strong feature/tower configuration and persistent supervised-feature cache, package both datasets, validate, download, and submit.
- **Non-goals**:
  - Do not redesign the LightGBM feature set or introduce a new tower.
  - Do not run a broad hyperparameter sweep before this first aligned-metric submission.
  - Do not overwrite the prior 1.3402985197047874 package or checkpoint.
- **Deferred work**:
  - LightGBM leaves/depth/learning-rate sweep.
  - Segment-aware fusion or per-dataset LightGBM parameter sets beyond this controlled run.
- **Verification rule**: A unit test must distinguish grouped MRR from flattened MAP; server logs must show MRR as the LightGBM early-stop metric; the final zip must contain valid full dataset1/dataset2 CSVs and the submission service must accept it.
- **Evidence source**: RED/GREEN pytest output, server log excerpts, artifact sizes and SHA-256 hashes, submission validation output, and leaderboard/submission response.
- **Pass criteria**: Custom eval returns the exact grouped MRR; early stopping monitors only that metric for `selection_metric=mrr`; both dataset outputs complete; package validation passes; submission receives an accepted/queued/scored identifier.
- **Confidence note**: Local full-candidate MRR is still a proxy for leaderboard MRR, so online score—not local improvement—owns the final model judgment.
- **Judgment owner**: Unit tests for implementation, package validator for artifact validity, competition response for submission completion, leaderboard score for model quality.

## Current State

- `fit_fusion_lgbm()` trains LambdaRank with `metric="map"` and LightGBM early stopping therefore follows MAP even when `selection_metric="mrr"`.
- Final post-training MRR is already calculated from the full `val_features` candidate dimension, but it does not select `best_iteration`.
- The current best online package is recorded at `1.3402985197047874` and remains the rollback baseline.
- Persistent supervised feature caching is available for repeated fusion-only parameter changes, but the first matching feature configuration may still need to populate it on the server.

## Priority Rationale

- Prove the grouped metric mathematically before launching a long GPU job.
- Keep feature-producing parameters fixed so the experiment isolates early-stop alignment.
- Validate and hash artifacts before submission to prevent an operational failure from being confused with a model result.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
| --- | --- | --- | --- |
| “完整候选 MRR” means MRR over every candidate in each validation tensor row | confirmed | Matches the available supervised validation boundary without flattening groups | Encode in custom LightGBM eval |
| `selection_metric=mrr` disables built-in MAP for early stopping | confirmed | Prevents mixed metrics from choosing an unintended iteration | Test captured LightGBM call |
| Existing seed-60 strong tower configuration is the control | confirmed | Isolates the metric change | Recover exact command from logs/docs before launch |
| Submission endpoint/credentials are present in the authorized workspace | assumed | Required for the final external action | Discover read-only; never print secrets |

## Phases

### Phase 1: Metric Contract

- **Purpose**: Prove full-candidate grouped MRR and early-stop wiring before production changes.
- **Entry condition**: Goal document exists and current fusion tests are available.
- **Phase rules**:
  - Write RED tests first.
  - Test a score matrix where MRR and flattened MAP imply different behavior.
  - Preserve the existing AP behavior.
- **Todos**:
  - [ ] Add RED test for exact grouped candidate MRR.
    - **Surface**: LightGBM fusion tests.
    - **Proof**: Fails because the grouped eval API does not exist.
    - **Depends on**: None.
  - [ ] Add RED test that `selection_metric=mrr` configures LightGBM early stopping around only grouped MRR.
    - **Surface**: `fit_fusion_lgbm()` call contract.
    - **Proof**: Captured train call currently shows built-in MAP and no MRR evaluator.
    - **Depends on**: First metric test.
  - [ ] Implement the minimal custom evaluator and early-stop configuration.
    - **Surface**: Hybrid LightGBM fusion module.
    - **Proof**: Focused tests pass.
    - **Depends on**: RED evidence.
- **Exit proof**: Exact MRR and LightGBM wiring tests are green; relevant fusion regressions pass.
- **Stop condition**: Stop if installed LightGBM cannot use a custom validation metric for early stopping.

### Phase 2: Controlled Server Experiment

- **Purpose**: Build a submission candidate while isolating the early-stop metric change.
- **Entry condition**: Phase 1 is green and code is synchronized to the server.
- **Phase rules**:
  - Preserve feature/tower parameters from the current strong seed-60 configuration.
  - Use one new run name and checkpoint path; never overwrite the champion.
  - Enable supervised feature cache for later fusion iterations.
- **Todos**:
  - [ ] Recover the exact prior strong command and establish available server/cache/disk state.
    - **Surface**: Experiment logs and remote filesystem.
    - **Proof**: Recorded command, free disk, GPU state, and cache status.
    - **Depends on**: Phase 1.
  - [ ] Run dataset1 and dataset2 to completion with grouped-MRR LightGBM early stopping.
    - **Surface**: Remote training process/logs.
    - **Proof**: Logs show cache hit/miss, LightGBM `mrr` early stopping, local AP/MRR, output row counts, and checkpoint completion.
    - **Depends on**: Server preflight.
- **Exit proof**: Both full CSVs and a complete checkpoint/result package exist remotely.
- **Stop condition**: Stop before submission if either dataset is partial, NaN/Inf appears, disk is unsafe, or package validation fails.

### Phase 3: Package, Download, and Submit

- **Purpose**: Turn the experiment into a traceable competition submission.
- **Entry condition**: Both dataset outputs are complete.
- **Phase rules**:
  - Validate zip root contents and CSV row counts before external submission.
  - Record local and remote hashes.
  - Submit exactly the validated artifact once unless the platform explicitly reports a retryable failure.
- **Todos**:
  - [ ] Build/validate the result zip and download it with its checkpoint/log.
    - **Surface**: Remote result directory and local result/checkpoint directories.
    - **Proof**: File sizes, CSV row counts, zip members, and SHA-256 hashes match.
    - **Depends on**: Phase 2.
  - [ ] Submit the validated package and record platform response.
    - **Surface**: Configured competition service.
    - **Proof**: Accepted/queued/scored response with submission identifier and, when available, score.
    - **Depends on**: Artifact validation.
- **Exit proof**: A traceable local artifact and platform submission record are both available.
- **Stop condition**: Do not resubmit on an ambiguous response until submission history is checked to avoid duplicates.

## Dry-Run Findings

- The custom evaluator must reshape predictions using the validation candidate count captured from `val_features`; labels alone do not encode group boundaries reliably enough for MRR.
- Built-in metrics should be disabled for the MRR path or early stopping may monitor more than the intended metric.
- LightGBM `best_iteration` must remain the model used for final validation prediction and serialization.
- A cache hit accelerates supervised train/validation construction but the final full-history encoder and prediction still run for a submission package.
- The competition response may queue scoring asynchronously; acceptance and score availability are separate completion signals.

## Final Validation

- Focused RED/GREEN LightGBM metric tests.
- Relevant hybrid/CLI/cache regression tests, Ruff, and compileall.
- Remote log audit for the selected early-stop metric and both dataset completions.
- Submission zip validation, SHA-256 verification, and platform submission response.

## First Execution Step

Add a failing pure-unit test for grouped full-candidate MRR using a validation score matrix with known reciprocal ranks.
