# Goal Document: Dataset2 Candidate-Sampling Sweep

## Go / No-Go

- **Judgment**: Go
- **Reason**: The `1.3402985197047874` submission is a verified online baseline, and the proposed sweep changes one dataset and one parameter family at a time.

## Target Outcome

Find a reproducible dataset2 candidate-sampling configuration that improves validation MRR over the new champion protocol without changing dataset1 or overwriting champion artifacts.

## Goal Definition

- **Type**: learning and quality
- **Boundary**: Run dataset2-only screening for test-candidate negative ratio, then negative count, using the current hybrid model, seed 60, K=512 prediction limits, 50-epoch tower caps, MRR selection, and ensemble fusion.
- **Non-goals**:
  - Do not retrain dataset1.
  - Do not change model code or feature definitions.
  - Do not produce submission candidates during screening.
- **Deferred work**:
  - Multi-seed confirmation and full prediction are deferred until a screening configuration passes the MRR gate.
  - Listwise fusion and segment-aware fusion remain separate follow-up experiments.
- **Verification rule**: Compare each run's selected dataset2 validation MRR under the same protocol.
- **Evidence source**: Persistent server run logs, status files, and the recorded fusion selection metrics.
- **Pass criteria**: A candidate must improve MRR by at least 0.003 over the ratio=1.00, negatives=31 baseline before multi-seed confirmation.
- **Confidence note**: Validation MRR is only a screening proxy; final acceptance still requires online leaderboard improvement.
- **Judgment owner**: Validation metric selects the screening winner; leaderboard score declares a new champion.

## Current State

- Online champion score: `1.3402985197047874`.
- Champion run: `ensemble_towers50_k512_d1d2_seed60_20260721`.
- Dataset1 validation MRR: `0.84760` for ensemble versus `0.84745` for MLP.
- Champion result and checkpoint are present locally and must remain immutable.
- Dataset2 is a new-link/cold-target problem; candidate-distribution calibration is the highest-leverage parameter family.
- Local and online scores are not perfectly calibrated, so isolated comparisons and online confirmation are required.

## Priority Rationale

- Reuse the ratio=1.00, negatives=31 champion run as the control instead of rerunning it.
- Screen ratios first because this directly controls agreement with the test candidate distribution.
- Change negative count only after selecting the ratio, avoiding a full Cartesian sweep.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
| --- | --- | --- | --- |
| Server working tree is the same one that produced the champion | confirmed | Preserves the comparison protocol even though the deployed tree has no `.git` metadata | Reuse in place; do not sync code |
| Core `ranker.py`, `sampling.py`, and `fusion.py` match local SHA-256 hashes | confirmed | Confirms the scoring and sampling implementation under test | Recorded during preflight |
| Champion dataset2 MRR is `0.73030` for the pure-LightGBM ensemble choice | confirmed | Supplies the ratio=1.00, negatives=31 control | Extracted from the champion server log |
| `--limit-rows 2` limits prediction output but preserves training and validation | confirmed | Makes screening cheaper without changing validation | Existing submission runbook |
| Server has no conflicting `jgrec-build` process | confirmed | Avoids resource contention | Checked with GPU, memory, and disk preflight |

## Phases

### Phase 1: Ratio Screening

- **Purpose**: Select the dataset2 test-candidate negative ratio.
- **Entry condition**: Server code revision, baseline log, resources, and artifact names are verified.
- **Phase rules**:
  - Run sequentially, never concurrently.
  - Keep all parameters identical to the champion except dataset2-only screening flags and ratio.
  - Do not save a checkpoint or run full prediction.
- **Todos**:
  - [ ] Extract the champion dataset2 fusion MRR for ratio=1.00.
    - **Surface**: Server champion run log.
    - **Proof**: Logged `fusion-select`, `fusion-lgbm-select`, and `ensemble-weight` lines.
    - **Depends on**: None.
  - [ ] Run ratio=0.60 with negatives=31.
    - **Surface**: Server run directory and log.
    - **Proof**: Successful status plus selected validation MRR.
    - **Depends on**: Server preflight.
  - [ ] Run ratio=0.75 with negatives=31.
    - **Surface**: Server run directory and log.
    - **Proof**: Successful status plus selected validation MRR.
    - **Depends on**: ratio=0.60 completion.
- **Exit proof**: A three-row comparison for ratios 0.60, 0.75, and 1.00 under negatives=31.
- **Stop condition**: Stop on code revision mismatch, resource pressure, non-comparable protocol, or repeated runtime failure.

### Phase 2: Negative-Count Screening

- **Purpose**: Test whether harder candidate sets improve the winning ratio.
- **Entry condition**: Phase 1 has a winner that is not materially worse than the baseline.
- **Phase rules**:
  - Keep the Phase 1 winning ratio fixed.
  - Test negatives=63, then negatives=99 sequentially.
  - Do not advance a gain below 0.003 to full prediction.
- **Todos**:
  - [ ] Run the winning ratio with negatives=63.
    - **Surface**: Server run directory and log.
    - **Proof**: Successful status plus selected validation MRR.
    - **Depends on**: Phase 1 winner.
  - [ ] Run the winning ratio with negatives=99.
    - **Surface**: Server run directory and log.
    - **Proof**: Successful status plus selected validation MRR.
    - **Depends on**: negatives=63 completion.
- **Exit proof**: A ranked comparison with one recommended configuration or an explicit keep-baseline decision.
- **Stop condition**: Stop if increased negatives cause memory failure, excessive runtime, or clear MRR regression.

### Phase 3: Confirmation Gate

- **Purpose**: Decide whether the screening winner deserves a full submission run.
- **Entry condition**: A candidate improves seed-60 validation MRR by at least 0.003.
- **Phase rules**:
  - Run the ratio=1.00, negatives=31 seed-42 control before confirming the winner with seed 42.
  - Preserve the current dataset1 output for any later submission package.
- **Todos**:
  - [ ] Run the champion protocol control with seed 42.
    - **Surface**: Server run directory and log.
    - **Proof**: Successful status plus selected validation MRR.
    - **Depends on**: Seed-60 candidate passing the 0.003 gate.
  - [ ] Re-run the winner with seed 42.
    - **Surface**: Server run directory and log.
    - **Proof**: At least 0.003 MRR improvement over the matched seed-42 control.
    - **Depends on**: Seed-42 control.
  - [ ] Decide full-run Go / No-Go.
    - **Surface**: Experiment record.
    - **Proof**: Both seeds pass or the candidate is rejected.
    - **Depends on**: Seed-42 result.
- **Exit proof**: A documented full-run decision.
- **Stop condition**: Reject if the second seed reverses the gain.

## Dry-Run Findings

- The ratio=1.00, negatives=31 champion already supplies the control and should not be rerun.
- Sequential execution is required because the full hybrid feature pipeline is memory intensive.
- Screening must use unique result and log names because the CLI refuses to overwrite existing run directories.
- Full prediction and checkpoint generation would add cost without improving the screening evidence.

## Final Validation

- Verify every completed run exited successfully and contains dataset2 MLP, LightGBM, and ensemble MRR lines.
- Compare seed-60 results against the champion control; confirm any passing winner with seed 42.
- Only an online score above `1.3402985197047874` may replace the champion.

## First Execution Step

Run a read-only server preflight for revision, processes, GPU, memory, disk, champion status, and champion dataset2 fusion metrics.
