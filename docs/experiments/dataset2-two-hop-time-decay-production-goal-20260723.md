# Goal Document: Dataset2 Time-Decayed Two-Hop Production Feature

## Go / No-Go

- **Judgment**: No-Go after exact validation; do not package.
- **Reason**: The full-100 fixed blend regressed by `-0.00088407`, including regressions in chronological slices 1 and 3. The proxy lift did not survive the production candidate distribution.

## Target Outcome

Add one causal, time-decayed two-hop feature for Dataset2, reuse the champion MLP unchanged, retrain only the Dataset2 LightGBM expert on cached full candidates, and produce a candidate checkpoint/package only if exact 20,000-query validation improves by at least `+0.002` in full MRR with no chronological slice regression.

## Goal Definition

- **Type**: technical, quality, and delivery.
- **Boundary**: Dataset2 temporal co-occurrence state; one fixed horizon `tau = 0.05 * prefix_time_span`; one appended feature; independent MLP/LightGBM feature masks; supervised cache rebuild/extension; Dataset2 LightGBM retraining; exact full-candidate validation; combine with unchanged Dataset1 champion only after passing.
- **Non-goals**:
  - Tune tau, source-history limit, co-occurrence history limit, or multiple variants on the same validation set.
  - Retrain towers or the champion MLP.
  - Change Dataset1 predictions.
  - Replace raw `cooccur_score`.
  - Use test labels, B-board feedback, or public candidate positions as supervision.
- **Deferred work**:
  - Multiple decay horizons, learned temporal kernels, and segment gating involving this feature.
- **Verification rule**: RED/GREEN contracts for temporal aggregation, snapshot/hydrate, old-checkpoint compatibility, cache identity, and distinct expert feature masks; then one frozen Dataset2 LightGBM fit and exact 20,000-query evaluation against the `0.5428303297309955` champion baseline and three fixed chronological slices.
- **Evidence source**: Unit/integration tests, cache manifest, memory/runtime log, exact validation report, checkpoint reload prediction equality, and result ZIP verification.
- **Pass criteria**: Candidate full MRR delta at least `+0.002`; every chronological slice delta nonnegative; Dataset1 output byte-identical to the champion; checkpoint reload predictions equal pre-save predictions; no material inference-memory regression beyond the declared budget.
- **Confidence note**: The proxy proves the signal but used 31 sampled negatives and standalone ranking. Exact full-candidate ensemble validation remains the production decision owner.
- **Judgment owner**: The frozen exact validation gate controls checkpoint/package creation; tests control correctness and compatibility.

## Current State

- Champion online score is `1.3426473547970703`; Dataset2 exact validation MRR is `0.5428303297309955`.
- The successful proxy used 2,000 evenly spaced positives, 31 deterministic public-distribution negatives, history 64, co-occurrence history 128, and tau `15,016,320` seconds.
- Existing future-only temporal state stores compact integer co-occurrence counts and discards event time.
- Existing inference selects the MLP feature mask and reuses it for LightGBM, so it cannot yet preserve the 63-feature MLP while giving LightGBM the appended feature.
- Appending the feature conditionally is required to keep old Dataset1/checkpoint behavior stable.

## Priority Rationale

- Prove compact causal state and backward compatibility before rebuilding expensive features.
- Preserve the champion MLP and raw co-occurrence feature; isolate the new signal in the cheaper Dataset2 LightGBM expert.
- Freeze one production attempt to limit validation overfitting before the immutable B board.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| Use one anchor-normalized decayed sum per item pair | confirmed design | Avoids retaining all timestamps; exact query-time value is recovered by a common exponential factor | Temporal-state tests |
| Append the feature after all existing 63 features | confirmed design | Keeps all champion feature indices stable | Compatibility test |
| Keep MLP indices `0..62`, allow LightGBM to use `0..63` | confirmed design | Avoids expensive MLP retraining | Separate-mask prediction test |
| Float-map memory fits the server | assumed | Main implementation risk | Measure peak RSS before cache build |
| Proxy lift survives 100 candidates and blending | rejected | Full delta `-0.00088407`; slices `[-0.00255136, +0.00059562, -0.00069644]` | No package |

## Phases

### Phase 1: Causal Compact State and Compatibility

- **Purpose**: Add the signal without breaking old checkpoints or retaining timestamp lists.
- **Entry condition**: Proxy report is passed and frozen.
- **Phase rules**:
  - TDD first.
  - Preserve raw co-occurrence counts.
  - Append the feature only when enabled; old configs/checkpoints remain 63-dimensional.
- **Todos**:
  - [x] Implement a compact sparse float map of anchor-normalized decayed co-occurrence sums.
    - **Surface**: temporal index, snapshot/shallow-copy/compaction, tests.
    - **Proof**: Matches a direct timestamp calculation before/after hydrate and future-only compaction.
    - **Depends on**: none.
  - [x] Append `cooccur_time_decay_score` through an opt-in Dataset2 feature path.
    - **Surface**: encoder/config/feature naming and structure scoring.
    - **Proof**: Old checkpoint features/predictions are unchanged; enabled feature matches the proxy formula.
    - **Depends on**: compact float map.
  - [x] Select MLP and LightGBM feature indices independently at inference.
    - **Surface**: hybrid ranker prediction and checkpoint tests.
    - **Proof**: MLP receives 63 columns, LightGBM receives 64, and reload predictions are identical.
    - **Depends on**: appended feature.
- **Exit proof**: Focused tests, checkpoint compatibility suite, and Ruff pass locally and on Linux.
- **Stop condition**: Stop if old Dataset1/champion predictions change or compact state exceeds the declared memory budget.

### Phase 2: Frozen Full-Candidate Dataset2 Evaluation

- **Purpose**: Test whether proxy signal improves the actual ensemble setting.
- **Entry condition**: Phase 1 passes and memory is acceptable.
- **Phase rules**:
  - Freeze cache key, feature order, LightGBM params/rounds, MLP weight, and validation slices before scoring.
  - Run one tau and one feature definition only.
  - Reuse the champion MLP and retrain only Dataset2 LightGBM.
- **Todos**:
  - [x] Build or extend cached train/validation features with the appended temporal column.
    - **Surface**: cache manifest and `.npy` artifacts.
    - **Proof**: Shapes, hashes, feature names, causal split identity, peak RSS, and runtime.
    - **Depends on**: Phase 1.
  - [x] Train the frozen Dataset2 LightGBM and score all 20,000 validation groups with 100 candidates.
    - **Surface**: model/report only; no checkpoint mutation yet.
    - **Proof**: Full and three-slice MRR deltas versus the exact champion baseline.
    - **Depends on**: feature cache.
- **Exit proof**: Report declares pass/reject under the `+0.002` all-slice gate.
- **Stop condition**: Stop on any slice regression, full delta below `+0.002`, cache alignment failure, or excessive memory.

### Phase 3: Conditional Checkpoint and Submission

- **Purpose**: Package only a validated improvement while leaving Dataset1 untouched.
- **Entry condition**: Phase 2 passes.
- **Phase rules**:
  - Start from the champion combined checkpoint.
  - Replace only Dataset2 temporal encoder state and LightGBM result.
  - Verify the ZIP before handoff; do not submit automatically.
- **Todos**:
  - [ ] Save and reload a separate candidate checkpoint.
    - **Surface**: new checkpoint path.
    - **Proof**: Dataset1 byte-identical predictions and Dataset2 pre/post-reload equality.
    - **Depends on**: Phase 2 pass.
  - [ ] Build a separate result ZIP and copy it locally.
    - **Surface**: result directory and ZIP.
    - **Proof**: expected files, row counts, size, and SHA-256.
    - **Depends on**: checkpoint reload proof.
- **Exit proof**: A local candidate ZIP exists with a validation report and checksum, or no ZIP exists after rejection.
- **Stop condition**: Any compatibility or packaging mismatch.

## Dry-Run Findings

- A proxy pass is not enough to submit because its 32-candidate standalone ranking is easier than the exact 100-candidate blended problem.
- Storing all pair timestamps in the final checkpoint would be unnecessarily large; the exponential kernel permits a single anchor-normalized float aggregate per pair.
- Reusing the MLP requires fixing inference to respect LightGBM's own feature indices; otherwise the appended feature would never reach LightGBM or would break the MLP shape.
- Adding the feature inside the middle of the existing structure tuple would shift downstream indices and risk Dataset1 compatibility, so the new column must be appended conditionally.

## Final Validation

- Focused and regression tests on Windows/Linux.
- Cache manifest/hash and causal split alignment.
- Exact 20,000-query full/per-slice MRR gate.
- Peak memory and inference runtime comparison.
- Candidate checkpoint reload equality, Dataset1 identity, ZIP contents, and SHA-256.

## Execution Result

- Baseline full fixed-blend MRR: `0.5428303297309955`.
- Candidate full fixed-blend MRR: `0.5419462615742707` (`-0.0008840681567248154`).
- Chronological slice deltas: `[-0.0025513638756688994, +0.0005956229735167851, -0.0006964354245194704]`.
- Gate: failed both the `+0.002` full-delta rule and the all-slices-non-decreasing rule.
- Safety recovery: 351/1,600,000 train positions and 566/2,000,000 validation positions used conservative maximum decay; both stayed within the frozen limits.
- Packaging: prohibited and not run; no result ZIP was generated.
- Evidence: `result/dataset2_two_hop_decay_full100_seed60_20260723/full100-report.json`.

## First Execution Step

Add a failing test showing that an anchor-normalized compact decay aggregate reproduces direct timestamp decay before and after future-only compaction, while an old disabled config still emits exactly 63 features.
