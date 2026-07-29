# Goal Document: Dataset1 Full-100 Setwise

## Go / No-Go

- **Judgment**: No-Go for packaging after completed Linux CUDA execution
- **Reason**: The candidate improved full MRR by `+0.0014274006`, below the
  frozen `+0.002` threshold, and regressed slice 0 by `-0.0000677395`.

## Target Outcome

Train fixed Dataset1 Setwise candidates on recent `100,000 x 100` and
`200,000 x 100` full-candidate groups, select the training scale and
Setwise/champion-LightGBM blend using only the first two chronological validation
slices, open the final slice only for an unseen gate, and generate a submission
only if the candidate beats the Dataset1 champion while preserving the current
Dataset2 champion bytes exactly.

## Goal Definition

- **Type**: learning / technical / delivery
- **Boundary**: Dataset1 only for cache construction, training, validation, and
  inference; one `200,000 x 100` recent-train cache; a suffix view for the
  `100,000 x 100` control; one `20,000 x 100` chronological validation cache;
  an adaptive context boundary that moves earlier only when required to leave
  exactly 200,000 pre-validation training rows; Setwise seed 60; a predeclared
  blend grid; conditional package generation.
- **Non-goals**:
  - Retraining or changing Dataset2.
  - Adding new features, towers, GNN variants, or segment rules.
  - Choosing any hyperparameter from the forward-held validation slice.
  - Overwriting current checkpoints, caches, reports, or submission artifacts.
- **Deferred work**:
  - Multi-seed Setwise ensembling.
  - Dataset1-specific feature engineering if full-100 Setwise is rejected.
  - Leaderboard submission, which remains a user action.
- **Verification rule**: The train and validation reports must prove one
  process/build ID and exact feature-hash binding; the `100k` cache must be an
  exact chronological suffix view of the `200k` cache; a pure selector must be
  invariant to mutations after row 13,334; the locked candidate must then pass
  the full and forward-slice gates; Dataset2 ZIP-member SHA-256 must remain
  `d7e0d574789aa6b507e592fffb5da054839aba663f63ed58134ea2d951b1ae1e`.
- **Evidence source**: RED/GREEN tests, cache reports, frozen config, training
  histories, exact full/three-slice MRR, model/checkpoint hashes, submission
  validation, and ZIP-member hashes.
- **Pass criteria**:
  - Train tensor is `200000 x 100 x F`; validation tensor is
    `20000 x 100 x F`; every candidate row is unique with the positive at
    position zero.
  - `100k` uses rows `[100000:200000]` from the same train tensor and matching
    sidecars, without a second encoder run.
  - Two Setwise models use the Dataset2 winning settings: seed 60, 10 epochs,
    patience 2, batch size 256, hidden dimension 32, learning rate `0.001`,
    negative log-softmax of candidate zero, and raw/row-mean/row-max context
    features.
  - Candidate scale and Setwise weight are selected only on validation rows
    `[0:13334]`. The predeclared Setwise-weight grid is `0.05..1.00` in `0.05`
    increments; ties favor the higher Setwise weight, then the larger `200k`
    training scale.
  - The original fixed MLP/LightGBM blend is the gate baseline; the candidate
    is the persistable Setwise/champion-LightGBM blend used by ranker inference.
  - After selection is frozen, full MRR improves by at least `+0.002` versus
    the original Dataset1 champion, every chronological slice is
    non-decreasing, and slice 2 is strictly non-decreasing.
  - A generated ZIP contains exactly `dataset1.csv` and `dataset2.csv`;
    Dataset2 is byte-identical to the current online champion.
- **Confidence note**: Same-process features and a genuinely unseen final
  validation slice control the main offline leakage risks. MRR is still only a
  proxy for the competition leaderboard.
- **Judgment owner**: Automated cache, selection, metric, schema, and hash
  gates authorize packaging; the leaderboard owns final online quality.

## Current State

- Dataset1 has 690,848 training events; 72.60% are repeats, and historical-pair
  hit rate rises from 58.91% to 74.61% across the observed time range.
- Dataset2's successful path used same-process recent-200k/full-100 training
  and chronological 20k/full-100 validation, Setwise context features, and
  forward-held weight selection.
- Before this run, Dataset1 had no corresponding full-100 Setwise cache, model,
  or report.
- Remote split preflight found `train_end=587221`; the configured
  `context_end=440415` leaves only 146,806 supervised rows. The frozen exact
  200k protocol therefore uses `context_end=387221`, a 53,194-row context
  backoff that still keeps every training row strictly before validation.
- The current Dataset1 online-champion CSV SHA-256 is
  `6760fd966fde3a8de4693d06f27ec0f2458e49c44e2ea48fc79e1393434e6e2b`.
- The current Dataset2 online-champion CSV SHA-256 is
  `d7e0d574789aa6b507e592fffb5da054839aba663f63ed58134ea2d951b1ae1e`.
- The prior Dataset1 segment-policy package scored `1.351329395102109`, below
  the `1.3530197200911278` champion, so it is excluded from this experiment.
- The replacement Linux CUDA endpoint was reachable and used for the complete
  production run. Local Windows remained a test-only environment.

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| Dataset2 same-process 200k/20k cache protocol | keep and parameterize | It is the proven correctness boundary for learned features. |
| Separate 100k cache build | replace with 200k suffix view | Avoids a second encoder lifecycle and makes the scale comparison exact. |
| Dataset2 fixed high-weight scan | widen before results | Dataset1's stronger incumbent may need a lower Setwise contribution. |
| First two slices select, final slice gates | keep | Prevents forward-slice selection leakage. |
| Dataset2 conditional package path | invert dataset ownership | Replace only Dataset1 and preserve Dataset2 bytes. |
| Multi-seed ensemble | defer | First prove one seed and two data scales have value. |

## Drift Diagnosis

- **Goal drift**: New feature families would no longer isolate whether
  full-candidate Setwise training itself transfers to Dataset1.
- **Phase drift**: Training before same-process provenance passes would repeat
  the invalid independent-encoder experiment.
- **Validation drift**: Choosing scale or weight on all 20,000 rows would expose
  the final gate and invalidate the result.
- **Compatibility drift**: Dataset2 inference/checkpoint state must not be
  regenerated merely to package a Dataset1 candidate.
- **Cleanup drift**: Existing user artifacts and rejected packages remain
  untouched.

## Priority Rationale

- The highest-risk issue is experimental validity, not Setwise model code:
  cached learned features must share one live encoder state.
- Building 200k once and deriving the 100k suffix makes the expensive stage
  reusable and gives a clean data-scale comparison.
- A broad but frozen blend grid is cheaper and safer than assuming Dataset2's
  `0.80` weight transfers to Dataset1.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| Current champion checkpoint is present on the Linux host | unresolved | Required to reproduce the Dataset1 baseline and final inference | Remote preflight |
| Dataset1 champion uses the same feature schema for train and validation | assumed | Required for cache/model compatibility | Checkpoint/cache report |
| Recent 200k rows fit before `train_end` | confirmed with adaptive context boundary | Required for both scale candidates | Unit contract plus cache report |
| Linux host has at least 12 GiB free disk and sufficient RAM/GPU headroom | confirmed | 1.4 TiB disk and 24 GiB available RAM at preflight | Remote resource preflight |
| The current Dataset2 ZIP member is the frozen package source | confirmed | Prevents Dataset2 regression | SHA-256 before and after packaging |

## Phases

### Phase 1: Freeze reusable Dataset1 contracts

- **Purpose**: Make scale selection and forward-slice isolation testable before
  production code.
- **Entry condition**: This goal document is complete.
- **Phase rules**:
  - RED before implementation.
  - Pure selection cannot inspect rows at or after `selection_stop`.
  - Existing Dataset2 behavior and APIs remain compatible.
- **Todos**:
  - [x] Add a test that selects across the 100k/200k models and blend grid.
    - **Surface**: fusion analysis API and unit tests.
    - **Proof**: RED because the multi-model prefix selector is missing.
    - **Depends on**: none.
  - [x] Prove changing only forward-held rows cannot change model scale,
    weight, or selection MRR.
    - **Surface**: fusion analysis test.
    - **Proof**: focused GREEN pytest.
    - **Depends on**: selector implementation.
- **Exit proof**: Focused tests and Ruff pass.
- **Stop condition**: Selection cannot be expressed without leaking slice 2.

### Phase 2: Dataset1 same-process cache and trainer adapters

- **Purpose**: Reuse the Dataset2 protocol without dataset-specific hard-coded
  paths, shapes, or state keys.
- **Entry condition**: Phase 1 is green.
- **Phase rules**:
  - Build one 200k train cache and one 20k validation cache in one process.
  - The 100k candidate is a suffix view, not a second cache build.
  - No production run until local/static and Linux CUDA preflight pass.
- **Todos**:
  - [x] Add a Dataset1 full-100 joint cache CLI with exact provenance reports.
    - **Surface**: scripts and existing full-100 utilities.
    - **Proof**: CLI/static test and report-contract unit tests.
    - **Depends on**: Phase 1.
  - [x] Add a dual-scale Dataset1 Setwise trainer/evaluator.
    - **Surface**: scripts, frozen config, model artifacts, evaluation report.
    - **Proof**: dry-run fixtures and focused tests.
    - **Depends on**: cache contract.
- **Exit proof**: Local tests/Ruff and remote `--help`/preflight pass.
- **Stop condition**: Feature schema, checkpoint, candidate, temporal, hash, or
  resource preflight mismatch.

### Phase 3: Production cache, training, and unseen gate

- **Purpose**: Obtain an honest offline decision.
- **Entry condition**: Linux CUDA host is reachable and Phase 2 passes there.
- **Phase rules**:
  - Freeze all settings before cache/training starts.
  - Select scale and weight only from rows `[0:13334]`.
  - Persist the locked choice before computing forward metrics.
- **Todos**:
  - [x] Build and verify the `200k x 100` train and `20k x 100` validation
    cache pair.
    - **Surface**: remote memmaps, sidecars, reports, logs.
    - **Proof**: shapes, finite values, candidate contract, matching build
      ID/PID, and exact hashes.
    - **Depends on**: remote preflight.
  - [x] Train the 100k and 200k Setwise models and freeze the prefix winner.
    - **Surface**: models, histories, selection report.
    - **Proof**: complete grid, selected scale/weight, prefix MRR.
    - **Depends on**: cache pair.
  - [x] Open slice 2 and apply the full/three-slice gate.
    - **Surface**: final evaluation report.
    - **Proof**: baseline/candidate MRR and deltas for full plus three slices.
    - **Depends on**: frozen selection report.
- **Exit proof**: Passing report or explicit rejection with no package.
- **Stop condition**: OOM, non-finite training, hash drift, forward leakage, or
  gate failure.

### Phase 4: Conditional Dataset1 package

- **Purpose**: Deliver only an offline-authorized online candidate.
- **Entry condition**: Phase 3 passes.
- **Phase rules**:
  - Replace Dataset1 only.
  - Copy Dataset2 from the current Setwise champion ZIP byte-for-byte.
  - Validate and hash before copy-back.
- **Todos**:
  - [ ] Persist the selected Dataset1 Setwise state/weight and generate
    Dataset1 test predictions.
    - **Surface**: checkpoint and CSV.
    - **Proof**: reload prediction equality and CSV validation.
    - **Depends on**: passing gate.
  - [ ] Compose, verify, and copy back the final ZIP.
    - **Surface**: submission package and provenance report.
    - **Proof**: exact ZIP inventory, row counts, probability bounds, Dataset2
      SHA-256 equality, and remote/local ZIP SHA-256 equality.
    - **Depends on**: Dataset1 inference.
- **Exit proof**: Submit-ready local ZIP or a documented no-package decision.
- **Stop condition**: Reload, schema, source-member, or hash mismatch.

## Dry-Run Findings

- The expensive learned-feature replay cannot be split across processes; the
  previous Dataset2 replay already proved candidates can match while learned
  features differ.
- `100k` can be compared fairly by taking the recent suffix of the same
  chronological `200k` feature and sidecar arrays.
- Dataset1's configured supervised pool is only 146,806 rows; choosing
  `min(configured_context_end, train_end-requested_rows)` is the smallest
  boundary change that makes exact 200k possible and leaves Dataset2 unchanged
  whenever its configured pool is already large enough.
- The forward slice cannot select the training scale, weight, epoch, or retry.
- Local Windows execution is not a fallback: Jittor native compilation fails,
  and the project declares Linux/POSIX support.
- Remote connectivity is therefore a hard entry condition for Phase 3, but it
  does not block implementing and testing the reusable contracts.

## Final Validation

- Focused RED/GREEN/REFACTOR evidence is recorded.
- Local and remote targeted pytest plus Ruff pass.
- Cache pair, suffix-view, feature-schema, and hash contracts pass.
- Selection report proves rows `[13334:20000]` were not used to choose the
  model.
- Final gate records exact full and slice deltas.
- A package exists only after a pass and preserves Dataset2 member SHA-256
  `d7e0d574789aa6b507e592fffb5da054839aba663f63ed58134ea2d951b1ae1e`.

## Execution Status

- Protocol implementation is complete.
- Local verification after the adaptive-context RED/GREEN cycle:
  `11 passed` focused; the final broader suite reports
  `31 passed, 4 skipped`; the sparse sentinel regression passes separately;
  Ruff and `py_compile` pass.
- Remote preflight verification: `35 passed`; the sparse-boundary regression
  suite later passed `47` tests; Ruff passes.
- The new Linux CUDA endpoint is reachable. The champion checkpoint exists,
  and the frozen Dataset2 CSV hash matches
  `d7e0d574789aa6b507e592fffb5da054839aba663f63ed58134ea2d951b1ae1e`.
- The first production launch stopped before cache allocation because the
  configured Dataset1 supervised pool had only 146,806 rows. No complete or
  partial `.npy` cache artifact was written; the adaptive-context fix is now
  locally and remotely green.
- The next launch reached 20,480 cache rows and exposed an existing
  `SparseCountMap.batch_get_counts()` sentinel-boundary bug. Its focused RED
  reproduced the exact `IndexError`; the minimal GREEN fix passes 47 related
  Linux tests. Invalid `.part` arrays were removed, failure evidence retained,
  and a clean same-seed build was restarted.
- The clean joint build completed with train shape `200000 x 100 x 63`,
  validation shape `20000 x 100 x 63`, build ID
  `6c6ddd8647d442c598c9ae854adc8cd7`, and the same Python PID `603645`.
- Prefix selection locked recent-100k at Setwise weight `1.00`, selection MRR
  `0.7949377455`.
- The unseen gate rejected the candidate: full delta `+0.0014274006` is below
  `+0.002`, and slice 0 delta is `-0.0000677395`. Slice 1 and slice 2 improved
  by `+0.0005012482` and `+0.0038490563`, respectively.
- `package_authorized=false`; no checkpoint, CSV, or ZIP was generated.
- Dataset2 remained byte-identical with SHA-256
  `d7e0d574789aa6b507e592fffb5da054839aba663f63ed58134ea2d951b1ae1e`.

## Final Decision

The goal was executed completely and ended in a controlled **No-Go for
packaging**. Preserve the caches and Setwise models as reusable experiment
inputs. If another run is authorized, test a frozen recency-ramped
champion/100k-Setwise blend without rebuilding full-100 features.
