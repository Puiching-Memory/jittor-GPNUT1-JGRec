# Goal Document: Dataset2 Cooccur Lift Auxiliary Expert v1

## Go / No-Go

- **Judgment**: Go
- **Reason**: The consumed validation error profile supplies a concrete
  mechanism hypothesis, the lift columns have not been tested in the
  repository, and the experiment has causal feature definitions, exact
  rolling boundaries, a frozen search space, and a no-rescan stop rule.

## Target Outcome

Determine whether two causal cooccurrence-lift columns can turn the
remaining raw cooccurrence signal into a stable long-tail discriminator.
Train one four-epoch 65-column Setwise auxiliary per exact-rolling fold,
run the existing robust selector once, open the external holdout only if a
lock exists, and package only after external acceptance.

## Goal Definition

- **Type**: technical / learning / quality
- **Boundary**:
  - Dataset2 200k chronological train cache and the existing three
    exact-rolling folds.
  - One causal `TemporalInteractionIndex` built from
    `data/dataset2/train.csv`.
  - Two appended columns at indices 63 and 64; the 63-column champion
    checkpoint and fold champion probabilities remain unchanged.
  - One Setwise auxiliary head per fold and the six frozen integration
    weights.
- **Non-goals**:
  - Do not enable, reuse, or modify `cooccur_time_decay_score`.
  - Do not bag heads, scan new salts, alter the lift formula/window after
    metrics, or tune on the repeatedly consumed validation set.
  - Do not change the champion, graph-score overlay, selector gates, or
    external evaluator.
  - Do not implement `tail_weighted_aux_expert_v1` in this run.
- **Deferred work**:
  - Register `tail_weighted_aux_expert_v1` only after this run is closed
    with a written failure-mode result.
  - Formal checkpoint wiring and double replay are deferred until rolling,
    external, and online validation all pass.
- **Verification rule**:
  1. Unit tests hand-check full/short lift, strict causal cutoffs,
     popularity normalization, augmented column order, and config drift.
  2. The run contract binds the lift matrix SHA-256, shape
     `(200000, 100, 2)`, realized short-window value, source manifest,
     graph-score overlay, seed, and frozen config.
  3. Selection uses the unchanged global hard gates: every fold MRR and
     NDCG@10 non-decreasing; pooled Hit@1/3/10 non-decreasing; pooled mean
     rank non-increasing; improved queries greater than worsened.
  4. Reports include the causal positive-destination-popularity Q1
     segment's query count, baseline/candidate MRR, and delta for mechanism
     diagnosis. This segment is diagnostic only and cannot select a
     weight, model, formula, window, or salt.
  5. External is opened at most once after a selection lock, and packaging
     requires external status `accepted`.
- **Evidence source**: frozen config, RED/GREEN tests, run contract and
  SHA-256 hashes, rolling manifest, selector report/lock, diagnostic
  report, external receipt/report, and package artifact when authorized.
- **Pass criteria**: all global rolling hard gates select one frozen
  weight, the one-shot external gate accepts it, and the Q1 diagnostic is
  reported without affecting selection.
- **Confidence note**: rolling isolation and contract hashes strongly
  protect execution semantics. The consumed Q1 profile motivates but does
  not validate the model; only new rolling folds and the gated external
  opening can decide acceptance.
- **Judgment owner**: the existing robust selector owns rolling lock
  creation; the existing one-shot evaluator owns external acceptance.

## Current State

- The weak profile is destination-popularity Q1: 5,086 queries and MRR
  0.309 on the repeatedly consumed validation.
- Ten of fifteen structural features are constant there; only raw
  `cooccur_score` remains active, but its count scale is confounded with
  destination popularity.
- Repository search found no prior causal log-lift experiment.
- `cooccur_time_decay_score` was rejected on 2026-07-23 and is explicitly
  outside this candidate.
- The source exact-rolling champion manifest and robust selection/external
  state machine already exist.

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| Freeze config before metrics | keep and strengthen | Bind formulas, index flags, columns, window ratio, salt, folds, weights, and stop rules |
| Build one shared 200k lift matrix | keep | Fold prefixes share identical causal row features and avoid three repeated CPU passes |
| 65-column Setwise auxiliary | keep | Tests the proposed signal without changing the champion contract |
| Q1 diagnostic | rewrite | Report mechanism evidence for every weight, but never feed it to selection |
| Existing selector/external/package sequence | keep | Preserves the established anti-leakage boundary |
| Tail-weighted fallback | defer | It is a distinct hypothesis and requires a new ID after this result is closed |

## Drift Diagnosis

- **Goal drift**: selecting on Q1 improvement would turn diagnosis into
  segment overfitting; only global hard gates may select.
- **Phase drift**: opening external before the rolling lock would make the
  external set another selection fold.
- **Validation drift**: a correct lift value alone is insufficient; the
  run must also prove causal cutoffs, artifact identity, and global
  ranking stability.
- **Compatibility drift**: replacing a 63-column champion feature would
  change the baseline; columns 63 and 64 must be auxiliary-only additions.
- **Cleanup drift**: touching the rejected time-decay feature or unrelated
  structural features is excluded.

## Priority Rationale

- Freeze the hypothesis and execution contract before code or metrics.
- Prove causal feature semantics on a microscopic index before spending
  CPU/GPU time.
- Materialize and hash the shared lift matrix before fold training.
- Keep diagnosis downstream of materialization and outside the selector.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| Cache rows are chronological train events | confirmed | Strict-left cutoffs exclude the current row and future events | Runner validates monotonic time and candidate zero |
| `fit_grouped` exposes the required grouped-time maps | confirmed by interface | Enables deterministic micro-tests without an `InteractionTable` | Unit tests |
| Full train time span defines the short window | confirmed | Prevents post-metric window tuning | Runner records realized value |
| Q1 definition | confirmed | Per fold, stable lower quartile of causal positive-destination popularity; pooled diagnostics sum fold counts and reciprocal ranks | Diagnostic helper tests |
| External materializer | deferred unless lock | Avoids opening or even wiring consumed external assets for a rejected family | Implement only after selection lock |

## Phases

### Phase 1: Freeze and causal RED

- **Purpose**: make the hypothesis, causal semantics, and drift failures
  executable before implementation.
- **Entry condition**: no cooccur-lift candidate metric has been read.
- **Phase rules**:
  - Only frozen docs and tests may change before RED.
  - No production module or runner implementation yet.
  - `cooccur_time_decay_ratio` stays exactly `0.0`.
- **Todos**:
  - [x] Freeze the complete experiment contract.
    - **Surface**: frozen JSON and this goal document
    - **Proof**: config exists and is parseable before RED
    - **Depends on**: none
  - [x] Add micro-index tests for full/short lift and popularity
    normalization.
    - **Surface**: `tests/test_cooccur_lift.py`
    - **Proof**: focused test fails because the module is missing
    - **Depends on**: frozen config
  - [x] Add augmented-view and config-drift tests.
    - **Surface**: same test file
    - **Proof**: RED names the absent public API
    - **Depends on**: frozen config
- **Exit proof**: focused RED fails for missing cooccur-lift behavior.
- **Stop condition**: index interfaces contradict strict-left causal
  semantics or the frozen formula cannot be implemented without time decay.

### Phase 2: GREEN and runner contract

- **Purpose**: implement the minimum causal feature module, then the
  auditable one-pass materializer and fold trainer.
- **Entry condition**: correct RED evidence exists.
- **Phase rules**:
  - Each behavior returns to RED before implementation.
  - Keep the module at or below 300 lines.
  - Do not compute selector metrics in the producer.
- **Todos**:
  - [x] Implement lift scoring, augmented view, and frozen loader.
    - **Surface**: `src/jgrec/rankers/hybrid/cooccur_lift.py`
    - **Proof**: focused tests GREEN
    - **Depends on**: Phase 1 RED
  - [x] Implement one-pass 200k lift materialization and single-head fold
    training.
    - **Surface**: `scripts/train_dataset2_cooccur_lift_rolling.py`
    - **Proof**: compile, Ruff, remote preflight, run-contract hashes
    - **Depends on**: module GREEN
  - [x] Record filled RED/GREEN/REFACTOR evidence.
    - **Surface**: TDD document
    - **Proof**: commands and observed results are concrete
    - **Depends on**: focused and related regressions
- **Exit proof**: tests, Ruff, compileall, and remote preflight pass.
- **Stop condition**: materialized lift shape/hash or source baseline
  fingerprint differs from the frozen contract.

### Phase 3: Rolling decision and conditional continuation

- **Purpose**: let the pre-existing state machine accept or close the
  candidate without adaptive changes.
- **Entry condition**: three fold candidate matrices and diagnostics are
  complete.
- **Phase rules**:
  - Invoke the selector exactly once.
  - Q1 diagnostics cannot alter the selector inputs.
  - A rejection immediately closes the family with no rescan.
  - External and package paths require their preceding authorization.
- **Todos**:
  - [x] Run the existing selector and audit lock/external isolation.
    - **Surface**: remote result directory
    - **Proof**: selection report plus lock count
    - **Depends on**: complete rolling manifest
  - [x] If and only if locked, materialize and evaluate external once.
    - **Surface**: external manifest/state directory
    - **Proof**: exclusive receipt and accepted/rejected report
    - **Depends on**: selection lock
  - [x] Package only after acceptance; otherwise write the closing result.
    - **Surface**: package or result document
    - **Proof**: authorization evidence or zero-package audit
    - **Depends on**: external outcome or rolling rejection
- **Exit proof**: accepted package exists, or a rejected result documents
  the failure and confirms zero unauthorized artifacts.
- **Stop condition**: any request to alter weights, formula, window, salt,
  or Q1 definition after metrics.

## Dry-Run Findings

- The short-window value must be derived once from the complete train CSV
  span and recorded; deriving it per fold would silently change semantics.
- The 200k lift matrix can be shared because every value uses its row time
  with strict-left searches; it is not fit on a fold-specific statistic.
- Destination popularity counts are needed again for Q1 diagnosis and
  should be materialized as a non-model diagnostic sidecar rather than
  inferred from lift values.
- The external materializer is intentionally absent from Phase 2: a
  rolling rejection must not cause external asset access.

## Final Validation

```text
uv run --group dev pytest tests/test_cooccur_lift.py
uv run --group dev pytest \
  tests/test_cooccur_lift.py \
  tests/test_listwise_mlp_exact_blend.py \
  tests/test_robust_weight_selection.py
uv run --group dev ruff check \
  src/jgrec/rankers/hybrid/cooccur_lift.py \
  tests/test_cooccur_lift.py \
  scripts/train_dataset2_cooccur_lift_rolling.py
uv run python -m compileall \
  src/jgrec/rankers/hybrid/cooccur_lift.py \
  scripts/train_dataset2_cooccur_lift_rolling.py
```

Then verify the remote selection report, Q1 diagnostic report, lock count,
external receipt count, and package count.

## First Execution Step

Add `tests/test_cooccur_lift.py` against the public module API and run the
focused test to capture the missing-module RED.
