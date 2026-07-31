# Goal Document: Dataset2 OOF Utility Router v3

## Go / No-Go

- **Judgment**: Go
- **Reason**: frozen short/medium/long OOF experts and bounded top-k corrections
  already exist; the remaining hypothesis is narrow and testable without changing
  any expert trunk.

## Target Outcome

Implement and run a pure-Jittor abstaining utility router that distinguishes
`gain / no-change / loss` for each bounded medium/long action, defaults to the
short decoder, and routes at most 1% of rows. Only if this router passes the
frozen temporal gate may a second phase fine-tune LambdaMRR residuals on rows
that the router predicts will change reciprocal rank.

## Goal Definition

- **Type**: learning and quality
- **Boundary**:
  - reuse the existing short/medium/long OOF decoder logits and feature cache;
  - build label-free action features and OOF utility targets;
  - train all trainable components with `jt.nn.Module`;
  - select a route threshold on pre-gate rolling-origin slices;
  - persist the selection lock before reading gate rewards;
  - conditionally run change-only LambdaMRR fine-tuning.
- **Non-goals**:
  - changing decoder, CST, sequence encoder, or expert checkpoints;
  - using candidate IDs or the known positive column as an input feature;
  - replacing the Dataset2 champion or creating a submission without a passed
    gate;
  - treating the previously observed gate as statistically unseen.
- **Deferred work**:
  - full-data retraining and test-logit generation;
  - external submission evaluation;
  - broader Transformer, seed, cap, top-k, or Lambda-loss searches.
- **Verification rule**: tests must establish target construction, abstention,
  route coverage, unavailable-action masking, checkpoint replay, and feature
  permutation invariance; the experiment must then satisfy every frozen
  selection and gate criterion.
- **Evidence source**: pytest output, frozen protocol, selection lock, gate
  report, saved route arrays, checkpoint hashes, and Jittor provenance.
- **Pass criteria**:
  - selection delta is positive on both consecutive rolling-origin validation
    slices and no selection time slice is negative;
  - gate delta is positive and no gate time slice is negative;
  - route coverage is no more than 1%;
  - at least 12% of routed rows change reciprocal rank;
  - bounded top-k safety and checkpoint replay pass;
  - `trainable_frameworks == ["jittor"]` and
    `non_jittor_trainable_models == []`.
- **Confidence note**: the gate interval has been observed by earlier
  experiments, so it is a strict diagnostic replay rather than a fresh unbiased
  holdout. Promotion beyond research still requires later external evidence.
- **Judgment owner**: the frozen experiment runner; it must not enter Phase 2
  unless every pass criterion is true.

## Current State

- The selected joint LambdaMRR experiment routed 407 gate rows but only 16
  changed reciprocal rank; its gate delta was `-0.0000061120`.
- The same gate has an oracle delta of `+0.0017491889`, so action opportunities
  exist but route-row discrimination is weak.
- A prior high-confidence router was positive at roughly 0.8% coverage, while
  5% coverage was unstable.
- OOF availability is asymmetric: medium has 81,184 rows and long has 40,196
  rows. A utility router must train with per-action validity instead of reducing
  both actions to the long-horizon intersection.
- The repository has a dirty worktree containing user experiments. This work
  must add isolated v3 files and preserve all unrelated changes.

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| multi-horizon decoder experts | keep and freeze | they are audited pure-Jittor OOF assets |
| joint reward-regression router | replace for routing | dense reward regression selected mostly no-op rows |
| bounded top-k action construction | keep | it contains correction blast radius |
| LambdaMRR residual head | defer | stronger correction is not justified until route precision passes |
| 5% route coverage | remove | the observed no-op rate makes this unsafe |
| final gate | keep as diagnostic replay | row boundaries remain comparable, but the gate is no longer statistically unseen |

## Drift Diagnosis

- **Goal drift**: more cap/top-k/Lambda sweeps improve residual strength without
  proving row selection.
- **Phase drift**: joint training mixed route calibration and candidate
  correction, hiding which component failed.
- **Validation drift**: a positive aggregate selection delta masked sparse
  changes and temporal concentration.
- **Compatibility drift**: no public compatibility shim is required; v3 is an
  isolated research module.
- **Cleanup drift**: no unrelated hybrid-ranker cleanup belongs in this goal.

## Priority Rationale

- Route-row precision is the current limiting factor and can be tested while
  every expert remains frozen.
- Separate action validity recovers medium-horizon training evidence before any
  increase in model capacity.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| candidate column 0 is the offline positive only for target construction | confirmed | it must never enter input features | tests enforce permutation equivariance |
| old gate can prove unbiased generalization | rejected | it is diagnostic evidence only | report must state this |
| medium and long can share feature schema but use separate valid rows | assumed | enables 81k/40k training coverage | shape and mask tests |
| 12% change-hit rate is attainable at at most 1% coverage | unresolved | this is the stop-rule threshold | frozen run decides |

## Phases

### Phase 1: OOF Utility Router v3

- **Purpose**: prove that an abstaining pure-Jittor classifier can identify
  high-value bounded actions without changing experts.
- **Entry condition**: audited OOF arrays, timestamps, and candidate feature
  cache align.
- **Phase rules**:
  - tests precede production implementation;
  - experts and bounded alternatives are read-only;
  - features must be label-free and candidate-permutation invariant;
  - per-action validity masks are mandatory;
  - route coverage is hard-capped at 1%;
  - selection lock is written before gate rewards are computed.
- **Todos**:
  - [ ] Define gain/no-change/loss and conditional magnitude targets.
    - **Surface**: test and hybrid utility-router module
    - **Proof**: focused target-construction pytest
    - **Depends on**: none
  - [ ] Define an abstaining utility route with unavailable-action masking.
    - **Surface**: test and hybrid utility-router module
    - **Proof**: focused route-policy pytest
    - **Depends on**: target contract
  - [ ] Train and checkpoint a pure-Jittor hurdle model.
    - **Surface**: hybrid utility-router module
    - **Proof**: gradient and exact checkpoint-replay tests
    - **Depends on**: route-policy contract
  - [ ] Build and run the Dataset2 rolling-origin experiment.
    - **Surface**: experiment runner and result directory
    - **Proof**: frozen config, selection lock, gate report, hashes
    - **Depends on**: green unit tests
- **Exit proof**: every Phase 1 pass criterion is recorded in
  `evaluation-report.json`.
- **Stop condition**: any gate criterion fails, any non-Jittor trainable
  dependency appears, or any gate reward is read before the selection lock.

### Phase 2: Change-only LambdaMRR fine-tuning

- **Purpose**: improve correction quality only after v3 has proved route
  precision.
- **Entry condition**: Phase 1 `evaluation-report.json` has
  `"passed": true`.
- **Phase rules**:
  - training rows are restricted to OOF rows predicted as `change`;
  - the expert trunk and v3 router stay frozen;
  - residual remains row-centered, top-k bounded, and capped;
  - Phase 2 needs its own pre-gate lock and cannot inherit a passing decision.
- **Todos**:
  - [ ] Add a change-mask-aware LambdaMRR training contract.
    - **Surface**: test and conditional residual module/runner
    - **Proof**: test shows zero gradient contribution from non-change rows
    - **Depends on**: Phase 1 pass
  - [ ] Run conditional fine-tuning and compare against the frozen v3 route.
    - **Surface**: result artifact
    - **Proof**: independent selection/gate report and bounded audit
    - **Depends on**: change-mask test
- **Exit proof**: conditional residual improves the passed v3 baseline without
  violating any safety or temporal criterion.
- **Stop condition**: Phase 1 fails, or conditional fine-tuning is non-positive
  on either validation slice.

## Dry-Run Findings

- The only available final common interval is the same gate previously read by
  other experiments; it cannot be relabeled as unseen.
- Action-specific training must be implemented before assembling the common
  evaluation interval, otherwise the medium data advantage is lost.
- The runner can reuse the existing mmap arrays and feature cache; no 4 GB cache
  rebuild is required.
- Phase 2 is structurally blocked behind a machine-readable Phase 1 pass, so a
  failed v3 cannot accidentally trigger more LambdaMRR work.

## Final Validation

Run the focused and neighboring tests with `uv run pytest`, execute the complete
remote Dataset2 experiment, verify artifact hashes and Jittor provenance, and
inspect the frozen gate decision. Run Phase 2 only when the Phase 1 report passes.

## First Execution Step

Add failing tests for utility targets, unavailable-action abstention, hard route
coverage, and pure-Jittor checkpoint replay.
