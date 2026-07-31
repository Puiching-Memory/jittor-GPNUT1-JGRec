# Goal Document: Dataset2 Cooccur Lift Online Promotion

## Go / No-Go

- **Judgment**: Go
- **Completion**: Achieved on 2026-07-29; final status is
  `accepted/promoted`.
- **Reason**: The user-reported online score `1.357529740346302` is
  `+0.0018295152278673` above the preregistered promotion threshold
  `1.3557002251184347`. Rolling, one-shot external, package integrity, and
  online confirmation now authorize the deferred checkpoint wiring and
  double replay.

## Target Outcome

Create a new contest checkpoint that preserves the current Dataset1 state
and every existing Dataset2 champion field, installs the accepted
`cooccur_lift_aux_expert_v1` head at locked weight `0.50`, and produces two
byte-identical full Dataset2 standard-load replays. Bind the resulting
champion to checkpoint SHA-256
`796d8d21a0c706ad11f244385b314d471d522c3b807748a54fe4ac78722f5880`
and separately prove raw-model equivalence and tie-safe service
equivalence.

## User-authorized protocol amendment

- **Authorization**: explicit user instruction on 2026-07-29 to bind the
  checkpoint, persist the replay report, distinguish raw-model from
  tie-safe service equivalence, and set the promotion to
  accepted/promoted.
- **Reason**: the original rule compared an eight-decimal package assembled
  from a previously tie-safe champion CSV with a seventeen-digit standard
  service replay that applies tie handling after the final blend. That is a
  comparison across two different serialization/postprocessing boundaries,
  not a raw-model identity check.
- **Replacement rule**:
  1. Raw-model equivalence is gated by immutable model/feature artifact
     identity and the bounded sequential raw replay covering the
     worst-known drift row, with maximum absolute error at most `5e-7` and
     zero Top-1 disagreements.
  2. Tie-safe service equivalence is gated by two byte-identical full
     standard-load replays, zero exact service ties, and zero Top-1
     disagreements against the accepted online ZIP.
  3. Full-score numeric deltas and Top-3/Top-10 set deltas between the two
     serialization paths remain mandatory diagnostics, but are not reused
     as a raw-model gate.
- **Unchanged protocol**: no weight/formula/window/salt/model change, no
  retraining, no metric-based reselection, and no rescan.
- **Evidence preservation**: the earlier failed auto-finalizer status and
  immutable pre-wiring receipt remain untouched; the promoted manifest
  supersedes only the obsolete combined replay gate.

## Goal Definition

- **Type**: technical / quality / delivery
- **Boundary**:
  - Dataset2 runtime integration only; Dataset1 checkpoint state is copied
    unchanged.
  - Source checkpoint SHA-256 is fixed at
    `0b0a846cb6c7f5b5403a75a04ed12707340f024dee99a27cded3258bb108a7aa`.
  - Auxiliary model SHA-256 is fixed at
    `9bed5d911eb7ca168b9d8ac535d1866608716fc4cf904248b66a3011968a4f83`.
  - Test lift SHA-256, query fingerprint, selection lock, external report,
    online ZIP, online score, and weight `0.50` are bound before checkpoint
    construction.
  - Runtime stores the actual auxiliary head plus the verified causal lift
    sidecar. It looks up lift rows using 256-bit query fingerprints and
    computes the Setwise auxiliary probability during inference.
  - The final formula remains
    `0.50 * champion_probability + 0.50 * auxiliary_probability`.
- **Non-goals**:
  - No weight, formula, feature, short-window, seed, or training change.
  - No new metric read, leaderboard rescan, or model retraining.
  - No mutation or overwrite of the existing champion checkpoint/package.
  - No reuse of `cooccur_time_decay_score`.
  - No embedding of the already blended final probability matrix as a
    substitute for the real auxiliary head.
- **Deferred work**:
  - General-query causal lift construction outside the official Dataset2
    test contract.
  - Deleting or archiving the previous champion.
- **Verification rule**:
  1. A frozen online-promotion receipt binds the exact score and online ZIP
     hash before production code is changed.
  2. Query fingerprints are order-independent, collision-checked, and reject
     missing or malformed queries.
  3. Snapshot/hydrate preserves the auxiliary head, lift sidecar, locked
     weight, and provenance while old checkpoints hydrate unchanged.
  4. Checkpoint state audit allows exactly one Dataset2 top-level addition;
     all existing Dataset2 fields and Dataset1 pickle hashes remain stable.
  5. Two independent full standard-load replays are byte-identical.
  6. Raw-model equivalence passes the amended raw contract, and replay
     versus the online-tested package passes the amended tie-safe service
     contract on all 153,420 rows.
- **Evidence source**: user score, frozen promotion receipt, RED/GREEN tests,
  checkpoint state audit, two replay CSVs, replay comparison report, hashes.
- **Pass criteria**: all six verification rules and both amended
  equivalence contracts pass; otherwise the online ZIP remains accepted
  but the formal checkpoint is not promoted.
- **Confidence note**: the online score owns the model decision. Replay
  checks prove reproducibility and serving equivalence, not a second model
  selection.
- **Judgment owner**: the user-provided online score authorizes entry; tests
  and replay audits authorize checkpoint promotion.

## Current State

- Accepted online candidate ZIP SHA-256:
  `7ebfeb7ea29d8dcd03a43a7433a43ddc8de0e24245d93115ecbf8ebd17ef50eb`.
- New 5,136,908,105-byte checkpoint already exists remotely with SHA-256
  `796d8d21a0c706ad11f244385b314d471d522c3b807748a54fe4ac78722f5880`;
  standard hydrate and protected-state audits passed.
- Full replay A and B already exist, are byte-identical, and share SHA-256
  `2b4012edb3a9d18675b00417553b0366438db44a9d5680a7566e693cd20b21e0`.
- The old combined gate failed only because 165 rows exceeded `5e-7` after
  comparing different serving/serialization boundaries; all 153,420
  Top-1 choices agree.
- Standard prediction already groups future queries by source and restores
  CSV order, so the actual auxiliary head can reuse the proven memory-safe
  inference path.
- The workspace contains many unrelated changes; only cooccur-lift
  promotion surfaces may be edited.

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| “Wire checkpoint after online win” | keep and activate | The online score crossed the frozen threshold |
| Reuse packaged final probabilities | remove | It would not be a real model checkpoint |
| Recompute the 13 GB causal index inside each prediction batch | replace | Store the already verified lift sidecar keyed by exact query fingerprints |
| Two full standard-load replays | keep | This is the promised reproducibility proof |
| Replace current champion in place | remove | Promotion must publish a new immutable artifact |
| One numeric gate after service tie handling | rewrite | Apply numeric tolerance to raw evidence and a separate deterministic ranking contract to service output |

## Drift Diagnosis

- **Goal drift**: retraining or changing the formula would turn promotion
  into a new experiment.
- **Phase drift**: building the 5 GB checkpoint before a focused runtime RED
  would make artifact construction the first behavior test.
- **Validation drift**: two equal files alone do not prove equivalence to the
  online-tested package, so a separate numeric/Top-1 audit is required.
- **Compatibility drift**: making the new field mandatory would break all
  old checkpoints; absent state must retain current behavior.
- **Cleanup drift**: unrelated checkpoint or runner cleanup is excluded.

## Priority Rationale

- Freeze the user score and artifact identity first.
- Test fingerprint lookup and snapshot/hydrate before writing a multi-GB
  checkpoint.
- Run one small equivalence replay before paying for two full replays.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| User score belongs to the downloaded ZIP hash | confirmed by conversation | Authorizes promotion | Bind both in receipt |
| Test lift rows align to official test CSV | confirmed by prior materialization report | Required for runtime lookup | Revalidate fingerprint and hashes |
| Runtime batch ordering may differ | confirmed | Cursor lookup would be unsafe | Use 256-bit per-query fingerprints |
| Replay numeric tolerance `5e-7` is sufficient for raw evidence | confirmed | Bounds model/runtime drift before serving postprocessing | Require zero raw Top-1 disagreements |
| Service equivalence is Top-1 exact and tie-free, not byte identity to the eight-decimal ZIP | user-authorized protocol amendment | Separates scoring identity from serialization/tie policy | Persist all residual rank diagnostics |

## Phases

### Phase 1: Freeze promotion authorization

- **Purpose**: make the online decision and immutable inputs auditable.
- **Entry condition**: exact user score is available.
- **Phase rules**:
  - Only documentation and immutable receipt creation are allowed.
  - No checkpoint or runtime code change before receipt hash exists.
- **Todos**:
  - [x] Write the online-promotion receipt.
    - **Surface**: experiment JSON.
    - **Proof**: score delta, ZIP SHA, lock SHA, external report SHA.
    - **Depends on**: none.
- **Exit proof**: receipt status is
  `online_score_passed_before_checkpoint_wiring`.
- **Stop condition**: score is not strictly above the frozen threshold or
  candidate ZIP hash differs.

### Phase 2: TDD the runtime contract

- **Purpose**: protect exact lookup, legacy compatibility, and blend
  semantics before modifying the ranker.
- **Entry condition**: Phase 1 receipt exists.
- **Phase rules**:
  - RED must fail because the public checkpoint integration API is absent.
  - Each GREEN slice is minimal; no artifact construction yet.
- **Todos**:
  - [x] RED/GREEN query fingerprint lookup and drift rejection.
    - **Surface**: new cooccur-lift checkpoint module and focused tests.
    - **Proof**: correct missing-API RED, then focused GREEN.
    - **Depends on**: Phase 1.
  - [x] RED/GREEN snapshot, hydrate, and probability blend.
    - **Surface**: hybrid ranker tests/runtime.
    - **Proof**: in-memory and restored scores match; legacy snapshot remains
      unchanged.
    - **Depends on**: lookup GREEN.
- **Exit proof**: focused and related checkpoint tests pass; Ruff and
  compileall pass.
- **Stop condition**: runtime needs changes outside the allowlisted
  cooccur-lift state and final blend hook.

### Phase 3: Build and audit the checkpoint

- **Purpose**: publish a new immutable checkpoint without champion drift.
- **Entry condition**: Phase 2 GREEN.
- **Phase rules**:
  - Refuse overwrite and refuse any hash/provenance mismatch.
  - Only `cooccur_lift_auxiliary_state` may be added to Dataset2.
- **Todos**:
  - [x] Build the new two-dataset checkpoint and integration report.
    - **Surface**: checkpoint builder and remote artifact.
    - **Proof**: Dataset1/state audits, standard hydrate, output SHA-256.
    - **Depends on**: Phase 2.
  - [x] Run a bounded sample equivalence replay.
    - **Surface**: official test prefix/sample.
    - **Proof**: runtime versus materialized auxiliary blend within tolerance.
    - **Depends on**: checkpoint build.
- **Exit proof**: sample passes and protected state hashes are stable.
- **Stop condition**: source state drift, lookup miss, or score disagreement.

### Phase 4: Double replay and promotion report

- **Purpose**: prove deterministic standard serving and online-package
  equivalence.
- **Entry condition**: Phase 3 sample passes.
- **Phase rules**:
  - Run two independent full standard loads.
  - Do not change code or artifacts between replay A and B.
- **Todos**:
  - [x] Generate replay A and replay B.
    - **Surface**: full Dataset2 CSVs.
    - **Proof**: byte-identical SHA-256.
    - **Depends on**: Phase 3.
  - [x] Apply the amended raw/service equivalence contracts and close docs.
    - **Surface**: replay report and experiment result.
    - **Proof**: bounded raw max delta `<=5e-7`; byte-identical, tie-free
      full service replays; zero Top-1 disagreements; promoted manifest.
    - **Depends on**: both replays.
- **Exit proof**: checkpoint status is promoted with complete evidence.
- **Stop condition**: either replay differs or package equivalence fails.

## Dry-Run Findings

- The source checkpoint uses a compact future-only structure index, so it
  cannot recreate short-window causal lift timestamps directly.
- Embedding the verified lift sidecar is necessary for exact official-test
  replay; 256-bit fingerprints avoid ordering assumptions.
- Standard runner source grouping already removes the major inference
  performance problem discovered during packaging.
- The online package uses eight-decimal CSV persistence while standard
  replay writes 17 significant digits; equivalence must be numeric plus
  Top-1, not raw byte identity across those two formats.
- The accepted package blends an already tie-safe champion CSV with the
  auxiliary matrix and then rounds to eight decimals. Standard serving
  blends raw checkpoint heads first, applies final tie handling, and writes
  seventeen significant digits. The old final-CSV numeric gate therefore
  crossed a postprocessing boundary.
- The bounded sequential replay processed twelve source-grouped batches
  through the worst-known drift row: maximum raw error
  `6.938893903907228e-18`, zero values above `5e-7`, and zero Top-1
  disagreements.

## Final Validation

- Focused RED/GREEN tests and existing checkpoint regressions.
- Ruff and compileall on every promotion file.
- Frozen receipt/hash audit.
- Dataset1 and protected Dataset2 pickle hashes unchanged.
- Two full replay files byte-identical.
- Raw-model contract: artifact identity plus bounded worst-row replay,
  maximum absolute delta `<=5e-7`, zero Top-1 disagreements.
- Tie-safe service contract: byte-identical full replays, no exact ties,
  and zero Top-1 disagreements against the accepted online ZIP.
- Final replay report status `accepted`; promoted manifest binds checkpoint
  SHA-256 and status `promoted`.

## First Execution Step

Completed: the amended equivalence classifier passed RED/GREEN, and the
read-only finalizer published the accepted replay report and promoted
manifest against the already completed checkpoint and replay artifacts.
