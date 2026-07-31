# Goal Document: Dataset2 Multi-Interest Production Path

## Go / No-Go

- **Judgment**: Go
- **Reason**: The frozen proxy passed full, three-slice, and new-edge gates;
  the remaining work is feature parity between validation and final queries.

## Target Outcome

Produce a loadable contest checkpoint and validated submission package whose
Dataset2 prediction path computes the same nine frozen multi-interest features
for final test queries.

## Goal Definition

- **Type**: delivery / technical
- **Boundary**: Optional checkpoint proxy state, final-prefix `gnn_recent`
  embeddings and temporal2/K2/K4 centers, augmented Setwise prediction,
  Dataset1 byte copy, Dataset2 package generation.
- **Non-goals**:
  - No K/history/feature/weight tuning.
  - No learned multi-interest graph tower.
  - No LightGBM or Dataset1 changes.
- **Deferred work**:
  - End-to-end multi-interest training.
- **Verification rule**: A fresh-process proxy rebuild must independently pass
  the frozen validation gates before final test prediction; generated CSV/zip
  must pass submission validation.
- **Evidence source**: RED/GREEN tests, validation replay, checkpoint reload,
  artifact hashes, submission validator.
- **Pass criteria**: Fresh-process full MRR improves by at least `+0.002` with
  all three slices non-decreasing; checkpoint loads; Dataset1 bytes match
  champion; Dataset2 row count and zip validate.
- **Confidence note**: Final-query proxy uses the full training prefix, matching
  the contest inference boundary.
- **Judgment owner**: Automated replay and artifact validators.

## Current State

- Frozen proxy fixed-blend MRR: `0.5509812280`.
- Nine feature definitions and Setwise model are frozen.
- Current ranker has no optional proxy state in snapshot/hydrate/predict.

## Priority Rationale

- Prove optional checkpoint compatibility before building a large final state.
- Replay validation through the same production augmentation function before
  generating any package.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| LightGBM remains on original 63 features | confirmed | Avoids schema mismatch | Prediction contract |
| Setwise receives 72 raw / 216 context features | confirmed | Model compatibility | RED test |
| Missing source/target maps to zero proxy features | frozen | Handles cold IDs | Unit test |
| Blend weight remains 0.80 | frozen | Prevents post-validation tuning | Builder assertion |

## Phases

### Phase 1: Optional prediction contract

- **Purpose**: Add backward-compatible proxy augmentation.
- **Entry condition**: Goal frozen.
- **Phase rules**:
  - RED before production code.
  - Old checkpoints must remain unchanged.
- **Todos**:
  - [x] Test proxy feature ordering, zero handling, and optional append.
    - **Surface**: proxy module, ranker snapshot/hydrate/predict.
    - **Proof**: RED/GREEN plus checkpoint tests.
    - **Depends on**: none.
- **Exit proof**: Targeted tests green.
- **Stop condition**: Existing checkpoint behavior changes without proxy state.

### Phase 2: Final-state builder

- **Purpose**: Build an auditable final Dataset2 state and package.
- **Entry condition**: Phase 1 green locally and remotely.
- **Phase rules**:
  - Frozen definitions only.
  - Replay validation before test prediction.
  - No source artifact overwrite.
- **Todos**:
  - [x] Build final-prefix embeddings/centers and attach proxy state.
  - [x] Attach frozen Setwise state and weight.
  - [x] Reload checkpoint and independently revalidate the frozen metric gate.
  - [x] Generate and validate result.zip.
- **Exit proof**: Candidate report with hashes and replay metrics.
- **Stop condition**: Replay mismatch, schema mismatch, or submission failure.

## Dry-Run Findings

- The augmented features must feed only Setwise; LightGBM continues to consume
  the original feature tensor.
- Proxy state can be optional, preserving all prior checkpoints.
- Validation replay needs prefix-specific proxy state, while the saved contest
  checkpoint needs full-training-prefix state.

## Final Validation

- [x] Targeted and checkpoint tests pass.
- [x] Independent validation replay passes the frozen gate.
- [x] Checkpoint reload and prediction succeed.
- [x] CSV/zip validators pass; the submission package and report are local.

## Outcome

- Server checkpoint:
  `checkpoints/d1_champion_d2_multi_interest_proxy_v3_seed60_20260725.pkl`
  (`5,087,185,389` bytes,
  SHA-256 `26bd659c10022e108777b7f0dc7772b4077a022252b8defa9b58abc9f0983028`).
- Local submission:
  `result/d1_champion_d2_multi_interest_proxy_v3_package_seed60_20260726/result.zip`
  (`62,612,282` bytes,
  SHA-256 `12c832e9c07448c4bb05c95f92df2a031e56e6991f8433523caf5669689812e1`).
- The package contains `61,051` Dataset1 and `153,420` Dataset2 rows.
- The full checkpoint remains on the server because the measured private-link
  transfer rate would require about one hour; incomplete local transfer files
  were removed so they cannot be mistaken for a valid checkpoint.

## First Execution Step

Write a failing test for optional query-time proxy augmentation with exact
feature order and cold-ID zeros.
