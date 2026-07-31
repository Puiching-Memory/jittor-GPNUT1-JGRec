# Goal Document: Dataset2 Multi-Interest Proxy

## Go / No-Go

- **Judgment**: Go
- **Reason**: The proxy can test whether source-interest collapse is a real
  error source without committing to a new graph tower.

## Target Outcome

Determine whether candidate similarity to multiple source-interest centroids
adds stable Dataset2 ranking signal beyond the current `gnn_recent` feature.

## Goal Definition

- **Type**: learning / technical
- **Boundary**: Reuse `gnn_recent` item embeddings; build temporal-2,
  deterministic K=2, and deterministic K=4 source interests; append
  max/top-2/coverage candidate features; retrain only Dataset2 Setwise.
- **Non-goals**:
  - No production multi-interest graph tower.
  - No Dataset1 changes.
  - No LightGBM or other tower retraining.
- **Deferred work**:
  - Learned interest routing and end-to-end multi-interest GNN training.
- **Verification rule**: Compare a fixed `0.80 Setwise + 0.20 LightGBM` blend
  on the same full-100 validation cache and three chronological slices.
- **Evidence source**: RED/GREEN tests, cache hashes, full/slice/new-edge MRR,
  and the remote run report.
- **Pass criteria**: Full MRR `+0.002`, all three slices non-decreasing, and
  new-edge MRR `+0.003` versus the current champion.
- **Confidence note**: This is a representation proxy, not evidence that an
  end-to-end multi-interest tower will achieve the same gain.
- **Judgment owner**: Frozen offline metric gate.

## Current State

- Champion fixed-blend MRR: `0.5469178184`.
- GNN removal costs `0.0062360`.
- Repeat/time-decay edge weighting failed.
- Rebuilt unweighted `gnn_recent` control reached `0.5485319`, suggesting
  representation retraining has more value than edge reweighting.

## Priority Rationale

- Test interest separation with cached embeddings before paying for a new
  architecture.
- Use fixed temporal prefixes so training and validation proxies cannot see
  future interaction labels.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| Positive candidate is cache position 0 | confirmed | Defines listwise label | Cache contract |
| Fixed-prefix histories avoid future leakage | confirmed | Makes proxy auditable | Script assertion |
| K-means cost is bounded by last 64 targets/source | frozen | Controls runtime | Experiment config |
| Cosine features are sufficient proxy | assumed | Limits conclusion | Metric gate |

## Phases

### Phase 1: Proxy contract

- **Purpose**: Define deterministic interest centers and candidate features.
- **Entry condition**: Goal frozen.
- **Phase rules**:
  - RED before implementation.
  - Pure NumPy boundary; no Jittor dependency.
- **Todos**:
  - [x] Test temporal and clustered centers plus max/top-2/coverage output.
    - **Surface**: proxy module and tests.
    - **Proof**: RED then GREEN.
    - **Depends on**: none.
- **Exit proof**: Deterministic tests pass locally.
- **Stop condition**: Feature construction requires query-future events.

### Phase 2: Full-cache experiment

- **Purpose**: Evaluate proxy signal with the current reranker.
- **Entry condition**: Phase 1 green.
- **Phase rules**:
  - Histories stop at `context_end` for training and `train_end` for
    validation.
  - Original feature caches remain read-only.
  - Same Setwise hyperparameters and LightGBM expert.
- **Todos**:
  - [x] Rebuild `gnn_recent` embeddings for both prefixes.
  - [x] Generate nine proxy channels for 200k×100 training and 20k×100
    validation rows.
  - [x] Retrain Setwise and evaluate full/slices/new-edge.
- **Exit proof**: Atomic report with hashes and gate verdict.
- **Stop condition**: Cache mismatch, future leakage, non-finite feature, or
  any source cache mutation.

## Dry-Run Findings

- K=2/4 must be deterministic to prevent seed search.
- Interest histories must be capped before clustering to keep CPU work bounded.
- Proxy arrays can be appended lazily, avoiding a second 5.04 GB feature cache.

## Final Validation

- Unit tests pass.
- Remote pipeline exits zero.
- Full, three-slice, and new-edge metrics are present.
- No submission package is generated automatically.

## Outcome

- Remote pipeline exited `0` in `357.03` seconds.
- Fixed-blend MRR improved from `0.5469178184` to `0.5509812280`
  (`+0.0040634096`).
- Slice deltas were `+0.0047543407`, `+0.0044276391`, and
  `+0.0030080907`.
- The validation set contained 20,000 new-edge rows under the prefix contract;
  their MRR improved by `+0.0040634096`.
- All frozen gates passed. A formal multi-interest direction is authorized,
  but no submission package was generated.

## First Execution Step

Write a failing pure-NumPy test for temporal/K-means interest centers and the
three candidate affinity channels.
