# Goal Document: Dataset2 Targeted GNN Improvements

## Go / No-Go

- **Judgment**: Go
- **Reason**: Corrected perturbation and retrained controls confirmed stable GNN
  contribution, with short and recent graph windows providing the largest
  marginal value.

## Target Outcome

Evaluate exactly three GNN improvements—short-window repeat/time-decay edge
weighting, recent-window time decay, and full-100-candidate-aligned graph
training—while holding graph dimensions, layers, non-GNN features, Setwise,
and Dataset1 constant.

## Goal Definition

- **Type**: technical / learning
- **Boundary**:
  - `gnn_short`: `repeat` and `time_decay`.
  - `gnn_recent`: `time_decay`.
  - Best passing edge configuration: full-100-candidate graph negatives or
    listwise objective.
  - Existing 128-dimensional, two-layer graph architecture.
- **Non-goals**:
  - No deeper/wider GNN.
  - No TGN, GAT, new graph tower, or changes to full-history GNN.
  - No changes to statistics, structure, Two-Tower, sequence, or source-profile
    towers.
  - No submission package before all offline gates pass.
- **Deferred work**:
  - GNN architecture replacement.
  - Query-segment-specific graph fusion.
- **Verification rule**: Select variants on chronological slices 0 and 1,
  freeze the winner, and use slice 2 only as a forward gate. Compare the final
  `0.80 Setwise + 0.20 LightGBM` MRR against the current champion.
- **Evidence source**: RED/GREEN tests, cache/model hashes, per-variant
  full/three-slice MRR, and runtime reports.
- **Pass criteria**:
  - Baseline graph scores and fixed-blend metrics reproduce exactly.
  - Each variant changes only the named graph window and its three Setwise
    context channels.
  - Edge-weight winner improves full MRR by at least `0.001` with no slice
    declining.
  - Candidate-aligned objective is attempted only with the best passing edge
    configuration; it must add another `0.001` full MRR with no slice declining.
- **Confidence note**: Offline/online GNN calibration remains risky; slice 2 is
  protected from selection, and no package is produced in this goal.
- **Judgment owner**: Frozen temporal metric gates.

## Current State

- Champion fixed-blend MRR: `0.5469178184`.
- GNN removal reduces full MRR by `0.0062360204`, with all slices declining.
- Individual permutation degradation:
  `short 0.0057518441`, `recent 0.0031077490`, `full 0.0010428983`.
- Existing code supports global `none/repeat/time_decay` edge weighting, but
  window-specific configuration and candidate-aligned graph training require
  verified contracts.

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| Global GNN edge weighting | split by window | Only short/recent are authorized to change. |
| More GNN epochs/dimensions | remove | Ablation supports signal value, not capacity expansion. |
| Random/sampled graph negatives | keep as baseline | Needed for an isolated objective comparison. |
| Complete 100-candidate target | defer until edge gate | Higher implementation and overfit risk. |

## Drift Diagnosis

- **Goal drift**: General GNN tuning would exceed the three authorized changes.
- **Phase drift**: Candidate-listwise training cannot precede a stable
  per-window edge baseline.
- **Validation drift**: No variant may be selected from slice 2.
- **Compatibility drift**: Default global edge-weight behavior must remain
  unchanged.
- **Cleanup drift**: No unrelated graph/cache refactor is included.

## Priority Rationale

- Short/recent edge weighting directly targets the two graph features with
  confirmed marginal contribution.
- Candidate-aligned training has more leverage but also more complexity, so it
  follows the cheaper edge experiment.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| Per-window weighting can preserve the old global default | assumed | Compatibility | RED contract |
| Existing cache sidecars identify all 200k×100 query candidates | confirmed | Enables full-candidate objective | Cache reports |
| GNN columns can be rebuilt or substituted without changing 60 non-GNN columns | unresolved | Determines runtime | Inspect cache/encoder boundary after goal freeze |
| Time-decay ratio `0.05` is the first frozen value | frozen | Avoids an unbounded sweep | Existing implemented smoke setting |

## Phases

### Phase 1: Window-specific edge contract

- **Purpose**: Allow short/recent weighting without changing full-history GNN.
- **Entry condition**: Goal document is frozen.
- **Phase rules**:
  - RED before production changes.
  - Existing global config remains backward compatible.
- **Todos**:
  - [x] Test window-specific `none/repeat/time_decay` resolution.
    - **Surface**: graph config, edge construction, tests.
    - **Proof**: RED then GREEN.
    - **Depends on**: none.
- **Exit proof**: Full stays `none`; short/recent receive only the requested
  modes; existing tests remain green.
- **Stop condition**: Window isolation cannot be represented without changing
  checkpoint compatibility.

### Phase 2: Edge-weight variants

- **Purpose**: Compare short repeat, short decay, and recent decay.
- **Entry condition**: Phase 1 passes locally and remotely.
- **Phase rules**:
  - Dimension/layers/epochs and other towers remain fixed.
  - Select on slices 0/1; slice 2 is forward-only.
- **Todos**:
  - [x] Build and evaluate the three graph feature variants.
    - **Surface**: GNN features, Setwise control models, reports.
    - **Proof**: exact per-slice metrics and artifact hashes.
    - **Depends on**: Phase 1.
- **Exit proof**: One frozen edge winner or evidence-backed rejection.
- **Stop condition**: Full gain below `0.001` or any slice declines.

### Phase 3: Full-100 candidate-aligned objective

- **Purpose**: Align GNN training with the actual candidate-ranking task.
- **Entry condition**: Phase 2 has a passing edge winner.
- **Phase rules**:
  - Use all 99 candidate negatives per positive query.
  - Prefer grouped softmax/listwise loss; no test-label use.
  - Keep architecture and winning edge configuration fixed.
- **Todos**:
  - [ ] Test candidate group construction and listwise loss. (Not entered:
    Phase 2 gate failed.)
    - **Surface**: graph sampler/loss and tests.
    - **Proof**: RED/GREEN with positive at candidate index 0.
    - **Depends on**: Phase 2.
  - [ ] Train and evaluate the aligned graph objective. (Not authorized:
    Phase 2 gate failed.)
    - **Surface**: graph model and cached GNN columns.
    - **Proof**: full/three-slice MRR versus edge winner.
    - **Depends on**: candidate contract test.
- **Exit proof**: Final offline winner/rejection report.
- **Stop condition**: Candidate cache mismatch, leakage, or forward-slice
  regression.

## Dry-Run Findings

- A single global `edge_weighting` switch is insufficient because changing it
  would also alter the weak full-history graph feature.
- Full-candidate listwise training must use training-cache candidate groups,
  never validation or test labels.
- Rebuilding only graph columns is preferable, but feasibility must be proven
  before launching an expensive cache job.

## Final Validation

- Targeted and existing GNN tests pass locally and remotely.
- All source artifacts remain hash-stable.
- Each stage produces an auditable pass/reject report.
- No submission package is generated.

## Outcome

- Phase 1 passed: window-specific configuration and weighted message passing
  are implemented and covered by RED/GREEN evidence.
- Phase 2 rejected all requested edge modes:
  - `gnn_short repeat`: full `0.5474764`, only `+0.0005586` versus champion;
    slice 0 declined.
  - `gnn_short time_decay`: full `0.5455183`, below champion.
  - `gnn_recent time_decay`: full `0.5468777`, below champion and below the
    isolated recent-none control by `0.0016542`.
- The strongest controls were `short_none=0.5484923` and
  `recent_none=0.5485319`; this indicates that controlled GNN retraining is
  more promising than the tested edge formulas.
- Phase 3 was not entered because no edge-weight configuration met the frozen
  `+0.001` full-MRR and all-slices-non-decreasing gate.
- No submission package was generated.

## First Execution Step

Write a failing test proving short/recent windows can override edge weighting
while full-history retains the existing global default.
