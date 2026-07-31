# Goal Document: Dataset2 GNN Marginal-Contribution Ablation

## Go / No-Go

- **Judgment**: Go
- **Reason**: The validation cache already contains the three GNN features, so
  their marginal contribution can be measured before committing to expensive
  graph-tower retraining.

## Target Outcome

Determine whether `gnn_full`, `gnn_recent`, and `gnn_short` make a stable,
material contribution to the current Dataset2
`0.80 Setwise + 0.20 LightGBM` champion, and authorize GNN improvement work
only when the evidence passes frozen thresholds.

## Goal Definition

- **Type**: learning / quality
- **Boundary**: Existing 20k validation cache, existing Setwise and LightGBM
  models, GNN feature columns only; optional no-GNN Setwise retraining from the
  existing 200k training cache.
- **Non-goals**:
  - No GNN architecture, dimension, layer, or loss changes.
  - No final encoder rebuild or submission package.
  - No leaderboard-based feature decision.
- **Deferred work**:
  - Time-decayed/repeat-weighted GNN.
  - Candidate-listwise GNN training.
  - TGN, GAT, or additional graph towers.
- **Verification rule**: First neutralize and deterministically permute GNN
  features without modifying the source cache. Only if joint perturbation
  materially hurts the champion may a fixed-hyperparameter no-GNN Setwise
  control be trained.
- **Evidence source**: Cache/model hashes, exact baseline reproduction,
  perturbation MRR by three chronological slices, and conditional retrained
  control.
- **Pass criteria**:
  - Baseline reproduces full/slice MRR within `1e-10`.
  - Source cache remains byte-identical.
  - Joint neutralization or permutation reduces full blend MRR by at least
    `0.002` with no chronological slice improving.
  - If retraining is triggered, the no-GNN Setwise control must trail by at
    least `0.001` full MRR and not beat the champion on any slice.
- **Confidence note**: Perturbation measures current-model dependence and may
  overstate irreplaceability; retraining is the stricter causal control.
- **Judgment owner**: Frozen offline diagnostic gates.

## Current State

- Online champion score: `1.3530197201`.
- Offline fixed blend: full MRR `0.5469178184`.
- The current 63-feature cache includes three graph scores.
- Historical GNN experiments show severe offline/online calibration risk, so
  feature presence alone is insufficient evidence.

## Priority Rationale

- Perturbation is minutes and has no training risk.
- Retraining is conditional because it is more expensive but distinguishes
  true information value from a model that merely depends on a feature.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| GNN feature names are exactly discoverable from cache metadata | assumed | Prevents wrong-column ablation | Assert exact names before inference |
| Setwise payload contains training mean/std | confirmed | Enables neutralization at standardized zero | Model artifact |
| Query-row permutation is deterministic | frozen | Makes diagnostic reproducible | Seed 60 |
| Existing caches are read-only | frozen | Protects expensive artifacts | Pre/post SHA-256 |

## Phases

### Phase 1: Ablation contract

- **Purpose**: Define safe, deterministic feature perturbation.
- **Entry condition**: Goal document is frozen.
- **Phase rules**:
  - RED before implementation.
  - Never write into memory-mapped cache arrays.
- **Todos**:
  - [x] Test neutralization and candidate permutation on selected columns.
    - **Surface**: reusable GNN ablation helper and tests.
    - **Proof**: RED then GREEN, including source-array immutability.
    - **Depends on**: none.
- **Exit proof**: Targeted tests and Ruff pass.
- **Stop condition**: Feature columns cannot be mapped exactly.

### Phase 2: Cached perturbation

- **Purpose**: Measure current champion dependence on each GNN feature.
- **Entry condition**: Phase 1 passes locally and on the server.
- **Phase rules**:
  - Reproduce baseline before ablation.
  - Record individual and joint neutralization/permutation.
  - Do not select a variant from the third slice.
- **Todos**:
  - [x] Run the cached Dataset2 GNN perturbation report.
    - **Surface**: JSON report.
    - **Proof**: full and three-slice metrics, deltas, hashes.
    - **Depends on**: Phase 1.
- **Exit proof**: Complete deterministic report and gate decision.
- **Stop condition**: Hash or baseline reproduction fails.

### Phase 3: Conditional no-GNN retraining

- **Purpose**: Test whether Setwise can recover without graph features.
- **Entry condition**: Joint perturbation passes the `0.002` dependence gate.
- **Phase rules**:
  - Same train/validation caches and training hyperparameters.
  - Remove only the three GNN columns.
- **Todos**:
  - [x] Train and compare a no-GNN Setwise control if authorized.
    - **Surface**: control model and evaluation report.
    - **Proof**: full/three-slice MRR versus champion.
    - **Depends on**: Phase 2 gate.
- **Exit proof**: Evidence-backed Do/Defer judgment for GNN improvement.
- **Stop condition**: Perturbation gate fails.

## Dry-Run Findings

- Neutralizing at each feature's training mean corresponds to standardized
  zero and avoids unrealistic raw zeros.
- Permuting complete query rows preserves within-query candidate score
  structure while breaking alignment with the source query.
- A retrained no-GNN control is necessary only when perturbation suggests a
  material dependency.

## Final Validation

- Tests and Ruff pass locally and remotely.
- Baseline and artifact hashes reproduce exactly.
- Report states whether no-GNN retraining ran and whether GNN improvement is
  justified.

## Final Outcome

- Corrected joint mean-neutralization reduced fixed-blend full MRR by
  `0.0084609169`; corrected within-query candidate permutation reduced it by
  `0.0107181971`. All three slices declined in both joint variants.
- Individual permutation degradation ranked:
  `gnn_short` (`0.0057518441`) >
  `gnn_recent` (`0.0031077490`) >
  `gnn_full` (`0.0010428983`).
- The strict no-GNN Setwise control reduced the fixed-blend full MRR from
  `0.5469178184` to `0.5406817980`, a degradation of `0.0062360204`.
- No-GNN slice degradation was
  `(0.0064579529, 0.0077648010, 0.0044850447)`.
- Source training and validation cache hashes were identical before and after.
- **Decision**: GNN contribution is confirmed. GNN improvement experiments are
  authorized, with short/recent windows ahead of the full-history window.

## First Execution Step

Add a failing test for immutable mean-neutralization and deterministic
query-row permutation of selected feature columns.
