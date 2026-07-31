# Goal Document: Dataset2 Multi-Interest Gated OOF

## Go / No-Go

- **Judgment**: Go
- **Reason**: The multi-interest candidate regressed online by `-0.00057222`,
  so the next useful question is whether its gains are confined to a
  label-free, reproducible query segment. The experiment has an explicit
  `+0.002` stop rule.

## Target Outcome

Determine whether a label-free query gate can select the multi-interest model
only where it is reliably better than the `1.3530197200911278` champion, with
stable improvement across three temporal slices and temporal OOF folds.

## Goal Definition

- **Type**: learning / quality
- **Boundary**: Dataset2 query-level champion-versus-multi-interest diagnosis,
  confidence features, gated fallback, three temporal slices, and temporal
  OOF evaluation.
- **Non-goals**:
  - No new tower or embedding training.
  - No Dataset1 changes.
  - No test-label or leaderboard-label fitting.
  - No submission package unless the offline gate passes.
- **Deferred work**:
  - New graph architectures and end-to-end multi-interest training.
- **Verification rule**: Compare the gate against the champion on identical
  full-candidate validation queries, then require temporal OOF and slice
  stability.
- **Evidence source**: Query-level reciprocal-rank deltas, error slices,
  RED/GREEN tests, temporal OOF metrics, and an experiment report.
- **Pass criteria**: Mean full-candidate MRR improvement is at least `+0.002`
  over the champion across OOF folds; every fold and each of the three
  chronological slices is non-decreasing.
- **Confidence note**: OOF gating prevents a query from being selected by a
  gate fitted on its own label. It remains an offline proxy, so no package is
  generated below the margin.
- **Judgment owner**: Automated temporal OOF evaluator and the fixed stop rule.

## Current State

- Champion online score: `1.3530197200911278`.
- Multi-interest online score: `1.3524474995709168`.
- Online regression: `-0.000572220520211`.
- The frozen offline blend previously reported `+0.00406341`, demonstrating a
  validation-to-leaderboard calibration gap.
- Champion and multi-interest checkpoints and cached validation artifacts are
  available on the server.

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| Compare Dataset2 rankings | keep, execute first | Establishes which queries improve or regress |
| High-confidence fallback gate | rewrite | Gate may use label-free inference features only |
| Three time periods | keep | Detects temporal instability |
| Multi-fold OOF | move before packaging | Prevents in-sample gate selection |
| Generate a candidate | defer | Allowed only after the fixed `+0.002` gate |

## Drift Diagnosis

- **Goal drift**: A higher in-sample MRR alone would not explain or repair the
  online regression.
- **Phase drift**: Gate implementation must follow error diagnosis, not precede
  it.
- **Validation drift**: The earlier single replay looked positive while the
  leaderboard declined; OOF and slice gates are now mandatory.
- **Compatibility drift**: Champion fallback must be exact when the gate is
  false.
- **Cleanup drift**: No unrelated model or cache refactors are included.

## Priority Rationale

- Query-level disagreement reveals whether a usable high-confidence segment
  exists before spending effort on a gate.
- Exact fallback behavior is protected before fitting any selector.
- Temporal OOF is the final decision boundary, not an optional report.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| Identical validation query IDs can be reconstructed for both models | assumed | Required for paired deltas | Artifact inventory |
| Gate features are available at inference without labels | confirmed | Prevents leakage | Feature schema test |
| Three chronological slices use equal query-count boundaries | assumed | Makes comparisons auditable | Evaluator |
| Three forward temporal OOF folds are feasible from cached queries | assumed | Controls fitting bias | Cache inventory |

## Phases

### Phase 1: Paired error diagnosis

- **Purpose**: Locate champion-versus-multi-interest wins and losses.
- **Entry condition**: Goal document frozen.
- **Phase rules**:
  - Pair only identical queries and candidates.
  - Labels may be used for diagnosis, never as inference-time gate inputs.
- **Todos**:
  - [x] Reconstruct or load paired full-candidate predictions.
    - **Surface**: server validation caches and checkpoints.
    - **Proof**: identical query count, labels, and candidate ordering.
    - **Depends on**: none.
  - [x] Report RR delta distributions and error slices.
    - **Surface**: experiment report.
    - **Proof**: win/loss/tie counts and MRR by repeat/new edge, history,
      popularity, prior strength, and recency/long-memory signals.
    - **Depends on**: paired predictions.
- **Exit proof**: At least one label-free segment has meaningful lift and
  coverage, or the direction stops immediately.
- **Stop condition**: Predictions cannot be paired exactly or improvements are
  diffuse with no reproducible segment.

### Phase 2: Exact-fallback confidence gate

- **Purpose**: Use multi-interest only for high-confidence queries.
- **Entry condition**: Phase 1 identifies candidate gate signals.
- **Phase rules**:
  - RED before production logic.
  - Gate false must reproduce champion scores/ranks exactly.
  - No query label, true rank, or fold identity may enter gate features.
- **Todos**:
  - [x] Test exact fallback and query-level all-candidate routing.
    - **Surface**: gate module and tests.
    - **Proof**: RED/GREEN behavior tests.
    - **Depends on**: phase 1 schema.
  - [x] Fit a small regularized gate with explicit coverage control.
    - **Surface**: cached evaluator.
    - **Proof**: training-fold metrics and gate coverage.
    - **Depends on**: behavior tests.
- **Exit proof**: In-fold implementation works without leakage and exact
  champion fallback is proven.
- **Stop condition**: Gate requires labels or candidate-specific routing that
  changes a query inconsistently.

### Phase 3: Temporal OOF stop decision

- **Purpose**: Decide whether the direction survives.
- **Entry condition**: Exact-fallback gate is green.
- **Phase rules**:
  - Fit each fold only on earlier or disjoint temporal folds.
  - Hyperparameters are fixed before final OOF aggregation.
  - No package below the fixed margin.
- **Todos**:
  - [x] Evaluate three temporal OOF folds and three global time slices.
    - **Surface**: evaluator and JSON report.
    - **Proof**: champion, multi-interest, gate MRR; deltas and coverage.
    - **Depends on**: phase 2.
  - [x] Apply the `+0.002` and non-decreasing stop rule.
    - **Surface**: report decision.
    - **Proof**: automatic `pass` or `stop`.
    - **Depends on**: OOF metrics.
- **Exit proof**: Report has a deterministic decision and reproducible command.
- **Stop condition**: Mean delta below `+0.002`, any fold decline, or any
  chronological slice decline.

## Dry-Run Findings

- The two submitted CSVs contain only test scores and no labels, so paired
  error attribution must use validation caches/checkpoint replay rather than
  leaderboard output.
- The same query must route all 100 candidates to one expert; per-candidate
  gating would alter the ranking semantics and invite leakage.
- Because the previous online delta contradicted the offline delta, passing
  one chronological slice is insufficient.

## Final Validation

- [x] RED/GREEN/refactor evidence for exact fallback.
- [x] Paired-query diagnostic JSON.
- [x] Three-fold temporal OOF plus three-slice report.
- [x] Automatic stop/pass decision using the fixed `+0.002` rule.

## Outcome

- Paired validation queries: `20,000`; multi-interest improves `3,353`,
  worsens `3,088`, and ties `13,559`.
- High-confidence OOF configuration: depth `3`, minimum leaf `2,000`,
  predicted-lift threshold `0.01`.
- OOF MRR delta versus champion: `+0.0034699741`.
- Fold/slice deltas:
  `+0.0046322256 / +0.0032618536 / +0.0025157002`.
- OOF query coverage: `28.065%`.
- The final full-validation tree uses only `champion_top_margin` and
  `expert_top1_agreement`; its coverage is `12.665%`, full delta is
  `+0.0025822553`, and all three slices remain positive.
- Test coverage: `23,657 / 153,420` queries (`15.4198%`); all other queries
  preserve champion scores exactly.
- Local package:
  `result/d1_champion_d2_multi_interest_confidence_gate_seed60_20260726/result.zip`
  (`63,698,089` bytes), SHA-256
  `900d7b7679e47a6054d0b8810b7cf65766242cd609127a46f1239ea4e621a2b0`.

## First Execution Step

Completed: inventoried the server-side caches and reused the paired
full-candidate validation tensors.
