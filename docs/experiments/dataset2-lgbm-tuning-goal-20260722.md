# Goal Document: Dataset2-Only LightGBM Tuning

## Go / No-Go

- **Judgment**: Go
- **Reason**: Dataset2's cached supervised features already exist on the server, LightGBM is the component with the clearest Dataset2 validation gain, and the experiment can freeze Dataset1 and every neural tower.

## Target Outcome

Tune only Dataset2's LightGBM reranker on cached features, select a configuration without using the late temporal holdout, verify it on that untouched pseudo-B slice, and produce a submission candidate that preserves the current `1.3411684997174933` Dataset1 prediction byte-for-byte only if the Dataset2 result passes the robustness gate.

## Goal Definition

- **Type**: learning and delivery
- **Boundary**: Dataset2 LightGBM hyperparameters, full-candidate MRR early stopping, cached train/validation tensors, fine MLP/LightGBM probability blending, checkpoint patching, Dataset2 inference, and candidate packaging.
- **Non-goals**:
  - Retrain or change Dataset1.
  - Retrain GNN, GRU, two-tower, source-profile, or fusion MLP components.
  - Change feature generation, negative sampling, or candidate construction.
  - Tune against leaderboard feedback.
- **Deferred work**:
  - Independent feature-cache generation for additional temporal folds.
  - Segment-aware LightGBM models or gates.
- **Verification rule**: Hyperparameters and blend weight are selected on the first 10,000 validation queries only; the last 10,000 queries remain unread by selection and must improve over the current Dataset2 checkpoint before packaging.
- **Evidence source**: Unit tests, tuning JSON/TSV, per-slice full-candidate MRR, best iteration, checkpoint inspection, prediction validation, CSV hashes, and ZIP hash.
- **Pass criteria**: The chosen LightGBM beats the current Dataset2 stored ensemble on both the tuning half and untouched late half, has no non-finite predictions, and the packaged Dataset1 CSV hash equals the current champion Dataset1 CSV hash.
- **Confidence note**: A single temporal pseudo-B slice is stronger than reusing the full validation set but remains an offline proxy; the current online champion remains the fallback.
- **Judgment owner**: The frozen pseudo-B metric decides whether a package may be built; artifact validators decide delivery completion; the competition score remains the external quality judge.

## Current State

- Current online champion score: `1.3411684997174933`.
- Dataset2 cache key: `4baa722bf26e5d50356da26ac5f479cb54324ddb`.
- Cached train tensor: `50,000 × 32 × 63`; cached validation tensor: `20,000 × 100 × 63`.
- Current Dataset2 checkpoint uses 63 leaves, learning rate 0.05, minimum child samples 20, feature/bagging fractions 0.8, and full-candidate MRR early stopping.
- Current Dataset2 stored ensemble validation MRR is `0.54088842`; the LightGBM component alone is `0.54043570`.
- Current checkpoint-wide blend comparisons were selected on the same validation split and are not evidence for this isolated LightGBM search.

## Priority Rationale

- Preserve a late temporal slice before running a parameter grid so tuning cannot consume all available validation evidence.
- Reuse cached tensors so the experiment measures LightGBM configuration, not feature/tower randomness.
- Patch and infer Dataset2 only after the robustness gate; Dataset1 remains a byte-identical control.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
| --- | --- | --- | --- |
| Validation rows are in temporal order | assumed | Required for the late half to act as pseudo-B | Confirm from cache construction/code before execution. |
| Cache features match the current champion checkpoint's 63 selected features | confirmed | Required for fair replacement | Recheck feature indices in the operator script. |
| A small, predeclared grid is preferable to an unconstrained search | confirmed | Limits holdout overfitting and runtime | Encode the fixed grid in the tuning report. |
| Dataset1 must not change | confirmed | Keeps the experiment attributable to Dataset2 | Enforce SHA-256 equality during packaging. |

## Phases

### Phase 1: Reproducible Dataset2 Tuning Primitive

- **Purpose**: Make temporal splitting, selection-only scoring, holdout scoring, and deterministic tie-breaking testable.
- **Entry condition**: The cache shape and current checkpoint are known.
- **Phase rules**:
  - Write failing tests before implementation.
  - Selection code cannot receive the pseudo-B scores.
  - The parameter grid and tie rule must be serialized in the report.
- **Todos**:
  - [ ] Test and implement chronological validation splitting.
    - **Surface**: tuning script helpers and focused tests.
    - **Proof**: RED then GREEN tests prove disjoint ordered halves.
    - **Depends on**: none.
  - [ ] Test and implement deterministic configuration and blend selection using only the tuning half.
    - **Surface**: tuning script helpers and focused tests.
    - **Proof**: Synthetic results choose the expected configuration without accessing holdout metrics.
    - **Depends on**: temporal split helper.
- **Exit proof**: Focused tests pass and selection inputs exclude the late half.
- **Stop condition**: Stop if cache row order is not temporal or the positive-candidate contract is inconsistent.

### Phase 2: Cached Dataset2 Search and Pseudo-B Gate

- **Purpose**: Find a more robust Dataset2 LightGBM without recomputing supervised features.
- **Entry condition**: Phase 1 is green, server is idle, and cache/checkpoint identities match.
- **Phase rules**:
  - Train only on the cached Dataset2 training tensor.
  - Use validation rows `0:10000` for early stopping, model selection, and blend selection.
  - Do not read rows `10000:20000` until one configuration and weight are frozen.
  - Keep the fixed parameter grid small; do not expand it based on pseudo-B results.
- **Todos**:
  - [ ] Reproduce the current LightGBM baseline under the split protocol.
    - **Surface**: server tuning log/report.
    - **Proof**: Baseline configuration, best iteration, tune MRR, and pseudo-B MRR are recorded.
    - **Depends on**: Phase 1.
  - [ ] Run the predeclared grid and freeze the best tune-half configuration and blend weight.
    - **Surface**: server LightGBM process and JSON/TSV report.
    - **Proof**: Every configuration has parameters, iteration, tune MRR, and resource status; only the winner is evaluated on pseudo-B.
    - **Depends on**: baseline reproduction.
  - [ ] Apply the pseudo-B robustness gate.
    - **Surface**: final evaluation record.
    - **Proof**: Candidate-vs-current deltas on tune and late halves.
    - **Depends on**: frozen winner.
- **Exit proof**: A frozen Dataset2 candidate either passes both-half improvement criteria or is rejected with evidence.
- **Stop condition**: Stop on cache mismatch, memory pressure, non-finite scores, or any attempt to choose parameters using pseudo-B metrics.

### Phase 3: Dataset2-Only Candidate Delivery

- **Purpose**: Deliver a valid candidate without changing Dataset1.
- **Entry condition**: Phase 2 candidate passes the robustness gate.
- **Phase rules**:
  - Patch a new checkpoint/artifact; never overwrite the champion.
  - Copy Dataset1 CSV from the champion byte-for-byte.
  - Generate Dataset2 predictions from the frozen LightGBM and blend only.
- **Todos**:
  - [ ] Create a new checkpoint with only Dataset2 LightGBM state changed.
    - **Surface**: contest checkpoint.
    - **Proof**: Dataset1 snapshot hash/serialized record is preserved and Dataset2 LightGBM parameters match the tuning report.
    - **Depends on**: robustness gate.
  - [ ] Infer Dataset2, package both CSVs, validate, hash, and download.
    - **Surface**: result directory and local artifact.
    - **Proof**: Dataset1 CSV hash equality, valid Dataset2 dimensions/probabilities, two ZIP members, and matching remote/local ZIP hashes.
    - **Depends on**: patched checkpoint.
- **Exit proof**: A locally available submission ZIP with a traceable tuning report and rollback baseline.
- **Stop condition**: Do not build a submission if pseudo-B regresses; stop if any existing artifact would be overwritten.

## Dry-Run Findings

- Tuning on all 20,000 validation rows would leave no untouched temporal evidence and would repeat the overfitting risk the user highlighted.
- The MLP probabilities must be computed once and reused across every LightGBM configuration; the MLP itself must not be retrained.
- The final blend weight must be chosen on the tuning half at `0.01` resolution and frozen before pseudo-B evaluation.
- The current champion's Dataset2 output cannot be reused because the experiment changes Dataset2; Dataset1 can and should be copied unchanged.
- A passing offline candidate is eligible for submission, not guaranteed to improve the hidden B leaderboard.

## Final Validation

- Focused tests and relevant LightGBM/checkpoint regressions pass.
- Server report proves cache identity, fixed grid, frozen selection, and tune/pseudo-B deltas.
- No tower or MLP training occurs.
- Candidate checkpoint and ZIP validate; Dataset1 is byte-identical to the champion.

## First Execution Step

Add failing tests for chronological split isolation and tune-only deterministic selection before implementing the cached Dataset2 tuner.

## Execution Result

- Focused behavior tests: `5 passed` locally and on the server; checkpoint regression set: `11 passed, 4 skipped`.
- Fixed grid completed `12/12` trials in 118 seconds using only cached Dataset2 features.
- Frozen winner: `learning_rate=0.03`, 63 leaves, minimum child samples 20, truncation level 30, best iteration 308, MLP blend weight `0.07`.
- Tune-half MRR: `0.57491055`, delta versus current stored ensemble `+0.00363050`.
- Untouched pseudo-B MRR: `0.51075010`, delta versus current stored ensemble `+0.00025333`.
- Combined validation MRR after the frozen decision: `0.54283033`.
- Robustness gate: passed; pseudo-B was not used during configuration or blend selection.
- New checkpoint: `checkpoints/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722.pkl`, `5,009,171,397` bytes, SHA-256 `a8dada300ff7aa87292fcff9c35498997e1c4013d4a3309451ba90e25666cf3f`.
- Local submission candidate: `result/d1_champion_d2_lgbm_lr003_pseudob_seed60_20260722/result.zip`, `61,548,216` bytes, SHA-256 `08c146af7d8edbfac2ac05802bf336707ff6dcad33e74747aa1dbdd362750833`.
- Dataset1 has 61,051 rows and is byte-identical to the `1.3411684997174933` champion CSV; Dataset2 has 153,420 rows and is the only changed prediction.
