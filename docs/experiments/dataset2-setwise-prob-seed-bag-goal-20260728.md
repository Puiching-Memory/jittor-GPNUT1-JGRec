# Goal Document: Dataset2 Setwise Probability Seed Bag v1

## Go / No-Go

- **Judgment**: Go after freeze.
- **Reason**: The candidate is one bounded variance-reduction test with a fixed
  two-seed probability mean, fixed six weights, exact rolling origins, one
  external opening, and an explicit close-on-failure rule.

## Target Outcome

Produce exactly one `setwise_prob_seed_bag_v1` integrated candidate. Train two
new four-epoch Setwise heads inside each of the three exact rolling folds,
average their probabilities, run the existing robust rolling selector, open
the external holdout once only after a selection lock exists, and package only
after the external gate accepts the locked candidate.

## Goal Definition

- **Type**: technical / learning / delivery.
- **Boundary**: Dataset2 Setwise probability variance reduction only; folds,
  weights, salts, training hyperparameters, graph-score semantics, selector,
  external evaluator, and stop rule are frozen in
  `setwise-prob-seed-bag-v1.frozen.json`.
- **Non-goals**:
  - Ranking aggregation, more seeds, or alternative probability transforms.
  - Searching epochs, salts, weights, folds, or package variants.
  - Reopening external evidence after either acceptance or rejection.
- **Deferred work**:
  - Any follow-up hypothesis must use a new fold and a new integration id.
- **Verification rule**: A package is authorized only when the frozen config
  hash matches, all two-seed/fold artifacts exist and verify, the existing
  rolling selector writes a selection lock, and the single external receipt
  reports `accepted`.
- **Evidence source**: RED/GREEN tests, frozen-config hash, model and score
  hashes, rolling manifest, selection report/lock, external receipt/report,
  and package audit.
- **Pass criteria**: Every existing rolling hard gate passes for one frozen
  weight; the locked candidate then passes every external hard gate.
- **Confidence note**: The consumed folds can only select and reject this
  preregistered candidate. They cannot justify protocol changes.
- **Judgment owner**: Existing automated rolling and external gates.

## Current State

- Three-seed rank aggregation has already been rejected.
- Probability averaging has not been evaluated under this exact protocol.
- Setwise is the high-variance 0.80 component of the Dataset2 main path.
- Today’s fold-exact Setwise retraining path already supports four epochs.

## Plan Rewrite Notes

| Existing item | Decision | Reason |
|---|---|---|
| Six partial weights | keep | Reuse the preregistered robust selector surface |
| Two new seeds | rewrite as two fixed salts | Prevent collision with historical `17/41/60` seeds |
| Three exact rolling origins | keep | Preserve today’s candidate and graph-score contract |
| External evaluation | keep, one-shot | Existing evaluator enforces receipt-based single use |
| Packaging | defer until accepted external | Avoid producing an unauthorized candidate |

## Drift Diagnosis

- **Goal drift**: Adding seeds or rank aggregation would test another idea.
- **Phase drift**: External materialization before selection lock would consume
  evidence too early.
- **Validation drift**: Changing salts or weights after rolling metrics would
  turn diagnosis into a rescan.
- **Compatibility drift**: A new bespoke selector would bypass established
  hard gates.
- **Cleanup drift**: No unrelated model or documentation cleanup is in scope.

## Priority Rationale

- Freeze first because every later artifact must be bound to one protocol hash.
- Test probability arithmetic and lock guards before spending GPU time.
- Train all rolling folds before the selector; external work remains impossible
  until a lock is present.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| Salt values `10007/20011` are unused | confirmed by code search | Preserves a genuinely new seed bag | Frozen config |
| Seed mapping is `60 + fold*1009 + salt` | confirmed | Matches current fold seed convention | Training runner |
| Exact-rolling baseline/graph assets are immutable | assumed until hash preflight | Required for comparable fold scores | Read-only remote preflight |
| Packaging implementation is needed only after external acceptance | confirmed | Avoids speculative package code | Stop at external rejection |

## Phases

### Phase 1: Freeze and contract tests

- **Purpose**: Make post-metric protocol changes impossible.
- **Entry condition**: Existing code contracts have been inspected without
  evaluating this candidate.
- **Phase rules**:
  - No candidate metric may be read before the frozen config exists.
  - Production code starts only after a failing behavior test.
- **Todos**:
  - [ ] Validate the exact config, probability mean, fold seeds, and lock guard.
    - **Surface**: tests and integration helper.
    - **Proof**: focused RED then GREEN.
    - **Depends on**: frozen config.
- **Exit proof**: Frozen config is parseable and all protocol invariants are
  enforced by tests.
- **Stop condition**: Any required value is not represented in the freeze.

### Phase 2: Rolling production and selection

- **Purpose**: Produce exactly three fold score families and let the existing
  selector make the only weight decision.
- **Entry condition**: Phase 1 GREEN and remote assets pass hash/shape preflight.
- **Phase rules**:
  - Two new heads per fold, four epochs each.
  - Auxiliary is the arithmetic probability mean only.
  - No external feature or metric is read.
- **Todos**:
  - [ ] Train and persist six heads plus fold probabilities.
  - [ ] Build one exact-integrated rolling manifest.
  - [ ] Run `scripts/select_robust_integrated_weight.py` once.
- **Exit proof**: A valid selection lock exists.
- **Stop condition**: Training/preflight failure or selector rejection closes
  the integration without a rescan.

### Phase 3: One-shot external and conditional package

- **Purpose**: Test the locked weight on untouched long-span evidence.
- **Entry condition**: Phase 2 selection lock exists and its hash matches.
- **Phase rules**:
  - Materialize only the locked external candidate.
  - `scripts/evaluate_locked_weight_external.py` may create one receipt.
  - No weight or seed changes are authorized.
- **Todos**:
  - [ ] Train the two frozen full-origin heads and create the external manifest.
  - [ ] Run the existing external evaluator once.
  - [ ] Package only when status is `accepted`.
- **Exit proof**: Accepted external report and package audit, or a final closed
  rejection report.
- **Stop condition**: Any external rejection ends the experiment.

## Dry-Run Findings

- The existing selector consumes score manifests and already enforces all
  cross-fold stability gates.
- The existing external evaluator consumes a lock-bound manifest and writes an
  irreversible open receipt.
- A new rolling producer is required; external/package code is intentionally
  deferred until rolling selection passes.

## Final Validation

- Focused pytest and Ruff checks.
- Frozen config hash and six model/score hashes.
- Existing selector result and optional selection lock.
- At most one external receipt.
- Package audit only after external acceptance.

## First Execution Step

Add a failing test for frozen seed mapping and exact arithmetic probability
averaging before implementing the rolling producer.
