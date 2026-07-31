# Goal Document: Dataset2 bounded source-sequence decoder

## Go / No-Go

- **Judgment**: Go
- **Reason**: The end-to-end D model has a verified failure mechanism: its
  unconstrained candidate-ID scale grew from `0.1` to `1.407`, while the
  source scale remained `0.162`, and external MRR fell `-0.073425` behind the
  champion. A frozen-CST plus bounded source decoder directly removes that
  amplification path while preserving the useful sequence interaction.

## Target Outcome

Produce and evaluate a pure-Jittor Dataset2 model that keeps the validated CST
ranking as an exact frozen fallback and allows candidate identity and source
history to affect ranking only through a zero-mean, frequency-shrunk,
hard-bounded source-sequence residual.

## Goal Definition

- **Type**: technical / quality / learning
- **Boundary**:
  - freeze the existing Variant-A CST logits for every rolling fold;
  - train a Source-Sequence Decoder residual with shared item embeddings;
  - prohibit a standalone candidate-ID additive branch;
  - compare fixed residual caps `0.02`, `0.05`, and `0.10`;
  - select on Fold0/Fold1, gate on unseen Fold2, then allow one external run.
- **Non-goals**:
  - retraining the CST trunk;
  - tuning on external validation;
  - reintroducing LightGBM or sklearn;
  - changing Dataset1;
  - building a confidence router in the same experiment.
- **Deferred work**:
  - long-gap rolling folds;
  - sparse top-k routing of the bounded decoder;
  - test-set packaging unless the frozen external gate passes.
- **Verification rule**: unit contracts plus rolling-origin and external
  metrics must prove that the residual cannot dominate the base and that its
  ranking gain is temporally stable.
- **Evidence source**: focused tests, exact residual audits, three
  rolling-origin folds, activity/time slices, and one external validation.
- **Pass criteria**:
  - cap and fallback audits pass exactly;
  - Fold0 and Fold1 deltas versus frozen A are both non-negative and their mean
    is at least `+0.0001`;
  - Fold2 delta is non-negative, the three-fold mean is at least `+0.0001`,
    and the worst activity delta is at least `-0.0005`;
  - external candidate beats the current champion by at least `+0.0002` and
    is non-negative in every time slice.
- **Confidence note**: rolling folds verify local temporal stability; the
  external set remains the final authority because prior experiments exposed
  a much longer horizon shift.
- **Judgment owner**: the frozen metric gates and exact audits.

## Current State

- Variant D is implemented and pure Jittor.
- D passed three short rolling folds but external MRR was `0.47447143` versus
  champion `0.54789665`.
- Candidate identity contributed about `+0.04852` internally, but its scale
  grew approximately fourteen-fold.
- The source decoder contributed only about `+0.00104` on average and was
  coupled to the same unbounded item-embedding route.
- Variant-A fold logits, causal source sequences, score-frozen sequences, and
  full external source sequences already exist.
- Earlier bounded ID-only residuals were safe but nearly rank-neutral
  externally; this experiment must test whether sequence interaction adds
  stable information without reopening the ID shortcut.

## Priority Rationale

- First prove the architectural bounds and fallback behavior; otherwise metric
  gains cannot demonstrate that overfitting was actually removed.
- Then run the cheapest representative smoke and two selection folds.
- Fold2 and external stay unread until their respective locks permit them.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| Frozen A logits are the correct fallback | confirmed | Prevents decoder failure from replacing the base ranking | Existing replay and fold artifacts |
| Source and candidate IDs share one embedding table | assumed | Enables candidate-to-history matching | Lock in model contract tests |
| Residual caps `0.02/0.05/0.10` cover the useful safe range | confirmed | Matches prior bounded-residual protocol | Frozen before training |
| Item support shrinkage uses only pre-origin history | confirmed | Prevents future frequency leakage | Runner audit |
| External validation is still authoritative | confirmed | Required because rolling horizons are short | One frozen evaluation only |

## Phases

### Phase 1: Bound and fallback contracts

- **Purpose**: Prove that the new decoder cannot recreate the unbounded ID
  shortcut.
- **Entry condition**: This goal document is committed to disk.
- **Phase rules**:
  - tests must fail before production code exists;
  - no metric experiment may start yet;
  - public behavior, not internal layer names, is tested.
- **Todos**:
  - [ ] Add a test that zero history reproduces the base logits exactly.
    - **Surface**: unit tests
    - **Proof**: expected RED, then exact array equality when GREEN
    - **Depends on**: none
  - [ ] Add a test that every residual is finite, row-centered, and bounded by
        the configured cap.
    - **Surface**: unit tests
    - **Proof**: numerical assertions over extreme raw residuals
    - **Depends on**: none
  - [ ] Add candidate-permutation and support-shrinkage tests.
    - **Surface**: unit tests
    - **Proof**: permuted output equivalence and weaker rare-ID residual
    - **Depends on**: none
- **Exit proof**: focused RED failures are caused by the missing bounded
  decoder API.
- **Stop condition**: Stop if exact fallback and hard bounds cannot coexist.

### Phase 2: Minimal pure-Jittor decoder

- **Purpose**: Implement only the architecture required by Phase 1.
- **Entry condition**: Phase 1 has valid RED evidence.
- **Phase rules**:
  - the frozen base logits are inputs and are never trainable;
  - candidate IDs may form attention queries but may not add directly to base
    candidate hidden states or logits;
  - empty history must force an exact zero residual;
  - residuals must be centered and rescaled after every learned transform.
- **Todos**:
  - [ ] Implement shared item/time/position embeddings and masked
        candidate-to-history attention.
    - **Surface**: pure-Jittor model module
    - **Proof**: focused tests
    - **Depends on**: Phase 1
  - [ ] Implement frequency shrinkage and hard row-centered projection.
    - **Surface**: model module
    - **Proof**: cap/fallback/permutation tests
    - **Depends on**: Phase 1
  - [ ] Implement checkpoint save/load and inference parity.
    - **Surface**: model module and tests
    - **Proof**: identical pre/post-load scores
    - **Depends on**: model GREEN
- **Exit proof**: all focused tests pass under CPU Jittor.
- **Stop condition**: Stop on any non-Jittor trainable dependency.

### Phase 3: Rolling-origin runner

- **Purpose**: Train and evaluate the bounded decoder without metric leakage.
- **Entry condition**: Phase 2 is green.
- **Phase rules**:
  - use causal train sequences and origin-frozen score sequences;
  - item support is computed only from the model's visible train prefix;
  - caps and optimizer settings are frozen before Fold0;
  - no Fold2 access during selection.
- **Todos**:
  - [ ] Build smoke, selection, gate, and external phases.
    - **Surface**: experiment runner
    - **Proof**: CLI, smoke report, and frozen config
    - **Depends on**: Phase 2
  - [ ] Train caps on Fold0 and Fold1 and lock one candidate.
    - **Surface**: remote CUDA experiment
    - **Proof**: selection lock
    - **Depends on**: runner smoke
  - [ ] Evaluate only the selected cap on Fold2.
    - **Surface**: remote CUDA experiment
    - **Proof**: gate report
    - **Depends on**: selection lock
- **Exit proof**: exact audits and metric gate are machine-readable.
- **Stop condition**: If no cap passes Fold0/Fold1, do not read Fold2.

### Phase 4: External decision

- **Purpose**: Decide whether the bounded decoder replaces the champion.
- **Entry condition**: Fold2 gate passes.
- **Phase rules**:
  - fix full-training epochs from selection folds;
  - do not scan caps, blends, or thresholds externally;
  - do not package a rejected candidate.
- **Todos**:
  - [ ] Train the selected full model and run one external evaluation.
    - **Surface**: remote CUDA experiment
    - **Proof**: external evaluation report
    - **Depends on**: Fold2 gate
  - [ ] Record pass/reject and submission status.
    - **Surface**: result documentation
    - **Proof**: final evaluation report
    - **Depends on**: external evaluation
- **Exit proof**: final report states metrics, framework compliance, and whether
  a submission was generated.
- **Stop condition**: Reject immediately on external full or time-slice gate
  failure.

## Dry-Run Findings

- The existing D checkpoint cannot be safely repaired by clipping its final
  scale after training: its internal representation was learned while the ID
  path dominated. A new residual-only training path is required.
- Using frozen logits rather than a live CST module reduces memory and makes
  exact fallback auditable.
- The source decoder still needs candidate IDs for matching, but the new
  boundary permits them only inside a capped interaction residual.
- Existing sequence caches satisfy the required causal and frozen-origin
  boundaries, so no 4 GB feature-cache rebuild is required.

## Final Validation

```bash
.venv/bin/python -m pytest \
  tests/test_hybrid_bounded_source_decoder.py \
  tests/test_hybrid_source_conditioned_cst.py -q

.venv/bin/ruff check \
  src/jgrec/rankers/hybrid/bounded_source_decoder.py \
  scripts/train_dataset2_bounded_source_decoder.py \
  tests/test_hybrid_bounded_source_decoder.py

bash scripts/run_dataset2_bounded_source_decoder_20260727.sh
```

## First Execution Step

Add the exact-fallback, hard-cap, candidate-permutation, support-shrinkage, and
checkpoint-parity tests before creating the bounded decoder module.
