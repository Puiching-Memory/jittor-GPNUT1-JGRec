# Goal Document: Dataset2 Full-100 Candidate Training

## Go / No-Go

- **Judgment**: Staged Go. The original byte-identical replay route is No-Go; the
  matched 32-vs-100 route is Go with the same frozen validation gate.
- **Reason**: The current champion trains its fusion experts on 32-candidate groups but is selected and deployed on 100-candidate groups. Recent feature and objective experiments improved isolated proxies without transferring to the complete blend, making candidate-distribution alignment the highest-value unresolved variable.

## Execution Reframe After Replay

- The bounded replay completed in `1496.7s` and correctly reconstructed the split,
  shapes, positive-first labels, and unique 32-candidate groups, but did not reproduce
  the old cached feature tensor.
- The old run and replay used the same Dataset2 Two-Tower stop point
  (`epoch=8`, `best_epoch=5`), while their validation losses differed. This is
  consistent with the old combined Dataset1→Dataset2 run having advanced Jittor's
  unsaved global model-initialization state.
- Requiring byte-identical learned-tower values is therefore not reproducible from the
  persisted checkpoint/cache alone. It remains a valid falsifier for the original
  route, which is now stopped.
- The replacement protocol fits one new Dataset2 prefix encoder once and builds one
  `50,000 x 100 x 63` tensor. The first 32 positions of every group are the exact
  nested control view, so the 32- and 100-candidate LightGBMs differ only in whether
  the final 68 candidates participate in LambdaRank. Both are evaluated on the
  unchanged old `20,000 x 100 x 63` validation tensor.
- Packaging additionally requires the 100-candidate model's full MRR to be no lower
  than its matched 32-candidate control. The original `+0.002` versus champion and
  three non-decreasing chronological slices remain mandatory.

## Target Outcome

Build one reproducible Dataset2 supervised training cache with 50,000 groups of
100 candidates and a nested first-32 control view, train only the two Dataset2 LightGBM
controls with the frozen champion parameters, evaluate both on the unchanged
20,000 x 100 champion validation tensor, and create a candidate package only if the
100-candidate fixed blend improves by at least `+0.002` over the champion, has no
chronological-slice regression, and does not trail the matched 32-candidate control.

## Goal Definition

- **Type**: technical, quality, learning, and delivery.
- **Boundary**: Dataset2 only; full-100 supervised training candidates; candidate/query identity sidecars; unchanged 63 feature definitions and tower configurations; fixed `lr003` LightGBM with 308 rounds; fixed MLP weight `0.07`; exact champion validation tensor and three fixed chronological slices.
- **Non-goals**:
  - Do not change or add towers.
  - Do not tune LightGBM, blend weight, negative mixture, seed, or feature mask.
  - Do not retrain the champion MLP.
  - Do not change Dataset1.
  - Do not submit automatically.
- **Deferred work**:
  - Candidate-aware Two-Tower and GNN training.
  - Per-tower block caches and multi-seed confirmation.
- **Verification rule**: TDD proves candidate identity persistence and exact width contracts; a replay preflight proves the rebuilt prefix encoder/sampler matches the existing 32-candidate cache; the frozen run writes shapes, hashes, model, full/per-slice MRR, and a pass/reject decision.
- **Evidence source**: Focused tests, cache manifests and SHA-256 hashes, replay error statistics, exact validation report, checkpoint reload prediction equality, and result ZIP inspection.
- **Pass criteria**:
  - Train features and candidate IDs have shape `50,000 x 100`; validation remains the existing `20,000 x 100 x 63` champion cache.
  - Candidate position zero is the positive for every training group; remaining IDs contain no positive duplicates.
  - The completed replay must prove the split and candidate contracts. A learned-tower
    value mismatch routes execution to the declared matched-control protocol rather
    than authorizing a direct one-arm comparison.
  - Baseline fixed-blend MRR reproduces `0.5428303297309955`.
  - Candidate full MRR delta is at least `+0.002`; each of three chronological slice deltas is nonnegative.
  - Candidate full MRR is not below the matched 32-candidate control.
  - Packaging runs only after the gate passes.
- **Confidence note**: Reusing the unchanged champion validation tensor removes validation-candidate drift. A single frozen seed is still not proof of B-board lift, so the all-slice gate remains mandatory.
- **Judgment owner**: Automated cache contracts and the frozen exact MRR gate.

## Current State

- The champion cache key is `4baa722bf26e5d50356da26ac5f479cb54324ddb`.
- Existing train shape is `50,000 x 32 x 63`; validation shape is `20,000 x 100 x 63`.
- Existing cache files store feature tensors and the post-build fusion RNG state, but not query/candidate IDs.
- The CLI already supports independent `train_num_negatives` and `val_num_negatives`.
- The source champion Dataset2 fixed-blend MRR is `0.5428303297309955`.
- The server has enough memory for the estimated `~1.17 GiB` float32 full-100 training tensor, but prefix tower fitting and feature generation are expensive.

## Priority Rationale

- Freeze validation identity first; otherwise a changed RNG stream would make candidate MRR incomparable.
- Prove replay of the old cache before spending hours on a 5-million-candidate feature build.
- Store candidate identities with the new cache so future tower experiments do not repeat ambiguous feature-to-ID recovery.
- Train one fixed LightGBM before considering any tower change.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| Existing 100-candidate validation tensor is the comparison owner | confirmed | Prevents validation distribution drift | Reuse byte-for-byte |
| Prefix encoder and sampler can replay the old cache | assumed | Controls whether new train features are comparable | Run bounded replay preflight |
| Full-100 tensor fits server memory/disk | assumed | Could block the build | Preflight free memory/disk and use memmap |
| One frozen negative mixture is sufficient for the first test | confirmed | Avoids validation-driven tuning | Reuse champion sampling ratios |
| Candidate IDs should be persisted in a backward-compatible sidecar | confirmed | Enables reproducibility and future per-tower work | TDD cache contract |

## Phases

### Phase 1: Candidate Identity and Width Contracts

- **Purpose**: Make full-candidate caches reproducible and auditable before running expensive models.
- **Entry condition**: Existing cache tests pass.
- **Phase rules**:
  - RED before production code.
  - Existing version-1 feature-only caches must remain loadable.
  - Candidate sidecars are optional for legacy caches and mandatory for the new full-100 cache.
- **Todos**:
  - [ ] Add failing tests for saving/loading train and validation candidate-ID sidecars.
    - **Surface**: supervised feature cache API and tests.
    - **Proof**: RED fails because identity sidecars are unsupported.
    - **Depends on**: none.
  - [ ] Add a failing test that rejects feature/candidate width or row mismatches.
    - **Surface**: cache manifest validation.
    - **Proof**: malformed identities are rejected before write/load.
    - **Depends on**: identity API.
  - [ ] Implement the smallest backward-compatible sidecar support.
    - **Surface**: supervised feature cache and manifest.
    - **Proof**: legacy and new cache tests pass.
    - **Depends on**: both RED tests.
- **Exit proof**: Focused cache tests and Ruff pass.
- **Stop condition**: Stop if the change invalidates the existing champion cache.

### Phase 2: Replay Preflight and Full-100 Train Cache

- **Purpose**: Build comparable full-100 training features without changing towers or validation.
- **Entry condition**: Phase 1 is green and server resources pass.
- **Phase rules**:
  - Reconstruct the champion temporal split and prefix encoder from the frozen checkpoint config.
  - Clone the post-encoder RNG state: one branch replays 32 candidates for comparison, one builds 100 candidates.
  - Keep the old validation feature file byte-identical.
- **Todos**:
  - [ ] Implement a bounded replay preflight against the first cached training rows.
    - **Surface**: full-100 cache builder.
    - **Proof**: old-width generated features match cached features within tolerance.
    - **Depends on**: Phase 1.
  - [ ] Build `50,000 x 100 x 63` train features and `50,000 x 100` candidate IDs using memmap/atomic outputs.
    - **Surface**: server cache artifacts and manifest.
    - **Proof**: shapes, hashes, positive position, uniqueness, finite values, memory/runtime log.
    - **Depends on**: replay pass.
- **Exit proof**: A complete immutable cache report exists; old validation hash is unchanged.
- **Stop condition**: Stop on replay mismatch, candidate-label mismatch, OOM, disk shortage, or non-finite features.

### Phase 3: Frozen LightGBM and Exact Gate

- **Purpose**: Isolate the value of 100-candidate training.
- **Entry condition**: Phase 2 passes.
- **Phase rules**:
  - Train only Dataset2 LightGBM.
  - Use `lr=0.03`, 308 rounds, 63 leaves, truncation 30, and MLP weight `0.07`.
  - No early stopping or parameter selection on the exact validation set.
- **Todos**:
  - [ ] Train on full-100 groups and score the unchanged champion validation tensor.
    - **Surface**: model and validation report.
    - **Proof**: baseline/candidate full and per-slice MRR with exact deltas.
    - **Depends on**: Phase 2.
  - [ ] Apply the frozen gate.
    - **Surface**: report status.
    - **Proof**: deterministic pass/reject and package authorization flag.
    - **Depends on**: scoring.
- **Exit proof**: Report declares pass or reject.
- **Stop condition**: Stop on full delta below `+0.002` or any slice regression.

### Phase 4: Conditional Packaging

- **Purpose**: Produce a safe submission artifact only for a validated improvement.
- **Entry condition**: Phase 3 passes.
- **Phase rules**:
  - Start from the current combined champion checkpoint.
  - Replace only Dataset2 LightGBM state; preserve Dataset1 CSV byte-for-byte.
- **Todos**:
  - [ ] Save/reload the candidate checkpoint and compare predictions.
    - **Surface**: new checkpoint.
    - **Proof**: pre/post reload equality and both datasets load.
    - **Depends on**: gate pass.
  - [ ] Build and copy a separate result ZIP.
    - **Surface**: result directory.
    - **Proof**: CSV row counts, Dataset1 hash identity, ZIP size and SHA-256.
    - **Depends on**: checkpoint proof.
- **Exit proof**: A verified local ZIP exists, or no ZIP exists after gate rejection.
- **Stop condition**: Any checkpoint, Dataset1, or packaging mismatch.

## Dry-Run Findings

- Merely setting `--train-num-negatives 99` would also advance the shared RNG differently and rebuild validation candidates, invalidating an exact comparison; the old validation cache must remain the scoring owner.
- The current feature cache cannot prove which candidates produced each row, so identity sidecars must be added before the new cache is trusted.
- Rebuilding all towers is expensive but necessary for the prefix training encoder; replaying a bounded old-width sample is the cheapest early falsifier.
- Full-100 train features are approximately 3.125 times the current training-cache size but remain feasible with float32 memmap on the server.

## Final Validation

- Focused RED/GREEN cache tests and broader cache/ranker regression tests.
- Replay report against the champion cache.
- Full-100 train cache manifest, IDs, hashes, and resource log.
- Exact unchanged validation baseline and full/per-slice MRR gate.
- Conditional checkpoint/ZIP verification only after a pass.

## First Execution Step

Add a failing cache test requiring candidate-ID sidecars to round-trip while a legacy feature-only cache still loads unchanged.
