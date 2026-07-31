# Goal Document: Persistent Supervised Feature Cache for Fusion Tuning

## Go / No-Go

- **Judgment**: Go
- **Reason**: Supervised train/validation tensors are the expensive, fusion-independent boundary; persisting them can remove repeated tower fitting and feature construction while preserving the existing full-training path.

## Target Outcome

Repeated hybrid runs over the same interactions and feature-producing configuration can load cached train/validation supervised tensors and immediately tune the fusion backend. Changing a feature dependency produces a different cache entry; changing only fusion hyperparameters reuses the entry.

## Goal Definition

- **Type**: technical and operational
- **Boundary**: Add an opt-in persistent cache directory, deterministic feature fingerprinting, atomic cache writes, mmap-backed cache reads, cache hit/miss logging, CLI wiring, and tests.
- **Non-goals**:
  - Do not cache final predictions, fusion models, or contest checkpoints.
  - Do not change candidate sampling, feature values, temporal splits, or default training behavior.
  - Do not add a general experiment scheduler or hyperparameter search engine.
- **Deferred work**:
  - Cache eviction and size quotas.
  - Cross-version migration of cached tensors; incompatible entries are invalidated instead.
  - Distributed writers sharing a network filesystem.
- **Verification rule**: Tests must prove miss/write/hit behavior, fusion-only reuse, feature-dependent invalidation, corrupted-entry recovery, and unchanged behavior when caching is disabled.
- **Evidence source**: Focused pytest RED/GREEN cycles, CLI help, relevant regression suite, Ruff, and compileall.
- **Pass criteria**: On the second identical run, supervised feature construction is not called and both tensors are loaded with identical values/shapes; fusion-only config changes keep the same key; feature config or interaction changes do not.
- **Confidence note**: Unit tests exercise the persistent boundary with real NumPy files; a full GPU timing benchmark is deferred because correctness and cache reuse can be proven without Jittor training.
- **Judgment owner**: Automated tests.

## Current State

- `_learn_fusion()` always fits temporal train/validation encoders and rebuilds both supervised feature tensors before fitting fusion.
- `supervised_feature_memmap` reduces peak memory but does not provide reuse across runs.
- Fusion backends consume train and validation tensors independently, so cached tensors are a stable handoff boundary.
- Existing commands and checkpoints must remain valid when no cache directory is provided.

## Priority Rationale

- Define cache identity before storage code because a false cache hit would silently corrupt experiments.
- Prove the storage component independently before routing `_learn_fusion()` around expensive encoder construction.
- Keep cache opt-in until it has real server timing evidence and an eviction policy.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
| --- | --- | --- | --- |
| CLI flag is `--supervised-feature-cache-dir` | confirmed | One shared directory can serve multiple fusion sweeps | Implement in CLI/config |
| Cache values are `.npy` arrays loaded with mmap | assumed | Fast load and low resident-memory overhead | Prove in storage tests |
| Cache fingerprint excludes fusion-only settings | confirmed | `epochs`, fusion mode/width, optimizer, early stop, and selection metric can be tuned without a rebuild | Cover with key test |
| Cache fingerprint includes source code schema version | confirmed | Feature-layout changes invalidate old entries | Use an explicit cache format/schema version |
| Corrupt/incomplete entries are treated as misses | confirmed | Interrupted writes cannot poison later experiments | Cover with recovery test |

## Phases

### Phase 1: Cache Identity and Storage

- **Purpose**: Establish a trustworthy persistent boundary before changing the training route.
- **Entry condition**: Goal document is committed to the workspace and current relevant tests pass.
- **Phase rules**:
  - Add failing tests before production code.
  - Use atomic manifest publication; a manifest is the marker of a complete entry.
  - Do not include fusion-only parameters in the fingerprint.
- **Todos**:
  - [x] Add RED tests for deterministic identity, fusion-only reuse, and feature-dependent invalidation.
    - **Surface**: New hybrid feature-cache tests.
    - **Proof**: Tests fail because no cache identity API exists.
    - **Depends on**: None.
  - [x] Implement cache identity and persistent NumPy storage.
    - **Surface**: Hybrid cache module.
    - **Proof**: Round-trip arrays load read-only/mmap-backed; corruption returns a miss.
    - **Depends on**: RED evidence.
- **Exit proof**: Storage tests are green and cache entries are self-describing through a manifest.
- **Stop condition**: Stop if a complete feature dependency cannot be separated from fusion-only configuration.

### Phase 2: Hybrid Training Integration

- **Purpose**: Skip expensive supervised encoders and feature construction on cache hits.
- **Entry condition**: Phase 1 is green.
- **Phase rules**:
  - Cache disabled means the current call path is unchanged.
  - A miss computes both tensors and publishes only after both succeed.
  - Final full-history encoder fitting remains unchanged.
- **Todos**:
  - [x] Add a RED integration test proving the second run does not call supervised feature construction.
    - **Surface**: `_learn_fusion()` boundary.
    - **Proof**: Second run currently calls the failing fake builder.
    - **Depends on**: Phase 1.
  - [x] Route cache hit/miss behavior through `_learn_fusion()`.
    - **Surface**: Hybrid ranker and logs.
    - **Proof**: Integration test passes and cached tensors reach fusion unchanged.
    - **Depends on**: RED evidence.
- **Exit proof**: Identical second run fits fusion from cached tensors without constructing supervised encoders/features.
- **Stop condition**: Stop if cache-hit routing skips state required by the final encoder or checkpoint.

### Phase 3: CLI, Documentation, and Regression

- **Purpose**: Make the optimization usable and safe for real fusion sweeps.
- **Entry condition**: Phase 2 is green.
- **Phase rules**:
  - Preserve defaults and old checkpoint loading.
  - Document which parameters reuse versus invalidate the cache.
- **Todos**:
  - [x] Add CLI/config wiring and visible cache status/path.
    - **Surface**: CLI, TrainingConfig, run panel.
    - **Proof**: CLI test and help output expose the opt-in path.
    - **Depends on**: Phase 2.
  - [x] Document a two-command fusion tuning workflow and run regressions.
    - **Surface**: Submission operations documentation and test suite.
    - **Proof**: Relevant pytest suite, Ruff, diff check, and compileall pass.
    - **Depends on**: CLI wiring.
- **Exit proof**: A documented fusion-only rerun hits the same cache and all validations pass.
- **Stop condition**: Stop on a backward-compatibility or tensor-equivalence regression.

## Dry-Run Findings

- Cache lookup must happen before fitting `train_encoder` and `val_encoder`; otherwise most of the intended speedup is lost.
- The cache key needs an interaction identity plus the temporal split/sampling/tower feature configuration, but must exclude all fusion-training knobs.
- Loading via `np.load(..., mmap_mode="r")` lets fusion train without copying the full tensors; the existing cleanup path must not delete persistent cache files.
- The cache manifest should record shapes, dtypes, key, and format version so partial or stale entries fail closed.

## Final Validation

- Cache identity/storage and cache-hit `_learn_fusion()` RED/GREEN tests passed.
- `uv run --no-sync pytest tests/test_cli.py tests/test_hybrid_supervised_feature_cache.py tests/test_hybrid_negatives.py tests/test_hybrid_supervised_features.py tests/test_hybrid_checkpoint.py -q`: `47 passed, 4 skipped`.
- `uv run --no-sync ruff check ...`: passed.
- `uv run --no-sync jgrec-build --help`: exposes `--supervised-feature-cache-dir`.
- `uv run --no-sync python -m compileall -q src tests`: passed.

## First Execution Step

Inspect the supervised feature lifecycle and enumerate the exact configuration dependencies used to build the cache fingerprint, then add the first failing identity test.
