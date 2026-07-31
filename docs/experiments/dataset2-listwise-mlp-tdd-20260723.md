# Hai TDD: Dataset2 Cached-Feature Listwise MLP

## Target Behavior

Train the existing fusion MLP on cached query groups with softmax cross-entropy targeting candidate zero, run a fixed number of epochs without validation selection, and expose a checkpoint-compatible final state plus its epoch losses.

## RED 1: Listwise Loss Contract

- **Test added**: `tests/test_hybrid_fusion_listwise.py`
- **Behavior asserted**: The Jittor loss matches a NumPy log-sum-exp reference for 2, 3, and 100 candidates; it is invariant to a per-query logit shift; raising candidate zero lowers the loss.
- **Linux command**: `uv run --no-sync pytest tests/test_hybrid_fusion_listwise.py -q`
- **Observed failure**: Collection failed with `ImportError: cannot import name '_listwise_positive_loss'`.
- **Failure is correct because**: No grouped listwise objective existed in the production fusion module.
- **Environment note**: The first local Windows attempt was intercepted by an unrelated Jittor/MSVC build failure, so the valid RED and all numerical verification were recorded on the Linux training server.

## GREEN 1

- **Minimal implementation**: Added `_listwise_positive_loss`, implemented as the negative mean log-softmax probability at candidate zero.
- **Linux command**: `uv run --no-sync pytest tests/test_hybrid_fusion_listwise.py -q`
- **Observed pass**: Six focused tests passed.

## RED 2: Fixed Training Does Not Select on Validation

- **Test added**: `test_fixed_listwise_trainer_evaluates_validation_only_after_all_epochs`
- **Behavior asserted**: A two-epoch run reports exactly two finite losses and invokes validation metrics exactly once, after training.
- **Observed failure**: The same RED import failed because `fit_fusion_mlp_listwise_fixed` did not exist.
- **Failure is correct because**: The only existing fusion trainers evaluated validation before and after every epoch and restored the best validation state.

## GREEN 2

- **Minimal implementation**: Added a fresh-initialized fixed-epoch streaming trainer with no early-stop or best-state path. It snapshots the final state and evaluates validation once at the end.
- **Linux command**: `uv run --no-sync pytest tests/test_hybrid_fusion_listwise.py tests/test_hybrid_checkpoint.py -q`
- **Observed pass**: 15 related tests passed.

## REFACTOR

- **Refactor done**: yes
- **Change**: Reused the existing streaming normalizer, deterministic initializer, state snapshot, metric evaluator, memory-release path, and `FusionResult` contract instead of creating a second model/state format.
- **Static command**: `uv run --no-sync ruff check src/jgrec/rankers/hybrid/fusion.py tests/test_hybrid_fusion_listwise.py scripts/train_dataset2_listwise_mlp_cached.py`
- **Observed result**: Ruff passed. The server run completed five fixed epochs, produced finite monotonic losses, and correctly rejected packaging when the frozen temporal gate failed.

## Next Behavior

If pursued, calibrate the improved listwise expert against the champion LightGBM on a genuinely separate calibration interval, with the weight frozen before a later untouched interval is read. Do not reinterpret the current reused validation gain as a submission signal.
