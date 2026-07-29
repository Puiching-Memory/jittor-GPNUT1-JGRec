# Dataset2 K512 Successor V2 Automatic Pipeline — TDD Evidence

## Target behavior

One fail-closed entry point must wait for a validated K=512 full prediction,
build a fresh same-process 200k/20k cache, retrain every frozen near/gapped
head under the weighted-normalizer fix, run the dual-horizon selector, and
open external at most once. No external command may run before selection, no
unsupported candidate may be substituted, and no tolerance or V1 weight may
change.

The accelerated cache path must compare its first sequential batch with the
four-process result using exact array equality and byte hashes, and must stop
unless the measured speedup is at least 1.5x.

## RED

- Added `tests/test_cooccur_lift_automatic_pipeline.py`.
- Initial focused run failed during collection because
  `jgrec.cooccur_lift_automatic_pipeline` did not exist.
- The tests required:
  - both full-predict exit codes to be zero;
  - immutable baseline SHA-256 propagation into the duel manifest;
  - a new matching gap-aware selection report/lock before external;
  - external to remain the terminal stage after dual-horizon selection.

## GREEN

- Added the policy/validation module and the automatic stage controller.
- Added fresh cache and sidecar-lineage validation, generated V1/duel/external
  contracts, and new K=512 cache-report bindings.
- Corrected the duel manifest to publish the frozen top-level baseline hash
  before any score is selected.
- Ported the already-tested forked structure/source-profile tower to the
  200k/20k cache builder with an exact first-batch parity and speed gate.
- Related regression suite: **57 passed**.

Command:

```bash
.venv/bin/python -m pytest -q \
  tests/test_cooccur_lift_automatic_pipeline.py \
  tests/test_parallel_structure.py \
  tests/test_cooccur_lift_successor.py \
  tests/test_cooccur_lift_successor_external.py \
  tests/test_standard_validation_protocol.py \
  tests/test_hybrid_fusion_listwise.py
```

## Refactor decision

The policy checks live in `jgrec.cooccur_lift_automatic_pipeline`; the
long-running controller only sequences immutable commands and artifacts.
Historical result directories remain untouched. A failed or unmarked partial
stage is not deleted or silently resumed; it stops for inspection.

The deterministic gapped feature cache is reused after its frozen report hash
is checked, but every learned V1/full-only/gap-aware/prior head is retrained.

## Additional verification

- Ruff: all changed pipeline files passed.
- Python compilation: all changed pipeline files passed.
- Dry run proved the fixed order:
  `full predict -> cache -> near lift -> frozen contracts -> duel -> selector
  -> full-origin heads -> preflight -> one external gate`.
- The dry run explicitly records K=512/K=512, 200k/20k, four structure
  workers, exact parity, minimum 1.5x speedup, safety-gate-only external, and
  no package generation.

## Runtime proof to be filled by artifacts

The active run writes immutable evidence beneath
`result/dataset2_k512_cooccur_lift_successor_v2_rerun_20260729/`. The cache
reports will contain observed worker PIDs, exact hashes, measured speedup,
checkpoint SHA-256, K=512 limits, joint build ID/PID, shapes, and artifact
hashes. Later reports provide replay, fold, selection, preflight, receipt, and
seven-gate evidence.
