# Hai TDD: K=512 successor online-package preflight

## Target Behavior

Before any test scoring, accept only the exact current K=512 V1 and
gap-aware-v2 lineage authorized by the seven-gate external safety report, and
reject a historical/substituted V1 model even when the surrounding package
shape is otherwise valid.

## RED

- **Test added**:
  `tests/test_cooccur_lift_online_package_contract.py`
- **Behavior asserted**: Exact current-run inputs and implementation hashes
  pass; changing the current V1 model bytes raises
  `input hash differs: bugfixed_v1_model`.
- **Command**:
  `uv run --no-sync pytest tests/test_cooccur_lift_online_package_contract.py -q`
- **Observed failure**:
  `ModuleNotFoundError: No module named
  'jgrec.cooccur_lift_online_package_contract'`.
- **Failure is correct because**: The package-lineage validator did not exist,
  so neither the accepted path nor the V1-substitution guard could run.

## GREEN

- **Minimal implementation**: Added one pure preflight validator that checks
  the frozen contract header, 17 input hashes, implementation hashes,
  selection/external/receipt/materialization bindings, current V1 replay
  evidence, source ZIP member hashes, and absence of all output directories;
  added a thin CLI that writes an exclusive preflight receipt.
- **Command**:
  `uv run --no-sync pytest tests/test_cooccur_lift_online_package_contract.py -q`
- **Observed pass**: `2 passed`.

## REFACTOR

- **Refactor done**: yes
- **Change**: Centralized safe root-relative path resolution and streaming
  SHA-256 verification; kept orchestration in a hash-frozen shell operator and
  package contracts in JSON.
- **Command after refactor**:
  `uv run --no-sync ruff check
  src/jgrec/cooccur_lift_online_package_contract.py
  scripts/verify_dataset2_k512_successor_online_package.py
  tests/test_cooccur_lift_online_package_contract.py`
- **Observed result**: `All checks passed!`

Related regression command:

`uv run --no-sync pytest
tests/test_cooccur_lift_online_package_contract.py
tests/test_cooccur_lift_bugfixed_v1.py
tests/test_cooccur_lift_successor_external.py
tests/test_partial_listwise_submission.py
tests/test_submission.py -q`

Result: `27 passed`.

## Next Behavior

Done. The remaining action is the user's manual upload of the audited ZIP;
leaderboard interpretation is outside this package-generation behavior.
