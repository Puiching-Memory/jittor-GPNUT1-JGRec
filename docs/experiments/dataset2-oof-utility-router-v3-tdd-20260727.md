# Hai TDD: Dataset2 OOF Utility Router v3

## Target Behavior

Build a pure-Jittor hurdle router that learns per-action
`gain / no-change / loss`, uses no positive-column or candidate-ID feature,
abstains on low-utility or unavailable actions, enforces a hard route quota,
and exactly replays its checkpoint.

## RED

- **Test added**: `tests/test_hybrid_oof_utility_router.py`
- **Behavior asserted**:
  - gain, no-change, loss, and unavailable targets remain distinct;
  - action features are invariant to a shared candidate permutation;
  - unavailable actions cannot be routed and the quota cannot be exceeded;
  - all four hurdle heads receive gradients;
  - a saved pure-Jittor checkpoint exactly replays utility predictions.
- **Command**:
  `.venv/bin/python -m pytest tests/test_hybrid_oof_utility_router.py -q`
  on the Linux/Jittor GPU host.
- **Observed failure**:
  `ModuleNotFoundError: No module named
  'jgrec.rankers.hybrid.oof_utility_router'`.
- **Failure is correct because**: the public module and behaviors named by the
  tests did not exist. Two earlier local attempts failed during Windows
  `pymetis`/Jittor environment setup and were explicitly not counted as RED.

## GREEN

- **Minimal implementation**:
  - added `OOFUtilityRouter`, a Jittor MLP with change, direction, gain
    magnitude, and loss magnitude heads;
  - added class-weighted hurdle loss and regret-weighted expected utility;
  - added action-specific label-free feature construction;
  - added unavailable-action masking, abstention thresholds, and hard quota;
  - added pure-NumPy checkpoint serialization of Jittor parameters and
    normalizers;
  - added a single fixed-protocol Dataset2 runner using frozen
    `top10/cap0.02` decoder actions.
- **Command**:
  `.venv/bin/python -m pytest tests/test_hybrid_oof_utility_router.py -q`
- **Observed pass**: `5 passed`.

## REFACTOR

- **Refactor done**: yes
- **Change**:
  - factored one candidate-permutation-invariant action feature contract shared
    by medium and long;
  - trained on per-action validity ranges instead of zero-filling or reducing
    to the long-horizon intersection;
  - separated warm-up false-positive mining from the final from-scratch model;
  - removed unused imports and satisfied repository lint.
- **Command after refactor**:
  - `.venv/bin/python -m pytest
    tests/test_hybrid_oof_utility_router.py
    tests/test_hybrid_high_confidence_topk_router.py
    tests/test_hybrid_multi_horizon_oof.py -q`
  - `uv run --no-sync ruff check
    src/jgrec/rankers/hybrid/oof_utility_router.py
    scripts/train_dataset2_oof_utility_router_v3.py
    tests/test_hybrid_oof_utility_router.py`
- **Observed result**: `18 passed`; ruff passed.

## Next Behavior

The conditional change-only LambdaMRR behavior was not entered: v3 failed the
frozen two-slice selection gate, so the goal document's stop rule blocked Phase
2 before a production test was added.
