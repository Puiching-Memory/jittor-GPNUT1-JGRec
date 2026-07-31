# Hai TDD: Two-Tower 2×2 tie handling and in-batch context bugfix

## Target Behavior

- Exact score ties use their average rank, so an all-equal candidate row does
  not award rank 1 to the positive merely because it occupies column 0.
- The in-batch destination encoder uses the positive event's actual
  popularity, recency, and time context. Bucket index 0 is a real model value
  and must not be used as a synthetic neutral context.
- Model parameter shapes and legacy checkpoint compatibility remain unchanged.

## RED

### Tie-neutral ranking

- **Test added**:
  `tests/test_tower_optimization_experiment.py::test_positive_ranks_are_neutral_to_exact_score_ties`
- **Behavior asserted**: exact ties contribute half a rank per tied negative;
  an all-equal four-candidate row has rank 2.5 and does not count as Hit@1.
- **Command**:
  `C:\Users\75556\anaconda3\python.exe -m pytest tests/test_tower_optimization_experiment.py::test_positive_ranks_are_neutral_to_exact_score_ties -q`
- **Observed failure**: implementation returned integer ranks `[1, 1, 2]`
  instead of `[2.5, 1.5, 2.5]`.
- **Failure is correct because**: the old implementation counted only scores
  strictly greater than the positive and silently awarded every exact tie to
  candidate column 0.

### Positive-event destination context

- **Test added**:
  `tests/test_hybrid_two_tower.py::test_two_tower_in_batch_destination_keeps_each_positive_event_context`
- **Behavior asserted**: destination ID, popularity, recency, and time are
  taken from candidate column 0 of each positive training event.
- **Command**:
  `C:\Users\75556\anaconda3\python.exe -m pytest tests/test_hybrid_two_tower.py::test_two_tower_in_batch_destination_keeps_each_positive_event_context -q`
- **Observed failure**: import failed because
  `_in_batch_positive_destination_columns` did not exist.
- **Failure is correct because**: the old in-batch path discarded all three
  destination context columns and replaced them with bucket index 0.

## GREEN

- **Minimal implementation**:
  - `positive_ranks` now computes average ranks from the counts of greater and
    equal negative scores;
  - the in-batch helper validates aligned candidate matrices and selects the
    positive event's four destination columns;
  - `TwoTower._two_tower_in_batch_loss` passes those actual context columns to
    the existing destination encoder;
  - no parameters, config defaults, or checkpoint tensor shapes changed.
- **Command**:
  `C:\Users\75556\anaconda3\python.exe -m pytest tests/test_tower_optimization_experiment.py tests/test_tower_optimization.py tests/test_hybrid_two_tower.py -q`
- **Observed pass**: `26 passed, 8 skipped`; skips are existing
  Jittor-dependent tests in the local non-Jittor environment.

## REFACTOR

- **Refactor done**: no.
- **Change**: the minimal implementation already isolates the NumPy context
  contract from the Jittor training path; no additional refactor was needed.
- **Command after refactor**:
  `C:\Users\75556\anaconda3\python.exe -m ruff check src/jgrec/rankers/hybrid/in_batch_negatives.py src/jgrec/rankers/hybrid/two_tower.py src/jgrec/rankers/hybrid/tower_optimization_experiment.py tests/test_hybrid_two_tower.py tests/test_tower_optimization_experiment.py`
- **Observed result**: `All checks passed`; `py_compile` also passed for all
  three changed production modules.

## Next Behavior

The historical 2×2 metrics are superseded for model-effect inference. A new
run must regenerate every arm under the tie-neutral evaluator and corrected
in-batch destination representation. No server run was started as part of
this local bugfix.

## Environment Note

`uv run pytest` was attempted first, but dependency setup stopped before test
collection because `pymetis` requires the unavailable Windows header
`sys/resource.h`. The focused local tests therefore used the repository's
existing Python 3.12 test environment.
