# Hai TDD: Dataset2 K512 4/8-Worker Selection

## Target Behavior
On the first real cache batch, compare sequential, 4-worker, and 8-worker
features. Select 8 only when all byte hashes agree, every requested worker is
observed, 8 is at least 1.10x faster than 4, and at least 8 GiB memory remains;
otherwise select 4.

## RED
- **Test added**: `tests/test_parallel_structure.py`
- **Behavior asserted**: Exact faster 8-worker selection, speed/memory
  fallback, and cross-arm byte-drift rejection.
- **Command**:
  `.venv/bin/python -m pytest -q tests/test_parallel_structure.py`
- **Observed failure**: Collection failed because
  `select_parallel_worker_trial` did not exist.
- **Failure is correct because**: The missing public selection policy was the
  exact behavior required by the trial, not an environment or fixture error.

## GREEN
- **Minimal implementation**: Added the evidence-only selection policy and
  integrated one real first-batch sequential/4/8 comparison. The 4-worker pool
  closes before 8 starts; a rejected 8 arm is closed and a fresh proven
  4-worker pool resumes.
- **Command**:
  `.venv/bin/python -m pytest -q tests/test_parallel_structure.py`
- **Observed pass**: 10 passed.

## REFACTOR
- **Refactor done**: yes
- **Change**: Isolated worker-arm evaluation, selection policy, memory reading,
  and pool cleanup so exactness validation remains shared.
- **Command after refactor**: Ruff plus automatic-pipeline, successor,
  validation-protocol, parallel-structure, and fusion regression tests.
- **Observed result**: Ruff passed; 61 tests passed.

## Next Behavior
Runtime evidence: record the real first-batch sequential/4/8 hashes, times,
worker PIDs, memory reserve, and selected worker count.
