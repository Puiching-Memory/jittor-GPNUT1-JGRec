# Dataset2 OOF / temporal signal correction TDD report

## Target behavior

Replace the static-ID correction direction with deterministic, leakage-safe
signals:

- multi-expert OOF disagreement;
- source-to-candidate support built only from events before the scoring origin;
- a fixed equal consensus of the two.

Every correction must:

- reorder only the frozen-base top 10;
- preserve the exact score multiset of each row;
- leave top-10-external candidates unchanged;
- leave every unselected row byte-identical to the frozen base;
- route at most 5% of rows;
- use only a pure-Jittor `jt.nn.Module` for the trainable router.

## RED

The tests were written before the implementation in
`tests/test_hybrid_disagreement_temporal_correction.py`.

The first authoritative Linux run failed during collection with:

```text
ModuleNotFoundError:
No module named 'jgrec.rankers.hybrid.disagreement_temporal_correction'
```

This was the expected failure: the new signal and proposal module did not
exist.

The local Windows `uv run pytest` attempt stopped earlier while building
`pymetis`, because the Windows compiler environment lacked
`sys/resource.h`. This was an environment failure, not the behavioral RED
evidence. All authoritative tests therefore ran in the competition Linux
environment.

The RED contracts cover:

- exact top-k scope and row score-multiset preservation;
- candidate-permutation behavior;
- unanimous and disagreeing OOF expert behavior;
- strict exclusion of equal-time and future history;
- temporal/hybrid candidate-permutation behavior;
- label-free router-feature permutation invariance;
- route-level multiset and sparsity audits.

## GREEN

The minimal implementation was added in:

- `src/jgrec/rankers/hybrid/disagreement_temporal_correction.py`;
- `scripts/train_dataset2_disagreement_temporal_correction.py`;
- `scripts/run_dataset2_disagreement_temporal_correction_20260727.sh`.

The implementation contains no trainable correction residual. It:

1. transforms expert logits to tie-neutral within-row percentile ranks;
2. builds a fixed consensus with an expert-rank disagreement penalty;
3. builds strict pre-origin pair-count, last-hit, and recent-global support;
4. produces the hybrid by an untuned equal percentile average;
5. reassigns the original descending top-10 scores according to the signal;
6. trains only the existing small Jittor confidence router.

Authoritative focused verification:

```text
14 passed in 2.75s
```

This includes the 8 new signal tests and 6 relevant existing router tests.

Static verification:

```text
python -m py_compile ...   passed
ruff check ...             All checks passed!
```

## Refactor decision

No broad refactor was performed. The existing confidence-router trainer,
checkpoint format, and hard-route implementation were reused. Signal
construction and score-multiset auditing were isolated in a new deterministic
module so they cannot accidentally import or reuse the old item-embedding
corrector.

A separate read-only diagnostic was added after the frozen experiment:

`scripts/diagnose_dataset2_temporal_correction_external.py`.

It does not mutate scores, checkpoints, selection locks, or the final report.

## Smoke evidence

The CUDA/Jittor smoke run covered all three candidates. Each candidate:

- routed exactly 25 of 512 rows, below the 5% cap;
- preserved the proposal and routed score multisets exactly;
- kept all top-10-external positions exact;
- kept all unrouted rows exact.

The router holdout contained positive correction labels for all three signals:

- OOF disagreement: 65 / 512;
- strict temporal support: 136 / 512;
- hybrid consensus: 142 / 512.

## Final verification commands

```bash
.venv/bin/python -m pytest \
  tests/test_hybrid_disagreement_temporal_correction.py \
  tests/test_hybrid_confidence_routed_topk_id.py -q

.venv/bin/ruff check \
  src/jgrec/rankers/hybrid/disagreement_temporal_correction.py \
  scripts/train_dataset2_disagreement_temporal_correction.py \
  scripts/inspect_dataset2_disagreement_temporal_inputs.py \
  scripts/diagnose_dataset2_temporal_correction_external.py \
  tests/test_hybrid_disagreement_temporal_correction.py

bash scripts/run_dataset2_disagreement_temporal_correction_20260727.sh
```

## TDD judgment

The requested behavior is protected and the implementation is green. The
model experiment itself is rejected by the frozen external gate; that negative
model result does not invalidate the implementation contracts.
