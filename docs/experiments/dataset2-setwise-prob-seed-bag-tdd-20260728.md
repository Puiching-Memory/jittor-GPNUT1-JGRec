# Setwise Probability Seed Bag v1 — TDD Evidence

## Target behavior

For `integration_id=setwise_prob_seed_bag_v1`, freeze the six candidate
weights, two seed salts, three exact-rolling boundaries, and four-epoch
training contract before reading candidate metrics. Train exactly two new
Setwise heads per fold, use their arithmetic probability mean as the
auxiliary expert, and bind every integrated score matrix to the existing
per-fold champion and candidate fingerprint.

The rolling producer must not select a model or weight. Only
`scripts/select_robust_integrated_weight.py` may compute the decision panel
and create a lock. A rolling rejection must leave external unopened and
must not authorize packaging.

## RED

Command:

```text
uv run --no-project --with pytest==9.0.3 --with numpy==1.26.4 \
  python -m pytest tests/test_setwise_prob_seed_bag.py -q -o addopts=
```

Observed failure:

```text
ModuleNotFoundError: No module named 'jgrec.setwise_prob_seed_bag'
```

This was the intended RED: the public contract module did not exist. A
second RED then failed because
`load_verified_source_baseline` did not exist, proving that baseline hash,
shape, and candidate-fingerprint binding had not yet been implemented.

## GREEN

Minimal implementation:

- `src/jgrec/setwise_prob_seed_bag.py`
  - rejects drift in weights, salts, epochs, status, probability formula,
    or fold boundaries;
  - derives the two deterministic training seeds for each fold;
  - accepts exactly two finite, row-normalized probability matrices and
    returns their arithmetic mean;
  - verifies the source exact-rolling manifest identity, baseline SHA-256,
    score shape, and candidate fingerprint.
- `scripts/train_dataset2_setwise_prob_seed_bag_rolling.py`
  - retrains six four-epoch Setwise heads with the frozen graph-score
    overlay;
  - writes two seed probability matrices and models per fold;
  - materializes only the six predeclared integrated weights;
  - reports losses and provenance but computes no ranking panel.

Focused GREEN:

```text
19 passed in 0.35s
```

Related local regression:

```text
uv run --no-project --with pytest==9.0.3 --with numpy==1.26.4 \
  python -m pytest \
  tests/test_setwise_prob_seed_bag.py \
  tests/test_listwise_mlp_exact_blend.py \
  tests/test_robust_weight_selection.py -q -o addopts=
```

Result:

```text
31 passed in 0.63s
```

The same 31 tests passed on the GPU server before launch.

## Refactor decision

The reusable validation and probability-mean logic lives in
`jgrec.setwise_prob_seed_bag`; GPU/Jittor orchestration remains in the
script. The existing candidate materializer, rolling manifest builder,
selector, and external evaluator were reused unchanged. No external
materializer or package path was added because the real rolling gate
rejected the family.

## Static verification

```text
uv run --no-project --with ruff==0.15.6 ruff check \
  src/jgrec/setwise_prob_seed_bag.py \
  tests/test_setwise_prob_seed_bag.py \
  scripts/train_dataset2_setwise_prob_seed_bag_rolling.py

uv run --no-project python -m py_compile \
  scripts/train_dataset2_setwise_prob_seed_bag_rolling.py \
  src/jgrec/setwise_prob_seed_bag.py
```

Result:

```text
All checks passed!
```

