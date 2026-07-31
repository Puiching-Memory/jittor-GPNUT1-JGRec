# Hai TDD: Dataset2 bounded source-sequence decoder

## Target Behavior

The frozen CST must remain the exact fallback. Candidate identity may affect
ranking only by querying a visible source sequence, and the resulting residual
must be frequency-shrunk, row-centered, finite, and hard-bounded by a fixed
cap no greater than `0.10`.

## RED

- **Test added**:
  `tests/test_hybrid_bounded_source_decoder.py`
- **Behavior asserted**:
  - empty source history reproduces base logits exactly, even with a non-zero
    residual head;
  - extreme raw residuals remain row-centered and within the cap;
  - unseen-item shrinkage is zero and support shrinkage is monotonic;
  - candidate permutation only permutes output scores;
  - a trained pure-Jittor checkpoint reloads identical scores.
- **Command**:

  ```bash
  .venv/bin/python -m pytest \
    tests/test_hybrid_bounded_source_decoder.py -q
  ```

- **Observed failure**:

  ```text
  ModuleNotFoundError:
  No module named 'jgrec.rankers.hybrid.bounded_source_decoder'
  ```

- **Failure is correct because**: the public bounded decoder API did not exist;
  the failure was not caused by syntax, fixtures, CUDA, or an unrelated
  dependency.

## GREEN

- **Minimal implementation**:
  - added a shared candidate/source item embedding used only inside
    candidate-to-history attention;
  - added time and position embeddings for the source sequence;
  - prohibited direct candidate-ID addition to the frozen base;
  - added support shrinkage `sqrt(count / (count + tau))`;
  - projected each residual row to exact zero mean and an absolute cap;
  - forced empty-history rows to an exact zero residual;
  - added pure-Jittor fit, inference, checkpoint, and audit APIs.
- **Command**:

  ```bash
  .venv/bin/python -m pytest \
    tests/test_hybrid_bounded_source_decoder.py \
    tests/test_hybrid_source_conditioned_cst.py -q
  ```

- **Observed pass**:

  ```text
  18 passed in 2.85s
  ```

## REFACTOR

- **Refactor done**: yes
- **Change**: separated the mathematical residual projection and support
  shrinkage from the model so the safety boundary can be tested independently;
  reused the existing Jittor listwise loss and checkpoint state helpers.
- **Command after refactor**:

  ```bash
  .venv/bin/python -m py_compile \
    src/jgrec/rankers/hybrid/bounded_source_decoder.py \
    scripts/train_dataset2_bounded_source_decoder.py

  .venv/bin/ruff check \
    src/jgrec/rankers/hybrid/bounded_source_decoder.py \
    scripts/train_dataset2_bounded_source_decoder.py \
    tests/test_hybrid_bounded_source_decoder.py
  ```

- **Observed result**: compilation passed and Ruff reported
  `All checks passed!`.

## CUDA smoke evidence

The real-cache smoke trained all three caps over 512 rows:

| Cap | Maximum residual | Maximum row mean | Empty-history exact | Replay error |
|---:|---:|---:|---:|---:|
| 0.02 | 0.0200000014 | 7.45e-11 | true | 0 |
| 0.05 | 0.0500000045 | 1.48e-10 | true | 0 |
| 0.10 | 0.1000000089 | 3.54e-10 | true | 0 |

All smoke audits passed.

## Next Behavior

Done for this experiment. The model result was evaluated under the frozen
rolling-origin and external gates; no additional cap or blend was selected
after external validation.
