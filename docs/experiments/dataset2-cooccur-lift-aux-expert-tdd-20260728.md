# Hai TDD: Dataset2 Causal Cooccur Lift Auxiliary Expert

## Target Behavior

Given a causal `TemporalInteractionIndex`, a source, query time, candidate
destinations, and the frozen short window:

- return full and short `log1p(cooccur) - log1p(destination popularity)`
  columns with strict-left upper cutoffs and an open lower short-window
  boundary;
- make an equally cooccurring but less popular destination receive a
  higher full lift;
- expose a read-only 65-column view whose first 63 columns preserve the
  base cache except for the frozen `gnn_short` overlay and whose final two
  columns are the lift features;
- reject any drift in the preregistered config.

## RED

- **Test added**: `tests/test_cooccur_lift.py`
- **Behavior asserted**: hand-counted full/short lift, strict exclusion of
  events at `time >= query_time`, exclusion at `time <= query_time -
  short_window`, popularity normalization, augmented column order, exact
  seeds/folds, and frozen-config drift rejection.
- **Command**:

  ```text
  uv run --no-project \
    --with pytest==9.0.3 \
    --with numpy==1.26.4 \
    python -m pytest tests/test_cooccur_lift.py -q -o addopts=
  ```

- **Observed failure**:

  ```text
  ModuleNotFoundError:
  No module named 'jgrec.rankers.hybrid.cooccur_lift'
  ```

- **Failure is correct because**: the public causal-lift module and all
  target APIs were absent. The failure occurred during import for the
  missing production behavior, not from a malformed fixture.

The requested local command
`uv run --group dev pytest tests/test_cooccur_lift.py -q` was also tried
first, but Windows attempted to build the Linux-oriented `pymetis`
dependency and failed on missing `sys/resource.h`. That environment setup
failure is explicitly not counted as RED. The project command will be
rerun in the Linux server environment.

## GREEN

- **Minimal implementation**:
  - added `cooccur_lift_scores` with the frozen 64-destination latest-unique
    history and strict causal `searchsorted` cutoffs;
  - added `CooccurLiftAugmentedView` for the 63 + 2 column contract;
  - added a strict frozen-config loader and deterministic fold seed;
  - added the rolling runner that materializes one shared lift matrix,
    binds its SHA-256 in the run contract, trains one 195-context-column
    Setwise head per fold, and writes Q1 diagnostics outside selector
    inputs.
- **Command**:

  ```text
  uv run --no-project \
    --with pytest==9.0.3 \
    --with numpy==1.26.4 \
    python -m pytest tests/test_cooccur_lift.py -q -o addopts=
  ```

- **Observed pass**:

  ```text
  13 passed
  ```

## REFACTOR

- **Refactor done**: yes
- **Change**: the first GREEN implementation was 309 lines. Constants and
  repeated formatting were compacted without changing behavior, bringing
  `cooccur_lift.py` to 278 lines and preserving the frozen public API.
- **Command after refactor**:

  ```text
  uv run --no-project \
    --with pytest==9.0.3 \
    --with numpy==1.26.4 \
    python -m pytest \
    tests/test_cooccur_lift.py \
    tests/test_listwise_mlp_exact_blend.py \
    tests/test_robust_weight_selection.py \
    -q -o addopts=

  uv run --no-project --with ruff==0.15.6 ruff check \
    src/jgrec/rankers/hybrid/cooccur_lift.py \
    tests/test_cooccur_lift.py \
    scripts/train_dataset2_cooccur_lift_rolling.py

  uv run --no-project python -m compileall -q \
    src/jgrec/rankers/hybrid/cooccur_lift.py \
    scripts/train_dataset2_cooccur_lift_rolling.py
  ```

- **Observed result**:

  ```text
  49 related tests passed on the remote Linux environment
  All checks passed!
  cooccur_lift.py: 278 lines
  ```

## Runtime Memory RED/GREEN

The direct Python index was exact but exceeded the 32 GB host envelope
before producing any candidate metric:

- the original materializer and one compact-array retry both reached about
  29.4 GB RSS with only about 1.18 GB available;
- each owned process group was terminated cleanly before fold training or
  selector invocation;
- neither attempt produced a candidate metric or changed the frozen
  configuration.

The GREEN runtime path is an exact native streaming materializer:

- symmetric cooccurrence counts use two dense triangular `uint8` arrays
  for full and short counts, with exact overflow maps;
- a short-window event queue expires old events while the stream advances;
- queries at a timestamp are scored before events at that timestamp are
  absorbed, preserving strict `searchsorted(side="left")` behavior;
- a micro-index fixture compares the native outputs to
  `TemporalInteractionIndex` exactly.

Observed production evidence:

```text
native materialization: 200000 x 100 x 2 float32 in 18.42 s
peak RSS: approximately 13.1 GB
lift SHA-256:
c4ca979de5c56de772edd87c0c9792868f3e22cb47e333ade4c2ca8fe7ff64f4
```

## External Guard RED/GREEN

`tests/test_cooccur_lift_external.py` first asserted missing lock-bound
external behavior, then passed after adding:

- exact selection-lock hash and integration-ID validation;
- the preregistered next full-origin seed (`33100`);
- a manifest that cannot be constructed before rolling acceptance;
- the existing evaluator's exclusive one-shot receipt semantics.

The focused cooccur-lift and external-guard suites report 19 passing tests.
The external cache had 19 rows tied with the training maximum timestamp;
the materializer deterministically retained the 19,981 rows strictly after
training, without reading any target label or score to make that cut.

## Online Materialization Equivalence

The 153,420 test rows contain only 2,180 unique sources. Chronological
feature scoring repeatedly evicted expensive exact source summaries, so
the delivery-only scorer was changed to stable source grouping and restores
probabilities to original CSV row order. This changes neither features nor
the accepted model.

An overlap audit against 6,400 already completed chronological rows found:

```text
shared completed rows: 601
maximum absolute probability difference: 2.4961035205439686e-07
mean absolute probability difference: 1.8826458136065617e-09
top-1 disagreements: 0
```

The grouped path improved observed throughput from one 4,096-row batch in
more than 27 minutes to about 8,000 rows/minute and completed all 153,420
rows in 1,089.31 seconds. The final probability matrix was finite and
normalized with maximum row-sum error `1.11e-15`.

## Packaging Regression

The shared packaging helper gained a default-preserving optional
`dataset2_mode` field so this candidate is not mislabeled as a Two-Tower
blend. The existing submission tests plus the new mode assertion pass:

```text
uv run --group dev pytest tests/test_partial_listwise_submission.py -q
4 passed
ruff: All checks passed!
compileall: passed
```

## Next Behavior

The source-grouped test materialization and one accepted locked-weight
package are complete. Dataset1 byte identity and the Dataset2 blend formula
were verified. Online submission remains a separate user action; no
checkpoint promotion is authorized by packaging.
