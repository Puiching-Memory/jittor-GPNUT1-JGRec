# TDD Evidence: Dataset2 Multi-Interest Proxy

## Target Behavior

Given ordered target embeddings, deterministically build temporal and
clustered interest centers, then return candidate max similarity, second-best
similarity, and positive-interest coverage.

## RED

- **Test added**: `tests/test_hybrid_multi_interest_proxy.py`
- **Behavior asserted**: temporal separation, deterministic K=2 modes, and
  exact max/top-2/coverage values.
- **Command**:
  `uv run --no-sync pytest -q tests/test_hybrid_multi_interest_proxy.py`
- **Observed failure**:
  `ModuleNotFoundError: jgrec.rankers.hybrid.multi_interest_proxy`
- **Failure is correct because**: the proxy behavior did not exist; test
  collection and the Python environment otherwise worked.

## GREEN

- **Minimal implementation**: Added normalized temporal centroids,
  deterministic recent/farthest cosine K-means, and three candidate affinity
  channels in a pure NumPy module.
- **Command**:
  `uv run --no-sync pytest -q tests/test_hybrid_multi_interest_proxy.py`
- **Observed pass**: `3 passed`.

## REFACTOR

- **Refactor done**: yes
- **Change**: Shared finite-matrix validation and row normalization; bounded
  source histories to 64 items; lazily concatenated nine proxy columns instead
  of copying the 5.04 GB source cache.
- **Command after refactor**:
  `uv run --no-sync pytest -q tests/test_hybrid_multi_interest_proxy.py tests/test_hybrid_gnn_window_config.py`
- **Observed result**: `4 passed` locally; `3 passed` remotely for the proxy
  module.

## Experiment Evidence

- Script: `scripts/evaluate_dataset2_multi_interest_proxy.py`
- Run: `dataset2_multi_interest_proxy_seed60_20260725`
- Exit: `0`
- Runtime: `357.0325` seconds.
- Full fixed-blend delta: `+0.0040634096`.
- Slice deltas: `+0.0047543407`, `+0.0044276391`,
  `+0.0030080907`.
- New-edge delta: `+0.0040634096` over 20,000 rows.
- Gate: passed.
- Local report:
  `result/dataset2_multi_interest_proxy_seed60_20260725/artifacts/multi-interest-report.json`
- Model SHA-256:
  `ac0b6702b746bbf8773878986c2ed5fd44da10884609bfcb8a395df32cd1c982`

## Next Behavior

Either construct a production-safe final-query feature path and submission
package using these frozen proxy definitions, or use the passed proxy as the
entry criterion for an end-to-end multi-interest graph tower. Do not tune K or
the feature definitions on the protected validation slices first.
