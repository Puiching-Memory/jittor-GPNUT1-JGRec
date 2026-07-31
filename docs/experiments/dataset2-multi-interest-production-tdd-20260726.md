# Dataset2 Multi-Interest Production TDD Evidence

## Target Behavior

Dataset2 final-test prediction must compute the same frozen nine
multi-interest proxy features used by validation, append them only to the
Setwise input, and remain backward-compatible with checkpoints that have no
proxy state.

## RED

- Added query-time tests for exact temporal2/K2/K4 feature ordering, cold
  source/target zero handling, optional append behavior, and checkpoint
  compatibility.
- Initial result: collection failed because
  `append_multi_interest_features` and the query-time feature API did not
  exist. This was the intended missing production behavior.

## GREEN

- Implemented query-time proxy generation in
  `src/jgrec/rankers/hybrid/multi_interest_proxy.py`.
- Added optional `multi_interest_proxy_state` to snapshot/hydrate/predict.
- Preserved the original 63 features for LightGBM and the original MLP.
  Setwise alone receives 72 raw features and 216 context features.
- Local verification:
  `uv run --no-sync pytest tests/test_hybrid_multi_interest_proxy.py
  tests/test_hybrid_checkpoint.py -q` -> `10 passed, 4 skipped`.
- Remote verification of the complete related suite -> `21 passed`.
- Ruff verification of the production module, ranker, tests, builder, and
  package-only runner -> passed.

## Production Replay and Packaging

- A fresh-process validation rebuild independently passed the frozen full-MRR
  and three-time-slice gate. Exact feature-array replay was intentionally not
  used as the acceptance rule because Jittor GNN reconstruction is not
  bitwise deterministic across processes.
- Final-prefix temporal2/K2/K4 state was stored in the Dataset2 checkpoint.
- The checkpoint was reloaded in a fresh process and all `153,419` Dataset2
  test rows were generated through the actual `ranker.predict` path.
- Dataset1 was copied byte-for-byte from the champion package.
- Submission validation passed for both CSV files and the ZIP.

## Artifacts

- Server checkpoint:
  `checkpoints/d1_champion_d2_multi_interest_proxy_v3_seed60_20260725.pkl`
  - bytes: `5,087,185,389`
  - SHA-256:
    `26bd659c10022e108777b7f0dc7772b4077a022252b8defa9b58abc9f0983028`
- Local submission:
  `result/d1_champion_d2_multi_interest_proxy_v3_package_seed60_20260726/result.zip`
  - bytes: `62,612,282`
  - SHA-256:
    `12c832e9c07448c4bb05c95f92df2a031e56e6991f8433523caf5669689812e1`
- Candidate report:
  `result/d1_champion_d2_multi_interest_proxy_v3_package_seed60_20260726/candidate-report.json`

## Refactor Decision

The proxy state remains optional and isolated from the base feature tensor.
No general feature-schema expansion was introduced, which protects old
checkpoints and avoids LightGBM/MLP dimensionality changes.
