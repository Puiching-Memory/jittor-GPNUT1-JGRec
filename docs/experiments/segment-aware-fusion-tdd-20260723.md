# Hai TDD: Query-Level Segment-Aware MLP/LightGBM Fusion

## Target Behavior

Choose one MLP/LightGBM mixture weight per query from label-free, order-invariant segment evidence; preserve legacy scalar blending when no gate exists; serialize the gate in contest checkpoints; and activate it only when it improves full-candidate MRR outside its fit interval.

## RED 1: Gate Contract

- **Tests added**: `tests/test_hybrid_segment_fusion.py`
- **Behavior asserted**: Five descriptor families are order-invariant; rank ties prefer the current global weight; scalar and per-query blending are both valid.
- **Command**: `uv run --no-sync pytest tests/test_hybrid_segment_fusion.py -q`
- **Observed failure**: Import failed because `jgrec.rankers.hybrid.segment_fusion` did not exist.
- **Failure is correct because**: The query-segment contract and gate implementation had not been created.

## GREEN 1

- **Minimal implementation**: Added descriptor extraction, deterministic candidate-weight selection, gate fitting/prediction, and scalar/vector expert blending.
- **Command**: `uv run --no-sync pytest tests/test_hybrid_segment_fusion.py -q`
- **Observed pass**: Descriptor, oracle-weight, and blending tests passed.

## RED 2: Checkpoint Compatibility

- **Test added**: `test_hybrid_snapshot_carries_optional_segment_gate_state`
- **Behavior asserted**: A gate survives snapshot/hydration, while snapshots without the field retain scalar behavior.
- **Command**: `uv run --no-sync pytest tests/test_hybrid_segment_fusion.py -q`
- **Observed failure**: Snapshot did not contain `segment_gate_result`.
- **Failure is correct because**: Runtime checkpoint serialization had no segment-gate field.

## GREEN 2

- **Minimal implementation**: Added optional gate state to hybrid snapshot/hydration and prediction, using `snapshot.get(...)` for old-checkpoint compatibility.
- **Command**: `uv run --no-sync pytest tests/test_hybrid_segment_fusion.py -q`
- **Observed pass**: Five focused tests passed locally and on the Linux server.

## RED 3: Float Weight Labels

- **Test changed**: `test_segment_gate_learns_different_query_weights_from_observable_descriptors`
- **Behavior asserted**: Real mixture weights such as `0.07` and `0.90` are valid discrete gate outputs.
- **Command**: `uv run --no-sync pytest tests/test_hybrid_segment_fusion.py::test_segment_gate_learns_different_query_weights_from_observable_descriptors -q`
- **Observed failure**: scikit-learn raised `ValueError: Unknown label type: continuous`.
- **Failure is correct because**: `DecisionTreeClassifier` cannot use floating-point mixture weights directly as class labels.

## GREEN 3

- **Minimal implementation**: Encoded candidate weights as integer class IDs during fitting and decoded IDs during prediction, with exact membership validation.
- **Command**: `uv run --no-sync pytest tests/test_hybrid_segment_fusion.py -q`
- **Observed pass**: Five focused tests passed; the server integration advanced into real cached evaluation.

## RED 4: Optimize the Actual Metric

- **Test added**: `test_policy_gate_selects_each_leaf_weight_by_reciprocal_rank_reward`
- **Behavior asserted**: A leaf selects the weight with the best aggregate reciprocal-rank reward, rather than the most frequent noisy per-query class.
- **Command**: `uv run --no-sync pytest tests/test_hybrid_segment_fusion.py::test_policy_gate_selects_each_leaf_weight_by_reciprocal_rank_reward -q`
- **Observed failure**: Import failed because `fit_segment_policy_gate` did not exist.
- **Failure is correct because**: The initial classifier optimized label accuracy, and the real cached run showed Dataset1 regression and Dataset2 degeneration to one weight.

## GREEN 4

- **Minimal implementation**: Added a multi-output shallow regression partition and selected each leaf's discrete weight directly by summed full-candidate reciprocal-rank reward, preferring the global weight on ties.
- **Command**: `uv run --no-sync pytest tests/test_hybrid_segment_fusion.py -q`
- **Observed pass**: Six focused tests passed locally and on the Linux server. The frozen cached experiment accepted Dataset1 and rejected Dataset2 independently.

## REFACTOR

- **Refactor done**: yes
- **Change**: Kept descriptor extraction and inference serialization shared; isolated the MRR policy in `MRRPolicyTree`; retained the earlier classifier path for compatibility and diagnostics; added a guarded builder that overlays only accepted gates.
- **Command after refactor**: `uv run --no-sync ruff check ...`, `uv run --no-sync python -m compileall -q ...`, and `uv run --no-sync pytest tests/test_hybrid_segment_fusion.py tests/test_contest_checkpoint.py tests/test_submission.py -q`
- **Observed result**: Ruff and compileall passed; 18 related tests passed; full server inference, checkpoint round-trip, CSV validation, ZIP integrity, and remote/local SHA-256 checks passed.

## Next Behavior

Done. The only remaining quality evidence is one leaderboard submission of the guarded candidate; do not tune this gate further from that score if it is intended for hidden B.
