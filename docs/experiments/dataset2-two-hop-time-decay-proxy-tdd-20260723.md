# Hai TDD: Dataset2 Time-Decayed Two-Hop Proxy

## Target Behavior

Collect causal item-pair co-occurrence event times, calculate raw and exponentially decayed two-hop scores, rank sparse ties neutrally, and enforce the frozen proxy continuation gate.

## RED

- **Test added**: `tests/test_hybrid_two_hop_decay_proxy.py`.
- **Behavior asserted**: Canonical symmetric pairs, latest unique history, duplicate/eviction semantics, strict exclusion of events at or after query time, analytical decay values, average tie ranks, and all proxy gate conditions.
- **Command**: `uv run --no-sync pytest tests/test_hybrid_two_hop_decay_proxy.py -q`.
- **Observed failure**: Collection failed with `ModuleNotFoundError` because `jgrec.rankers.hybrid.two_hop_decay_proxy` did not exist.
- **Failure is correct because**: The new temporal contract had no implementation before the test.

## GREEN

- **Minimal implementation**: Added pure helpers for canonical pairs, latest unique targets, required-pair event collection, raw/decayed scoring, tie-neutral MRR, and gate evaluation.
- **Command**: `uv run --no-sync pytest tests/test_hybrid_two_hop_decay_proxy.py -q`.
- **Observed pass**: Six focused tests passed locally and on the Linux server.

## RED: Compressed Event Arrays

- **Test changed**: `test_two_hop_scores_exclude_current_and_future_events` now supplies NumPy event arrays, matching the server script's compact representation.
- **Behavior asserted**: Scoring accepts compact event arrays without changing causal filtering or analytical values.
- **Command**: `uv run --no-sync pytest tests/test_hybrid_two_hop_decay_proxy.py -q`.
- **Observed failure**: `ValueError: The truth value of an array with more than one element is ambiguous`.
- **Failure is correct because**: The production-shaped proxy input exposed an unsupported array truth check.

## GREEN: Compressed Event Arrays

- **Minimal implementation**: Replaced the truth-value check with an explicit `None`/length check.
- **Command**: `uv run --no-sync pytest tests/test_hybrid_two_hop_decay_proxy.py -q`.
- **Observed pass**: Six tests passed on Windows and Linux; the server proxy then completed successfully.

## REFACTOR

- **Refactor done**: yes.
- **Change**: Kept the experiment driver separate from pure temporal/ranking contracts and normalized imports.
- **Command after refactor**: `uv run --no-sync ruff check src/jgrec/rankers/hybrid/two_hop_decay_proxy.py tests/test_hybrid_two_hop_decay_proxy.py scripts/evaluate_dataset2_two_hop_decay_proxy.py`.
- **Observed result**: All checks passed.

## Next Behavior

Done for the proxy. Production temporal-state, feature-cache, checkpoint compatibility, and leaderboard packaging are a separate goal.
