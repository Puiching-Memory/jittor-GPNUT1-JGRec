# Hai TDD: Dataset2 Multi-Interest Confidence Gate

## Target Behavior

Route every Dataset2 query wholly to either the champion or multi-interest
expert, preserve champion scores exactly on fallback queries, use only
label-free permutation-invariant descriptors at inference, and reject a gate
unless three blocked temporal OOF folds and all three chronological slices
pass the fixed `+0.002` rule.

## RED

- **Test added**:
  `tests/test_hybrid_multi_interest_gate.py`
- **Behavior asserted**:
  exact whole-query fallback, descriptor permutation invariance, disjoint
  blocked folds, OOF routing, high-confidence trial selection, serializable
  final gate prediction, and score-only production descriptor parity.
- **Command**:
  `uv run --no-sync pytest tests/test_hybrid_multi_interest_gate.py -q`
- **Observed failure**:
  the cycles first failed on missing module/API imports:
  `multi_interest_gate`, `ConfidenceGateConfig`,
  `select_stable_high_confidence_trial`, `fit_confidence_gate`, and
  `expert_score_descriptors`.
- **Failure is correct because**:
  each failure identified the next missing public behavior rather than a
  syntax, fixture, environment, or assertion problem.

## GREEN

- **Minimal implementation**:
  added label-free descriptors, exact query routing, three-fold blocked OOF,
  the stability stop rule, high-confidence selection with threshold/coverage
  constraints, and a serializable final decision-tree gate.
- **Command**:
  `uv run --no-sync pytest tests/test_hybrid_multi_interest_gate.py -q`
- **Observed pass**:
  `7 passed` locally and `7 passed` on the server.

## REFACTOR

- **Refactor done**: yes
- **Change**:
  extracted the seven score-derived descriptors into
  `expert_score_descriptors`. The final trained gate uses only two of them, so
  production packaging can route existing champion/candidate CSVs without
  rebuilding graph features.
- **Command after refactor**:
  `uv run --no-sync pytest tests/test_hybrid_multi_interest_gate.py -q`
  and
  `uv run --no-sync ruff check src/jgrec/rankers/hybrid/multi_interest_gate.py
  tests/test_hybrid_multi_interest_gate.py
  scripts/package_dataset2_multi_interest_confidence_gate.py`
- **Observed result**:
  `7 passed`; Ruff passed locally and remotely.

## Next Behavior

Done. The produced package was validated for `61,051` Dataset1 rows,
`153,420` Dataset2 rows, exact champion fallback, and matching local/remote
SHA-256.
