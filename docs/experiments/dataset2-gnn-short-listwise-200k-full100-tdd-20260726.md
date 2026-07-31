# TDD Evidence: Dataset2 gnn_short Listwise 200k/100

## Target behavior

Train only `gnn_short` with 200,000 cached Dataset2 query groups, exactly 100 candidates per group, the positive at column 0, group-softmax loss, and full-candidate validation MRR early stopping.

## RED

Command:

```bash
uv run --no-sync pytest -q tests/test_hybrid_gnn_listwise.py
```

The server test failed during collection with:

```text
ModuleNotFoundError: No module named 'jgrec.rankers.hybrid.gnn_listwise'
```

This was the intended failure: the candidate-group and GNN listwise contract did not yet exist.

## GREEN

Added `gnn_listwise.py` with:

- the positive-at-zero/full-width cache validator;
- group-softmax positive loss;
- complete-candidate MRR;
- differentiable graph candidate logits.

Added the isolated Dataset2 runner and detached launcher. The focused server test result was:

```text
3 passed
```

Ruff also reported:

```text
All checks passed!
```

## Refactor decision

The listwise helpers were kept outside the existing pointwise/BPR `GraphTower.fit` path. This prevents an experiment-only objective from changing champion behavior or the other GNN windows.

## Runtime verification

The corrected `v2` job exited successfully:

- best epoch: 16;
- best full-100 validation MRR: `0.4659887151`;
- early stop: epoch 19;
- exit code: 0;
- no submission package generated.
