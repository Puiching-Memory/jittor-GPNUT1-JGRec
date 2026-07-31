# TDD Evidence: Dataset2 Targeted GNN Improvements

## Target Behavior

Allow `gnn_short` and `gnn_recent` to select edge weighting independently,
preserve the existing global defaults and old checkpoints, and ensure
repeat/time-decay weights reach LightGCN/XSimGCL message passing.

## RED

- Test: `tests/test_hybrid_gnn_window_config.py`
- Command:
  `uv run --no-sync pytest -q tests/test_hybrid_gnn_window_config.py`
- Expected failure:
  `ImportError: cannot import name 'graph_window_edge_parameters'`
- Why it failed for the right reason: the production configuration had no
  per-window resolution contract.

The initial attempt to collect `tests/test_hybrid_gnn_edges.py` locally was not
counted as RED because the Windows Jittor compiler environment failed before
test collection.

## GREEN

- Added optional full/recent/short edge-weighting and time-decay overrides.
- Added `graph_window_edge_parameters`, with global fallback for old
  checkpoints.
- Added `_graph_window_data`, returning edge indices and aligned weights.
- Propagated symmetric weights into LightGCN/XSimGCL and passed them to
  `gcn_norm`.
- Preserved `_graph_window_edges` for existing callers and tests.

Verification:

- Local:
  `uv run --no-sync pytest -q tests/test_hybrid_gnn_window_config.py tests/test_hybrid_checkpoint.py tests/test_cli.py`
  -> `28 passed, 4 skipped`.
- Remote:
  `uv run --no-sync pytest -q tests/test_hybrid_gnn_window_config.py tests/test_hybrid_gnn_edges.py`
  -> `8 passed`.
- Remote CUDA execution trained weighted XSimGCL windows for 50 epochs without
  non-finite values or runtime errors.

## REFACTOR

- Kept edge sampling and message-passing weights aligned through shared sampled
  indices.
- Updated graph tail limiting so any weighted window retains the event history
  required to aggregate repeats and time decay.
- Used a lazy one-column cache overlay, avoiding five copies of the 5.04 GB
  full feature cache.

## Experiment Evidence

- Script: `scripts/evaluate_dataset2_targeted_gnn_edges.py`
- Run: `dataset2_targeted_gnn_edges_seed60_20260725`
- Exit status: `0`
- Runtime: `1536.9234` seconds.
- Local report:
  `result/dataset2_targeted_gnn_edges_seed60_20260725/artifacts/edge-weight-report.json`

| Variant | Full MRR | Slice 0 | Slice 1 | Slice 2 | Verdict |
|---|---:|---:|---:|---:|---|
| Champion | 0.5469178 | 0.5863014 | 0.5482467 | 0.5061992 | Baseline |
| short none control | 0.5484923 | 0.5887274 | 0.5488706 | 0.5078728 | Control |
| short repeat | 0.5474764 | 0.5849254 | 0.5482687 | 0.5092293 | Reject |
| short time decay | 0.5455183 | 0.5856268 | 0.5453703 | 0.5055517 | Reject |
| recent none control | 0.5485319 | 0.5883693 | 0.5495771 | 0.5076431 | Control |
| recent time decay | 0.5468777 | 0.5847169 | 0.5464956 | 0.5094150 | Reject |

## Final Judgment

The requested repeat/time-decay edge formulations do not pass the frozen
offline gate. The full-100-candidate graph listwise phase remains RED/deferred
because its entry condition was not met; proceeding would add expensive OOF
temporal training without a stable edge winner. No package was generated.
