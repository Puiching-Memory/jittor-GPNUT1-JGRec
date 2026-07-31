# TDD Evidence: Dataset2 Setwise High-Weight Scan

## Target Behavior

1. Scan the inclusive Setwise-weight grid `0.80..1.00` at `0.01`.
2. Select a weight from chronological slices 0 and 1 only.
3. Keep slice 2 as a forward holdout and authorize packaging only when full
   MRR improves by at least `0.002` and no slice regresses.
4. Persist and reload the exact selected weight.
5. Generate a schema-valid package without rebuilding the eight-hour cache.

## RED

- Selector tests initially failed because the high-weight scan API did not
  exist. The test also changes only slice-2 predictions and requires the
  selected weight to remain unchanged.
- CUDA regression test initially failed because `jgrec.core.cuda` did not
  exist.
- The first server package attempt failed at GNN fitting with
  `jt.flags.use_cuda == 0`, proving the CUDA precondition was missing.
- The completed second package attempt exposed a final integration RED:
  `HybridRankerAdapter` has no direct `lgbm_result`; the persisted expert is
  owned by `adapter.impl`.

## GREEN

- Added an inclusive, deterministic prefix selector and an authorization
  helper in `fusion_analysis.py`.
- Added `require_jittor_cuda` and invoked it before final encoder fitting.
- Candidate packaging now consumes the authorized report weight, stores it in
  the Dataset2 checkpoint, and reads it back through `adapter.impl`.
- Added a finalize-only validator so the integration assertion could be fixed
  without rerunning test prediction. It checks:
  - evaluation authorization;
  - checkpoint and metadata weight equality;
  - both CSV schemas and row counts;
  - ZIP presence and all artifact hashes.

## Refactor Decision

The scan and weight authorization live in the reusable fusion-analysis module.
Artifact recovery is isolated in a finalize-only script because it has a
different responsibility from training and prediction and must never trigger
an expensive rerun.

## Verification Evidence

Local:

```text
ruff check: All checks passed
pytest: 16 passed, 4 skipped
```

Server:

```text
ruff check: All checks passed
scan runtime: 15.5658 seconds
selected weight: 0.80
stored checkpoint blend weight: 0.80
candidate status: complete
```

Metric gate:

| Metric | Champion | Selected 0.80 | Delta |
|---|---:|---:|---:|
| Full MRR | 0.4958900939 | 0.5469178184 | +0.0510277246 |
| Slice 0 | 0.5010874461 | 0.5863014322 | +0.0852139861 |
| Slice 1 | 0.4828043225 | 0.5482466914 | +0.0654423689 |
| Slice 2 holdout | 0.5037796964 | 0.5061992242 | +0.0024195278 |

Final artifact:

```text
result.zip bytes: 63843429
SHA-256: 6b8fdf96d3fbded938865b644fdf103cfcb67f7df38e4915e4f62aba9d8cab26
ZIP entries: dataset1.csv, dataset2.csv
local/server hash match: true
```
