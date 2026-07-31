# Dataset2 GNN Short Capacity Result — 2026-07-27

## Verdict

**Reject `gnn_max_train_edges=200000`; keep the matched `50 epochs / 40000
edges` short-window control.**

The 200k-edge candidate remained slightly above the original champion on full
MRR, but failed the frozen `+0.001`/all-slices-non-decreasing gate and regressed
against the matched 40k-edge control on the full metric and every chronological
slice.

## Corrected Baseline

Repository defaults alone were misleading:

- `TrainingConfig.gnn_epochs` is 10, but the production CLI/checkpoint used by
  the Dataset2 champion resolves to 50 epochs.
- The matched targeted-GNN experiment had therefore already tested
  `50 epochs / 40000 edges`, not `10 / 40000`.
- The only untested capacity axis in this comparison was
  `max_train_edges: 40000 -> 200000`.

All other inputs were frozen:

- seed: 60;
- model: XSimGCL, 128 dimensions, 2 layers;
- variant: `gnn_short`, no edge weighting;
- checkpoint SHA-256:
  `a8dada300ff7aa87292fcff9c35498997e1c4013d4a3309451ba90e25666cf3f`;
- train-cache SHA-256:
  `b277fc53407f3b28d92a663dbcc071c5c37ac245d5f9d49833e70e80d52aa33d`;
- validation-cache SHA-256:
  `7c2cfb763a2803fa7b7bd754dc7f44fb40bedfa15c0015f2c1ca9bcd717ecbcf`;
- fusion: Setwise with fixed `0.80 Setwise + 0.20 LightGBM` evaluation.

## Result

| Metric | 50 / 40k control | 50 / 200k candidate | Delta |
|---|---:|---:|---:|
| Full MRR | 0.5484923183 | 0.5475115740 | -0.0009807443 |
| Slice 0 | 0.5887274003 | 0.5870364072 | -0.0016909931 |
| Slice 1 | 0.5488706206 | 0.5479325797 | -0.0009380409 |
| Slice 2 | 0.5078728415 | 0.5075597427 | -0.0003130988 |

Against the older champion (`0.5469178184`), the 200k candidate was
`+0.0005937556` on full MRR, but slice 1 declined by `-0.0003141116`.
The candidate therefore failed both parts of the frozen gate.

The Setwise expert alone also fell from `0.5473624488` at 40k edges to
`0.5454726396` at 200k edges (`-0.0018898092`).

## Runtime and Stability

- Remote GPU: NVIDIA GeForce RTX 4090, 48 GB.
- Exit code: 0.
- Elapsed time: 454.72 seconds.
- Two graph fits completed 50 epochs each.
- No traceback, OOM, non-finite loss, NaN, or Inf was found in the run log.
- Peak observed process RSS during fusion was about 14.2 GB; graph training
  used about 1.1 GB GPU memory.

## Decision

1. Do not increase the current short-window graph edge budget to 200k.
2. Keep `50 / 40k` as the offline winner for the pointwise short-window tower.
3. Do not spend a repeat run on 200k: the candidate missed the champion gate
   and lost to 40k on all four matched metrics, so the replication phase was
   not entered.
4. If A1 continues into production integration, start from the existing
   `short_none 50 / 40k` candidate rather than the 200k candidate.

The separate listwise GNN has already been integrated offline in two forms:
validation-only replacement regressed full MRR by `-0.0228498`, and
chronological OOF integration regressed it by `-0.0025563`. It should not be
treated as “never integrated”; any further rescue belongs under partial-blend
calibration rather than capacity expansion.

## Evidence

- Local report:
  `result/dataset2_gnn_short_capacity_e50_edges200k_seed60_20260727/artifacts/gnn-capacity-report.json`
- Local run log:
  `result/dataset2_gnn_short_capacity_e50_edges200k_seed60_20260727/run.log`
- Report SHA-256:
  `a10b6bbce6a41cb6fd58f2c591905c6995755d27fdf0d2cbe1a8297bcd01af32`
- Log SHA-256:
  `fe32887b45ccd60bf891f9290c9ffe3dd74810ca210692b022e6cb15b0ec895b`
- Remote score artifacts remain available under:
  `result/dataset2_gnn_short_capacity_e50_edges200k_seed60_20260727/artifacts/`

