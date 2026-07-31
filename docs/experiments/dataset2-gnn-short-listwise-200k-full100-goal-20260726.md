# Dataset2 gnn_short Listwise 200k/100 Goal

## Verifiable target

Start one detached server experiment that retrains only `gnn_short` on Dataset2 using:

- the most recent 200,000 training events;
- complete 100-candidate groups;
- positive candidate fixed at column 0;
- group-softmax/listwise loss;
- the champion's unchanged 128-dimensional XSimGCL encoder and seed 60;
- complete-candidate validation MRR for model selection and early stopping.

This turn is complete when the remote process is demonstrably stable: it remains alive, advances beyond preflight into training, writes fresh batches/metrics to its log, consumes plausible bounded resources, and shows no traceback, OOM, or non-finite loss.

## Boundary

In scope:

- reuse the existing Dataset2 200k train and 20k validation candidate caches;
- add the smallest isolated training path needed for `gnn_short`;
- add contract tests for 100-candidate grouping, positive-at-zero, and listwise loss;
- launch exactly one detached remote job and monitor it only until stable.

Out of scope:

- changing `gnn_recent`, Two-Tower, MLP, LightGBM, Setwise, or multi-interest features;
- changing embedding dimension, graph layers, or encoder family;
- rebuilding the complete champion;
- producing a submission package;
- waiting for the experiment to finish.

## Current state

- The current strongest online package scores `1.3530197200911278`.
- Controlled ablation indicates `gnn_short` contributes useful signal, while previous edge-weighting variants did not improve it.
- Existing caches already encode Dataset2 candidate groups for the recent-200k/full-100 setting, so candidate construction should not be repeated.

## Route

1. Prove the listwise grouping and loss contract with a failing test.
2. Implement the isolated `gnn_short` candidate-listwise training path.
3. Run focused local and remote tests plus a cache/config preflight.
4. Launch one named detached server job.
5. Verify process, log progress, resource use, and numerical stability; then stop monitoring.

## Phase rules

- Keep candidate order unchanged; candidate column 0 is the positive label.
- Reject any cache whose shapes, row counts, or candidate width disagree.
- Train only parameters owned by `gnn_short`.
- Select checkpoints only by full-100 validation MRR.
- Do not silently fall back to sampled candidates or pointwise BCE.
- Do not launch a second process if the named job already exists.

## Proof checklist

- [ ] RED: focused contract test fails for the missing listwise behavior.
- [ ] GREEN: focused test passes after the minimal implementation.
- [ ] Local lint/import checks pass.
- [ ] Remote cache/config preflight passes.
- [ ] One detached PID is recorded and alive.
- [ ] Log reaches real training work and continues advancing.
- [ ] Loss is finite; no traceback/OOM is present.
- [ ] CPU/GPU and memory use are plausible and bounded.

## Dry run

Given train candidates shaped `(200000, 100)` and validation candidates shaped `(20000, 100)`, the runner validates all sidecars, preserves positive column 0, initializes the champion-matched 128-dimensional `gnn_short` model, computes one group-softmax loss over each candidate row, evaluates validation MRR over all 100 candidates, and saves only the best `gnn_short` state.

## Go / No-Go

**GO**, provided the cache contract and focused tests pass. **NO-GO** if the cache is inconsistent, the loss is non-finite, another identical run is active, or the implementation updates parameters outside `gnn_short`.
