# Dataset2 Bug-Fixed v1 Submission Result

## Status

Complete. The user manually uploaded the audited package and reported an
online score of `1.3577315048069973`. The bug-fixed V1 therefore passes the
external non-shrink safety gate. This result does not authorize automatic
promotion or deployment.

## Root Cause of `0.10733`

The old V1-vs-current replay error was not a tolerance-edge case. Repeating
the current fixed-epoch trainer twice on identical real features, with the same
seed, NumPy-initialized MLP, batch order, and hyperparameters, produced:

| Training execution | Rows | Maximum probability error | Gate |
|--------------------|-----:|--------------------------:|------|
| Default CUDA | 20,000 | `0.05319197303453227` | fail |
| CUDA + deterministic cuBLAS workspace | 20,000 | `0.048016706738819914` | fail |
| CPU | 2,000 | `0.0` | pass |
| CPU | 20,000 | `0.0` | pass |

The CUDA path also changed epoch losses and final state arrays. This proves
that the currently frozen seed and deterministic NumPy initialization are
necessary but insufficient for reproducible CUDA training. The original
`0.10732766809698313` guard therefore exposed execution-level training
nondeterminism; it must not be accepted by widening the replay tolerance.

## Refrozen Candidate

- Candidate:
  `cooccur_lift_aux_expert_v1_bugfixed_refit_20260729`.
- V1 structure unchanged: full + short cooccur-lift channels, 195-column
  Setwise context, four epochs, batch size 256, seed 33100.
- Blend weight unchanged: `0.50`; no rescan.
- Training device: deterministic CPU.
- Test scoring device: CUDA.
- Historical model and historical external result are not reused as evidence.
- Model, source checkpoint, frozen config, selection lock, training report,
  test probabilities, and package are SHA-bound end to end.

## Runtime Evidence

- Full-data dual training replayed exactly:
  `matched=true`, `state_matched=true`, `loss_matched=true`,
  `probability_matched=true`, `max_abs_error=0.0`,
  `mean_abs_error=0.0`.
- Replay tolerances remained frozen at `rtol=2e-5`, `atol=2e-6`;
  `tolerance_relaxed=false`.
- Both runs produced identical epoch losses:
  `[2.3397681310658567, 2.155058836693044, 2.1289023127397306,
  2.1131875859502025]`.
- Deterministic CPU training took `150.8558006286621` seconds.
- CUDA scoring materialized shape `[153420, 100]` in
  `1094.2295274734497` seconds, with maximum row-sum error
  `1.1102230246251565e-15`.
- Dataset1 is the byte-identical frozen champion member:
  61,051 rows, 100 columns,
  SHA-256 `81285eddfba612b9c05e96075be29a15718d8a94839d143e3083bf5353644369`.
- Dataset2 contains 153,420 rows and 100 columns, using the frozen
  `0.50 * champion + 0.50 * expert` formula, with SHA-256
  `702f46d6a14b36e5330cac315ceefb130e54d4a68f9a173ce9d65c5a1d06192f`.
- Model SHA-256:
  `73788cb3c71c2b4768d7456f3995ddbd131812a700eabd9e9ee914a359b98d52`.
- Test auxiliary probabilities SHA-256:
  `4fd99424f94e472f42c618d309ab1c55f3ec74973eb9c188c6c62533fb1f16f3`.
- Audited ZIP SHA-256:
  `b90960c3427f70e2745bcb381289fca4625c208ebfaefb43ecdbc7a7387ff2f0`.
- Local audited package:
  `artifacts/dataset2_cooccur_lift_bugfixed_v1_refit_cpu_retry1_20260729/result.zip`.
- No V2 candidate ran, no V1-family weight rescan occurred, and no historical
  external metric was read during training, materialization, or packaging.

## External Interpretation

- Previous promoted V1 score: `1.357529740346302`.
- Bug-fixed V1 score: `1.3577315048069973`.
- Raw change: `+0.0002017644606953` (`+0.014862618084803833%`).
- Under the frozen `19.5x` transport discount, the interpretation-only
  calibrated change is `+0.000010346895420271795`.
- The external observation is retained only as evidence that the rebuilt V1
  did not shrink online. It must not be reported as a measured effect size.
- Promotion remains unauthorized pending an explicit user decision.
- Machine-readable record:
  `docs/experiments/cooccur-lift-aux-expert-v1-bugfixed-refit.online-result.json`.
