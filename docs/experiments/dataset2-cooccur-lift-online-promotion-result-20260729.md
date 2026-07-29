# Dataset2 Cooccur Lift Online Promotion Result

## Verdict

`cooccur_lift_aux_expert_v1` is formally **accepted and promoted** at locked
weight `0.50`. The current champion is bound to:

`796d8d21a0c706ad11f244385b314d471d522c3b807748a54fe4ac78722f5880`

The checkpoint is
`checkpoints/d1_time_ramp_g050_d2_cooccur_lift_aux_w050_20260729.pkl`
and is 5,136,908,105 bytes.

## Artifact Binding

| Artifact | Frozen SHA-256 / state |
|---|---|
| Champion checkpoint | `796d8d21a0c706ad11f244385b314d471d522c3b807748a54fe4ac78722f5880` |
| Accepted online ZIP | `7ebfeb7ea29d8dcd03a43a7433a43ddc8de0e24245d93115ecbf8ebd17ef50eb` |
| Immutable pre-wiring receipt | `ff45301fcc3ec797477c17c72ea9687a5507dc32e1dd2299fac178245b60a7ee` |
| Replay A and B | `2b4012edb3a9d18675b00417553b0366438db44a9d5680a7566e693cd20b21e0` |
| Replay report | `ca6ba78d75a6c48f573ee2992613ff6f688f03545b9ab1a181d43a79b2014494` |
| Decision / promotion | `accepted / promoted` |

## Equivalence Contracts

### Raw-model equivalence: passed

- Full model and feature identity is bound by checkpoint, source checkpoint,
  auxiliary model, lift feature, query fingerprint, Dataset1 pickle, and
  protected Dataset2 state hashes.
- The bounded sequential replay processed 12 source-grouped batches through
  the worst-known drift row 97,576.
- Maximum raw absolute error:
  `6.938893903907228e-18`.
- Values/rows above `5e-7`: `0 / 0`.
- Raw Top-1 disagreements: `0`.
- This is artifact-wide identity plus targeted worst-case numeric replay; it
  does not claim a separate 153,420-row pre-tie raw replay.

### Tie-safe service equivalence: passed

- Two independent 153,420-row standard-load replays are byte-identical.
- Served rows with exact ties: `0`.
- Top-1 disagreements against the accepted online ZIP: `0`.
- Top-3 set disagreements: `0`.
- The service contract is deterministic, tie-free, Top-1 exact equivalence;
  it does not claim byte, full-order, or raw numeric identity to the
  eight-decimal accepted ZIP.

## Mandatory Residual Diagnostics

The old combined gate compared different boundaries:

- Accepted ZIP: blend an already tie-safe champion CSV with stored auxiliary
  probabilities, then write eight decimals.
- Standard service: blend raw checkpoint heads, apply final tie handling,
  then write seventeen significant digits.

Consequently, 165 rows and 923 values exceed `5e-7`; maximum absolute delta
is `0.00177944374565564`, mean absolute delta is
`4.82333312981554e-09`. There is one Top-10 set disagreement and three
Top-10 prefix-order disagreements. These are recorded as diagnostics and
do not participate in the user-authorized tie-safe Top-1 service gate.

## Protocol Closure

- No weight, formula, window, salt, or model change.
- No retraining, metric reselection, or rescan.
- The old failed auto-finalizer status remains unchanged and is explicitly
  superseded by the new accepted/promoted status.
- The old checkpoint integration report remains immutable even though its
  pre-replay `package_authorized=false` field is stale; the final replay
  report and promoted manifest are the newer authority.

## Successor Protocol Gate

The v1 promotion above remains accepted and promoted. It is not retroactively
reselected. However, the transport audit found that the validation protocol
used for v1 cannot fairly decide a time-local successor such as
`cooccur-lift full-only v2`:

- strict external attributes `36.18%` of separate first-layer lift energy to
  the short channel;
- online materialization has `39.9720%` all-zero short rows;
- the observed external-to-online effect ratio is `19.50x`.

The successor plan is now preregistered before any successor metric, with
exactly two candidates: `cooccur_lift_full_only_v2` and
`cooccur_lift_gap_aware_v2`. The plan binds the v1 champion checkpoint as
baseline and freezes this eligibility conjunction:

- every near fold has MRR/NDCG@10 delta `>= 0`;
- every P75/P90/P100 gapped fold has MRR delta `> 0` and NDCG@10 delta
  `>= 0`.

The deployment mixture remains a diagnostic panel but cannot compensate for a
failed near or gapped fold in this duel. If both candidates pass, ordering is
mean gapped MRR, worst gapped MRR, mean near MRR, then preregistered tie-break.
The v1 integration weight remains exactly `0.50`; weight rescans are forbidden.

An optional near-fold `zero_short` arm may be reported as a mechanism
cross-check but cannot participate in selection. For all time-local candidate
families, external is now `safety_gate_only`; raw external delta is not an
authorized online effect-size estimate, and the report records the conservative
`raw_delta / 19.5` calibration proxy.

Machine-readable forward status and the exact `plan-v2` lock are bound into the
replay report, promoted manifest, and canonical status. The old under-specified
plan/lock remains preserved only as superseded history. The three real gapped
folds were materialized from the full Dataset2 history under the exact
P75/P90/P100 schedule. The frozen selector selected
`cooccur_lift_gap_aware_v2`: its three near MRR deltas are
`+0.012193/+0.012395/+0.014808`, and its gapped deltas are
`+0.006620/+0.007734/+0.004013`; all corresponding NDCG@10 gates also pass.
`cooccur_lift_full_only_v2` was rejected because all three gapped MRR deltas
are negative.

Selection report SHA-256 is
`dc25b5b445e6b8f188f072bfec03e80b19e6f8fa1acfb34991c90d3cb9f25344`;
selection lock SHA-256 is
`b52b529534b717ef136c82b17a090889b5aa4d67aed8618605ecbfda828e7e30`.
The selected candidate is eligible for the one-shot external safety gate, but
external remains unauthorized and unopened; no external receipt, checkpoint,
ZIP, or package was generated.

## Canonical Evidence

- [Promoted manifest](cooccur-lift-aux-expert-v1.promoted.json)
- [Successor validation plan](cooccur-lift-successor-v2-duel.validation-plan.json)
- [Full-only candidate](cooccur-lift-full-only-v2.preregistered.json)
- [Gap-aware candidate](cooccur-lift-gap-aware-v2.preregistered.json)
- [Replay report](../../result/dataset2_cooccur_lift_online_promotion_20260729/promotion-v2/replay-report.json)
- [Canonical status](../../result/dataset2_cooccur_lift_online_promotion_20260729/promotion-v2/status.json)
- [Successor plan-v2 lock](../../result/dataset2_cooccur_lift_successor_v2_duel_20260729/plan-v2/validation-plan-lock.json)
- [Successor selection report](../../result/dataset2_cooccur_lift_successor_v2_duel_20260729/selection-v5-cpu-replay-wiringfix/selection-report.json)
- [Successor selection lock](../../result/dataset2_cooccur_lift_successor_v2_duel_20260729/selection-v5-cpu-replay-wiringfix/selection-lock.json)
- [Goal](dataset2-cooccur-lift-online-promotion-goal-20260729.md)
- [TDD evidence](dataset2-cooccur-lift-online-promotion-tdd-20260729.md)
