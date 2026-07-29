# Goal Document: Segment-Aware MLP/LightGBM Fusion

## Go / No-Go

- **Judgment**: Go
- **Reason**: Both datasets already have cached 100-candidate validation features and complementary MLP/LightGBM outputs. A shallow, inference-compatible gate can test the user's segmentation hypothesis without retraining towers or consuming leaderboard feedback.

## Target Outcome

Replace each dataset's single global MLP/LightGBM weight with an optional query-level segment gate that uses only observable candidate-set evidence for repeat/new-link behavior, source-history length, target coldness/popularity, candidate-prior strength, and recent-versus-long-term memory. Activate a dataset's gate only if it beats that dataset's current global blend on both a calibration interval and a final frozen temporal holdout, then produce a validated candidate package without changing either expert.

## Goal Definition

- **Type**: technical, learning, quality, and delivery
- **Boundary**: Query-level segment descriptors, full-candidate reciprocal-rank rewards, shallow policy-tree gates, checkpoint serialization/hydration, batch inference, cached validation evaluation, guarded per-dataset activation, full prediction, packaging, and local delivery.
- **Non-goals**:
  - Retrain or change MLP, LightGBM, GNN, sequence, two-tower, source-profile, statistics, or candidate generation.
  - Use the true positive candidate, candidate position zero, validation rank, or labels as gate inputs.
  - Fit a per-candidate gate whose weights could distort probability normalization differently inside one query.
  - Expand the gate grid after observing the final holdout or leaderboard.
- **Deferred work**:
  - New independently generated temporal feature caches.
  - Neural mixture-of-experts gating.
  - Segment-specific expert retraining.
- **Verification rule**: Train gate partitions and choose each leaf's expert weight by full-candidate reciprocal-rank reward on validation rows `0:10000`, choose one predeclared tree configuration on rows `10000:15000`, freeze it, and read rows `15000:20000` only for final acceptance. A gate is stored for a dataset only if it strictly improves MRR over that dataset's current global blend on both the calibration and final intervals.
- **Evidence source**: RED/GREEN tests, label-invariance tests, cache/checkpoint identity, per-split MRR, exported decision-tree rules, per-segment weight counts, checkpoint round-trip, CSV validation/hashes, and ZIP hash.
- **Pass criteria**: Gate inputs are identical when labels/candidate order conventions change without changing observable features; each activated dataset improves on calibration and final holdout; inactive datasets fall back byte-for-byte in behavior to the current global blend; final CSVs have expected dimensions and finite probabilities.
- **Confidence note**: The final 5,000 rows were seen only as an aggregate in the preceding Dataset2 experiment, so they are clean for gate selection but not a fully independent model-development benchmark. Hidden B remains the decisive test.
- **Judgment owner**: Tests own implementation correctness; the frozen temporal protocol owns offline activation; the competition score owns the final quality judgment.

## Current State

- Current online champion: `1.3426473547970703`.
- Dataset1 in the champion is unchanged from the full-MRR checkpoint and favors repeat-memory/MLP behavior.
- Dataset2 uses the tuned `learning_rate=0.03` LightGBM, best iteration 308, and global MLP weight `0.07`; its pseudo-B delta was positive.
- Both validation caches contain 20,000 chronological queries with 100 candidates and 63 features.
- Runtime currently stores one scalar `lgbm_result.mlp_weight` and applies it to every query.
- Existing checkpoints have no segment-gate field, so snapshot/hydration must remain backward compatible.

## Plan Rewrite Notes

| Existing item | Decision | Reason |
| --- | --- | --- |
| Historical repeat edge vs new edge | rewrite | At inference the true edge is unknown; use repeat-candidate ratio and maximum pair evidence across the candidate set. |
| Source history length buckets | keep | `src_activity` is label-free and query-level; shallow tree thresholds form learned buckets. |
| Cold target vs popular target | rewrite | Use candidate-set unseen ratio plus aggregate/max destination popularity, not the positive target's feature. |
| Candidate prior strength | keep | Use max, mean, and top-gap summaries from observable prior features. |
| Recent hit vs long-term memory hit | rewrite | Use candidate-set recent-hit rate and short/long pair-decay summaries without referencing which candidate is correct. |
| Dataset1 repeat / Dataset2 new-link specialization | keep | Train and accept gates independently; never force shared rules or weights. |
| Add another generic tower | remove | It is outside this attribution experiment and would invalidate cached-feature reuse. |

## Drift Diagnosis

- **Goal drift**: Changing expert training or candidate features would test a different hypothesis than segmented fusion.
- **Phase drift**: Gate runtime support must be proven before fitting real gates; package generation must follow frozen-holdout acceptance.
- **Validation drift**: Choosing rules on all 20,000 rows or after leaderboard feedback would erase the intended holdout.
- **Compatibility drift**: Old checkpoints must continue using the scalar global weight when `segment_gate_result` is absent.
- **Cleanup drift**: No unrelated CLI, feature, or tower refactor belongs in this experiment.

## Priority Rationale

- First prevent label leakage and backward incompatibility at the code boundary.
- Then test whether segment evidence generalizes across two later time intervals before paying for full inference.
- Accept Dataset1 and Dataset2 independently so a weak gate cannot damage the stronger dataset.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
| --- | --- | --- | --- |
| Cached validation row order remains chronological after sampling | confirmed | Enables blocked temporal protocol | `_sample_events` sorts sampled indices; record cache keys. |
| One weight per query is the intended granularity | assumed | Preserves within-query probability mixture semantics | Implement query-level descriptors; report this interpretation. |
| Decision-tree thresholds count as source/popularity buckets | assumed | Keeps the gate interpretable while learning cut points | Export exact rules and feature importance. |
| Each dataset may reject its gate independently | confirmed | Prevents forced regressions | Fall back to current scalar blend for failed datasets. |
| Candidate weight choices include the current global weight | confirmed | Gives the gate a no-change action | Add global weight plus a small fixed expert/blend set before fitting. |

## Phases

### Phase 1: Leak-Free Gate Contract

- **Purpose**: Define observable query descriptors, oracle-weight labels, gate prediction, and backward-compatible runtime behavior.
- **Entry condition**: Goal document is complete and feature names are known.
- **Phase rules**:
  - Write RED tests before production code.
  - Descriptor extraction may aggregate candidate features but cannot inspect labels, candidate zero specially, or ranks.
  - A missing gate must preserve the existing scalar blend exactly.
- **Todos**:
  - [x] Test and implement the five descriptor families.
    - **Surface**: new segment-fusion module and tests.
    - **Proof**: Permuting candidate rows changes only order-invariant summaries; no positive-index input exists.
    - **Depends on**: none.
  - [x] Test and implement deterministic oracle-weight targets and gate prediction.
    - **Surface**: segment-fusion module and tests.
    - **Proof**: Synthetic ranks select expected discrete weights and prefer the global weight on ties.
    - **Depends on**: descriptors.
  - [x] Test and implement checkpoint snapshot/hydration plus batch blending fallback.
    - **Surface**: hybrid ranker and checkpoint tests.
    - **Proof**: Old snapshot predictions are unchanged; gated snapshots produce different per-query weights and survive round-trip.
    - **Depends on**: gate predictor.
- **Exit proof**: Focused tests pass and demonstrate no label-dependent gate input.
- **Stop condition**: Stop if any requested segment cannot be computed without knowing the correct candidate.

### Phase 2: Dataset-Specific Frozen Evaluation

- **Purpose**: Determine whether segmentation generalizes beyond the interval used to fit it.
- **Entry condition**: Phase 1 is green; both caches and current expert checkpoints match feature layouts.
- **Phase rules**:
  - Fit gates on rows `0:10000` only.
  - Select among the fixed shallow-tree configurations on rows `10000:15000` only.
  - Freeze tree and discrete weight set before evaluating rows `15000:20000`.
  - Do not expand rules or grid after final-holdout results.
- **Todos**:
  - [x] Compute current MLP/LightGBM outputs and segment descriptors for Dataset1 and Dataset2.
    - **Surface**: cached evaluation script.
    - **Proof**: Cache/checkpoint keys, feature names, scalar baselines, and shapes are recorded.
    - **Depends on**: Phase 1.
  - [x] Fit and select one interpretable gate per dataset.
    - **Surface**: tuning reports and serialized gate states.
    - **Proof**: Train/calibration metrics, chosen depth/leaf size, rules, and weight counts are recorded before final evaluation.
    - **Depends on**: component scores.
  - [x] Apply the final-holdout activation gate independently.
    - **Surface**: frozen evaluation report.
    - **Proof**: Candidate-versus-global deltas on calibration and final rows for each dataset.
    - **Depends on**: frozen gate.
- **Exit proof**: Each dataset has an explicit activate/fallback decision with exported rules and no grid expansion.
- **Stop condition**: Reject a dataset gate on any final-holdout regression, non-finite weight, or degenerate one-weight tree that gives no segment behavior.

### Phase 3: Guarded Checkpoint and Submission

- **Purpose**: Package only validated gates while retaining rollback and attribution.
- **Entry condition**: At least one dataset gate passes Phase 2.
- **Phase rules**:
  - Start from the `1.3426473547970703` checkpoint.
  - Add only accepted `segment_gate_result` states; rejected datasets retain scalar blending.
  - Never overwrite the champion checkpoint or result directory.
- **Todos**:
  - [x] Write and round-trip the gated checkpoint.
    - **Surface**: contest checkpoint.
    - **Proof**: Both datasets load; expert model hashes remain unchanged; accepted gate rules match reports.
    - **Depends on**: Phase 2.
  - [x] Run full inference, validate, hash, and download the submission.
    - **Surface**: server result and local artifact.
    - **Proof**: Expected row counts, finite probabilities, two ZIP members, and matching remote/local SHA-256.
    - **Depends on**: checkpoint round-trip.
- **Exit proof**: A local, traceable submission ZIP exists with a per-dataset gate/fallback report.
- **Stop condition**: Do not package if both gates fail or if expert states differ from the champion.

## Dry-Run Findings

- “历史重复边 vs 新边” cannot use the positive candidate's repeat flag at inference; candidate-set repeat evidence is the valid replacement.
- Query-level weights preserve the existing convex-mixture semantics; per-candidate weights would introduce a second normalization and a different model class.
- The gate must see all 63 raw features before feature selection because its descriptors refer to named statistical/prior columns; expert scoring still uses the checkpoint's selected columns.
- Dataset2 must use the tuned LightGBM checkpoint/model, while Dataset1 uses the same champion snapshot unchanged.
- A tree that always emits one weight is not segment-aware and must be rejected even if it slightly improves due only to a new global weight.

## Final Validation

- Focused segment-fusion RED/GREEN tests, hybrid checkpoint regressions, Ruff, and compileall.
- Server reports prove three-way temporal separation, fixed grid, exported rules, and independent dataset activation.
- Gated checkpoint loads both datasets and preserves expert states.
- Submission rows and ZIP members validate; local/remote SHA-256 match.

## First Execution Step

Add a failing unit test for label-free, candidate-order-invariant extraction of the five query-segment descriptor families.

## Execution Result

- **Status**: Complete; guarded candidate built.
- The initial per-query classification gate was rejected: Dataset1 regressed on calibration and final, while Dataset2 collapsed to the existing global weight. This exposed that classification accuracy was the wrong optimization target.
- The fixed replacement was a shallow MRR policy tree: splits model the vector of reciprocal-rank gains, and each leaf selects the candidate weight with the highest aggregate full-candidate MRR reward.
- Dataset1 accepted `depth2_leaf500`: calibration MRR `0.78911019 -> 0.79183659` (`+0.00272640`); final MRR `0.78048670 -> 0.78198509` (`+0.00149839`).
- Dataset2 rejected: calibration improved `+0.00124271`, but final regressed `-0.00024136`; the packaged candidate therefore preserves the online-champion Dataset2 CSV and scalar weight exactly.
- Dataset1 rules use only `source_recency` and `memory_short_minus_long`, emitting MLP weights `0.00`, `0.60`, or `1.00` per query.
- Local submission: `result/d1_segment_policy_d2_champion_seed60_20260723/result.zip`, SHA-256 `29e02653559971ee200011f74ddc26f99a3424d218b0a95ed64684eb3d892c71`.
- Server checkpoint: `checkpoints/d1_segment_policy_d2_champion_seed60_20260723.pkl`, SHA-256 `f25e5af532bcd2346eecfcea3d95b8481a526d76f44eaec7dda9a39f9b480eb6`.
- ZIP integrity: exactly `dataset1.csv` with 61,051 rows and `dataset2.csv` with 153,420 rows; local and remote hashes match.
- **Risk note**: The final interval was observed by the rejected classifier before the policy-tree objective was introduced. The policy-tree grid was not expanded in response, but this interval is no longer a pristine independent benchmark; hidden B remains the decisive generalization test.
