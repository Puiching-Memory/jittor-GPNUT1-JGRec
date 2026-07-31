# Goal Document: Dataset2 K512 Successor V2 Automatic Pipeline

## Go / No-Go
- **Judgment**: Go
- **Reason**: The user explicitly authorized an automatic continuation from
  the full K=512 prediction artifact through fresh 200k cache construction,
  all frozen fold-head retraining, dual-horizon selection, and one external
  safety-gate opening. Existing frozen candidate shapes, folds, weights,
  seeds, and gate thresholds remain unchanged.

## Target Outcome
After the current full K=512 Dataset2 prediction validates successfully, run
one durable, fail-closed pipeline that creates a new K512 cache lineage,
replays every frozen near and gapped trainable head under the corrected
weighted-normalizer implementation, applies the standard dual-horizon
selector, and opens external at most once only if the new selector authorizes
it.

## Goal Definition
- **Type**: technical / operational / delivery
- **Boundary**:
  - Source checkpoint SHA-256
    `0b0a846cb6c7f5b5403a75a04ed12707340f024dee99a27cded3258bb108a7aa`.
  - Dataset2 fresh joint cache: 200,000 train rows, 20,000 validation rows,
    100 candidates, 63 base features, K=512 for structure and source-profile.
  - Frozen V1/full-only/gap-aware shapes, three near folds, three gapped folds,
    fixed weight 0.50, CPU deterministic double replay, original
    `rtol=2e-5`/`atol=2e-6`.
  - One external safety gate after a new selection lock.
- **Non-goals**:
  - No V1-family weight rescan, seed/capacity/window scan, tolerance
    relaxation, or candidate mutation.
  - No use of external raw deltas as effect-size estimates.
  - No submission package generation in this pipeline.
  - No overwrite of historical cache, duel, selection, external, or package
    artifacts.
- **Deferred work**:
  - Online full-test V2 materialization and package generation after an
    accepted external result.
  - Source-conditioned/source-candidate joint audit unless the new candidate
    passes gapped folds but later shrinks online.
- **Verification rule**: Each stage publishes an atomic status and immutable
  report; a later stage starts only after its predecessor's exact row/shape,
  SHA-256, provenance, and policy checks pass.
- **Evidence source**: Full-predict validation, cache reports, overlay report,
  generated preregistration contracts, two-run replay reports, rolling
  manifest, standard selection report/lock, external receipt/report, logs,
  and exit codes.
- **Pass criteria**:
  - Fresh joint cache reports bind the exact source checkpoint and prove
    K=512, 200k/20k shapes, finite features, and same-process provenance.
  - Reused query-aligned sidecars are accepted only after byte-identical
    candidate/row/time/source/destination verification; otherwise stop.
  - Every listed near/gapped head passes state/loss/probability replay under
    unchanged tolerances.
  - Each near fold is non-decreasing and each gapped fold strictly improves
    MRR with non-decreasing NDCG@10 before external becomes eligible.
  - External writes at most one receipt and is interpreted only as a
    seven-gate pass/fail safety check.
- **Confidence note**: Cache rebuild and fold training consume no external
  metrics. The one irreversible read is isolated behind the newly generated
  selection lock and a fresh state directory.
- **Judgment owner**: Structural/hash validators own stage completion; the
  standard selector owns internal eligibility; the standard external
  evaluator owns the safety-gate verdict.

## Current State
- The full K=512 Dataset2 prediction is running and saving its own validated
  artifacts.
- The old 200k cache was built from a checkpoint that reports K=512, but its
  lineage predates the weighted-normalizer correction and is intentionally not
  reused as the new training-feature authority.
- The existing duel/external shell scripts hard-code old cache, model,
  selection-lock, and implementation hashes, so they cannot represent the new
  lineage.
- The existing gapped feature cache is already K=512 and contains no trained
  successor-head state; it can be reused only after its complete artifact
  report and hashes are revalidated.
- The repository is already dirty with user/previous-session work; changes
  must remain narrowly scoped and must not overwrite unrelated files.

## Plan Rewrite Notes
| Existing item | Decision | Reason |
|---|---|---|
| K=512 full predict | keep as hard prerequisite | Proves the exact source checkpoint and full service path complete safely |
| Old 200k train/validation cache | replace | New head lineage must bind a fresh K512 cache report |
| Completed K512 gapped feature cache | retain after hash validation | It contains deterministic features, not the buggy weighted head |
| Historical near manifest | diagnostic/query-alignment input only | It cannot become authoritative for newly trained V1 scores |
| Existing V1/duel/external contracts | copy and refreeze under a new result lineage | Their policy is still valid, but their artifact hashes are stale |
| Existing package phase | remove | User authorized external safety gate, not automatic submission |

## Drift Diagnosis
- **Goal drift**: Reusing the old full-origin V1 model would skip the requested
  corrected retraining.
- **Phase drift**: Opening external before the new selector would turn the
  holdout into a selection fold.
- **Validation drift**: Process exit zero alone is insufficient; every cache
  and score artifact must match the frozen shape/hash contracts.
- **Compatibility drift**: Historical immutable contracts remain preserved;
  new contracts receive new paths and hashes rather than being edited.
- **Cleanup drift**: No unrelated model, documentation, dependency, or
  historical result cleanup is included.

## Priority Rationale
- Make cache/provenance correctness the first irreversible compute boundary.
- Reuse only deterministic K512 gapped feature materialization; retrain all
  learned heads after the weighted-normalizer fix.
- Keep external last and one-shot.

## Assumptions and Open Decisions
| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| Fresh cache candidates and sidecars equal the historical frozen query contract | assumed, must prove | Allows exact reuse of short-none/prior score sidecars | Pipeline byte-comparison gate |
| Existing gapped cache was built from the promoted K512 checkpoint | confirmed by prior report, must rehash | Avoids several hours of feature-only recomputation | Cache validator |
| Gap-aware remains the selected candidate | unresolved by design | External implementation currently supports gap-aware only | Selector decides; stop safely if another candidate wins |
| Full predict finishes before the cache build | required | Prevents memory/resource overlap | Pipeline prerequisite |

## Phases

### Phase 1: Automation contract and tests
- **Purpose**: Define a fail-closed, resumable stage machine before launching
  expensive work.
- **Entry condition**: User authorization recorded.
- **Phase rules**:
  - RED before implementation.
  - Completed stages may be reused only if their validators still pass.
  - No external command exists in the stage graph before selection acceptance.
- **Todos**:
  - [x] Add tests for prerequisite failure, hash/shape failure, selection
    rejection, one-shot external ordering, and baseline-hash propagation.
    - **Surface**: pipeline tests and duel manifest construction.
    - **Proof**: focused RED/GREEN commands.
    - **Depends on**: none.
  - [x] Implement the pipeline controller and fresh-contract freezer.
    - **Surface**: scripts.
    - **Proof**: tests plus dry-run stage plan.
    - **Depends on**: tests.
- **Exit proof**: Focused tests green and shell/Python syntax checks pass.
- **Stop condition**: Any path can reach external without an accepted new
  selection lock.

### Phase 2: Fresh K512 joint cache and near overlays
- **Purpose**: Produce the authoritative 200k/20k feature lineage.
- **Entry condition**: Full predict final and validation exit codes are zero.
- **Phase rules**:
  - Fresh unique paths; no overwrite.
  - Exact parallel first-batch parity is required if multiple structure
    workers are used.
  - Reused short-none/prior sidecars require byte-identical queries.
- **Todos**:
  - [ ] Build 200k train plus 20k validation base-feature caches.
    - **Surface**: cache and cache reports.
    - **Proof**: report/artifact hashes, K512 provenance, finite arrays.
    - **Depends on**: Phase 1 and full predict.
  - [ ] Materialize fresh causal near lift and write the alignment report.
    - **Surface**: near overlay artifacts.
    - **Proof**: native materializer report and sidecar equality hashes.
    - **Depends on**: fresh cache.
- **Exit proof**: All cache/overlay artifacts validate and are frozen into the
  new training contracts.
- **Stop condition**: Candidate/row alignment differs, parity fails, OOM, disk
  reserve is insufficient, or any artifact is non-finite.

### Phase 3: All fold heads and dual-horizon selection
- **Purpose**: Recompute the complete internal decision under the corrected
  weighted normalizer.
- **Entry condition**: Phase 2 reports and contracts frozen.
- **Phase rules**:
  - CPU double replay for every frozen trainable head.
  - Original tolerances and all candidate/fold/weight/seed settings unchanged.
  - External remains closed.
- **Todos**:
  - [ ] Retrain V1, full-only, gap-aware, and required prior heads in all near
    and gapped folds.
    - **Surface**: duel scores/models/replay reports.
    - **Proof**: six fold reports and deterministic replay evidence.
    - **Depends on**: Phase 2.
  - [ ] Run the standard selector.
    - **Surface**: selection report/lock.
    - **Proof**: exact per-fold near/gapped gates and selected/rejected status.
    - **Depends on**: complete duel manifest.
- **Exit proof**: One terminal standard-selector result with no external
  receipt yet.
- **Stop condition**: Replay drift, missing baseline binding, any fold contract
  mismatch, or no eligible candidate.

### Phase 4: Full-origin V1/V2 and external safety gate
- **Purpose**: Open the one permitted near-horizon safety check.
- **Entry condition**: New selection lock selects the supported gap-aware V2.
- **Phase rules**:
  - Freeze new V1 and external execution contracts before their metrics.
  - Retrain full-origin V1 and gap-aware V2 twice on CPU.
  - Write receipt before reading external scores.
  - Treat the seven gates only as pass/fail; do not report raw delta as an
    effect-size estimate.
- **Todos**:
  - [ ] Train the corrected full-origin V1 and validate deterministic replay.
    - **Surface**: model/training report.
    - **Proof**: state/loss/probability replay.
    - **Depends on**: Phase 3 selection acceptance.
  - [ ] Materialize the selected external candidate under the fresh cache
    lineage and preflight it without opening metrics.
    - **Surface**: external scores/manifest/preflight.
    - **Proof**: hash-bound external manifest and zero prior opens.
    - **Depends on**: full-origin V1.
  - [ ] Invoke the standard external evaluator once.
    - **Surface**: external receipt/report.
    - **Proof**: one receipt and seven exact gates.
    - **Depends on**: preflight.
- **Exit proof**: External status is accepted or rejected, with no package
  generated.
- **Stop condition**: Non-gap-aware selection, any preflight/hash mismatch,
  prior receipt, replay drift, or evaluator execution error.

## Dry-Run Findings
- The old automatic scripts cannot be parameter-swapped safely because their
  frozen hashes intentionally reject the new cache and implementation.
- The current duel producer omits the standard top-level `baseline_sha256`;
  this must be fixed before the new run instead of patched after metrics.
- Fresh K512 near candidates are expected to match the historical frozen
  schedule because the sampling contract is unchanged, but the pipeline must
  prove equality before reusing short-none and prior score sidecars.
- If the selector chooses full-only, the pipeline must stop before external
  rather than silently evaluating gap-aware.

## Final Validation
- Focused RED/GREEN and related regression tests.
- Pipeline dry-run shows the exact ordered command graph and no external
  command before the selection gate.
- Remote script/source hashes equal the tested local files.
- Final status, reports, logs, and exit code are saved under one unique result
  root.

## First Execution Step
Add the failing automation tests that require baseline-hash propagation and
forbid an external stage without a newly accepted selection lock.
