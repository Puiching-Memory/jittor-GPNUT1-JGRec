# Dataset2 K=512 Gap-Aware V2 Online Package Result

## Status

Complete. The selected `cooccur_lift_gap_aware_v2` package was generated on
top of the current K=512 bugfixed-V1 baseline, validated on the server,
downloaded, and independently rehashed locally. It was not uploaded
automatically.

Local package:

`result/dataset2_k512_cooccur_lift_successor_v2_rerun_20260729/submission/result.zip`

SHA-256:

`a7f1a6522b977a70584aca3e4388f27f7e448ddaa792f4180edfe729448913b3`

## Frozen Lineage

- Online-package contract SHA-256:
  `f7b2e1713dc4237199d7632b72c2a95d944231971e2cb8a309581ac66e60e7eb`.
- Selection-lock SHA-256:
  `1343aceaf36d4718d0f52e2f927a2a1daf0b256ec34168ca2751001b0188e2b7`.
- Current K=512 V1 model SHA-256:
  `3a592856c828fbaca2da08d8ee155e2eca09232e3ba4737bbca110e841e2f8eb`.
- Gap-aware-v2 model SHA-256:
  `5567a4f26f8a06bcf74ed8c3f1ac83fcb31c31e9a18968b7503706d2d873a158`.
- The historical V1 model/package was not substituted.
- Both blend weights remained exactly `0.5`; no feature or weight rescan ran.
- External remained a seven-gate safety check; no external effect size was
  consumed by materialization or packaging.

## Current V1 Baseline

- V1 package SHA-256:
  `a6cbfeade42b4bcf437ad6faa9e02b5f8999d0739944181e2bf44cb161e9f209`.
- V1 Dataset1 member SHA-256:
  `81285eddfba612b9c05e96075be29a15718d8a94839d143e3083bf5353644369`.
- V1 Dataset2 member SHA-256:
  `a15508ec1a3f2f3fda4a945595e626a544390f4afe119f2f768009b527810304`.
- Baseline lock SHA-256:
  `afa0eb0a44f369ea1fc10a475f27dcd928135bf8f000cba7a49231aed2377385`.

## Gap-Aware Test Materialization

- Shape: `[153420, 100]`.
- Probability SHA-256:
  `1403714096c0eff9d721acc908fee66f1025d4bad455f036608b35cfc8e895c9`.
- Maximum row-sum error:
  `9.992007221626409e-16`.
- Supported rows: `92311`.
- Collapsed rows: `61109`.
- Collapsed fraction: `0.39831182375179247`.
- Availability rule: `min(query_time, train_history_end)`.
- Materialization elapsed time: `1086.5811202526093` seconds.

## Final Package

Formula:

`0.5 * current_K512_bugfixed_V1 + 0.5 * gap_aware_v2`

| Member | Rows | Columns | SHA-256 |
|---|---:|---:|---|
| `dataset1.csv` | 61051 | 100 | `81285eddfba612b9c05e96075be29a15718d8a94839d143e3083bf5353644369` |
| `dataset2.csv` | 153420 | 100 | `f97fd795447d9e998527156250d841e066f9a6fd61dc9d55c087342bf041ebc9` |

- ZIP bytes: `59165251`.
- Final validation status: `passed`.
- Package-pipeline status: `complete`.
- Package-pipeline exit code: `0`.
- Local ZIP/member hashes exactly equal the remote validation report.

## Operational Note

The first foreground SSH attempt exited after `6711` native rows when the
local wait channel timed out. Its partial directory and original preflight
receipt were moved without deletion to:

`result/dataset2_k512_cooccur_lift_successor_v2_rerun_20260729/recovery-archive/package-ssh-timeout-attempt1`

The successful run used a detached server-side session with an independent
log, PID, status file, and exit marker. No model, weight, tolerance, or data
contract changed during recovery.

## Verification

- Focused package-contract tests: `2 passed`.
- Related package/external/submission regressions: `27 passed`.
- Ruff: passed.
- Remote final-package validation: passed.
- Independent local ZIP and member hash validation: passed.
