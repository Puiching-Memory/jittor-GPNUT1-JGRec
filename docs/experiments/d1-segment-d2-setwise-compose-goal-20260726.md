# Goal Document: Dataset1 Segment Policy + Dataset2 Setwise Champion Package

## Go / No-Go

- **Judgment**: Go
- **Reason**: Both source submission ZIPs already exist locally with candidate
  reports and independently validated row counts and hashes. The requested
  work is a deterministic artifact composition with no training or model
  changes.

## Target Outcome

Create a new submission directory whose `result.zip` contains exactly:

- `dataset1.csv` from the accepted Dataset1 segment-policy package;
- `dataset2.csv` from the online `1.3530197200911278` Dataset2 Setwise champion
  package.

The package must be structurally valid, preserve both selected CSV members
byte-for-byte, and include an auditable composition report.

## Goal Definition

- **Type**: operational / quality / delivery
- **Boundary**: Read the two existing ZIPs, select one named CSV member from
  each, validate both CSVs, write one new flat ZIP, and record source/output
  hashes.
- **Non-goals**:
  - Retraining, inference, checkpoint composition, or score tuning.
  - Changing Dataset1 or Dataset2 probability values.
  - Using the rejected Dataset2 multi-interest confidence-gate output.
- **Deferred work**:
  - Leaderboard submission and interpretation of its external score.
- **Verification rule**: Hash each selected source member before composition,
  hash the materialized CSV afterward, validate expected rows and 100 columns,
  inspect the final ZIP member list, and verify final ZIP members have the same
  hashes as their selected sources.
- **Evidence source**: Focused RED/GREEN tests, submission validator output,
  ZIP member inventory, SHA-256 report, and final artifact.
- **Pass criteria**:
  - Dataset1 source/member/output hash is
    `94ac214a6076c5cb87e0229d69dc93aefcff3e7595cd8980b3b6f9e96b9328ea`.
  - Dataset2 source/member/output hash is
    `d7e0d574789aa6b507e592fffb5da054839aba663f63ed58134ea2d951b1ae1e`.
  - Dataset1 has 61,051 rows; Dataset2 has 153,420 rows; each row has 100
    probabilities in `[0, 1]`.
  - Final ZIP contains exactly `dataset1.csv` and `dataset2.csv` at its root.
- **Confidence note**: Byte hashes prove exact model-output preservation;
  structural validation proves submit readiness but cannot predict the
  leaderboard score.
- **Judgment owner**: Automated hash/CSV/ZIP checks declare delivery complete;
  the competition leaderboard owns model quality.

## Current State

- Dataset1 segment-policy source:
  `result/d1_segment_policy_d2_champion_seed60_20260723/result.zip`.
- Dataset2 Setwise champion source:
  `result/d1_champion_d2_setwise_w080_seed60_20260725/result.zip`.
- Both source directories contain reports, but their CSV files are stored only
  inside the ZIP archives.
- The worktree has unrelated user changes; this task must not modify or clean
  them.

## Priority Rationale

- Prove exact member selection and no-overwrite behavior before handling the
  large real artifacts.
- Materialize and validate the two selected CSVs before publishing the final
  ZIP.

## Assumptions and Open Decisions

| Item | Status | Impact | Owner / Next step |
|---|---|---|---|
| Each source ZIP contains flat `dataset1.csv` and `dataset2.csv` members | verified | Required for deterministic selection | Both inventories checked |
| Candidate-report CSV hashes describe the corresponding ZIP members | verified | Required for provenance | Both selected hashes matched |
| Output name `d1_segment_policy_d2_setwise_champion_20260726` is unused | verified | Prevents overwrite | Checked before materialization |

## Phases

### Phase 1: Freeze and test composition behavior

- **Purpose**: Define a safe, reusable ZIP-composition boundary.
- **Entry condition**: Goal document is present and source artifacts are
  readable.
- **Phase rules**:
  - RED before implementation.
  - Select members by exact filename.
  - Reject missing/duplicate members and existing output targets.
- **Todos**:
  - [x] Add a focused test for composing Dataset1 from one ZIP and Dataset2
    from another.
    - **Surface**: submission tests and composition API
    - **Proof**: RED failure because the API is missing
    - **Depends on**: none
  - [x] Implement the smallest composition API and CLI.
    - **Surface**: `jgrec.submission` and one script
    - **Proof**: Focused test passes and output members match input bytes
    - **Depends on**: failing test
- **Exit proof**: Focused GREEN test and Ruff pass.
- **Stop condition**: Existing public APIs cannot represent exact-byte
  composition without changing unrelated behavior.

### Phase 2: Compose and validate production artifact

- **Purpose**: Produce the requested submit-ready package.
- **Entry condition**: Phase 1 is green and source member hashes match their
  reports.
- **Phase rules**:
  - Never overwrite an existing output.
  - Do not read or use either source ZIP's non-selected dataset member.
  - Publish a composition report only after all checks pass.
- **Todos**:
  - [x] Extract the two selected CSV members into a new output directory.
    - **Surface**: result artifact
    - **Proof**: source/member/materialized hashes agree
    - **Depends on**: source preflight
  - [x] Validate CSVs and write the flat final ZIP.
    - **Surface**: result artifact and report
    - **Proof**: expected rows, 100 columns, valid probability range, exact ZIP
      inventory, and final member hashes
    - **Depends on**: selected CSV materialization
- **Exit proof**: Validated `result.zip` and `composition-report.json`.
- **Stop condition**: Any member name, row count, hash, probability, or ZIP
  inventory mismatch.

## Dry-Run Findings

- Direct filesystem copying is impossible because the selected CSVs are only
  stored inside their source ZIPs.
- Reusing `write_zip()` after extraction preserves CSV bytes because it does
  not rewrite CSV contents.
- A generic exact-member composition helper avoids one-off shell extraction
  and makes source selection independently testable.

## Final Validation

- Focused test passed, the complete submission test module reported `7 passed`,
  and Ruff reported `All checks passed!`.
- Composition CLI exited zero and wrote
  `result/d1_segment_policy_d2_setwise_champion_20260726/result.zip`.
- Dataset1 has 61,051 rows and SHA-256
  `94ac214a6076c5cb87e0229d69dc93aefcff3e7595cd8980b3b6f9e96b9328ea`.
- Dataset2 has 153,420 rows and SHA-256
  `d7e0d574789aa6b507e592fffb5da054839aba663f63ed58134ea2d951b1ae1e`.
- Final ZIP has exactly two root members and SHA-256
  `669b64fe2eb784d7bedadc553dfba7cf487aebc39082130c3b966e27b4849da5`.

## First Execution Step

Add a failing test that imports the missing composition API and proves it
selects Dataset1 and Dataset2 from different source ZIPs without changing
their bytes.
