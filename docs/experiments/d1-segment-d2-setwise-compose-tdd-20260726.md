# Hai TDD: Cross-source submission composition

## Target Behavior

Select `dataset1.csv` from one submission ZIP and `dataset2.csv` from another,
preserve the selected member bytes exactly, validate both CSVs, and publish a
flat two-member submission ZIP with provenance hashes.

## RED

- **Test added**:
  `tests/test_submission.py::test_compose_submission_package_selects_each_dataset_from_its_named_source`
- **Behavior asserted**: The composer selects each dataset only from its named
  source, writes the exact input bytes to disk and into the final ZIP, and
  reports the selected SHA-256 values.
- **Command**:
  `uv run --no-sync pytest tests/test_submission.py::test_compose_submission_package_selects_each_dataset_from_its_named_source -q`
- **Observed failure**:
  `ImportError: cannot import name 'compose_submission_package' from 'jgrec.submission'`
- **Failure is correct because**: The requested public composition boundary did
  not exist yet; the fixture and test collection reached the intended missing
  behavior.

## GREEN

- **Minimal implementation**: Added `compose_submission_package()` with exact
  member lookup, pre/post SHA-256 checks, existing CSV validation, flat ZIP
  publication, overwrite refusal, and a JSON composition report. Added a thin
  CLI adapter in `scripts/compose_submission_package.py`.
- **Command**:
  `uv run --no-sync pytest tests/test_submission.py::test_compose_submission_package_selects_each_dataset_from_its_named_source -q`
- **Observed pass**: `1 passed in 2.37s`.

## REFACTOR

- **Refactor done**: yes
- **Change**: Extracted streaming SHA-256 and unique ZIP-member helpers, then
  moved `Mapping` to its Python 3.12 `collections.abc` import.
- **Command after refactor**:
  `uv run --no-sync pytest tests/test_submission.py -q` and
  `uv run --no-sync ruff check src/jgrec/submission.py tests/test_submission.py scripts/compose_submission_package.py`
- **Observed result**: `7 passed in 2.12s`; `All checks passed!`.

## Next Behavior

Done. The production artifact independently passed member-name, row-count,
member-hash, and whole-ZIP hash verification.
