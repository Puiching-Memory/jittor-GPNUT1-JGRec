# Hai TDD: Bug-Fixed v1 Submission Provenance

## Target Behavior

A Dataset2 test materialization must be rejected before scoring when its
auxiliary model, source checkpoint, frozen configuration, selection lock, or
training evidence differs from the preregistered bug-fixed v1 refit. The
historical accepted external report must not authorize a newly trained model.

## RED

- **Test added**:
  `tests/test_cooccur_lift_bugfixed_v1.py`, especially
  `test_materialization_rejects_a_model_swap_before_scoring`.
- **Behavior asserted**: a training report bound to model A cannot be used to
  score model B; deterministic replay must retain the original
  `rtol=2e-5`, `atol=2e-6` gate.
- **Command**:
  `uv run --no-sync pytest -q tests/test_cooccur_lift_bugfixed_v1.py`
- **Observed failure**:
  test collection failed with
  `ModuleNotFoundError: No module named 'jgrec.cooccur_lift_bugfixed_v1'`.
- **Failure is correct because**: the repository had no model-to-training
  provenance validator, which is precisely why the old scorer could accept an
  arbitrary `--auxiliary-model` after checking unrelated historical external
  evidence.

## GREEN

- **Minimal implementation**:
  added `jgrec.cooccur_lift_bugfixed_v1` with pure validators that bind the
  candidate ID, model SHA-256, source checkpoint SHA-256, frozen config,
  selection lock, fixed weight, seed, and unchanged replay tolerances.
- **Command**:
  `uv run --no-sync pytest -q tests/test_cooccur_lift_bugfixed_v1.py`
- **Observed pass**: `7 passed`.

## REFACTOR

- **Refactor done**: yes.
- **Change**:
  reused the pure evidence validator in both test materialization and package
  generation; preserved the historical accepted-external mode while making it
  mutually exclusive with the bug-fixed training-evidence mode. Added one
  dedicated full-origin trainer that trains twice and publishes no model when
  the deterministic replay gate fails.
- **Command after refactor**:
  `uv run --no-sync pytest -q tests/test_cooccur_lift_bugfixed_v1.py
  tests/test_cooccur_lift_external.py tests/test_cooccur_lift.py` and remote
  `.venv/bin/ruff check` on the changed Python files.
- **Observed result**: `28 passed`; remote Ruff reported
  `All checks passed!`.

## Next Behavior

### Deterministic Training Device

- **RED test added**: a training report that claims `training_device=cuda`
  must be rejected when the refrozen candidate requires CPU training.
- **RED command**:
  `uv run --no-sync pytest -q tests/test_cooccur_lift_bugfixed_v1.py`
- **Observed RED**:
  `test_training_report_rejects_unbound_or_drifted_evidence` did not raise for
  the CUDA mutation (`1 failed, 7 passed`).
- **GREEN implementation**:
  bound `training_device=cpu` and `test_scoring_device=cuda` in the candidate,
  training report validator, trainer, scorer, and packager.
- **GREEN command**:
  `uv run --no-sync pytest -q tests/test_cooccur_lift_bugfixed_v1.py
  tests/test_cooccur_lift_external.py tests/test_cooccur_lift.py`
- **Observed GREEN**: `29 passed`; remote focused test reported `8 passed`
  and Ruff reported `All checks passed!`.
- **Preflight evidence**:
  real-feature 20,000-row dual training produced maximum probability errors
  `0.05319197303453227` on default CUDA and `0.048016706738819914` with
  deterministic cuBLAS workspace; CPU produced exact equality at 2,000 and
  20,000 rows (`max_abs_error=0.0`).

The full-data CPU refit completed with exact dual-run replay
(`max_abs_error=0.0`), one SHA-bound 153,420-row CUDA prediction, and one
structurally valid two-member submission ZIP. The user manually submitted the
audited ZIP (SHA-256
`b90960c3427f70e2745bcb381289fca4625c208ebfaefb43ecdbc7a7387ff2f0`)
and reported `1.3577315048069973`. This passes the external safety gate only;
the score is not used as an effect-size estimate.
