from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jgrec.cooccur_lift_automatic_pipeline import (
    automatic_stage_order,
    build_frozen_validation_recovery_marker,
    validate_external_transition,
    validate_full_predict_prerequisite,
    validate_stage_order,
)

EXPECTED_EXTERNAL_CANDIDATE = "cooccur_lift_gap_aware_v2"


@dataclass(frozen=True)
class PipelinePaths:
    root: Path
    run_dir: Path
    full_predict_dir: Path
    source_checkpoint: Path
    v1_checkpoint: Path
    train_csv: Path
    train_cache_prefix: Path
    validation_cache_prefix: Path
    train_cache_report: Path
    validation_cache_report: Path
    near_assets_dir: Path
    frozen_v1_contract: Path
    duel_contract: Path
    external_contract: Path
    duel_dir: Path
    selection_dir: Path
    v1_training_dir: Path
    external_dir: Path
    external_state_dir: Path
    preflight_report: Path

    @property
    def near_lift(self) -> Path:
        return self.near_assets_dir / "lift-features.npy"

    @property
    def rolling_manifest(self) -> Path:
        return self.duel_dir / "rolling-manifest.json"

    @property
    def selection_report(self) -> Path:
        return self.selection_dir / "selection-report.json"

    @property
    def selection_lock(self) -> Path:
        return self.selection_dir / "selection-lock.json"

    @property
    def v1_training_report(self) -> Path:
        return self.v1_training_dir / "training-report.json"

    @property
    def v1_model(self) -> Path:
        return (
            self.v1_training_dir
            / "cooccur-lift-bugfixed-v1-seed33100.npz"
        )

    @property
    def external_manifest(self) -> Path:
        return self.external_dir / "external-manifest.json"

    @property
    def external_materialization_report(self) -> Path:
        return (
            self.external_dir
            / "external-materialization-report.json"
        )


class PipelineController:
    def __init__(
        self,
        *,
        paths: PipelinePaths,
        poll_seconds: int,
        dry_run: bool,
    ) -> None:
        self.paths = paths
        self.poll_seconds = poll_seconds
        self.dry_run = dry_run
        self.stage_order = automatic_stage_order()
        validate_stage_order(self.stage_order)
        self.completed: list[str] = []
        self.started_at = _utc_now()
        self.status_path = paths.run_dir / "pipeline-status.json"
        self.marker_dir = paths.run_dir / "stage-markers"
        self.log_dir = paths.run_dir / "logs"

    def run(self) -> int:
        if self.dry_run:
            plan = self._dry_run_plan()
            print(
                json.dumps(
                    plan,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        self.paths.run_dir.mkdir(parents=True, exist_ok=True)
        self.marker_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._load_resume_state()
        self._write_status(status="running", current_stage=None)
        try:
            self._wait_for_full_predict()
            self._build_joint_cache()
            self._materialize_near_assets()
            self._freeze_v1_contract()
            self._freeze_duel_contract()
            self._train_duel()
            try:
                selected = self._select_dual_horizon()
            except UnsupportedSelection:
                return 0
            if not selected:
                self._write_status(
                    status="complete_internal_rejected",
                    current_stage=None,
                    terminal_reason=(
                        "No candidate passed the frozen near+gapped gate."
                    ),
                )
                return 0
            transition = validate_external_transition(
                selection_report_path=self.paths.selection_report,
                selection_lock_path=self.paths.selection_lock,
                expected_candidate_id=EXPECTED_EXTERNAL_CANDIDATE,
            )
            self._train_v1_full_origin()
            self._freeze_external_contract()
            self._materialize_external()
            self._preflight_external()
            accepted = self._open_external_gate()
            self._write_status(
                status=(
                    "complete_external_accepted"
                    if accepted
                    else "complete_external_rejected"
                ),
                current_stage=None,
                external_transition=transition,
                terminal_reason=(
                    "External safety gate completed exactly once; "
                    "no submission package was generated."
                ),
            )
            return 0
        except BaseException as exc:
            self._write_status(
                status="failed",
                current_stage=None,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

    def _load_resume_state(self) -> None:
        if not self.status_path.exists():
            return
        status = _read_json(self.status_path)
        completed = status.get("completed_stages", [])
        if not isinstance(completed, list):
            raise ValueError("pipeline completed stage list is malformed")
        expected_prefix = list(self.stage_order[: len(completed)])
        if completed != expected_prefix:
            raise ValueError("pipeline completed stages are not a prefix")
        for stage in completed:
            marker = self._load_valid_marker(stage)
            if marker.get("status") != "complete":
                raise ValueError(f"stage marker is incomplete: {stage}")
        self.completed = list(completed)
        self.started_at = str(status.get("started_at_utc", self.started_at))

    def _wait_for_full_predict(self) -> None:
        stage = "wait_full_predict"
        if self._is_completed(stage):
            validate_full_predict_prerequisite(self.paths.full_predict_dir)
            return
        self._write_status(status="running", current_stage=stage)
        while True:
            try:
                evidence = validate_full_predict_prerequisite(
                    self.paths.full_predict_dir
                )
            except ValueError as exc:
                if _full_predict_is_terminal(self.paths.full_predict_dir):
                    raise
                self._write_status(
                    status="waiting_for_full_predict",
                    current_stage=stage,
                    wait_reason=str(exc),
                )
                time.sleep(self.poll_seconds)
                continue
            self._complete_stage(stage, evidence=evidence)
            return

    def _build_joint_cache(self) -> None:
        if self._is_completed("build_joint_cache"):
            self._rebind_frozen_validation_recovery_if_needed()
        self._run_stage(
            "build_joint_cache",
            [
                self._python,
                "scripts/build_dataset2_full100_train_cache.py",
                "--checkpoint",
                self.paths.source_checkpoint,
                "--train-csv",
                self.paths.train_csv,
                "--output-prefix",
                self.paths.train_cache_prefix,
                "--report",
                self.paths.train_cache_report,
                "--validation-output-prefix",
                self.paths.validation_cache_prefix,
                "--validation-report",
                self.paths.validation_cache_report,
                "--candidate-count",
                "100",
                "--train-rows",
                "200000",
                "--validation-rows",
                "20000",
                "--train-selection",
                "recent",
                "--batch-rows",
                "4096",
                "--structure-workers",
                "8",
                "--minimum-parallel-speedup",
                "1.5",
                "--comparison-structure-workers",
                "4",
                "--minimum-comparison-speedup",
                "1.10",
                "--minimum-memory-reserve-gib",
                "8",
            ],
            outputs=(
                self.paths.train_cache_report,
                self.paths.validation_cache_report,
            ),
        )

    def _rebind_frozen_validation_recovery_if_needed(self) -> None:
        stage = "build_joint_cache"
        marker = self._load_valid_marker(stage)
        outputs = (
            self.paths.train_cache_report,
            self.paths.validation_cache_report,
        )
        try:
            self._validate_outputs(outputs, marker)
        except ValueError as output_error:
            rebound = build_frozen_validation_recovery_marker(
                marker=marker,
                train_report_path=self.paths.train_cache_report,
                validation_report_path=(
                    self.paths.validation_cache_report
                ),
                rebound_at_utc=_utc_now(),
            )
            audit_path = (
                self.marker_dir
                / "build_joint_cache.before-frozen-validation-recovery.json"
            )
            if audit_path.exists():
                raise FileExistsError(
                    "validation recovery audit marker already exists: "
                    f"{audit_path}"
                ) from output_error
            _write_json_exclusive(audit_path, marker)
            _write_json_atomic(
                self.marker_dir / f"{stage}.json",
                rebound,
            )

    def _materialize_near_assets(self) -> None:
        root = self.paths.root
        self._run_stage(
            "materialize_near_lift",
            [
                self._python,
                "scripts/materialize_dataset2_k512_near_assets.py",
                "--frozen-config",
                root
                / "docs/experiments/cooccur-lift-aux-expert-v1.frozen.json",
                "--train-csv",
                self.paths.train_csv,
                "--train-cache-prefix",
                self.paths.train_cache_prefix,
                "--train-cache-report",
                self.paths.train_cache_report,
                "--validation-cache-prefix",
                self.paths.validation_cache_prefix,
                "--validation-cache-report",
                self.paths.validation_cache_report,
                "--reference-train-cache-prefix",
                root
                / (
                    "cache/supervised_features/"
                    "dataset2_joint_recent200k_full100_seed60_20260725"
                ),
                "--reference-validation-cache-prefix",
                root
                / (
                    "cache/supervised_features/"
                    "dataset2_joint_recent200k_full100_val_seed60_20260725"
                ),
                "--reference-train-short-none",
                root
                / (
                    "result/dataset2_targeted_gnn_edges_seed60_20260725/"
                    "artifacts/short_none.train-scores.npy"
                ),
                "--reference-validation-short-none",
                root
                / (
                    "result/dataset2_targeted_gnn_edges_seed60_20260725/"
                    "artifacts/short_none.val-scores.npy"
                ),
                "--reference-prior-external",
                root
                / (
                    "result/dataset2_partial_listwise_expert_blend_20260728/"
                    "champion-probabilities.npy"
                ),
                "--output-dir",
                self.paths.near_assets_dir,
            ],
            outputs=(
                self.paths.near_assets_dir / "near-assets-report.json",
                self.paths.near_lift,
            ),
        )

    def _freeze_v1_contract(self) -> None:
        root = self.paths.root
        self._run_stage(
            "freeze_v1_contract",
            [
                self._python,
                "scripts/freeze_dataset2_k512_successor_contracts.py",
                "bugfixed-v1",
                "--base-contract",
                root
                / (
                    "docs/experiments/"
                    "cooccur-lift-aux-expert-v1-bugfixed-refit."
                    "preregistered.json"
                ),
                "--frozen-config",
                root
                / "docs/experiments/cooccur-lift-aux-expert-v1.frozen.json",
                "--selection-lock",
                root
                / (
                    "result/dataset2_cooccur_lift_aux_expert_v1_"
                    "20260728_compact_retry2/rolling-selection/"
                    "selection-lock.json"
                ),
                "--source-checkpoint",
                self.paths.source_checkpoint,
                "--train-cache-report",
                self.paths.train_cache_report,
                "--validation-cache-report",
                self.paths.validation_cache_report,
                "--train-lift-features",
                self.paths.near_lift,
                "--train-short-none",
                root
                / (
                    "result/dataset2_targeted_gnn_edges_seed60_20260725/"
                    "artifacts/short_none.train-scores.npy"
                ),
                "--fusion-source",
                root / "src/jgrec/rankers/hybrid/fusion.py",
                "--output",
                self.paths.frozen_v1_contract,
            ],
            outputs=(self.paths.frozen_v1_contract,),
        )

    def _freeze_duel_contract(self) -> None:
        root = self.paths.root
        self._run_stage(
            "freeze_duel_contract",
            [
                self._python,
                "scripts/freeze_dataset2_k512_successor_contracts.py",
                "duel",
                "--base-contract",
                root
                / (
                    "docs/experiments/"
                    "cooccur-lift-successor-v2-duel.execution."
                    "preregistered.json"
                ),
                "--validation-plan",
                root
                / (
                    "docs/experiments/"
                    "cooccur-lift-successor-v2-duel.validation-plan.json"
                ),
                "--plan-lock",
                self._plan_lock,
                "--bugfixed-v1-contract",
                self.paths.frozen_v1_contract,
                "--near-v1-manifest",
                self._historical_near_manifest,
                "--near-cache-report",
                self.paths.train_cache_report,
                "--gapped-cache-report",
                self._gapped_cache_report,
                "--runner",
                root
                / "scripts/train_dataset2_cooccur_lift_successor_v2_duel.py",
                "--execution-module",
                root / "src/jgrec/cooccur_lift_successor_execution.py",
                "--fusion-source",
                root / "src/jgrec/rankers/hybrid/fusion.py",
                "--pipeline-script",
                Path(__file__).resolve(),
                "--output",
                self.paths.duel_contract,
            ],
            outputs=(self.paths.duel_contract,),
        )

    def _train_duel(self) -> None:
        root = self.paths.root
        self._run_stage(
            "train_duel",
            [
                self._python,
                "scripts/train_dataset2_cooccur_lift_successor_v2_duel.py",
                "--v1-checkpoint",
                self.paths.v1_checkpoint,
                "--validation-plan",
                root
                / (
                    "docs/experiments/"
                    "cooccur-lift-successor-v2-duel.validation-plan.json"
                ),
                "--plan-lock",
                self._plan_lock,
                "--execution-contract",
                self.paths.duel_contract,
                "--bugfixed-v1-contract",
                self.paths.frozen_v1_contract,
                "--full-only-config",
                root
                / (
                    "docs/experiments/"
                    "cooccur-lift-full-only-v2.preregistered.json"
                ),
                "--gap-aware-config",
                root
                / (
                    "docs/experiments/"
                    "cooccur-lift-gap-aware-v2.preregistered.json"
                ),
                "--near-cache-prefix",
                self.paths.train_cache_prefix,
                "--near-cache-report",
                self.paths.train_cache_report,
                "--near-short-none",
                root
                / (
                    "result/dataset2_targeted_gnn_edges_seed60_20260725/"
                    "artifacts/short_none.train-scores.npy"
                ),
                "--near-lift",
                self.paths.near_lift,
                "--near-v1-manifest",
                self._historical_near_manifest,
                "--gapped-cache-dir",
                self._gapped_cache_dir,
                "--output-dir",
                self.paths.duel_dir,
            ],
            outputs=(
                self.paths.rolling_manifest,
                self.paths.duel_dir / "training-report.json",
            ),
        )

    def _select_dual_horizon(self) -> bool:
        return_code = self._run_stage(
            "select_dual_horizon",
            [
                self._python,
                "scripts/select_standard_rolling_candidate.py",
                "--manifest",
                self.paths.rolling_manifest,
                "--plan-lock",
                self._plan_lock,
                "--output-dir",
                self.paths.selection_dir,
            ],
            outputs=(self.paths.selection_report,),
            accepted_return_codes=(0, 2),
        )
        report = _read_json(self.paths.selection_report)
        if return_code == 2 or report.get("status") != "selected":
            return False
        if not self.paths.selection_lock.is_file():
            raise ValueError("selected candidate lacks a selection lock")
        selected = report.get("selected_candidate_id")
        if selected != EXPECTED_EXTERNAL_CANDIDATE:
            self._write_status(
                status="complete_unsupported_candidate",
                current_stage=None,
                terminal_reason=(
                    f"Selected {selected}; the frozen external "
                    "implementation supports only "
                    f"{EXPECTED_EXTERNAL_CANDIDATE}. External was not opened."
                ),
            )
            raise UnsupportedSelection(selected)
        return True

    def _train_v1_full_origin(self) -> None:
        root = self.paths.root
        self._run_stage(
            "train_v1_full_origin",
            [
                self._python,
                "scripts/train_dataset2_cooccur_lift_bugfixed_v1.py",
                "--candidate-contract",
                self.paths.frozen_v1_contract,
                "--frozen-config",
                root
                / "docs/experiments/cooccur-lift-aux-expert-v1.frozen.json",
                "--selection-lock",
                root
                / (
                    "result/dataset2_cooccur_lift_aux_expert_v1_"
                    "20260728_compact_retry2/rolling-selection/"
                    "selection-lock.json"
                ),
                "--source-checkpoint",
                self.paths.source_checkpoint,
                "--train-cache-prefix",
                self.paths.train_cache_prefix,
                "--train-cache-report",
                self.paths.train_cache_report,
                "--validation-cache-report",
                self.paths.validation_cache_report,
                "--train-lift-features",
                self.paths.near_lift,
                "--train-short-none",
                root
                / (
                    "result/dataset2_targeted_gnn_edges_seed60_20260725/"
                    "artifacts/short_none.train-scores.npy"
                ),
                "--output-dir",
                self.paths.v1_training_dir,
            ],
            outputs=(
                self.paths.v1_training_report,
                self.paths.v1_model,
            ),
        )

    def _freeze_external_contract(self) -> None:
        root = self.paths.root
        self._run_stage(
            "freeze_external_contract",
            [
                self._python,
                "scripts/freeze_dataset2_k512_successor_contracts.py",
                "external",
                "--base-contract",
                root
                / (
                    "docs/experiments/"
                    "cooccur-lift-gap-aware-v2.external-execution."
                    "preregistered.json"
                ),
                "--candidate-config",
                root
                / (
                    "docs/experiments/"
                    "cooccur-lift-gap-aware-v2.preregistered.json"
                ),
                "--selection-lock",
                self.paths.selection_lock,
                "--bugfixed-v1-contract",
                self.paths.frozen_v1_contract,
                "--bugfixed-v1-training-report",
                self.paths.v1_training_report,
                "--bugfixed-v1-model",
                self.paths.v1_model,
                "--source-checkpoint",
                self.paths.source_checkpoint,
                "--train-cache-report",
                self.paths.train_cache_report,
                "--validation-cache-report",
                self.paths.validation_cache_report,
                "--train-lift-features",
                self.paths.near_lift,
                "--train-short-none",
                self._train_short_none,
                "--validation-short-none",
                self._validation_short_none,
                "--train-csv",
                self.paths.train_csv,
                "--prior-external-probabilities",
                self._prior_external,
                "--materializer-script",
                root
                / (
                    "scripts/"
                    "materialize_dataset2_cooccur_lift_successor_v2_"
                    "external.py"
                ),
                "--external-module",
                root / "src/jgrec/cooccur_lift_successor_external.py",
                "--output",
                self.paths.external_contract,
            ],
            outputs=(self.paths.external_contract,),
        )

    def _materialize_external(self) -> None:
        root = self.paths.root
        self._run_stage(
            "materialize_external",
            [
                self._python,
                (
                    "scripts/"
                    "materialize_dataset2_cooccur_lift_successor_v2_"
                    "external.py"
                ),
                "--execution-contract",
                self.paths.external_contract,
                "--candidate-config",
                root
                / (
                    "docs/experiments/"
                    "cooccur-lift-gap-aware-v2.preregistered.json"
                ),
                "--selection-lock",
                self.paths.selection_lock,
                "--bugfixed-v1-contract",
                self.paths.frozen_v1_contract,
                "--bugfixed-v1-training-report",
                self.paths.v1_training_report,
                "--bugfixed-v1-model",
                self.paths.v1_model,
                "--source-checkpoint",
                self.paths.source_checkpoint,
                "--train-cache-prefix",
                self.paths.train_cache_prefix,
                "--train-cache-report",
                self.paths.train_cache_report,
                "--validation-cache-prefix",
                self.paths.validation_cache_prefix,
                "--validation-cache-report",
                self.paths.validation_cache_report,
                "--train-lift-features",
                self.paths.near_lift,
                "--train-short-none",
                self._train_short_none,
                "--validation-short-none",
                self._validation_short_none,
                "--train-csv",
                self.paths.train_csv,
                "--prior-external-probabilities",
                self._prior_external,
                "--output-dir",
                self.paths.external_dir,
            ],
            outputs=(
                self.paths.external_manifest,
                self.paths.external_materialization_report,
            ),
        )

    def _preflight_external(self) -> None:
        root = self.paths.root
        self._run_stage(
            "preflight_external",
            [
                self._python,
                (
                    "scripts/"
                    "preflight_dataset2_cooccur_lift_successor_v2_external.py"
                ),
                "--candidate-config",
                root
                / (
                    "docs/experiments/"
                    "cooccur-lift-gap-aware-v2.preregistered.json"
                ),
                "--selection-lock",
                self.paths.selection_lock,
                "--materialization-report",
                self.paths.external_materialization_report,
                "--manifest",
                self.paths.external_manifest,
                "--state-dir",
                self.paths.external_state_dir,
                "--output",
                self.paths.preflight_report,
            ],
            outputs=(self.paths.preflight_report,),
        )

    def _open_external_gate(self) -> bool:
        return_code = self._run_stage(
            "open_external_gate",
            [
                self._python,
                "scripts/evaluate_standard_external_gate.py",
                "--manifest",
                self.paths.external_manifest,
                "--selection-lock",
                self.paths.selection_lock,
                "--state-dir",
                self.paths.external_state_dir,
            ],
            outputs=(
                self.paths.external_state_dir
                / "external-open-receipt.json",
                self.paths.external_state_dir
                / "external-evaluation-report.json",
            ),
            accepted_return_codes=(0, 2),
        )
        report = _read_json(
            self.paths.external_state_dir
            / "external-evaluation-report.json"
        )
        if return_code == 0 and report.get("status") != "accepted":
            raise ValueError("external evaluator return/status differs")
        if return_code == 2 and report.get("status") != "rejected":
            raise ValueError("external evaluator rejection/status differs")
        return return_code == 0

    def _run_stage(
        self,
        stage: str,
        command: Sequence[str | Path],
        *,
        outputs: Sequence[Path],
        accepted_return_codes: tuple[int, ...] = (0,),
    ) -> int:
        command_text = [str(value) for value in command]
        if self._is_completed(stage):
            marker = self._load_valid_marker(stage)
            self._validate_outputs(outputs, marker)
            return int(marker["return_code"])
        self._require_next(stage)
        for output in outputs:
            if output.exists():
                raise FileExistsError(
                    f"unmarked output blocks safe resume: {output}"
                )
        self._write_status(status="running", current_stage=stage)
        log_path = self.log_dir / f"{stage}.log"
        if log_path.exists():
            raise FileExistsError(
                f"unmarked stage log blocks safe resume: {log_path}"
            )
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "OMP_NUM_THREADS": "24",
                "MKL_NUM_THREADS": "24",
                "OPENBLAS_NUM_THREADS": "24",
                "NUMEXPR_NUM_THREADS": "24",
                "PYTHONUNBUFFERED": "1",
            }
        )
        started = time.time()
        with log_path.open("x", encoding="utf-8") as log:
            log.write(
                json.dumps(
                    {
                        "stage": stage,
                        "command": command_text,
                        "started_at_utc": _utc_now(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            log.flush()
            process = subprocess.run(
                command_text,
                cwd=self.paths.root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if process.returncode not in accepted_return_codes:
            tail = _tail(log_path)
            raise RuntimeError(
                f"{stage} exited {process.returncode}; log tail:\n{tail}"
            )
        missing = [str(path) for path in outputs if not path.is_file()]
        if missing:
            raise RuntimeError(
                f"{stage} did not create required outputs: {missing}"
            )
        evidence = {
            "return_code": process.returncode,
            "elapsed_seconds": time.time() - started,
            "command": command_text,
            "command_sha256": _sha256_text(command_text),
            "log": str(log_path.resolve()),
        }
        self._complete_stage(
            stage,
            evidence=evidence,
            outputs=outputs,
            return_code=process.returncode,
        )
        return process.returncode

    def _complete_stage(
        self,
        stage: str,
        *,
        evidence: dict[str, Any],
        outputs: Sequence[Path] = (),
        return_code: int = 0,
    ) -> None:
        self._require_next(stage)
        marker = {
            "schema_version": 1,
            "status": "complete",
            "stage": stage,
            "completed_at_utc": _utc_now(),
            "return_code": return_code,
            "evidence": evidence,
            "outputs": [_descriptor(path) for path in outputs],
        }
        marker_path = self.marker_dir / f"{stage}.json"
        if marker_path.exists():
            raise FileExistsError(f"stage marker already exists: {marker_path}")
        _write_json_exclusive(marker_path, marker)
        self.completed.append(stage)
        self._write_status(status="running", current_stage=None)

    def _load_valid_marker(self, stage: str) -> dict[str, Any]:
        marker_path = self.marker_dir / f"{stage}.json"
        marker = _read_json(marker_path)
        if (
            marker.get("stage") != stage
            or marker.get("status") != "complete"
        ):
            raise ValueError(f"invalid stage marker: {marker_path}")
        return marker

    def _validate_outputs(
        self,
        outputs: Sequence[Path],
        marker: dict[str, Any],
    ) -> None:
        descriptors = marker.get("outputs")
        if not isinstance(descriptors, list):
            raise ValueError("stage output descriptors are malformed")
        frozen = {
            str(Path(str(item["path"])).resolve()): item
            for item in descriptors
            if isinstance(item, dict) and "path" in item
        }
        for path in outputs:
            descriptor = frozen.get(str(path.resolve()))
            if (
                descriptor is None
                or not path.is_file()
                or path.stat().st_size != descriptor.get("bytes")
                or _sha256(path) != descriptor.get("sha256")
            ):
                raise ValueError(f"completed stage output differs: {path}")

    def _is_completed(self, stage: str) -> bool:
        return stage in self.completed

    def _require_next(self, stage: str) -> None:
        expected = self.stage_order[len(self.completed)]
        if stage != expected:
            raise ValueError(
                f"stage order differs: expected={expected} actual={stage}"
            )

    def _write_status(
        self,
        *,
        status: str,
        current_stage: str | None,
        **extra: Any,
    ) -> None:
        payload = {
            "schema_version": 1,
            "protocol": "dataset2_k512_successor_v2_automatic_pipeline_v1",
            "status": status,
            "started_at_utc": self.started_at,
            "updated_at_utc": _utc_now(),
            "pid": os.getpid(),
            "root": str(self.paths.root),
            "run_dir": str(self.paths.run_dir),
            "current_stage": current_stage,
            "completed_stages": list(self.completed),
            "stage_order": list(self.stage_order),
            "external_decision_role": "safety_gate_only",
            "external_effect_size_estimation_authorized": False,
            "tolerance_relaxation_authorized": False,
            "submission_package_authorized": False,
            **extra,
        }
        _write_json_atomic(self.status_path, payload)

    def _dry_run_plan(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "protocol": "dataset2_k512_successor_v2_automatic_dry_run_v1",
            "stage_order": list(self.stage_order),
            "full_predict_prerequisite": str(
                self.paths.full_predict_dir
            ),
            "fresh_cache": {
                "train_prefix": str(self.paths.train_cache_prefix),
                "validation_prefix": str(
                    self.paths.validation_cache_prefix
                ),
                "train_rows": 200_000,
                "validation_rows": 20_000,
                "candidate_count": 100,
                "structure_predict_neighbor_limit": 512,
                "source_profile_predict_history_limit": 512,
                "requested_structure_workers": 8,
                "comparison_structure_workers": 4,
                "exact_first_batch_parity_required": True,
                "minimum_parallel_speedup": 1.5,
                "minimum_incremental_speedup": 1.10,
                "minimum_memory_reserve_gib": 8,
            },
            "selection": {
                "near_and_gapped": True,
                "external_before_selection": False,
                "unsupported_selection_action": "stop_without_external",
            },
            "external": {
                "maximum_opens": 1,
                "role": "safety_gate_only",
                "effect_size_estimation_authorized": False,
                "package_generation": False,
            },
        }

    @property
    def _python(self) -> str:
        return sys.executable

    @property
    def _plan_lock(self) -> Path:
        return (
            self.paths.root
            / (
                "result/dataset2_cooccur_lift_successor_v2_duel_20260729/"
                "plan-v2/validation-plan-lock.json"
            )
        )

    @property
    def _historical_near_manifest(self) -> Path:
        return (
            self.paths.root
            / (
                "result/dataset2_cooccur_lift_aux_expert_v1_"
                "20260728_compact_retry2/artifacts/rolling-manifest.json"
            )
        )

    @property
    def _gapped_cache_dir(self) -> Path:
        return (
            self.paths.root
            / (
                "result/dataset2_cooccur_lift_successor_v2_duel_20260729/"
                "gapped-cache-v3-parallel4-dual"
            )
        )

    @property
    def _gapped_cache_report(self) -> Path:
        return self._gapped_cache_dir / "cache-report.json"

    @property
    def _train_short_none(self) -> Path:
        return (
            self.paths.root
            / (
                "result/dataset2_targeted_gnn_edges_seed60_20260725/"
                "artifacts/short_none.train-scores.npy"
            )
        )

    @property
    def _validation_short_none(self) -> Path:
        return (
            self.paths.root
            / (
                "result/dataset2_targeted_gnn_edges_seed60_20260725/"
                "artifacts/short_none.val-scores.npy"
            )
        )

    @property
    def _prior_external(self) -> Path:
        return (
            self.paths.root
            / (
                "result/dataset2_partial_listwise_expert_blend_20260728/"
                "champion-probabilities.npy"
            )
        )


class UnsupportedSelection(RuntimeError):
    pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete fail-closed Dataset2 K512 successor V2 "
            "pipeline through one external safety gate."
        )
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "result/"
            "dataset2_k512_cooccur_lift_successor_v2_rerun_20260729"
        ),
    )
    parser.add_argument(
        "--train-cache-prefix",
        type=Path,
        default=Path(
            "cache/supervised_features/"
            "dataset2_k512_joint_recent200k_full100_seed60_20260729"
        ),
    )
    parser.add_argument(
        "--validation-cache-prefix",
        type=Path,
        default=Path(
            "cache/supervised_features/"
            "dataset2_k512_joint_recent200k_full100_val_seed60_20260729"
        ),
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("poll seconds must be positive")

    root = args.root.resolve()
    run_dir = _under_root(root, args.run_dir)
    paths = PipelinePaths(
        root=root,
        run_dir=run_dir,
        full_predict_dir=(
            root / "result/dataset2_k512_full_predict_20260729"
        ),
        source_checkpoint=(
            root
            / (
                "checkpoints/"
                "d1_time_ramp_g050_d2_gnn_short_none_e50_edges40000_"
                "setwise_w080_seed60_20260727.pkl"
            )
        ),
        v1_checkpoint=(
            root
            / (
                "checkpoints/"
                "d1_time_ramp_g050_d2_cooccur_lift_aux_w050_20260729.pkl"
            )
        ),
        train_csv=root / "data/dataset2/train.csv",
        train_cache_prefix=_under_root(root, args.train_cache_prefix),
        validation_cache_prefix=_under_root(
            root, args.validation_cache_prefix
        ),
        train_cache_report=run_dir / "cache/train-cache-report.json",
        validation_cache_report=(
            run_dir / "cache/validation-cache-report.json"
        ),
        near_assets_dir=run_dir / "near-assets",
        frozen_v1_contract=run_dir / "contracts/bugfixed-v1.json",
        duel_contract=run_dir / "contracts/duel-execution.json",
        external_contract=(
            run_dir / "contracts/external-execution.json"
        ),
        duel_dir=run_dir / "duel",
        selection_dir=run_dir / "selection",
        v1_training_dir=run_dir / "v1-full-origin",
        external_dir=run_dir / "external-materialization",
        external_state_dir=run_dir / "external-state",
        preflight_report=run_dir / "external-preflight.json",
    )
    controller = PipelineController(
        paths=paths,
        poll_seconds=args.poll_seconds,
        dry_run=args.dry_run,
    )
    return controller.run()


def _under_root(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _full_predict_is_terminal(result_dir: Path) -> bool:
    final_path = result_dir / "final-exit-code.txt"
    if not final_path.is_file():
        return False
    try:
        return int(final_path.read_text(encoding="utf-8").strip()) != 0
    except ValueError:
        return True


def _descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(values: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(values), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _tail(path: Path, limit: int = 16_384) -> str:
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - limit))
        return handle.read().decode("utf-8", errors="replace")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
