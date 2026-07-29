from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.cooccur_lift_bugfixed_v1 import (
    validate_bugfixed_v1_materialization_inputs,
)
from jgrec.cooccur_lift_successor_execution import (
    build_deterministic_replay_report,
)
from jgrec.cooccur_lift_successor_external import (
    build_standard_external_manifest,
    full_origin_copy_weights,
    short_window_support_from_availability,
    validate_successor_external_setup,
)
from jgrec.core.io import read_interactions
from jgrec.rankers.hybrid.cooccur_lift import CooccurLiftAugmentedView
from jgrec.rankers.hybrid.cooccur_lift_native import (
    materialize_compact_cooccur_lift,
)
from jgrec.rankers.hybrid.cooccur_lift_successor import (
    ConcatenatedFeatureView,
    CooccurLiftGapAwareView,
)
from jgrec.rankers.hybrid.setwise import SetwiseFeatureView
from materialize_dataset2_cooccur_lift_test import _load_auxiliary_model
from train_dataset2_cooccur_lift_successor_v2_duel import (
    _fit_head,
    _predict_probabilities,
    _release_jittor,
)

TRAIN_ROWS = 200_000
EXTERNAL_ROWS = 20_000
CANDIDATE_COUNT = 100
BASE_FEATURE_COUNT = 63
CONTEXT_FEATURE_COUNT = 198
REPLAY_RTOL = 2e-5
REPLAY_ATOL = 2e-6


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train the frozen full-origin gap-aware v2 head twice on CPU and "
            "materialize a metric-unread standard external manifest."
        )
    )
    parser.add_argument("--execution-contract", required=True, type=Path)
    parser.add_argument("--candidate-config", required=True, type=Path)
    parser.add_argument("--selection-lock", required=True, type=Path)
    parser.add_argument("--bugfixed-v1-contract", required=True, type=Path)
    parser.add_argument("--bugfixed-v1-training-report", required=True, type=Path)
    parser.add_argument("--bugfixed-v1-model", required=True, type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument("--train-cache-report", required=True, type=Path)
    parser.add_argument("--validation-cache-prefix", required=True, type=Path)
    parser.add_argument("--validation-cache-report", required=True, type=Path)
    parser.add_argument("--train-lift-features", required=True, type=Path)
    parser.add_argument("--train-short-none", required=True, type=Path)
    parser.add_argument("--validation-short-none", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--prior-external-probabilities", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    setup = validate_successor_external_setup(
        candidate_config_path=args.candidate_config,
        selection_lock_path=args.selection_lock,
    )
    execution = _validate_execution_contract(args)
    v1_contract = _read_json(args.bugfixed_v1_contract)
    v1_training = _read_json(args.bugfixed_v1_training_report)
    validate_bugfixed_v1_materialization_inputs(
        contract=v1_contract,
        contract_sha256=_sha256(args.bugfixed_v1_contract),
        training_report=v1_training,
        actual_model_sha256=_sha256(args.bugfixed_v1_model),
        actual_source_checkpoint_sha256=_sha256(args.source_checkpoint),
    )

    train_paths = _cache_paths(args.train_cache_prefix, split="train")
    validation_paths = _cache_paths(
        args.validation_cache_prefix,
        split="val",
    )
    train_features = _load(train_paths["features"])
    train_candidates = _load(train_paths["candidates"])
    train_sources = _load(train_paths["sources"])
    train_destinations = _load(train_paths["destinations"])
    train_times = _load(train_paths["times"])
    train_short_none = _load(args.train_short_none)
    train_lift = _load(args.train_lift_features)
    validation_features = _load(validation_paths["features"])
    validation_candidates = _load(validation_paths["candidates"])
    validation_sources = _load(validation_paths["sources"])
    validation_destinations = _load(validation_paths["destinations"])
    validation_times = _load(validation_paths["times"])
    validation_short_none = _load(args.validation_short_none)
    prior_external = _load(args.prior_external_probabilities)
    _validate_assets(
        train_features=train_features,
        train_candidates=train_candidates,
        train_sources=train_sources,
        train_destinations=train_destinations,
        train_times=train_times,
        train_short_none=train_short_none,
        train_lift=train_lift,
        validation_features=validation_features,
        validation_candidates=validation_candidates,
        validation_sources=validation_sources,
        validation_destinations=validation_destinations,
        validation_times=validation_times,
        validation_short_none=validation_short_none,
        prior_external=prior_external,
    )

    training_time_max = int(train_times[-1])
    strict_rows = np.asarray(validation_times > training_time_max)
    if int(strict_rows.sum()) != 19_981:
        raise ValueError("strict external row count differs from frozen holdout")

    train_report = _read_json(args.train_cache_report)
    validation_report = _read_json(args.validation_cache_report)
    _validate_cache_report_assets(train_report, train_paths)
    _validate_cache_report_assets(validation_report, validation_paths)
    state = load_checkpoint_dataset(args.source_checkpoint, "dataset2")
    feature_names = tuple(str(name) for name in state["feature_names"])
    if (
        feature_names != tuple(train_report["feature_names"])
        or feature_names != tuple(validation_report["feature_names"])
    ):
        raise ValueError("checkpoint and cache feature schemas differ")
    gnn_column = feature_names.index("gnn_short")
    hidden_dim = int(state["setwise_hidden_dim"])
    del state
    gc.collect()

    args.output_dir.mkdir(parents=True)
    started = time.time()
    interactions = read_interactions(args.train_csv).sort_by_time()

    stale_lifts: list[np.ndarray] = []
    native_training: list[dict[str, Any]] = []
    for index, gap_seconds in enumerate(setup.gap_seconds):
        gap_dir = args.output_dir / f"training-gap-{index}"
        lift_path = gap_dir / "lift.npy"
        popularity_path = gap_dir / "positive-popularity.npy"
        gap_dir.mkdir()
        contract = materialize_compact_cooccur_lift(
            interactions=interactions,
            sources=train_sources,
            candidates=train_candidates,
            destinations=train_destinations,
            event_time=train_times,
            availability_time=np.asarray(train_times) - gap_seconds,
            short_window=float(setup.short_window_seconds),
            lift_path=lift_path,
            positive_popularity_path=popularity_path,
            progress_path=gap_dir / "progress.json",
            work_dir=gap_dir / "work",
        )
        if contract["collapsed_short_rows"] != TRAIN_ROWS:
            raise ValueError("a frozen gapped training copy did not collapse")
        stale_lifts.append(_load(lift_path))
        native_training.append(contract)

    external_dir = args.output_dir / "external-features"
    external_dir.mkdir()
    external_lift_path = external_dir / "lift.npy"
    external_native = materialize_compact_cooccur_lift(
        interactions=interactions,
        sources=validation_sources,
        candidates=validation_candidates,
        destinations=validation_destinations,
        event_time=validation_times,
        availability_time=validation_times,
        short_window=float(setup.short_window_seconds),
        lift_path=external_lift_path,
        positive_popularity_path=external_dir / "positive-popularity.npy",
        progress_path=external_dir / "progress.json",
        work_dir=external_dir / "work",
    )
    if external_native["collapsed_short_rows"] != 0:
        raise ValueError("external near-horizon feature rows unexpectedly collapsed")
    external_lift = _load(external_lift_path)
    external_support = short_window_support_from_availability(
        validation_times,
        validation_times,
        short_window_seconds=setup.short_window_seconds,
    )
    if not np.all(external_support == 1.0):
        raise ValueError("external support indicator must be all one")

    v1_model, v1_result = _load_auxiliary_model(args.bugfixed_v1_model)
    jt.flags.use_cuda = 0
    v1_view = SetwiseFeatureView(
        CooccurLiftAugmentedView(
            validation_features,
            short_none_scores=validation_short_none,
            gnn_short_column=gnn_column,
            lift_features=external_lift,
        ),
        transform_version=1,
    )
    v1_auxiliary = _predict_probabilities(v1_model, v1_view, v1_result)
    v1_baseline = (
        0.5 * np.asarray(prior_external, dtype=np.float64)
        + 0.5 * np.asarray(v1_auxiliary, dtype=np.float64)
    )
    del v1_model, v1_result, v1_view, v1_auxiliary
    _release_jittor()

    near_raw = CooccurLiftGapAwareView(
        train_features,
        short_none_scores=train_short_none,
        gnn_short_column=gnn_column,
        lift_features=train_lift,
        short_window_supported=np.ones(TRAIN_ROWS, dtype=np.float32),
    )
    stale_raw = [
        CooccurLiftGapAwareView(
            train_features,
            short_none_scores=train_short_none,
            gnn_short_column=gnn_column,
            lift_features=lift,
            short_window_supported=np.zeros(TRAIN_ROWS, dtype=np.float32),
        )
        for lift in stale_lifts
    ]
    train_view = SetwiseFeatureView(
        ConcatenatedFeatureView((near_raw, *stale_raw)),
        transform_version=1,
    )
    external_view = SetwiseFeatureView(
        CooccurLiftGapAwareView(
            validation_features,
            short_none_scores=validation_short_none,
            gnn_short_column=gnn_column,
            lift_features=external_lift,
            short_window_supported=external_support,
        ),
        transform_version=1,
    )
    if train_view.shape != (
        TRAIN_ROWS * 4,
        CANDIDATE_COUNT,
        CONTEXT_FEATURE_COUNT,
    ):
        raise ValueError("full-origin gap-aware training view shape differs")
    copy_weights = full_origin_copy_weights(
        collapsed_fraction=setup.collapsed_fraction,
        gapped_copy_count=len(setup.gap_seconds),
    )
    row_weights = np.concatenate(
        [
            np.full(TRAIN_ROWS, weight, dtype=np.float32)
            for weight in copy_weights
        ]
    )

    first_model, first_result, first_losses = _fit_head(
        train_view,
        train_view[:1],
        hidden_dim=hidden_dim,
        seed=setup.full_origin_seed,
        candidate_name=f"{setup.candidate_id}-full-origin",
        feature_count=CONTEXT_FEATURE_COUNT,
        train_row_weights=row_weights,
    )
    first_auxiliary = _predict_probabilities(
        first_model,
        external_view,
        first_result,
    )
    first_state = {
        key: np.array(value, copy=True)
        for key, value in first_result.state.items()
    }
    model_path = args.output_dir / (
        f"{setup.candidate_id}-seed{setup.full_origin_seed}.npz"
    )
    _save_model(
        model_path,
        result=first_result,
        hidden_dim=hidden_dim,
        seed=setup.full_origin_seed,
    )
    del first_model, first_result
    _release_jittor()

    second_model, second_result, second_losses = _fit_head(
        train_view,
        train_view[:1],
        hidden_dim=hidden_dim,
        seed=setup.full_origin_seed,
        candidate_name=f"{setup.candidate_id}-full-origin-replay",
        feature_count=CONTEXT_FEATURE_COUNT,
        train_row_weights=row_weights,
    )
    second_auxiliary = _predict_probabilities(
        second_model,
        external_view,
        second_result,
    )
    replay = build_deterministic_replay_report(
        first_state=first_state,
        second_state=second_result.state,
        first_losses=first_losses,
        second_losses=second_losses,
        first_predictions={"external": first_auxiliary},
        second_predictions={"external": second_auxiliary},
        rtol=REPLAY_RTOL,
        atol=REPLAY_ATOL,
    )
    del second_model, second_result, second_auxiliary
    _release_jittor()
    if not replay["matched"]:
        raise RuntimeError("gap-aware full-origin deterministic replay failed")

    candidate = (
        0.5 * v1_baseline
        + 0.5 * np.asarray(first_auxiliary, dtype=np.float64)
    )
    auxiliary_path = args.output_dir / "external-auxiliary.npy"
    baseline_path = args.output_dir / "external-baseline-v1.npy"
    candidate_path = args.output_dir / "external-candidate-v2.npy"
    np.save(auxiliary_path, first_auxiliary[strict_rows])
    np.save(baseline_path, v1_baseline[strict_rows])
    np.save(candidate_path, candidate[strict_rows])
    fingerprint = hashlib.sha256(
        np.ascontiguousarray(
            validation_candidates[strict_rows]
        ).tobytes(order="C")
    ).hexdigest()
    manifest = build_standard_external_manifest(
        setup=setup,
        candidate_fingerprint=fingerprint,
        training_time_max=training_time_max,
        score_time_min=int(validation_times[strict_rows][0]),
        score_time_max=int(validation_times[strict_rows][-1]),
        baseline_path=baseline_path.resolve(),
        baseline_sha256=_sha256(baseline_path),
        candidate_path=candidate_path.resolve(),
        candidate_sha256=_sha256(candidate_path),
        scored_rows=int(strict_rows.sum()),
        supported_rows=int(external_support[strict_rows].sum()),
    )
    manifest_path = args.output_dir / "external-manifest.json"
    _write_json(manifest_path, manifest)
    report = {
        "schema_version": 1,
        "protocol": "cooccur_lift_successor_v2_external_materialization_v1",
        "status": "external_candidate_materialized_metrics_unread",
        "decision_role": "safety_gate_only",
        "effect_size_estimation_authorized": False,
        "candidate_id": setup.candidate_id,
        "candidate_config_sha256": setup.config_sha256,
        "selection_lock_sha256": setup.selection_lock_sha256,
        "execution_contract_sha256": _sha256(args.execution_contract),
        "baseline_role": "bugfixed_v1_new_champion",
        "selected_weight": setup.selected_weight,
        "full_origin_seed": setup.full_origin_seed,
        "training_copy_weights": list(copy_weights),
        "training_gap_seconds": list(setup.gap_seconds),
        "training_rows_per_copy": TRAIN_ROWS,
        "effective_training_rows": TRAIN_ROWS * 4,
        "short_window_seconds": setup.short_window_seconds,
        "external_short_window_support": {
            "supported_rows": int(external_support[strict_rows].sum()),
            "total_rows": int(strict_rows.sum()),
            "collapsed_fraction": 0.0,
            "unique_values": [1],
        },
        "deterministic_replay": replay,
        "model": str(model_path.resolve()),
        "model_sha256": _sha256(model_path),
        "external_manifest": str(manifest_path.resolve()),
        "external_manifest_sha256": _sha256(manifest_path),
        "external_auxiliary_sha256": _sha256(auxiliary_path),
        "external_baseline_sha256": _sha256(baseline_path),
        "external_candidate_sha256": _sha256(candidate_path),
        "native_training_materializers": native_training,
        "native_external_materializer": external_native,
        "external_ranking_metrics_computed": False,
        "external_evaluator_invoked": False,
        "strict_external_rows": int(strict_rows.sum()),
        "elapsed_seconds": time.time() - started,
        "execution_contract": execution,
    }
    _write_json(
        args.output_dir / "external-materialization-report.json",
        report,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def _validate_execution_contract(args: argparse.Namespace) -> dict[str, Any]:
    contract = _read_json(args.execution_contract)
    if (
        contract.get("protocol")
        != "cooccur_lift_successor_v2_external_execution_v1"
        or contract.get("status") != "frozen_before_external_open"
        or contract.get("external_authorized") is not True
        or contract.get("maximum_external_opens") != 1
        or contract.get("decision_role") != "safety_gate_only"
        or contract.get("effect_size_estimation_authorized") is not False
        or contract.get("external_collapsed_fraction") != 0.0
        or contract.get("replay_rtol") != REPLAY_RTOL
        or contract.get("replay_atol") != REPLAY_ATOL
        or contract.get("tolerance_relaxation_authorized") is not False
    ):
        raise ValueError("external execution contract differs")
    paths = {
        "candidate_config": args.candidate_config,
        "selection_lock": args.selection_lock,
        "bugfixed_v1_contract": args.bugfixed_v1_contract,
        "bugfixed_v1_training_report": args.bugfixed_v1_training_report,
        "bugfixed_v1_model": args.bugfixed_v1_model,
        "source_checkpoint": args.source_checkpoint,
        "train_cache_report": args.train_cache_report,
        "validation_cache_report": args.validation_cache_report,
        "train_lift_features": args.train_lift_features,
        "train_short_none": args.train_short_none,
        "validation_short_none": args.validation_short_none,
        "train_csv": args.train_csv,
        "prior_external_probabilities": args.prior_external_probabilities,
        "materializer_script": Path(__file__).resolve(),
        "external_module": Path(
            __import__(
                "jgrec.cooccur_lift_successor_external",
                fromlist=["__file__"],
            ).__file__
        ),
    }
    expected = contract.get("input_sha256")
    if not isinstance(expected, dict) or set(expected) != set(paths):
        raise ValueError("external execution input hash set differs")
    for name, path in paths.items():
        if _sha256(path) != expected[name]:
            raise ValueError(f"external execution input differs: {name}")
    return contract


def _validate_assets(**assets: np.ndarray) -> None:
    if assets["train_features"].shape != (
        TRAIN_ROWS,
        CANDIDATE_COUNT,
        BASE_FEATURE_COUNT,
    ):
        raise ValueError("training feature shape differs")
    if assets["validation_features"].shape != (
        EXTERNAL_ROWS,
        CANDIDATE_COUNT,
        BASE_FEATURE_COUNT,
    ):
        raise ValueError("external feature shape differs")
    matrix_shapes = {
        "train_candidates": (TRAIN_ROWS, CANDIDATE_COUNT),
        "train_short_none": (TRAIN_ROWS, CANDIDATE_COUNT),
        "prior_external": (EXTERNAL_ROWS, CANDIDATE_COUNT),
        "validation_candidates": (EXTERNAL_ROWS, CANDIDATE_COUNT),
        "validation_short_none": (EXTERNAL_ROWS, CANDIDATE_COUNT),
    }
    for name, shape in matrix_shapes.items():
        if assets[name].shape != shape:
            raise ValueError(f"{name} shape differs")
    if assets["train_lift"].shape != (
        TRAIN_ROWS,
        CANDIDATE_COUNT,
        2,
    ):
        raise ValueError("training lift shape differs")
    vector_shapes = {
        "train_sources": (TRAIN_ROWS,),
        "train_destinations": (TRAIN_ROWS,),
        "train_times": (TRAIN_ROWS,),
        "validation_sources": (EXTERNAL_ROWS,),
        "validation_destinations": (EXTERNAL_ROWS,),
        "validation_times": (EXTERNAL_ROWS,),
    }
    for name, shape in vector_shapes.items():
        if assets[name].shape != shape:
            raise ValueError(f"{name} shape differs")
    if not np.array_equal(
        assets["train_candidates"][:, 0],
        assets["train_destinations"],
    ):
        raise ValueError("training positive candidate differs")
    if not np.array_equal(
        assets["validation_candidates"][:, 0],
        assets["validation_destinations"],
    ):
        raise ValueError("external positive candidate differs")
    if not np.all(np.isfinite(assets["prior_external"])):
        raise ValueError("prior external probabilities are non-finite")


def _validate_cache_report_assets(
    report: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    descriptors = report.get("artifacts")
    report_names = {
        "features": "features",
        "candidates": "candidates",
        "sources": "src",
        "destinations": "dst",
        "times": "time",
    }
    if not isinstance(descriptors, dict):
        raise ValueError("cache report has no artifact descriptors")
    for path_name, report_name in report_names.items():
        path = paths[path_name]
        descriptor = descriptors.get(report_name)
        if (
            not isinstance(descriptor, dict)
            or Path(descriptor.get("path", "")).resolve() != path.resolve()
            or int(descriptor.get("bytes", -1)) != path.stat().st_size
            or descriptor.get("sha256") != _sha256(path)
        ):
            raise ValueError(f"cache report artifact differs: {path_name}")


def _save_model(
    path: Path,
    *,
    result: Any,
    hidden_dim: int,
    seed: int,
) -> None:
    payload = {
        "mean": np.asarray(result.mean, dtype=np.float32),
        "std": np.asarray(result.std, dtype=np.float32),
        "feature_indices": np.asarray(result.feature_indices, dtype=np.int32),
        "hidden_dim": np.asarray([hidden_dim], dtype=np.int32),
        "source_feature_count": np.asarray([66], dtype=np.int32),
        "context_transform_version": np.asarray([1], dtype=np.int32),
        "training_seed": np.asarray([seed], dtype=np.int64),
        "candidate_id": np.asarray(["cooccur_lift_gap_aware_v2"]),
    }
    payload.update(
        {
            f"state__{key}": np.asarray(value, dtype=np.float32)
            for key, value in result.state.items()
        }
    )
    np.savez_compressed(path, **payload)


def _cache_paths(prefix: Path, *, split: str) -> dict[str, Path]:
    base = str(prefix)
    return {
        "features": Path(f"{base}.{split}.npy"),
        "candidates": Path(f"{base}.{split}-candidates.npy"),
        "sources": Path(f"{base}.{split}-src.npy"),
        "destinations": Path(f"{base}.{split}-dst.npy"),
        "times": Path(f"{base}.{split}-time.npy"),
    }


def _load(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
