from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.cooccur_lift_bugfixed_v1 import (
    validate_bugfixed_v1_training_report,
)
from jgrec.rankers.hybrid import fusion as fusion_module
from jgrec.rankers.hybrid.cooccur_lift import (
    COOCCUR_LIFT_FEATURE_NAMES,
    CooccurLiftAugmentedView,
    load_frozen_cooccur_lift_config,
)
from jgrec.rankers.hybrid.full100_training import (
    validate_full100_cache_arrays,
    validate_joint_cache_reports,
)
from jgrec.rankers.hybrid.fusion import (
    FusionConfig,
    FusionResult,
    fit_fusion_mlp_listwise_fixed,
    predict_logits,
)
from jgrec.rankers.hybrid.setwise import SetwiseFeatureView

TRAIN_ROWS = 200_000
CANDIDATE_COUNT = 100
BASE_FEATURE_COUNT = 63
SOURCE_FEATURE_COUNT = 65
PROBE_ROWS_PER_EDGE = 256


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train the frozen cooccur-lift v1 shape twice with the corrected "
            "deterministic fusion path and publish a model only if replay passes."
        )
    )
    parser.add_argument("--candidate-contract", required=True, type=Path)
    parser.add_argument("--frozen-config", required=True, type=Path)
    parser.add_argument("--selection-lock", required=True, type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument("--train-cache-report", required=True, type=Path)
    parser.add_argument("--validation-cache-report", required=True, type=Path)
    parser.add_argument("--train-lift-features", required=True, type=Path)
    parser.add_argument("--train-short-none", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")

    started = time.time()
    contract = _read_json(args.candidate_contract)
    config = load_frozen_cooccur_lift_config(args.frozen_config)
    selection_lock = _read_json(args.selection_lock)
    train_report = _read_json(args.train_cache_report)
    validation_report = _read_json(args.validation_cache_report)
    training_assets = _mapping(contract, "training_assets")
    implementation = _mapping(contract, "implementation_contract")
    replay_gate = _mapping(contract, "deterministic_replay_gate")

    _validate_frozen_paths(
        training_assets=training_assets,
        frozen_config=args.frozen_config,
        selection_lock=args.selection_lock,
        source_checkpoint=args.source_checkpoint,
        train_cache_report=args.train_cache_report,
        validation_cache_report=args.validation_cache_report,
        train_lift_features=args.train_lift_features,
        train_short_none=args.train_short_none,
    )
    if config.integration_id != contract.get("integration_id"):
        raise ValueError("candidate and frozen config integration_id differ")
    if (
        selection_lock.get("integration_id") != config.integration_id
        or float(selection_lock.get("selected_weight", -1.0))
        != float(contract.get("selected_weight", -2.0))
    ):
        raise ValueError("selection lock differs from bugfixed v1 contract")

    fusion_sha256 = _sha256(Path(fusion_module.__file__).resolve())
    if fusion_sha256 != implementation.get("fusion_source_sha256"):
        raise ValueError("fusion implementation differs from frozen candidate")
    joint_contract = validate_joint_cache_reports(
        train_report,
        validation_report,
    )
    if (
        joint_contract["joint_build_id"]
        != training_assets.get("joint_build_id")
        or joint_contract["process_id"]
        != training_assets.get("joint_build_pid")
        or joint_contract["train_feature_sha256"]
        != training_assets.get("train_feature_sha256")
    ):
        raise ValueError("joint cache provenance differs from frozen candidate")

    paths = _cache_paths(args.train_cache_prefix)
    _validate_report_artifacts(train_report)
    if _sha256(paths["features"]) != training_assets.get(
        "train_feature_sha256"
    ):
        raise ValueError("training feature cache differs from frozen candidate")

    train_features = np.load(
        paths["features"],
        mmap_mode="r",
        allow_pickle=False,
    )
    train_candidates = np.load(
        paths["candidates"],
        mmap_mode="r",
        allow_pickle=False,
    )
    train_sources = np.load(
        paths["sources"],
        mmap_mode="r",
        allow_pickle=False,
    )
    train_destinations = np.load(
        paths["destinations"],
        mmap_mode="r",
        allow_pickle=False,
    )
    train_times = np.load(
        paths["times"],
        mmap_mode="r",
        allow_pickle=False,
    )
    train_rows = np.load(
        paths["row_indices"],
        mmap_mode="r",
        allow_pickle=False,
    )
    train_lift = np.load(
        args.train_lift_features,
        mmap_mode="r",
        allow_pickle=False,
    )
    train_short_none = np.load(
        args.train_short_none,
        mmap_mode="r",
        allow_pickle=False,
    )
    cache_contract = validate_full100_cache_arrays(
        features=train_features,
        candidates=train_candidates,
        src=train_sources,
        dst=train_destinations,
        time=train_times,
        row_indices=train_rows,
        expected_train_rows=TRAIN_ROWS,
        expected_candidate_count=CANDIDATE_COUNT,
        expected_feature_count=BASE_FEATURE_COUNT,
    )
    if train_lift.shape != (TRAIN_ROWS, CANDIDATE_COUNT, 2):
        raise ValueError("training lift features have the wrong shape")
    if train_short_none.shape != (TRAIN_ROWS, CANDIDATE_COUNT):
        raise ValueError("training short-none scores have the wrong shape")
    if not np.array_equal(train_candidates[:, 0], train_destinations):
        raise ValueError("training candidate zero differs from destination")

    checkpoint_state = load_checkpoint_dataset(
        args.source_checkpoint,
        "dataset2",
    )
    feature_names = tuple(
        str(name) for name in checkpoint_state["feature_names"]
    )
    if feature_names != tuple(train_report["feature_names"]):
        raise ValueError("source checkpoint and training cache schemas differ")
    hidden_dim = int(checkpoint_state["setwise_hidden_dim"])
    gnn_column = feature_names.index("gnn_short")
    del checkpoint_state
    gc.collect()

    augmented = CooccurLiftAugmentedView(
        train_features,
        short_none_scores=train_short_none,
        gnn_short_column=gnn_column,
        lift_features=train_lift,
    )
    train_view = SetwiseFeatureView(
        augmented,
        transform_version=1,
    )
    if train_view.shape != (TRAIN_ROWS, CANDIDATE_COUNT, 195):
        raise ValueError("bugfixed v1 Setwise context has the wrong shape")
    probe = np.concatenate(
        (
            np.asarray(train_view[:PROBE_ROWS_PER_EDGE], dtype=np.float32),
            np.asarray(train_view[-PROBE_ROWS_PER_EDGE:], dtype=np.float32),
        ),
        axis=0,
    )

    seed = int(contract["full_origin_seed"])
    training_config = FusionConfig(
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        hidden_dim=hidden_dim,
        selection_metric=config.selection_metric,
        early_stop_patience=config.early_stop_patience,
    )
    first = _fit_once(
        train_view=train_view,
        probe=probe,
        training_config=training_config,
        config=config,
        seed=seed,
        run_label="run1",
    )
    _release_jittor()
    second = _fit_once(
        train_view=train_view,
        probe=probe,
        training_config=training_config,
        config=config,
        seed=seed,
        run_label="run2",
    )
    replay = _deterministic_replay_report(
        first=first,
        second=second,
        rtol=float(replay_gate["rtol"]),
        atol=float(replay_gate["atol"]),
    )
    if not replay["matched"]:
        raise ValueError(
            "bugfixed v1 deterministic replay failed without publishing a model: "
            f"{replay}"
        )

    args.output_dir.mkdir(parents=True)
    model_path = args.output_dir / (
        f"cooccur-lift-bugfixed-v1-seed{seed}.npz"
    )
    _save_fusion_result(
        model_path,
        result=first["result"],
        hidden_dim=hidden_dim,
        seed=seed,
        salt=config.seed_salt,
    )
    report = {
        "schema_version": 1,
        "status": "complete_deterministic_bugfixed_v1_refit",
        "candidate_id": contract["candidate_id"],
        "integration_id": contract["integration_id"],
        "selected_weight": float(contract["selected_weight"]),
        "full_origin_seed": seed,
        "candidate_contract": str(args.candidate_contract.resolve()),
        "candidate_contract_sha256": _sha256(args.candidate_contract),
        "model": str(model_path.resolve()),
        "model_sha256": _sha256(model_path),
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": _sha256(args.source_checkpoint),
        "frozen_config_sha256": _sha256(args.frozen_config),
        "selection_lock_sha256": _sha256(args.selection_lock),
        "train_cache_report_sha256": _sha256(args.train_cache_report),
        "validation_cache_report_sha256": _sha256(
            args.validation_cache_report
        ),
        "train_lift_features_sha256": _sha256(
            args.train_lift_features
        ),
        "train_short_none_sha256": _sha256(args.train_short_none),
        "fusion_source_sha256": fusion_sha256,
        "training_device": "cpu",
        "joint_cache_contract": joint_contract,
        "cache_contract": cache_contract,
        "losses": {
            "run1": list(first["losses"]),
            "run2": list(second["losses"]),
        },
        "deterministic_replay": replay,
        "historical_model_reused": False,
        "external_data_read": False,
        "external_metric_computed": False,
        "elapsed_seconds": time.time() - started,
    }
    validate_bugfixed_v1_training_report(
        contract=contract,
        contract_sha256=_sha256(args.candidate_contract),
        report=report,
        actual_model_sha256=_sha256(model_path),
    )
    _write_json(args.output_dir / "training-report.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def _fit_once(
    *,
    train_view: Any,
    probe: np.ndarray,
    training_config: FusionConfig,
    config: Any,
    seed: int,
    run_label: str,
) -> dict[str, Any]:
    jt.flags.use_cuda = 0
    jt.set_global_seed(seed)
    model, result, losses = fit_fusion_mlp_listwise_fixed(
        train_view,
        train_view[:1],
        training_config,
        np.random.default_rng(seed),
        verbose=False,
        feature_indices=config.context_feature_indices,
        candidate_name=f"{config.integration_id}_bugfixed_{run_label}",
    )
    probabilities = _predict_probabilities(
        model,
        probe,
        result,
        batch_size=training_config.batch_size,
    )
    del model
    return {
        "result": result,
        "losses": losses,
        "probabilities": probabilities,
    }


def _deterministic_replay_report(
    *,
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    first_result = first["result"]
    second_result = second["result"]
    state_keys_match = first_result.state.keys() == second_result.state.keys()
    state_matched = state_keys_match and all(
        np.allclose(
            first_result.state[key],
            second_result.state[key],
            rtol=rtol,
            atol=atol,
        )
        for key in first_result.state
    )
    first_probabilities = np.asarray(first["probabilities"])
    second_probabilities = np.asarray(second["probabilities"])
    absolute_error = np.abs(first_probabilities - second_probabilities)
    probability_matched = bool(
        np.allclose(
            first_probabilities,
            second_probabilities,
            rtol=rtol,
            atol=atol,
        )
    )
    loss_matched = bool(
        np.allclose(
            first["losses"],
            second["losses"],
            rtol=rtol,
            atol=atol,
        )
    )
    return {
        "matched": bool(
            state_matched and probability_matched and loss_matched
        ),
        "runs": 2,
        "rtol": rtol,
        "atol": atol,
        "tolerance_relaxed": False,
        "state_matched": bool(state_matched),
        "probability_matched": probability_matched,
        "loss_matched": loss_matched,
        "probe_shape": list(first_probabilities.shape),
        "max_abs_error": float(np.max(absolute_error, initial=0.0)),
        "mean_abs_error": float(np.mean(absolute_error)),
    }


def _predict_probabilities(
    model: Any,
    features: np.ndarray,
    result: FusionResult,
    *,
    batch_size: int,
) -> np.ndarray:
    probabilities = np.empty(features.shape[:2], dtype=np.float64)
    for start in range(0, features.shape[0], batch_size):
        stop = min(start + batch_size, features.shape[0])
        logits = predict_logits(
            model,
            features[start:stop],
            result.mean,
            result.std,
        )
        shifted = np.asarray(logits, dtype=np.float64)
        shifted -= shifted.max(axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        probabilities[start:stop] = (
            exponentials / exponentials.sum(axis=1, keepdims=True)
        )
    return probabilities


def _validate_frozen_paths(
    *,
    training_assets: Mapping[str, Any],
    frozen_config: Path,
    selection_lock: Path,
    source_checkpoint: Path,
    train_cache_report: Path,
    validation_cache_report: Path,
    train_lift_features: Path,
    train_short_none: Path,
) -> None:
    paths = {
        "frozen_config": frozen_config,
        "selection_lock": selection_lock,
        "source_checkpoint": source_checkpoint,
        "train_cache_report": train_cache_report,
        "validation_cache_report": validation_cache_report,
        "train_lift_features": train_lift_features,
        "train_short_none": train_short_none,
    }
    for name, path in paths.items():
        expected = str(training_assets.get(f"{name}_sha256", ""))
        if not expected or _sha256(path) != expected:
            raise ValueError(f"{name} differs from frozen candidate")


def _validate_report_artifacts(report: Mapping[str, Any]) -> None:
    artifacts = _mapping(report, "artifacts")
    for name, item in artifacts.items():
        if not isinstance(item, Mapping):
            raise ValueError(f"training cache artifact {name} is malformed")
        path = Path(str(item.get("path", "")))
        expected = str(item.get("sha256", ""))
        if not path.is_file() or not expected or _sha256(path) != expected:
            raise ValueError(
                f"training cache artifact {name} differs from its report"
            )


def _cache_paths(prefix: Path) -> dict[str, Path]:
    base = str(prefix)
    return {
        "features": Path(f"{base}.train.npy"),
        "candidates": Path(f"{base}.train-candidates.npy"),
        "sources": Path(f"{base}.train-src.npy"),
        "destinations": Path(f"{base}.train-dst.npy"),
        "times": Path(f"{base}.train-time.npy"),
        "row_indices": Path(f"{base}.train-row-indices.npy"),
    }


def _save_fusion_result(
    path: Path,
    *,
    result: FusionResult,
    hidden_dim: int,
    seed: int,
    salt: int,
) -> None:
    payload = {
        "mean": np.asarray(result.mean, dtype=np.float32),
        "std": np.asarray(result.std, dtype=np.float32),
        "feature_indices": np.asarray(result.feature_indices, dtype=np.int32),
        "hidden_dim": np.asarray([hidden_dim], dtype=np.int32),
        "source_feature_count": np.asarray(
            [SOURCE_FEATURE_COUNT],
            dtype=np.int32,
        ),
        "context_transform_version": np.asarray([1], dtype=np.int32),
        "training_seed": np.asarray([seed], dtype=np.int64),
        "seed_salt": np.asarray([salt], dtype=np.int64),
        "feature_names": np.asarray(COOCCUR_LIFT_FEATURE_NAMES),
    }
    payload.update(
        {
            f"state__{key}": np.asarray(value, dtype=np.float32)
            for key, value in result.state.items()
        }
    )
    np.savez_compressed(path, **payload)


def _release_jittor() -> None:
    gc.collect()
    jt.sync_all()
    jt.clean()


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
