from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.contest_checkpoint import (
    ContestCheckpointWriter,
    load_checkpoint_dataset,
    load_checkpoint_metadata,
)
from jgrec.core.io import read_test_queries
from jgrec.rankers.hybrid.cooccur_lift import (
    BASE_FEATURE_COUNT,
    CONTEXT_FEATURE_COUNT,
    INTEGRATION_ID,
)
from jgrec.rankers.hybrid.cooccur_lift_checkpoint import (
    CHECKPOINT_FIELD,
    LOCKED_WEIGHT,
    CausalLiftFeatureStore,
    CooccurLiftAuxiliaryState,
    install_cooccur_lift_auxiliary_state,
    validate_online_promotion_receipt,
)
from jgrec.rankers.registry import create_ranker

RESERVED_METADATA_KEYS = {"format", "version", "model_name", "datasets"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Promote the accepted cooccur-lift auxiliary into a standalone "
            "Dataset2 checkpoint under the frozen online receipt."
        )
    )
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--promotion-receipt", required=True, type=Path)
    parser.add_argument("--candidate-report", required=True, type=Path)
    parser.add_argument("--candidate-zip", required=True, type=Path)
    parser.add_argument("--selection-lock", required=True, type=Path)
    parser.add_argument("--external-report", required=True, type=Path)
    parser.add_argument(
        "--test-materialization-report",
        required=True,
        type=Path,
    )
    parser.add_argument("--auxiliary-model", required=True, type=Path)
    parser.add_argument("--test-lift-features", required=True, type=Path)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--output-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    _refuse_overwrite(args.output_checkpoint, args.output_dir)
    started = time.time()
    receipt = _read_json(args.promotion_receipt)
    validate_online_promotion_receipt(receipt)

    hashes = _hash_and_bind_inputs(args, receipt)
    candidate_report = _read_json(args.candidate_report)
    selection_lock = _read_json(args.selection_lock)
    external_report = _read_json(args.external_report)
    materialization = _read_json(args.test_materialization_report)
    _validate_authorization_chain(
        receipt=receipt,
        candidate_report=candidate_report,
        selection_lock=selection_lock,
        external_report=external_report,
        materialization=materialization,
        hashes=hashes,
    )

    source_metadata = load_checkpoint_metadata(args.source_checkpoint)
    datasets = tuple(str(name) for name in source_metadata["datasets"])
    if "dataset1" not in datasets or "dataset2" not in datasets:
        raise ValueError("source checkpoint must contain dataset1 and dataset2")

    extra_metadata = {
        key: value
        for key, value in source_metadata.items()
        if key not in RESERVED_METADATA_KEYS
    }
    extra_metadata.update(
        {
            "derived_from": str(args.source_checkpoint.resolve()),
            "derived_from_sha256": hashes["source_checkpoint"],
            "dataset2_integration": (
                "cooccur_lift_aux_expert_v1_w050"
            ),
            "dataset2_online_score": float(receipt["online_score"]),
            "dataset2_promotion_receipt": str(
                args.promotion_receipt.resolve()
            ),
            "dataset2_promotion_receipt_sha256": hashes[
                "promotion_receipt"
            ],
            "dataset2_auxiliary_model_sha256": hashes[
                "auxiliary_model"
            ],
            "dataset2_test_lift_features_sha256": hashes[
                "test_lift_features"
            ],
            "dataset2_selected_weight": LOCKED_WEIGHT,
            "dataset2_encoder_retrained": False,
            "dataset2_allowed_state_changes": (CHECKPOINT_FIELD,),
        }
    )
    writer = ContestCheckpointWriter(
        args.output_checkpoint,
        model_name=str(source_metadata["model_name"]),
        expected_datasets=datasets,
        metadata=extra_metadata,
    )

    dataset1_hash: str
    protected_dataset2_hashes: dict[str, str]
    auxiliary_state_audit: dict[str, Any]
    feature_store_shape: tuple[int, int, int]
    try:
        dataset1 = load_checkpoint_dataset(
            args.source_checkpoint,
            "dataset1",
        )
        dataset1_hash = _pickle_sha256(dataset1)
        writer.add_dataset("dataset1", dataset1)
        del dataset1
        gc.collect()

        source_dataset2 = load_checkpoint_dataset(
            args.source_checkpoint,
            "dataset2",
        )
        auxiliary_state = _load_auxiliary_state(
            source_dataset2=source_dataset2,
            auxiliary_model=args.auxiliary_model,
            test_csv=args.test_csv,
            test_lift_features=args.test_lift_features,
            provenance={
                "promotion_receipt_sha256": hashes[
                    "promotion_receipt"
                ],
                "candidate_zip_sha256": hashes["candidate_zip"],
                "candidate_report_sha256": hashes["candidate_report"],
                "selection_lock_sha256": hashes["selection_lock"],
                "external_report_sha256": hashes["external_report"],
                "test_materialization_report_sha256": hashes[
                    "test_materialization_report"
                ],
                "auxiliary_model_sha256": hashes["auxiliary_model"],
                "test_lift_features_sha256": hashes[
                    "test_lift_features"
                ],
                "test_csv_sha256": hashes["test_csv"],
                "source_checkpoint_sha256": hashes[
                    "source_checkpoint"
                ],
                "online_score": str(receipt["online_score"]),
            },
        )
        candidate_dataset2 = install_cooccur_lift_auxiliary_state(
            source_dataset2,
            auxiliary_state,
        )
        protected_dataset2_hashes = _audit_in_memory_install(
            source_dataset2,
            candidate_dataset2,
        )
        auxiliary_state_audit = _audit_auxiliary_state(auxiliary_state)
        feature_store_shape = tuple(
            int(value)
            for value in auxiliary_state.feature_store.lift_features.shape
        )
        writer.add_dataset("dataset2", candidate_dataset2)
        writer.finalize()
        del (
            auxiliary_state,
            candidate_dataset2,
            source_dataset2,
        )
        gc.collect()
    except BaseException:
        writer.abort()
        raise

    reloaded_dataset1 = load_checkpoint_dataset(
        args.output_checkpoint,
        "dataset1",
    )
    reloaded_dataset1_hash = _pickle_sha256(reloaded_dataset1)
    del reloaded_dataset1
    gc.collect()
    if reloaded_dataset1_hash != dataset1_hash:
        raise RuntimeError("Dataset1 changed during checkpoint integration")

    reloaded_dataset2 = load_checkpoint_dataset(
        args.output_checkpoint,
        "dataset2",
    )
    _audit_reloaded_dataset2(
        reloaded_dataset2,
        protected_hashes=protected_dataset2_hashes,
        auxiliary_state_audit=auxiliary_state_audit,
    )
    _validate_standard_hydrate(reloaded_dataset2)
    del reloaded_dataset2
    gc.collect()

    output_hash = _sha256(args.output_checkpoint)
    report = {
        "status": "complete",
        "integration_id": INTEGRATION_ID,
        "selected_weight": LOCKED_WEIGHT,
        "online_score": float(receipt["online_score"]),
        "promotion_threshold": float(receipt["promotion_threshold"]),
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": hashes["source_checkpoint"],
        "output_checkpoint": str(args.output_checkpoint.resolve()),
        "output_checkpoint_bytes": args.output_checkpoint.stat().st_size,
        "output_checkpoint_sha256": output_hash,
        "promotion_receipt": str(args.promotion_receipt.resolve()),
        "promotion_receipt_sha256": hashes["promotion_receipt"],
        "candidate_zip_sha256": hashes["candidate_zip"],
        "candidate_report_sha256": hashes["candidate_report"],
        "selection_lock_sha256": hashes["selection_lock"],
        "external_report_sha256": hashes["external_report"],
        "test_materialization_report_sha256": hashes[
            "test_materialization_report"
        ],
        "auxiliary_model_sha256": hashes["auxiliary_model"],
        "test_lift_features_sha256": hashes["test_lift_features"],
        "test_csv_sha256": hashes["test_csv"],
        "feature_store_shape": list(feature_store_shape),
        "dataset1_pickle_sha256": dataset1_hash,
        "dataset1_reload_pickle_sha256": reloaded_dataset1_hash,
        "protected_dataset2_top_level_sha256": (
            protected_dataset2_hashes
        ),
        "auxiliary_state_audit": auxiliary_state_audit,
        "standard_hydrate_passed": True,
        "double_replay_required": True,
        "package_authorized": False,
        "memory_strategy": (
            "sequential dataset load/write/release; lift mmap until pickle; "
            "single reloaded dataset resident"
        ),
        "elapsed_seconds": time.time() - started,
    }
    args.output_dir.mkdir(parents=True)
    _write_json(
        args.output_dir / "checkpoint-integration-report.json",
        report,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


def _refuse_overwrite(checkpoint: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite: {checkpoint}")
    temporary = checkpoint.with_suffix(f"{checkpoint.suffix}.tmp")
    if temporary.exists():
        raise FileExistsError(
            f"refusing to resume unknown partial checkpoint: {temporary}"
        )


def _hash_and_bind_inputs(
    args: argparse.Namespace,
    receipt: dict[str, Any],
) -> dict[str, str]:
    paths = {
        "source_checkpoint": args.source_checkpoint,
        "promotion_receipt": args.promotion_receipt,
        "candidate_report": args.candidate_report,
        "candidate_zip": args.candidate_zip,
        "selection_lock": args.selection_lock,
        "external_report": args.external_report,
        "test_materialization_report": (
            args.test_materialization_report
        ),
        "auxiliary_model": args.auxiliary_model,
        "test_lift_features": args.test_lift_features,
        "test_csv": args.test_csv,
    }
    hashes = {key: _sha256(path) for key, path in paths.items()}
    bindings = {
        "source_checkpoint": "source_checkpoint_sha256",
        "candidate_report": "candidate_report_sha256",
        "candidate_zip": "candidate_zip_sha256",
        "selection_lock": "selection_lock_sha256",
        "external_report": "external_report_sha256",
        "auxiliary_model": "auxiliary_model_sha256",
    }
    for path_key, receipt_key in bindings.items():
        if hashes[path_key] != receipt[receipt_key]:
            raise ValueError(
                f"{path_key} hash differs from promotion receipt"
            )
    return hashes


def _validate_authorization_chain(
    *,
    receipt: dict[str, Any],
    candidate_report: dict[str, Any],
    selection_lock: dict[str, Any],
    external_report: dict[str, Any],
    materialization: dict[str, Any],
    hashes: dict[str, str],
) -> None:
    if (
        candidate_report.get("status") != "online_candidate"
        or candidate_report.get("submission_authorized_by_user") is not True
        or candidate_report.get("result_zip_sha256")
        != hashes["candidate_zip"]
        or float(candidate_report.get("promotion_threshold", -1.0))
        != float(receipt["promotion_threshold"])
        or float(candidate_report["dataset2"]["auxiliary_weight"])
        != LOCKED_WEIGHT
        or candidate_report["expert"]["model_sha256"]
        != hashes["auxiliary_model"]
        or candidate_report["expert"]["selection_lock_sha256"]
        != hashes["selection_lock"]
    ):
        raise ValueError("candidate report differs from promotion authority")
    if (
        selection_lock.get("integration_id") != INTEGRATION_ID
        or float(selection_lock.get("selected_weight", -1.0))
        != LOCKED_WEIGHT
        or selection_lock.get("external_holdout_read") is not False
    ):
        raise ValueError("selection lock differs from frozen selection")
    if (
        external_report.get("integration_id") != INTEGRATION_ID
        or external_report.get("status") != "accepted"
        or float(external_report.get("selected_weight", -1.0))
        != LOCKED_WEIGHT
        or external_report.get("failed_gates") != []
        or external_report.get("selection_lock_sha256")
        != hashes["selection_lock"]
        or external_report.get("weight_rescan_authorized") is not False
    ):
        raise ValueError("external report does not authorize integration")
    if (
        materialization.get("integration_id") != INTEGRATION_ID
        or materialization.get("status")
        != "complete_online_candidate_materialization"
        or float(materialization.get("selected_weight", -1.0))
        != LOCKED_WEIGHT
        or materialization.get("source_checkpoint_sha256")
        != hashes["source_checkpoint"]
        or materialization.get("auxiliary_model_sha256")
        != hashes["auxiliary_model"]
        or materialization.get("external_report_sha256")
        != hashes["external_report"]
        or materialization.get("selection_lock_sha256")
        != hashes["selection_lock"]
        or materialization.get("test_csv_sha256") != hashes["test_csv"]
        or materialization.get("production_checkpoint_modified") is not False
        or materialization.get("shape") != [153420, 100]
    ):
        raise ValueError("test materialization contract differs")


def _load_auxiliary_state(
    *,
    source_dataset2: dict[str, Any],
    auxiliary_model: Path,
    test_csv: Path,
    test_lift_features: Path,
    provenance: dict[str, str],
) -> CooccurLiftAuxiliaryState:
    feature_names = tuple(source_dataset2["feature_names"])
    if len(feature_names) != BASE_FEATURE_COUNT:
        raise ValueError("source Dataset2 does not have 63 base features")
    try:
        gnn_short_column = feature_names.index("gnn_short")
    except ValueError as error:
        raise ValueError("source Dataset2 has no gnn_short feature") from error

    with np.load(auxiliary_model, allow_pickle=False) as archive:
        hidden_dim = int(np.asarray(archive["hidden_dim"]).reshape(-1)[0])
        source_feature_count = int(
            np.asarray(archive["source_feature_count"]).reshape(-1)[0]
        )
        transform_version = int(
            np.asarray(
                archive["context_transform_version"]
            ).reshape(-1)[0]
        )
        model_state = {
            key.removeprefix("state__"): np.asarray(
                archive[key],
                dtype=np.float32,
            ).copy()
            for key in archive.files
            if key.startswith("state__")
        }
        mean = np.asarray(archive["mean"], dtype=np.float32).copy()
        std = np.asarray(archive["std"], dtype=np.float32).copy()
        feature_indices = tuple(
            int(value)
            for value in np.asarray(archive["feature_indices"]).tolist()
        )
    if source_feature_count != 65:
        raise ValueError("auxiliary model source feature count is not 65")
    if transform_version != 1:
        raise ValueError("auxiliary model context transform is not v1")
    if feature_indices != tuple(range(CONTEXT_FEATURE_COUNT)):
        raise ValueError("auxiliary model does not use the full context")

    queries = read_test_queries(test_csv)
    lift = np.load(
        test_lift_features,
        mmap_mode="r",
        allow_pickle=False,
    )
    store = CausalLiftFeatureStore.from_queries(queries, lift)
    del queries, lift
    gc.collect()
    return CooccurLiftAuxiliaryState(
        integration_id=INTEGRATION_ID,
        weight=LOCKED_WEIGHT,
        hidden_dim=hidden_dim,
        gnn_short_column=gnn_short_column,
        model_state=model_state,
        mean=mean,
        std=std,
        feature_indices=feature_indices,
        feature_store=store,
        provenance=provenance,
    )


def _audit_in_memory_install(
    source: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, str]:
    source_keys = set(source)
    candidate_keys = set(candidate)
    expected_keys = source_keys | {CHECKPOINT_FIELD}
    if candidate_keys != expected_keys:
        raise RuntimeError("Dataset2 install changed unexpected fields")
    for key in source_keys - {CHECKPOINT_FIELD}:
        if candidate[key] is not source[key]:
            raise RuntimeError(
                f"Dataset2 protected field was replaced: {key}"
            )
    if not isinstance(
        candidate.get(CHECKPOINT_FIELD),
        CooccurLiftAuxiliaryState,
    ):
        raise RuntimeError("Dataset2 auxiliary state was not installed")
    return {
        key: _pickle_sha256(value)
        for key, value in source.items()
        if key != CHECKPOINT_FIELD
    }


def _audit_reloaded_dataset2(
    dataset2: dict[str, Any],
    *,
    protected_hashes: dict[str, str],
    auxiliary_state_audit: dict[str, Any],
) -> None:
    if set(dataset2) != set(protected_hashes) | {CHECKPOINT_FIELD}:
        raise RuntimeError("reloaded Dataset2 fields differ")
    for key, expected in protected_hashes.items():
        if _pickle_sha256(dataset2[key]) != expected:
            raise RuntimeError(
                f"reloaded Dataset2 protected field differs: {key}"
            )
    state = dataset2[CHECKPOINT_FIELD]
    if (
        not isinstance(state, CooccurLiftAuxiliaryState)
        or _audit_auxiliary_state(state) != auxiliary_state_audit
    ):
        raise RuntimeError("reloaded Dataset2 auxiliary state differs")


def _audit_auxiliary_state(
    state: CooccurLiftAuxiliaryState,
) -> dict[str, Any]:
    return {
        "integration_id": state.integration_id,
        "weight": float(state.weight),
        "hidden_dim": int(state.hidden_dim),
        "gnn_short_column": int(state.gnn_short_column),
        "model_state_sha256": {
            key: _array_sha256(state.model_state[key])
            for key in sorted(state.model_state)
        },
        "mean_sha256": _array_sha256(state.mean),
        "std_sha256": _array_sha256(state.std),
        "feature_indices": list(state.feature_indices),
        "query_fingerprints_sha256": _array_sha256(
            state.feature_store.query_fingerprints
        ),
        "lift_features_sha256": _array_sha256(
            state.feature_store.lift_features
        ),
        "lift_features_shape": list(
            state.feature_store.lift_features.shape
        ),
        "provenance": dict(sorted(state.provenance.items())),
    }


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(
        json.dumps(list(array.shape), separators=(",", ":")).encode(
            "ascii"
        )
    )
    raw = memoryview(array).cast("B")
    block_size = 4 * 1024 * 1024
    for offset in range(0, raw.nbytes, block_size):
        digest.update(raw[offset : offset + block_size])
    return digest.hexdigest()


def _validate_standard_hydrate(dataset2: dict[str, Any]) -> None:
    import jittor as jt  # noqa: PLC0415

    jt.flags.use_cuda = 1
    ranker = create_ranker("hybrid", None)
    ranker.hydrate(dataset2)
    state = ranker.impl.cooccur_lift_auxiliary_state
    model = ranker.impl.cooccur_lift_auxiliary_model
    if (
        not isinstance(state, CooccurLiftAuxiliaryState)
        or model is None
        or float(state.weight) != LOCKED_WEIGHT
    ):
        raise RuntimeError("standard hydrate omitted cooccur-lift auxiliary")
    del ranker
    gc.collect()


def _pickle_sha256(value: Any) -> str:
    sink = _HashWriter()
    pickle.dump(value, sink, protocol=pickle.HIGHEST_PROTOCOL)
    return sink.hexdigest()


class _HashWriter:
    def __init__(self) -> None:
        self._digest = hashlib.sha256()

    def write(self, value: bytes | pickle.PickleBuffer) -> int:
        view = memoryview(value)
        self._digest.update(view)
        return view.nbytes

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
