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

from jgrec.contest_checkpoint import (
    load_checkpoint_dataset,
    set_model_state,
)
from jgrec.core.io import read_interactions
from jgrec.core.types import TestQueryArray
from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid.full100_training import (
    passes_full100_gate,
    validate_joint_cache_reports,
)
from jgrec.rankers.hybrid.fusion import (
    FusionConfig,
    fit_fusion_mlp_listwise_streaming,
)
from jgrec.rankers.hybrid.fusion_analysis import ranking_mrr_three_slices
from jgrec.rankers.hybrid.fusion_lgbm import predict_logits_lgbm
from jgrec.rankers.hybrid.gnn import (
    GRAPH_WINDOW_FRACTIONS,
    GraphTower,
    _graph_window_data,
    _mapped_edges,
)
from jgrec.rankers.hybrid.gnn_listwise import (
    full_candidate_mrr,
    replace_feature_column,
    validate_candidate_groups,
)
from jgrec.rankers.hybrid.setwise import SetwiseFeatureView

WINDOW_NAME = "gnn_short"
WINDOW_INDEX = 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replace only Dataset2 validation gnn_short with listwise-GNN "
            "scores, retrain Setwise fusion, and compare with the champion."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--gnn-model", required=True, type=Path)
    parser.add_argument("--gnn-training-report", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument("--train-cache-report", required=True, type=Path)
    parser.add_argument("--oof-training-features", type=Path)
    parser.add_argument("--oof-training-report", type=Path)
    parser.add_argument("--validation-cache-prefix", required=True, type=Path)
    parser.add_argument("--validation-cache-report", required=True, type=Path)
    parser.add_argument("--champion-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--fusion-batch-size", type=int, default=256)
    parser.add_argument("--fusion-epochs", type=int, default=10)
    parser.add_argument("--fusion-patience", type=int, default=2)
    parser.add_argument("--setwise-weight", type=float, default=0.80)
    parser.add_argument("--min-full-delta", type=float, default=0.002)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.time()

    train_report = _read_json(args.train_cache_report)
    validation_report = _read_json(args.validation_cache_report)
    champion_report = _read_json(args.champion_report)
    joint_contract = validate_joint_cache_reports(
        train_report,
        validation_report,
    )
    train_path = Path(f"{args.train_cache_prefix}.train.npy")
    validation_path = Path(f"{args.validation_cache_prefix}.val.npy")
    source_train_features = np.load(
        train_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    validation_features = np.load(
        validation_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    if source_train_features.shape != (200_000, 100, 63):
        raise ValueError(
            f"unexpected training shape: {source_train_features.shape}"
        )
    if validation_features.shape != (20_000, 100, 63):
        raise ValueError(
            f"unexpected validation shape: {validation_features.shape}"
        )
    _require_hash(
        train_path,
        train_report["artifacts"]["features"]["sha256"],
        "training cache",
    )
    if (args.oof_training_features is None) != (
        args.oof_training_report is None
    ):
        raise ValueError(
            "OOF training features and report must be provided together"
        )
    if args.oof_training_features is None:
        train_features = source_train_features
        training_feature_policy = (
            "keep leakage-safe historical champion gnn_short training feature"
        )
        oof_training_contract = None
    else:
        oof_report = _read_json(args.oof_training_report)
        if oof_report.get("status") != "complete":
            raise ValueError("OOF training feature report is incomplete")
        if not bool(oof_report.get("leakage_free")):
            raise ValueError("OOF training feature report is not leakage-free")
        _require_hash(
            args.oof_training_features,
            oof_report["oof_feature_sha256"],
            "OOF training features",
        )
        train_features = np.load(
            args.oof_training_features,
            mmap_mode="r",
            allow_pickle=False,
        )
        if train_features.shape != (175_000, 100, 63):
            raise ValueError(
                f"unexpected OOF training shape: {train_features.shape}"
            )
        training_feature_policy = (
            "chronological expanding-window OOF gnn_short rows 25000:200000"
        )
        oof_training_contract = {
            "features": str(args.oof_training_features.resolve()),
            "features_sha256": _sha256(args.oof_training_features),
            "report": str(args.oof_training_report.resolve()),
            "report_sha256": _sha256(args.oof_training_report),
            "covered_source_rows": oof_report["covered_source_rows"],
            "fold_count": oof_report["fold_count"],
            "leakage_free": oof_report["leakage_free"],
        }
    _require_hash(
        validation_path,
        validation_report["artifacts"]["features"]["sha256"],
        "validation cache",
    )

    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    config = state["config"]
    graph_config = config.graph_config()
    feature_names = tuple(str(name) for name in state["feature_names"])
    if feature_names != tuple(validation_report["feature_names"]):
        raise ValueError("checkpoint and validation feature schemas differ")
    gnn_column = feature_names.index(WINDOW_NAME)
    lgbm_result = state.get("lgbm_result")
    if lgbm_result is None:
        raise ValueError("champion checkpoint has no Dataset2 LightGBM")
    id_map = _node_id_map_from_snapshot(state["id_map"])

    train_src = _load_sidecar(args.train_cache_prefix, "src", split="train")
    train_dst = _load_sidecar(args.train_cache_prefix, "dst", split="train")
    train_time = _load_sidecar(args.train_cache_prefix, "time", split="train")
    train_candidates = _load_sidecar(
        args.train_cache_prefix,
        "candidates",
        split="train",
    )
    validation_src = _load_sidecar(
        args.validation_cache_prefix,
        "src",
        split="val",
    )
    validation_dst = _load_sidecar(
        args.validation_cache_prefix,
        "dst",
        split="val",
    )
    validation_time = _load_sidecar(
        args.validation_cache_prefix,
        "time",
        split="val",
    )
    validation_candidates = _load_sidecar(
        args.validation_cache_prefix,
        "candidates",
        split="val",
    )
    validate_candidate_groups(
        train_src,
        train_dst,
        train_candidates,
        width=100,
    )
    validate_candidate_groups(
        validation_src,
        validation_dst,
        validation_candidates,
        width=100,
    )
    validation_queries = TestQueryArray(
        src=validation_src,
        time=validation_time,
        candidates=validation_candidates,
    )

    interactions = read_interactions(args.train_csv).sort_by_time()
    cutoff_time = int(train_time[-1])
    context = interactions[interactions.time <= cutoff_time]
    mapped_edges = _mapped_edges(context, id_map, graph_config)
    edge_count = max(
        1,
        int(len(mapped_edges) * GRAPH_WINDOW_FRACTIONS[WINDOW_INDEX]),
    )
    rng = np.random.default_rng(args.seed)
    edge_index, edge_weight = _graph_window_data(
        mapped_edges[-edge_count:],
        graph_config,
        rng,
        window_name=WINDOW_NAME,
    )
    del mapped_edges, context, interactions
    gc.collect()

    jt.flags.use_cuda = 1
    jt.set_global_seed(args.seed)
    tower = GraphTower(id_map=id_map, config=graph_config)
    model = tower._build_model(edge_index, edge_weight)
    with np.load(args.gnn_model, allow_pickle=False) as archive:
        set_model_state(
            model,
            {name: np.asarray(archive[name]) for name in archive.files},
        )
    new_gnn_scores = _score_queries(
        model,
        validation_queries,
        id_map=id_map,
        batch_size=args.score_batch_size,
    )
    if not np.all(np.isfinite(new_gnn_scores)):
        raise FloatingPointError("listwise GNN validation scores are non-finite")
    standalone_mrr = full_candidate_mrr(new_gnn_scores)
    expected_standalone_mrr = float(
        _read_json(args.gnn_training_report)["best_full100_val_mrr"]
    )
    if abs(standalone_mrr - expected_standalone_mrr) > 1e-9:
        raise ValueError(
            "reconstructed GNN score MRR differs from its training report: "
            f"{standalone_mrr} != {expected_standalone_mrr}"
        )
    del model, tower
    gc.collect()

    replacement_path = args.output_dir / "validation-gnn-short-listwise.npy"
    replacement_contract = replace_feature_column(
        validation_features,
        new_gnn_scores,
        column=gnn_column,
        output_path=replacement_path,
        batch_rows=256,
    )
    candidate_validation = np.load(
        replacement_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    if not np.array_equal(
        candidate_validation[..., gnn_column],
        new_gnn_scores,
    ):
        raise RuntimeError("published validation cache has wrong gnn_short values")

    champion = {
        key: float(value)
        for key, value in champion_report["setwise"]["fixed_blend"].items()
    }
    frozen = {
        "status": "frozen_before_fusion_training",
        "scope": (
            "validation gnn_short replacement + Dataset2 Setwise retraining; "
            "training cache unchanged; no package"
        ),
        "seed": args.seed,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "gnn_model": str(args.gnn_model.resolve()),
        "gnn_model_sha256": _sha256(args.gnn_model),
        "source_train_cache_sha256": _sha256(train_path),
        "fusion_train_shape": list(train_features.shape),
        "oof_training_contract": oof_training_contract,
        "original_validation_cache_sha256": _sha256(validation_path),
        "replacement_validation_cache": str(replacement_path.resolve()),
        "replacement_validation_cache_sha256": _sha256(replacement_path),
        "replacement_contract": replacement_contract,
        "joint_cache_contract": joint_contract,
        "feature_name": WINDOW_NAME,
        "feature_column": gnn_column,
        "standalone_full100_mrr": standalone_mrr,
        "champion": champion,
        "setwise_weight": args.setwise_weight,
        "minimum_full_delta": args.min_full_delta,
        "all_three_slices_non_decreasing": True,
        "training_feature_policy": training_feature_policy,
    }
    _write_json(args.output_dir / "frozen-config.json", frozen)
    print(json.dumps(frozen, ensure_ascii=False, sort_keys=True), flush=True)

    candidate_lgbm = _softmax(
        predict_logits_lgbm(lgbm_result.model_text, candidate_validation)
    )
    fusion_config = FusionConfig(
        epochs=args.fusion_epochs,
        batch_size=args.fusion_batch_size,
        lr=0.001,
        weight_decay=0.0,
        hidden_dim=32,
        selection_metric="mrr",
        early_stop_patience=args.fusion_patience,
    )
    train_view = SetwiseFeatureView(train_features)
    validation_view = SetwiseFeatureView(candidate_validation)
    setwise_model, setwise_result, history = fit_fusion_mlp_listwise_streaming(
        train_view,
        validation_view,
        fusion_config,
        np.random.default_rng(args.seed),
        verbose=True,
        feature_indices=tuple(range(train_view.shape[-1])),
        candidate_name="dataset2_gnn_short_listwise_setwise",
    )
    model_path = args.output_dir / "dataset2-setwise-gnn-short-listwise.npz"
    _save_setwise_model(
        model_path,
        result=setwise_result,
        hidden_dim=fusion_config.hidden_dim,
        source_feature_count=train_features.shape[-1],
    )
    setwise_logits = _predict_streaming(
        setwise_model,
        validation_view,
        setwise_result.mean,
        setwise_result.std,
        setwise_result.feature_indices,
        args.fusion_batch_size,
    )
    setwise_probabilities = _softmax(setwise_logits)
    candidate_blend = (
        args.setwise_weight * setwise_probabilities
        + (1.0 - args.setwise_weight) * candidate_lgbm
    )
    candidate = ranking_mrr_three_slices(candidate_blend)
    delta = _metric_delta(candidate, champion)
    gate_passed = passes_full100_gate(
        baseline_full_mrr=champion["full"],
        candidate_full_mrr=candidate["full"],
        baseline_slice_mrrs=tuple(
            champion[f"slice_{index}"] for index in range(3)
        ),
        candidate_slice_mrrs=tuple(
            candidate[f"slice_{index}"] for index in range(3)
        ),
        min_full_delta=args.min_full_delta,
    )
    report = {
        "status": "passed" if gate_passed else "rejected",
        "gate_passed": gate_passed,
        "final_test_integration_authorized": gate_passed,
        "package_generated": False,
        "champion": champion,
        "candidate": candidate,
        "delta_vs_champion": delta,
        "candidate_setwise_expert": ranking_mrr_three_slices(
            setwise_probabilities
        ),
        "candidate_lightgbm_expert": ranking_mrr_three_slices(candidate_lgbm),
        "setwise_best_val_mrr": setwise_result.best_val_mrr,
        "setwise_best_val_ap": setwise_result.best_val_ap,
        "setwise_history": list(history),
        "setwise_model": str(model_path.resolve()),
        "setwise_model_sha256": _sha256(model_path),
        "replacement_validation_cache": str(replacement_path.resolve()),
        "replacement_validation_cache_sha256": _sha256(replacement_path),
        "gate": {
            "minimum_full_delta": args.min_full_delta,
            "full_delta_passed": delta["full"] >= args.min_full_delta,
            "all_three_slices_non_decreasing": all(
                delta[f"slice_{index}"] >= 0.0 for index in range(3)
            ),
        },
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.output_dir / "evaluation-report.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if gate_passed else 2


def _score_queries(
    model: Any,
    queries: TestQueryArray,
    *,
    id_map: NodeIdMap,
    batch_size: int,
) -> np.ndarray:
    user_ids = id_map.src_ids(queries.src)
    candidate_ids = id_map.dst_ids(queries.candidates)
    if np.any(user_ids < 0):
        raise ValueError("validation source is outside the checkpoint ID map")
    valid = candidate_ids >= 0
    safe = candidate_ids.clip(min=0)
    with jt.no_grad():
        user_all, item_all = model.get_all_embeddings()
        user_embeddings = np.asarray(user_all.numpy(), dtype=np.float32)
        item_embeddings = np.asarray(item_all.numpy(), dtype=np.float32)
    scores = np.empty(candidate_ids.shape, dtype=np.float32)
    for start in range(0, len(queries), batch_size):
        end = min(start + batch_size, len(queries))
        scores[start:end] = np.sum(
            item_embeddings[safe[start:end]]
            * user_embeddings[user_ids[start:end], None, :],
            axis=-1,
            dtype=np.float32,
        )
        scores[start:end][~valid[start:end]] = 0.0
    return scores


def _node_id_map_from_snapshot(snapshot: dict[str, Any]) -> NodeIdMap:
    src_values = tuple(int(value) for value in snapshot["src_values"])
    dst_values = tuple(int(value) for value in snapshot["dst_values"])
    return NodeIdMap(
        src_to_id={value: index for index, value in enumerate(src_values)},
        dst_to_id={value: index for index, value in enumerate(dst_values)},
        src_values=src_values,
        dst_values=dst_values,
    )


def _load_sidecar(prefix: Path, name: str, *, split: str) -> np.ndarray:
    path = Path(f"{prefix}.{split}-{name}.npy")
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _predict_streaming(
    model: Any,
    features: Any,
    mean: np.ndarray,
    std: np.ndarray,
    feature_indices: tuple[int, ...],
    batch_size: int,
) -> np.ndarray:
    scores = np.empty(features.shape[:2], dtype=np.float32)
    with jt.no_grad():
        for start in range(0, features.shape[0], batch_size):
            end = min(start + batch_size, features.shape[0])
            selected = np.asarray(features[start:end], dtype=np.float32)
            selected = selected[..., feature_indices]
            normalized = ((selected - mean) / std).astype(
                np.float32,
                copy=False,
            )
            scores[start:end] = np.asarray(
                model(jt.array(normalized, dtype=jt.float32)).numpy(),
                dtype=np.float32,
            )
    return scores


def _save_setwise_model(
    path: Path,
    *,
    result: Any,
    hidden_dim: int,
    source_feature_count: int,
) -> None:
    payload = {
        "mean": np.asarray(result.mean, dtype=np.float32),
        "std": np.asarray(result.std, dtype=np.float32),
        "feature_indices": np.asarray(result.feature_indices, dtype=np.int32),
        "hidden_dim": np.asarray([hidden_dim], dtype=np.int32),
        "source_feature_count": np.asarray(
            [source_feature_count],
            dtype=np.int32,
        ),
        "context_transform_version": np.asarray([1], dtype=np.int32),
    }
    payload.update(
        {
            f"state__{key}": np.asarray(value, dtype=np.float32)
            for key, value in result.state.items()
        }
    )
    np.savez_compressed(path, **payload)


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _metric_delta(
    candidate: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, float]:
    return {
        key: float(candidate[key] - baseline[key])
        for key in ("full", "slice_0", "slice_1", "slice_2")
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch: actual={actual} expected={expected}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
