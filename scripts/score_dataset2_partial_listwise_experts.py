from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import jittor as jt
import lightgbm as lgb
import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.core.io import read_interactions
from jgrec.core.types import TestQueryArray
from jgrec.idmap import NodeIdMap
from jgrec.rankers.common.temporal_index import TemporalInteractionIndex
from jgrec.rankers.hybrid.config import TwoTowerConfig
from jgrec.rankers.hybrid.fusion import (
    build_fusion_from_state,
    predict_logits,
)
from jgrec.rankers.hybrid.partial_listwise_blend import (
    descending_midrank_probabilities,
    ranking_mrr,
)
from jgrec.rankers.hybrid.setwise import setwise_context_features
from jgrec.rankers.hybrid.two_tower import TwoTower

EXPECTED_CHAMPION = {
    "full": 0.5485470648527594,
    "slice_0": 0.5882028774417708,
    "slice_1": 0.5493313411199712,
    "slice_2": 0.5081009093765456,
}
SLICES = ((0, 6667), (6667, 13334), (13334, 20000))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Score the frozen Dataset2 champion and two listwise auxiliary "
            "experts on one aligned 20k x 100 validation manifest."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--validation-cache-prefix", required=True, type=Path)
    parser.add_argument("--validation-cache-report", required=True, type=Path)
    parser.add_argument("--short-none-scores", required=True, type=Path)
    parser.add_argument("--listwise-mlp-model", required=True, type=Path)
    parser.add_argument("--listwise-two-tower-model", required=True, type=Path)
    parser.add_argument("--two-tower-report", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--frozen-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    output_paths = {
        "champion": args.output_dir / "champion-probabilities.npy",
        "listwise_mlp": args.output_dir / "listwise-mlp-probabilities.npy",
        "listwise_two_tower": (
            args.output_dir / "listwise-two-tower-probabilities.npy"
        ),
        "report": args.output_dir / "score-report.json",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            f"refusing to overwrite score artifacts: {existing}"
        )
    frozen = _read_json(args.frozen_config)
    if frozen.get("status") != "frozen_before_any_partial_blend_metric":
        raise ValueError("partial-blend config was not frozen before scoring")
    validation_report = _read_json(args.validation_cache_report)
    if validation_report.get("status") != "complete":
        raise ValueError("validation cache report is incomplete")

    prefix = str(args.validation_cache_prefix)
    feature_path = Path(f"{prefix}.val.npy")
    candidate_path = Path(f"{prefix}.val-candidates.npy")
    src_path = Path(f"{prefix}.val-src.npy")
    dst_path = Path(f"{prefix}.val-dst.npy")
    time_path = Path(f"{prefix}.val-time.npy")
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    candidates = np.load(
        candidate_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    query_src = np.load(src_path, mmap_mode="r", allow_pickle=False)
    query_dst = np.load(dst_path, mmap_mode="r", allow_pickle=False)
    query_time = np.load(time_path, mmap_mode="r", allow_pickle=False)
    short_none = np.load(
        args.short_none_scores,
        mmap_mode="r",
        allow_pickle=False,
    )
    expected_shape = (20_000, 100)
    if features.shape != (*expected_shape, 63):
        raise ValueError(f"unexpected validation feature shape: {features.shape}")
    if candidates.shape != expected_shape or short_none.shape != expected_shape:
        raise ValueError("candidate or short-none score shape differs")
    if not np.array_equal(candidates[:, 0], query_dst):
        raise ValueError("candidate zero differs from validation destination")
    for name, path in {
        "checkpoint": args.checkpoint,
        "listwise_mlp": args.listwise_mlp_model,
        "listwise_two_tower": args.listwise_two_tower_model,
        "validation_candidates": candidate_path,
        "validation_features": feature_path,
    }.items():
        expected = frozen["hashes"][name]
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(f"{name} hash mismatch: {actual} != {expected}")

    started = time.time()
    jt.flags.use_cuda = 1
    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    feature_names = tuple(str(name) for name in state["feature_names"])
    if feature_names != tuple(validation_report["feature_names"]):
        raise ValueError("checkpoint and validation feature schemas differ")
    gnn_column = feature_names.index("gnn_short")
    setwise_result = state.get("setwise_fusion_result")
    setwise_state = state.get("setwise_fusion_state")
    if setwise_result is None or setwise_state is None:
        raise ValueError("current champion has no persisted Setwise head")
    setwise_model = build_fusion_from_state(
        input_dim=len(setwise_result.feature_indices),
        hidden_dim=int(state["setwise_hidden_dim"]),
        state=setwise_state,
    )
    lgbm_result = state.get("lgbm_result")
    if lgbm_result is None:
        raise ValueError("current champion has no LightGBM expert")
    outer_weight = float(lgbm_result.mlp_weight)
    if abs(outer_weight - 0.80) > 1e-12:
        raise ValueError("current champion outer Setwise weight is not 0.80")
    booster = lgb.Booster(model_str=str(lgbm_result.model_text))
    lgbm_indices = tuple(int(index) for index in lgbm_result.feature_indices)

    mlp_payload = np.load(args.listwise_mlp_model, allow_pickle=False)
    mlp_indices = tuple(
        int(value) for value in mlp_payload["feature_indices"]
    )
    mlp_state = {
        key.removeprefix("state__"): np.asarray(
            mlp_payload[key],
            dtype=np.float32,
        )
        for key in mlp_payload.files
        if key.startswith("state__")
    }
    mlp_hidden_dim = int(mlp_state["linear1.bias"].shape[0])
    mlp_model = build_fusion_from_state(
        input_dim=len(mlp_indices),
        hidden_dim=mlp_hidden_dim,
        state=mlp_state,
    )
    mlp_mean = np.asarray(mlp_payload["mean"], dtype=np.float32)
    mlp_std = np.asarray(mlp_payload["std"], dtype=np.float32)
    mlp_payload.close()

    champion = np.empty(expected_shape, dtype=np.float64)
    listwise_mlp = np.empty(expected_shape, dtype=np.float64)
    for start in range(0, expected_shape[0], args.batch_size):
        stop = min(start + args.batch_size, expected_shape[0])
        source_batch = np.asarray(features[start:stop], dtype=np.float32)
        batch = source_batch.copy()
        batch[..., gnn_column] = short_none[start:stop]
        setwise_features = setwise_context_features(batch)
        setwise_indices = tuple(
            int(index) for index in setwise_result.feature_indices
        )
        if setwise_indices != tuple(range(setwise_features.shape[-1])):
            setwise_features = setwise_features[..., setwise_indices]
        setwise_probabilities = _softmax(
            predict_logits(
                setwise_model,
                setwise_features,
                setwise_result.mean,
                setwise_result.std,
            )
        )
        lgbm_features = _select_columns(source_batch, lgbm_indices)
        flat = np.ascontiguousarray(
            lgbm_features.reshape(-1, lgbm_features.shape[-1]),
            dtype=np.float32,
        )
        lgbm_logits = booster.predict(flat).reshape(batch.shape[:2])
        lgbm_probabilities = _softmax(lgbm_logits)
        champion[start:stop] = (
            outer_weight * setwise_probabilities
            + (1.0 - outer_weight) * lgbm_probabilities
        )
        mlp_features = _select_columns(batch, mlp_indices)
        listwise_mlp[start:stop] = _softmax(
            predict_logits(
                mlp_model,
                mlp_features,
                mlp_mean,
                mlp_std,
            )
        )
        if start % max(args.batch_size * 10, 1) == 0:
            print(
                f"[partial-listwise-score] fusion rows={stop}/{expected_shape[0]}",
                flush=True,
            )

    actual_champion = _metrics(champion)
    _require_metrics(actual_champion, EXPECTED_CHAMPION)
    np.save(output_paths["champion"], champion, allow_pickle=False)
    np.save(output_paths["listwise_mlp"], listwise_mlp, allow_pickle=False)

    del state, setwise_model, mlp_model, booster
    gc.collect()
    jt.sync_all()
    jt.clean()

    two_tower_report = _read_json(args.two_tower_report)
    tower_config_values = two_tower_report["candidate"]["config"]
    tower_config = TwoTowerConfig(**tower_config_values)
    interactions = read_interactions(args.train_csv).sort_by_time()
    context_end = int(
        two_tower_report["frozen_config"]["context_end"]
    )
    train_prefix = interactions[:context_end]
    id_map = NodeIdMap.from_interactions(train_prefix)
    model_payload = np.load(
        args.listwise_two_tower_model,
        allow_pickle=False,
    )
    model_state = {
        key: np.asarray(model_payload[key], dtype=np.float32)
        for key in model_payload.files
    }
    model_payload.close()
    expected_src = int(model_state["src_id_embedding.weight"].shape[0])
    expected_dst = int(model_state["dst_id_embedding.weight"].shape[0])
    actual_embedding_shape = (
        id_map.num_src + 1,
        id_map.num_dst + 1,
    )
    if actual_embedding_shape != (expected_src, expected_dst):
        raise ValueError(
            "Two-Tower id map differs from saved embedding state: "
            f"{actual_embedding_shape} != "
            f"{(expected_src, expected_dst)}"
        )
    index = TemporalInteractionIndex()
    index.fit(
        train_prefix,
        build_transitions=False,
        build_cooccurs=False,
    )
    min_time = int(train_prefix.time[0])
    max_time = int(train_prefix.time[-1])
    graph_span = max(max_time - min_time, 1)
    tower = TwoTower(id_map=id_map, config=tower_config)
    tower.hydrate(
        {
            "model_state": model_state,
            "index": index,
            "min_time": min_time,
            "max_time": max_time,
            "graph_span": graph_span,
        }
    )
    queries = TestQueryArray(
        src=np.asarray(query_src, dtype=np.int32),
        time=np.asarray(query_time, dtype=np.int64),
        candidates=np.asarray(candidates, dtype=np.int32),
    )
    raw_two_tower = tower.scores_for_query_array(queries)[:, :, 0]
    listwise_two_tower = descending_midrank_probabilities(raw_two_tower)
    np.save(
        output_paths["listwise_two_tower"],
        listwise_two_tower,
        allow_pickle=False,
    )

    score_hashes = {
        name: _sha256_file(path)
        for name, path in output_paths.items()
        if name != "report"
    }
    report = {
        "status": "passed",
        "selection_metrics_read": False,
        "candidate_manifest_sha256": _sha256_file(candidate_path),
        "candidate_shape": list(candidates.shape),
        "champion": actual_champion,
        "champion_reproduced": True,
        "expert_diagnostics": {
            "listwise_mlp_full_mrr": ranking_mrr(listwise_mlp),
            "listwise_two_tower_full_mrr": ranking_mrr(
                listwise_two_tower
            ),
            "listwise_two_tower_raw_full_mrr": ranking_mrr(raw_two_tower),
        },
        "two_tower_context": {
            "context_end": context_end,
            "id_map_num_src": id_map.num_src,
            "id_map_num_dst": id_map.num_dst,
            "min_time": min_time,
            "max_time": max_time,
            "graph_span": graph_span,
        },
        "score_artifacts": {
            name: {
                "path": str(path.resolve()),
                "sha256": score_hashes[name],
                "shape": list(np.load(path, mmap_mode="r").shape),
            }
            for name, path in output_paths.items()
            if name != "report"
        },
        "elapsed_seconds": time.time() - started,
    }
    _write_json(output_paths["report"], report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _metrics(scores: np.ndarray) -> dict[str, float]:
    result = {"full": ranking_mrr(scores)}
    for index, (start, stop) in enumerate(SLICES):
        result[f"slice_{index}"] = ranking_mrr(scores[start:stop])
    return result


def _require_metrics(
    actual: dict[str, float],
    expected: dict[str, float],
) -> None:
    for name, value in expected.items():
        if abs(actual[name] - value) > 1e-12:
            raise RuntimeError(
                f"champion reproduction failed for {name}: "
                f"{actual[name]} != {value}"
            )


def _select_columns(
    features: np.ndarray,
    indices: tuple[int, ...],
) -> np.ndarray:
    if indices == tuple(range(features.shape[-1])):
        return features
    return features[..., indices]


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
