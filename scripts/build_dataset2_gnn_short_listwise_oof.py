from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from jgrec.contest_checkpoint import (
    get_model_state,
    load_checkpoint_dataset,
)
from jgrec.core.io import read_interactions
from jgrec.core.memory import release_memory
from jgrec.idmap import NodeIdMap
from jgrec.rankers.common.early_stop import LossEarlyStopper
from jgrec.rankers.hybrid.gnn import (
    GRAPH_WINDOW_FRACTIONS,
    GraphTower,
    _graph_window_data,
    _mapped_edges,
)
from jgrec.rankers.hybrid.gnn_listwise import (
    expanding_oof_folds,
    full_candidate_mrr,
    graph_candidate_logits,
    listwise_positive_loss,
    replace_feature_column,
    validate_candidate_groups,
)

WINDOW_NAME = "gnn_short"
WINDOW_INDEX = 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-free expanding-window OOF gnn_short features "
            "for Dataset2 Setwise training."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument("--train-cache-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--burn-in", type=int, default=25_000)
    parser.add_argument("--fold-size", type=int, default=25_000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--internal-val-ratio", type=float, default=0.1)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.time()

    report = _read_json(args.train_cache_report)
    source_feature_path = Path(f"{args.train_cache_prefix}.train.npy")
    source_features = np.load(
        source_feature_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    candidates = _load_sidecar(args.train_cache_prefix, "candidates")
    src = _load_sidecar(args.train_cache_prefix, "src")
    dst = _load_sidecar(args.train_cache_prefix, "dst")
    event_time = _load_sidecar(args.train_cache_prefix, "time")
    if source_features.shape != (200_000, 100, 63):
        raise ValueError(f"unexpected source feature shape: {source_features.shape}")
    validate_candidate_groups(src, dst, candidates, width=100)
    if not np.all(event_time[1:] >= event_time[:-1]):
        raise ValueError("training cache must be chronological")
    _require_hash(
        source_feature_path,
        report["artifacts"]["features"]["sha256"],
        "training features",
    )
    feature_names = tuple(str(name) for name in report["feature_names"])
    gnn_column = feature_names.index(WINDOW_NAME)
    folds = expanding_oof_folds(
        row_count=len(src),
        burn_in=args.burn_in,
        fold_size=args.fold_size,
    )

    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    graph_config = replace(
        state["config"].graph_config(),
        epochs=args.epochs,
        early_stop_patience=args.patience,
        batch_size=args.batch_size,
    )
    id_map = _node_id_map_from_snapshot(state["id_map"])
    if tuple(str(name) for name in state["feature_names"]) != feature_names:
        raise ValueError("checkpoint and training cache feature schemas differ")
    del state
    gc.collect()

    user_ids = _map_values(src, id_map.src_values)
    positive_ids = _map_values(dst, id_map.dst_values)
    dst_lookup = _lookup_table(id_map.dst_values)
    if np.any(user_ids < 0) or np.any(positive_ids < 0):
        raise ValueError("positive OOF events fall outside checkpoint ID map")
    interactions = read_interactions(args.train_csv).sort_by_time()
    oof_scores_path = args.output_dir / "gnn-short-listwise-oof-scores.npy"
    oof_scores = np.lib.format.open_memmap(
        oof_scores_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(src) - args.burn_in, candidates.shape[1]),
    )

    frozen = {
        "status": "frozen_before_oof_training",
        "seed": args.seed,
        "objective": "complete_100_candidate_group_softmax",
        "selection_metric": "internal_prefix_full100_mrr",
        "row_count": len(src),
        "burn_in": args.burn_in,
        "fold_size": args.fold_size,
        "oof_rows": len(src) - args.burn_in,
        "feature_name": WINDOW_NAME,
        "feature_column": gnn_column,
        "graph_config": asdict(graph_config),
        "folds": [
            {
                "train_rows": list(fold.train_rows),
                "score_rows": list(fold.score_rows),
            }
            for fold in folds
        ],
        "source_feature_sha256": _sha256(source_feature_path),
        "checkpoint_sha256": _sha256(args.checkpoint),
    }
    _write_json(args.output_dir / "frozen-config.json", frozen)
    print(json.dumps(frozen, ensure_ascii=False, sort_keys=True), flush=True)

    jt.flags.use_cuda = 1
    fold_reports: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(folds):
        fold_started = time.time()
        train_start, train_stop = fold.train_rows
        score_start, score_stop = fold.score_rows
        if train_start != 0 or train_stop > score_start:
            raise RuntimeError("OOF chronology contract was violated")
        cutoff_time = int(event_time[train_stop - 1])
        context = interactions[interactions.time <= cutoff_time]
        mapped_edges = _mapped_edges(context, id_map, graph_config)
        edge_count = max(
            1,
            int(len(mapped_edges) * GRAPH_WINDOW_FRACTIONS[WINDOW_INDEX]),
        )
        fold_rng = np.random.default_rng(args.seed + fold_index * 1009)
        edge_index, edge_weight = _graph_window_data(
            mapped_edges[-edge_count:],
            graph_config,
            fold_rng,
            window_name=WINDOW_NAME,
        )
        del mapped_edges, context
        gc.collect()

        jt.set_global_seed(args.seed + fold_index * 1009)
        tower = GraphTower(id_map=id_map, config=graph_config)
        model = tower._build_model(edge_index, edge_weight)
        optimizer = jt.nn.Adam(
            model.parameters(),
            lr=graph_config.lr,
            weight_decay=graph_config.weight_decay,
        )
        internal_val_size = max(
            1,
            int((train_stop - train_start) * args.internal_val_ratio),
        )
        fit_stop = train_stop - internal_val_size
        fit_rows = np.arange(train_start, fit_stop, dtype=np.int32)
        internal_val_rows = np.arange(fit_stop, train_stop, dtype=np.int32)
        stopper = LossEarlyStopper(patience=args.patience)
        history: list[dict[str, float | int]] = []

        for epoch in range(1, args.epochs + 1):
            order = fold_rng.permutation(fit_rows)
            losses: list[float] = []
            for batch_number, batch_start in enumerate(
                range(0, order.shape[0], args.batch_size),
                start=1,
            ):
                row_idx = order[
                    batch_start : batch_start + args.batch_size
                ]
                candidate_ids = _map_with_lookup(
                    np.asarray(candidates[row_idx]),
                    dst_lookup,
                )
                logits = graph_candidate_logits(
                    model,
                    user_ids[row_idx],
                    candidate_ids,
                )
                loss = listwise_positive_loss(logits)
                optimizer.step(loss)
                loss_value = float(loss.item())
                if not np.isfinite(loss_value):
                    raise FloatingPointError(
                        f"fold={fold_index} epoch={epoch} has non-finite loss"
                    )
                losses.append(loss_value)
                if (
                    batch_number == 1
                    or batch_number % args.progress_every == 0
                ):
                    print(
                        f"[oof-fold {fold_index + 1}/{len(folds)}] "
                        f"epoch={epoch} batch={batch_number} "
                        f"loss={loss_value:.6f}",
                        flush=True,
                    )
            val_mrr = _score_mrr(
                model,
                user_ids[internal_val_rows],
                candidates[internal_val_rows],
                dst_lookup,
                batch_size=args.batch_size,
            )
            mean_loss = float(np.mean(losses))
            history.append(
                {
                    "epoch": epoch,
                    "loss": mean_loss,
                    "internal_val_mrr": val_mrr,
                }
            )
            print(
                f"[oof-fold {fold_index + 1}/{len(folds)}] epoch={epoch} "
                f"loss={mean_loss:.6f} internal_val_mrr={val_mrr:.9f}",
                flush=True,
            )
            if stopper.update(epoch, -val_mrr, model):
                print(
                    f"[oof-fold {fold_index + 1}/{len(folds)}] "
                    f"early_stop epoch={epoch} best_epoch={stopper.best_epoch}",
                    flush=True,
                )
                break
            release_memory()
        stopper.restore_best(model)

        fold_scores = _score_rows(
            model,
            user_ids[score_start:score_stop],
            candidates[score_start:score_stop],
            dst_lookup,
            batch_size=args.batch_size,
        )
        destination_start = score_start - args.burn_in
        destination_stop = score_stop - args.burn_in
        oof_scores[destination_start:destination_stop] = fold_scores
        oof_scores.flush()
        fold_model_path = args.output_dir / (
            f"fold-{fold_index:02d}-best-gnn-short.npz"
        )
        np.savez_compressed(
            fold_model_path,
            **get_model_state(model),
        )
        fold_report = {
            "fold": fold_index,
            "train_rows": [train_start, train_stop],
            "fit_rows": [train_start, fit_stop],
            "internal_val_rows": [fit_stop, train_stop],
            "score_rows": [score_start, score_stop],
            "graph_cutoff_time": cutoff_time,
            "graph_edges": int(edge_index.shape[1]),
            "best_epoch": stopper.best_epoch,
            "best_internal_val_mrr": -float(stopper.best_loss),
            "score_mrr": full_candidate_mrr(fold_scores),
            "score_sha256": _sha256_array(fold_scores),
            "model_sha256": _sha256(fold_model_path),
            "history": history,
            "elapsed_seconds": time.time() - fold_started,
        }
        fold_reports.append(fold_report)
        _write_json(
            args.output_dir / "oof-progress.json",
            {
                "status": "training",
                "completed_folds": len(fold_reports),
                "total_folds": len(folds),
                "folds": fold_reports,
                "elapsed_seconds": time.time() - started,
            },
        )
        print(json.dumps(fold_report, sort_keys=True), flush=True)
        del model, tower, optimizer, fold_scores, edge_index, edge_weight
        release_memory()
        gc.collect()

    oof_scores.flush()
    if not np.all(np.isfinite(oof_scores)):
        raise FloatingPointError("published OOF score matrix contains non-finite values")
    oof_feature_path = args.output_dir / "train-oof-gnn-short-listwise.npy"
    replacement_contract = replace_feature_column(
        source_features[args.burn_in :],
        oof_scores,
        column=gnn_column,
        output_path=oof_feature_path,
        batch_rows=128,
    )
    for name, values in (
        ("src", src),
        ("dst", dst),
        ("time", event_time),
        ("candidates", candidates),
    ):
        np.save(
            args.output_dir / f"train-oof-{name}.npy",
            np.asarray(values[args.burn_in :]),
            allow_pickle=False,
        )

    final = {
        "status": "complete",
        "leakage_free": True,
        "fold_count": len(folds),
        "covered_source_rows": [args.burn_in, len(src)],
        "oof_shape": list(oof_scores.shape),
        "oof_score_sha256": _sha256(oof_scores_path),
        "oof_feature_path": str(oof_feature_path.resolve()),
        "oof_feature_sha256": _sha256(oof_feature_path),
        "replacement_contract": replacement_contract,
        "folds": fold_reports,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.output_dir / "oof-build-report.json", final)
    print(json.dumps(final, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def _score_mrr(
    model: Any,
    user_ids: np.ndarray,
    candidates: np.ndarray,
    dst_lookup: np.ndarray,
    *,
    batch_size: int,
) -> float:
    return full_candidate_mrr(
        _score_rows(
            model,
            user_ids,
            candidates,
            dst_lookup,
            batch_size=batch_size,
        )
    )


def _score_rows(
    model: Any,
    user_ids: np.ndarray,
    candidates: np.ndarray,
    dst_lookup: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    candidate_ids = _map_with_lookup(candidates, dst_lookup)
    valid = candidate_ids >= 0
    safe = candidate_ids.clip(min=0)
    with jt.no_grad():
        user_all, item_all = model.get_all_embeddings()
        user_embeddings = np.asarray(user_all.numpy(), dtype=np.float32)
        item_embeddings = np.asarray(item_all.numpy(), dtype=np.float32)
    scores = np.empty(candidate_ids.shape, dtype=np.float32)
    for start in range(0, user_ids.shape[0], batch_size):
        end = min(start + batch_size, user_ids.shape[0])
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


def _lookup_table(raw_values: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(raw_values, dtype=np.int64)
    lookup = np.full(int(values.max()) + 1, -1, dtype=np.int32)
    lookup[values] = np.arange(values.shape[0], dtype=np.int32)
    return lookup


def _map_values(values: np.ndarray, raw_values: tuple[int, ...]) -> np.ndarray:
    return _map_with_lookup(values, _lookup_table(raw_values))


def _map_with_lookup(values: np.ndarray, lookup: np.ndarray) -> np.ndarray:
    raw = np.asarray(values)
    mapped = np.full(raw.shape, -1, dtype=np.int32)
    valid = (raw >= 0) & (raw < lookup.shape[0])
    mapped[valid] = lookup[raw[valid]]
    return mapped


def _load_sidecar(prefix: Path, name: str) -> np.ndarray:
    path = Path(f"{prefix}.train-{name}.npy")
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch: actual={actual} expected={expected}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
