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
    full_candidate_mrr,
    graph_candidate_logits,
    listwise_positive_loss,
    validate_candidate_groups,
)

WINDOW_NAME = "gnn_short"
WINDOW_INDEX = 2
CANDIDATE_WIDTH = 100


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train only Dataset2 gnn_short with recent-200k, complete-100 "
            "candidate groups and a listwise loss."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--train-prefix", required=True, type=Path)
    parser.add_argument("--validation-prefix", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--validation-batch-size", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite GNN experiment: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True)
    started = time.time()

    train = _load_groups(args.train_prefix, "train")
    validation = _load_groups(args.validation_prefix, "val")
    validate_candidate_groups(
        train["src"],
        train["dst"],
        train["candidates"],
        width=CANDIDATE_WIDTH,
    )
    validate_candidate_groups(
        validation["src"],
        validation["dst"],
        validation["candidates"],
        width=CANDIDATE_WIDTH,
    )
    if train["candidates"].shape != (200_000, CANDIDATE_WIDTH):
        raise ValueError(
            "training cache must contain exactly 200000 x 100 candidates"
        )
    if validation["candidates"].shape[1] != CANDIDATE_WIDTH:
        raise ValueError("validation cache must use complete 100-candidate rows")
    if not np.all(train["time"][1:] >= train["time"][:-1]):
        raise ValueError("training cache is not ordered by time")

    checkpoint_state = load_checkpoint_dataset(
        args.checkpoint,
        "dataset2",
    )
    base_config = checkpoint_state["config"].graph_config()
    graph_config = replace(
        base_config,
        enabled=True,
        epochs=args.epochs,
        early_stop_patience=args.patience,
        batch_size=args.batch_size,
    )
    id_map = _node_id_map_from_snapshot(checkpoint_state["id_map"])
    del checkpoint_state
    gc.collect()

    interactions = read_interactions(args.train_csv).sort_by_time()
    cutoff_time = int(train["time"][-1])
    context = interactions[interactions.time <= cutoff_time]
    if len(context) == 0:
        raise ValueError("no graph context exists before the training cutoff")
    mapped_edges = _mapped_edges(context, id_map, graph_config)
    edge_count = max(
        1,
        int(len(mapped_edges) * GRAPH_WINDOW_FRACTIONS[WINDOW_INDEX]),
    )
    window_edges = mapped_edges[-edge_count:]
    rng = np.random.default_rng(args.seed)
    edge_index, edge_weight = _graph_window_data(
        window_edges,
        graph_config,
        rng,
        window_name=WINDOW_NAME,
    )
    if edge_index.shape[1] == 0:
        raise ValueError("gnn_short graph window contains no mapped edges")
    del mapped_edges, window_edges, interactions, context
    gc.collect()

    train_user_ids = _map_values(
        train["src"],
        id_map.src_values,
    )
    validation_user_ids = _map_values(
        validation["src"],
        id_map.src_values,
    )
    if np.any(train_user_ids < 0):
        raise ValueError("training cache contains sources outside checkpoint ID map")
    if np.any(validation_user_ids < 0):
        raise ValueError("validation cache contains sources outside checkpoint ID map")
    dst_lookup = _lookup_table(id_map.dst_values)
    train_positive_ids = _map_with_lookup(train["dst"], dst_lookup)
    validation_positive_ids = _map_with_lookup(
        validation["dst"],
        dst_lookup,
    )
    if np.any(train_positive_ids < 0) or np.any(validation_positive_ids < 0):
        raise ValueError("positive candidates must exist in checkpoint ID map")

    frozen = {
        "status": "preflight_complete",
        "scope": "Dataset2 gnn_short only; no fusion/package generation",
        "objective": "group_softmax_listwise_positive_column_0",
        "selection_metric": "complete_100_candidate_mrr",
        "seed": args.seed,
        "train_rows": int(train["candidates"].shape[0]),
        "validation_rows": int(validation["candidates"].shape[0]),
        "candidate_width": CANDIDATE_WIDTH,
        "train_time_range": [
            int(train["time"][0]),
            int(train["time"][-1]),
        ],
        "validation_time_range": [
            int(validation["time"][0]),
            int(validation["time"][-1]),
        ],
        "graph_context_cutoff": cutoff_time,
        "graph_window": WINDOW_NAME,
        "graph_edges": int(edge_index.shape[1]),
        "graph_config": asdict(graph_config),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256_file(args.checkpoint),
        "train_prefix": str(args.train_prefix.resolve()),
        "validation_prefix": str(args.validation_prefix.resolve()),
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
    }
    _write_json(args.output_dir / "frozen-config.json", frozen)
    print(json.dumps(frozen, ensure_ascii=False, sort_keys=True), flush=True)

    jt.flags.use_cuda = 1
    jt.set_global_seed(args.seed)
    tower = GraphTower(id_map=id_map, config=graph_config)
    model = tower._build_model(edge_index, edge_weight)
    optimizer = jt.nn.Adam(
        model.parameters(),
        lr=graph_config.lr,
        weight_decay=graph_config.weight_decay,
    )
    stopper = LossEarlyStopper(patience=args.patience)
    best_mrr = float("-inf")
    train_rows = train_user_ids.shape[0]

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        order = rng.permutation(train_rows)
        losses: list[float] = []
        for batch_number, start in enumerate(
            range(0, train_rows, args.batch_size),
            start=1,
        ):
            row_idx = order[start : start + args.batch_size]
            candidate_ids = _map_with_lookup(
                np.asarray(train["candidates"][row_idx]),
                dst_lookup,
            )
            if not np.array_equal(
                candidate_ids[:, 0],
                train_positive_ids[row_idx],
            ):
                raise RuntimeError("mapped training positive moved from column 0")
            logits = graph_candidate_logits(
                model,
                train_user_ids[row_idx],
                candidate_ids,
            )
            loss = listwise_positive_loss(logits)
            optimizer.step(loss)
            loss_value = float(loss.item())
            if not np.isfinite(loss_value):
                raise FloatingPointError(
                    f"non-finite listwise loss at epoch={epoch} batch={batch_number}"
                )
            losses.append(loss_value)
            if (
                batch_number == 1
                or batch_number % args.progress_every == 0
            ):
                print(
                    f"[gnn:{WINDOW_NAME}:listwise] epoch={epoch} "
                    f"batch={batch_number} rows={min(start + args.batch_size, train_rows)}"
                    f"/{train_rows} loss={loss_value:.6f}",
                    flush=True,
                )
            if batch_number % args.progress_every == 0:
                release_memory()

        val_mrr = _validation_mrr(
            model=model,
            user_ids=validation_user_ids,
            positive_ids=validation_positive_ids,
            candidates=validation["candidates"],
            dst_lookup=dst_lookup,
            batch_size=args.validation_batch_size,
        )
        mean_loss = float(np.mean(losses))
        if not np.isfinite(val_mrr):
            raise FloatingPointError("validation MRR is non-finite")
        print(
            f"[gnn:{WINDOW_NAME}:listwise] epoch={epoch} "
            f"loss={mean_loss:.6f} full100_val_mrr={val_mrr:.9f} "
            f"elapsed_seconds={time.time() - epoch_started:.1f}",
            flush=True,
        )
        best_mrr = max(best_mrr, val_mrr)
        _write_json(
            args.output_dir / "progress.json",
            {
                "status": "training",
                "epoch": epoch,
                "loss": mean_loss,
                "full100_val_mrr": val_mrr,
                "best_full100_val_mrr": best_mrr,
                "elapsed_seconds": time.time() - started,
            },
        )
        if stopper.update(epoch, -val_mrr, model):
            print(
                f"[gnn:{WINDOW_NAME}:listwise] early_stop epoch={epoch} "
                f"best_epoch={stopper.best_epoch} "
                f"best_full100_val_mrr={-stopper.best_loss:.9f}",
                flush=True,
            )
            break
        release_memory()

    stopper.restore_best(model)
    model_path = args.output_dir / "best-gnn-short-listwise.npz"
    np.savez_compressed(model_path, **get_model_state(model))
    final = {
        "status": "complete",
        "best_epoch": stopper.best_epoch,
        "best_full100_val_mrr": -float(stopper.best_loss),
        "model_path": str(model_path.resolve()),
        "model_sha256": _sha256_file(model_path),
        "elapsed_seconds": time.time() - started,
        "package_generated": False,
    }
    _write_json(args.output_dir / "training-report.json", final)
    print(json.dumps(final, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def _load_groups(prefix: Path, split: str) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for name in ("candidates", "src", "dst", "time"):
        path = Path(f"{prefix}.{split}-{name}.npy")
        if not path.is_file():
            raise FileNotFoundError(path)
        values[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    return values


def _node_id_map_from_snapshot(snapshot: dict[str, Any]) -> NodeIdMap:
    src_values = tuple(int(value) for value in snapshot["src_values"])
    dst_values = tuple(int(value) for value in snapshot["dst_values"])
    return NodeIdMap(
        src_to_id={
            value: index for index, value in enumerate(src_values)
        },
        dst_to_id={
            value: index for index, value in enumerate(dst_values)
        },
        src_values=src_values,
        dst_values=dst_values,
    )


def _lookup_table(raw_values: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(raw_values, dtype=np.int64)
    if values.size == 0 or np.any(values < 0):
        raise ValueError("ID-map values must be non-empty non-negative integers")
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


def _validation_mrr(
    *,
    model: Any,
    user_ids: np.ndarray,
    positive_ids: np.ndarray,
    candidates: np.ndarray,
    dst_lookup: np.ndarray,
    batch_size: int,
) -> float:
    reciprocal_rank_sum = 0.0
    rows = 0
    with jt.no_grad():
        user_all, item_all = model.get_all_embeddings()
        user_embeddings = np.asarray(user_all.numpy(), dtype=np.float32)
        item_embeddings = np.asarray(item_all.numpy(), dtype=np.float32)
    for start in range(0, user_ids.shape[0], batch_size):
        end = min(start + batch_size, user_ids.shape[0])
        candidate_ids = _map_with_lookup(
            np.asarray(candidates[start:end]),
            dst_lookup,
        )
        if not np.array_equal(
            candidate_ids[:, 0],
            positive_ids[start:end],
        ):
            raise RuntimeError("mapped validation positive moved from column 0")
        valid = candidate_ids >= 0
        safe = candidate_ids.clip(min=0)
        scores = np.sum(
            item_embeddings[safe]
            * user_embeddings[user_ids[start:end], None, :],
            axis=-1,
            dtype=np.float32,
        )
        scores[~valid] = 0.0
        batch_rows = end - start
        reciprocal_rank_sum += full_candidate_mrr(scores) * batch_rows
        rows += batch_rows
    return reciprocal_rank_sum / rows


def _sha256_file(path: Path) -> str:
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
