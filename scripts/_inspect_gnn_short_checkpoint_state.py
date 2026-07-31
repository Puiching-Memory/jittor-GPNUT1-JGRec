from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset


def main() -> None:
    checkpoint = Path(sys.argv[1])
    state = load_checkpoint_dataset(checkpoint, "dataset2")
    config = state["config"]
    graph = state["encoder"]["graph"]
    feature_names = tuple(str(name) for name in state["feature_names"])
    fusion_result = state.get("fusion_result")
    setwise_result = state.get("setwise_fusion_result")
    lgbm_result = state.get("lgbm_result")

    payload = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "state_keys": sorted(state),
        "config": {
            "seed": int(config.seed),
            "gnn_epochs": int(config.gnn_epochs),
            "gnn_max_train_edges": int(config.gnn_max_train_edges),
            "gnn_edge_weighting": str(config.gnn_edge_weighting),
            "selection_metric": str(config.selection_metric),
            "fusion_mode": str(config.fusion_mode),
        },
        "feature_count": len(feature_names),
        "gnn_short_column": feature_names.index("gnn_short"),
        "id_map": {
            "src_count": len(state["id_map"]["src_values"]),
            "dst_count": len(state["id_map"]["dst_values"]),
        },
        "graph_windows": {
            name: {
                "user_shape": list(
                    np.asarray(graph["user_embeddings"][name]).shape
                ),
                "item_shape": list(
                    np.asarray(graph["item_embeddings"][name]).shape
                ),
                "user_sha256": _sha256_array(
                    graph["user_embeddings"][name]
                ),
                "item_sha256": _sha256_array(
                    graph["item_embeddings"][name]
                ),
            }
            for name in sorted(graph["user_embeddings"])
        },
        "fusion": _fusion_summary(
            fusion_result,
            state.get("fusion_state"),
            int(state["fusion_hidden_dim"]),
        ),
        "setwise": _fusion_summary(
            setwise_result,
            state.get("setwise_fusion_state"),
            int(state.get("setwise_hidden_dim", 64)),
        ),
        "lgbm": {
            "present": lgbm_result is not None,
            "candidate_name": (
                None
                if lgbm_result is None
                else str(lgbm_result.candidate_name)
            ),
            "mlp_weight": (
                None
                if lgbm_result is None
                else float(lgbm_result.mlp_weight)
            ),
            "model_text_sha256": (
                None
                if lgbm_result is None
                else hashlib.sha256(
                    lgbm_result.model_text.encode("utf-8")
                ).hexdigest()
            ),
        },
        "protected_field_pickle_sha256": {
            key: _pickle_sha256(value)
            for key, value in state.items()
            if key
            not in {
                "setwise_fusion_state",
                "setwise_fusion_result",
                "setwise_hidden_dim",
                "lgbm_result",
            }
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def _fusion_summary(
    result: Any,
    state: Any,
    hidden_dim: int,
) -> dict[str, Any]:
    return {
        "present": result is not None and state is not None,
        "candidate_name": (
            None if result is None else str(result.candidate_name)
        ),
        "best_val_mrr": (
            None if result is None else float(result.best_val_mrr)
        ),
        "hidden_dim": hidden_dim,
        "feature_count": (
            None if result is None else len(result.feature_indices)
        ),
        "state_sha256": None if state is None else _pickle_sha256(state),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: Any) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _pickle_sha256(value: Any) -> str:
    return hashlib.sha256(
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()


if __name__ == "__main__":
    main()
