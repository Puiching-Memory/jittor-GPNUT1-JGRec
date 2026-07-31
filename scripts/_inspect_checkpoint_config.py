from __future__ import annotations

import json
import sys
from pathlib import Path

from jgrec.contest_checkpoint import load_checkpoint_dataset


def main() -> None:
    state = load_checkpoint_dataset(Path(sys.argv[1]), sys.argv[2])
    config = state["config"]
    names = (
        "seed",
        "selection_metric",
        "max_train_events",
        "max_val_events",
        "num_negatives",
        "train_num_negatives",
        "val_num_negatives",
        "test_candidate_negative_ratio",
        "structure_predict_neighbor_limit",
        "source_profile_predict_history_limit",
        "fusion_mode",
        "two_tower_embedding_dim",
        "two_tower_hidden_dim",
        "two_tower_epochs",
        "two_tower_max_samples",
    )
    payload = {
        name: getattr(config, name, None)
        for name in names
    }
    payload["selected_fusion"] = state["fusion_result"].candidate_name
    payload["best_val_mrr"] = state["fusion_result"].best_val_mrr
    lgbm = state.get("lgbm_result")
    payload["lgbm_mlp_weight"] = None if lgbm is None else lgbm.mlp_weight
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
