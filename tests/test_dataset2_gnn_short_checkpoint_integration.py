from __future__ import annotations

import pickle
from dataclasses import replace

import numpy as np
import pytest

from jgrec.rankers.hybrid.config import TrainingConfig
from jgrec.rankers.hybrid.fusion import FusionResult
from jgrec.rankers.hybrid.fusion_lgbm import LGBMFusionResult
from jgrec.rankers.hybrid.gnn_short_checkpoint import (
    install_gnn_short_setwise_fusion,
)


def _pickle_bytes(value: object) -> bytes:
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def _source_state() -> dict[str, object]:
    feature_names = tuple(
        "gnn_short" if index == 59 else f"feature_{index}"
        for index in range(63)
    )
    return {
        "config": TrainingConfig(
            gnn_edge_weighting="none",
            gnn_short_edge_weighting=None,
            gnn_epochs=50,
            gnn_max_train_edges=40_000,
        ),
        "feature_names": feature_names,
        "encoder": {
            "graph": {
                "gnn_short": {
                    "user_embeddings": np.arange(12, dtype=np.float32).reshape(3, 4),
                    "item_embeddings": np.arange(20, dtype=np.float32).reshape(5, 4),
                }
            }
        },
        "fusion_state": {"linear1.weight": np.ones((2, 3), dtype=np.float32)},
        "fusion_result": object(),
        "lgbm_result": LGBMFusionResult(
            best_val_ap=0.4,
            best_val_mrr=0.5,
            model_text="tree\nmodel",
            feature_indices=(0, 1, 59),
            candidate_name="champion_lgbm",
            mlp_weight=0.07,
        ),
        "setwise_fusion_state": None,
        "setwise_fusion_result": None,
        "setwise_hidden_dim": 64,
        "id_map": {
            "src_values": np.asarray([1, 2], dtype=np.int32),
            "dst_values": np.asarray([3, 4], dtype=np.int32),
        },
    }


def _setwise_result() -> FusionResult:
    return FusionResult(
        best_val_ap=0.45,
        best_val_mrr=0.5484923183476225,
        state={"linear1.weight": np.ones((32, 189), dtype=np.float32)},
        mean=np.zeros(189, dtype=np.float32),
        std=np.ones(189, dtype=np.float32),
        feature_indices=tuple(range(189)),
        candidate_name="gnn_short_none_e50_edges40000_setwise",
    )


def test_install_changes_only_setwise_head_and_blend_weight() -> None:
    source = _source_state()
    source_before = _pickle_bytes(source)
    protected_keys = (
        "config",
        "feature_names",
        "encoder",
        "fusion_state",
        "fusion_result",
        "id_map",
    )
    protected_before = {
        key: _pickle_bytes(source[key])
        for key in protected_keys
    }

    setwise_result = _setwise_result()
    candidate = install_gnn_short_setwise_fusion(
        source,
        setwise_result=setwise_result,
        hidden_dim=32,
        setwise_weight=0.8,
    )

    assert _pickle_bytes(source) == source_before
    assert candidate is not source
    for key in protected_keys:
        assert _pickle_bytes(candidate[key]) == protected_before[key]
    assert candidate["setwise_fusion_result"] is setwise_result
    assert candidate["setwise_fusion_state"] is setwise_result.state
    assert candidate["setwise_hidden_dim"] == 32
    assert candidate["lgbm_result"].mlp_weight == pytest.approx(0.8)
    assert candidate["lgbm_result"].model_text == "tree\nmodel"
    assert candidate["lgbm_result"].candidate_name == "champion_lgbm"


@pytest.mark.parametrize(
    ("config_overrides", "match"),
    [
        ({"gnn_epochs": 49}, "50 epochs"),
        ({"gnn_max_train_edges": 200_000}, "40,000"),
        ({"gnn_edge_weighting": "time_decay"}, "unweighted"),
    ],
)
def test_install_rejects_non_winning_gnn_configuration(
    config_overrides: dict[str, object],
    match: str,
) -> None:
    source = _source_state()
    source["config"] = replace(source["config"], **config_overrides)

    with pytest.raises(ValueError, match=match):
        install_gnn_short_setwise_fusion(
            source,
            setwise_result=_setwise_result(),
            hidden_dim=32,
            setwise_weight=0.8,
        )


def test_install_rejects_setwise_schema_from_another_feature_set() -> None:
    source = _source_state()
    mismatched = FusionResult(
        best_val_ap=0.0,
        best_val_mrr=0.0,
        state={"weight": np.ones((1, 186), dtype=np.float32)},
        mean=np.zeros(186, dtype=np.float32),
        std=np.ones(186, dtype=np.float32),
        feature_indices=tuple(range(186)),
        candidate_name="wrong_schema",
    )

    with pytest.raises(ValueError, match="189"):
        install_gnn_short_setwise_fusion(
            source,
            setwise_result=mismatched,
            hidden_dim=32,
            setwise_weight=0.8,
        )
