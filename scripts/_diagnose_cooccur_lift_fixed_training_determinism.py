from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.rankers.hybrid.cooccur_lift import CooccurLiftAugmentedView
from jgrec.rankers.hybrid.fusion import (
    FusionConfig,
    fit_fusion_mlp_listwise_fixed,
    predict_logits,
)
from jgrec.rankers.hybrid.setwise import SetwiseFeatureView


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    rows = int(args.rows)
    use_cuda = args.device == "cuda"
    root = Path("/home/edu/workspace/jittor-GPNUT1-JGRec")
    prefix = root / (
        "cache/supervised_features/"
        "dataset2_joint_recent200k_full100_seed60_20260725"
    )
    features = np.load(
        f"{prefix}.train.npy",
        mmap_mode="r",
        allow_pickle=False,
    )[:rows]
    lift = np.load(
        root
        / (
            "result/dataset2_cooccur_lift_aux_expert_v1_20260728_compact_retry2/"
            "artifacts/lift-features.npy"
        ),
        mmap_mode="r",
        allow_pickle=False,
    )[:rows]
    short_none = np.load(
        root
        / (
            "result/dataset2_targeted_gnn_edges_seed60_20260725/"
            "artifacts/short_none.train-scores.npy"
        ),
        mmap_mode="r",
        allow_pickle=False,
    )[:rows]
    checkpoint = load_checkpoint_dataset(
        root
        / (
            "checkpoints/"
            "d1_time_ramp_g050_d2_gnn_short_none_e50_edges40000_"
            "setwise_w080_seed60_20260727.pkl"
        ),
        "dataset2",
    )
    feature_names = tuple(str(name) for name in checkpoint["feature_names"])
    hidden_dim = int(checkpoint["setwise_hidden_dim"])
    augmented = CooccurLiftAugmentedView(
        features,
        short_none_scores=short_none,
        gnn_short_column=feature_names.index("gnn_short"),
        lift_features=lift,
    )
    view = SetwiseFeatureView(augmented, transform_version=1)
    probe = np.asarray(view[:256], dtype=np.float32)
    config = FusionConfig(
        epochs=4,
        batch_size=256,
        lr=0.001,
        weight_decay=0.0,
        hidden_dim=hidden_dim,
        early_stop_patience=0,
        selection_metric="mrr",
    )
    first = _fit(view, probe, config, use_cuda=use_cuda)
    jt.sync_all()
    jt.clean()
    second = _fit(view, probe, config, use_cuda=use_cuda)
    state_error = max(
        float(
            np.max(
                np.abs(
                    first["state"][key] - second["state"][key]
                ),
                initial=0.0,
            )
        )
        for key in first["state"]
    )
    probability_error = float(
        np.max(
            np.abs(
                first["probabilities"] - second["probabilities"]
            ),
            initial=0.0,
        )
    )
    print(
        json.dumps(
            {
                "rows": rows,
                "device": args.device,
                "losses_run1": list(first["losses"]),
                "losses_run2": list(second["losses"]),
                "losses_matched": bool(
                    np.allclose(
                        first["losses"],
                        second["losses"],
                        rtol=2e-5,
                        atol=2e-6,
                    )
                ),
                "state_max_abs_error": state_error,
                "probability_max_abs_error": probability_error,
                "probability_matched": bool(
                    np.allclose(
                        first["probabilities"],
                        second["probabilities"],
                        rtol=2e-5,
                        atol=2e-6,
                    )
                ),
            },
            sort_keys=True,
        )
    )
    return 0


def _fit(
    view: Any,
    probe: np.ndarray,
    config: FusionConfig,
    *,
    use_cuda: bool,
) -> dict[str, Any]:
    seed = 33100
    jt.flags.use_cuda = int(use_cuda)
    jt.set_global_seed(seed)
    model, result, losses = fit_fusion_mlp_listwise_fixed(
        view,
        view[:1],
        config,
        np.random.default_rng(seed),
        verbose=False,
        feature_indices=tuple(range(195)),
        candidate_name="determinism-diagnostic",
    )
    logits = predict_logits(
        model,
        probe,
        result.mean,
        result.std,
    )
    logits = np.asarray(logits, dtype=np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    exponentials = np.exp(logits)
    return {
        "losses": losses,
        "state": result.state,
        "probabilities": (
            exponentials / exponentials.sum(axis=1, keepdims=True)
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
