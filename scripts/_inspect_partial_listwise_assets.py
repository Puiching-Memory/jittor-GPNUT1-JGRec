from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/edu/workspace/jittor-GPNUT1-JGRec")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _npz_shapes(path: Path) -> dict[str, list[int]]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            key: [int(value) for value in payload[key].shape]
            for key in sorted(payload.files)
        }


def main() -> None:
    validation_prefix = (
        ROOT
        / "cache/supervised_features/"
        "dataset2_joint_recent200k_full100_val_seed60_20260725"
    )
    candidates = np.load(
        Path(f"{validation_prefix}.val-candidates.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    src = np.load(
        Path(f"{validation_prefix}.val-src.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    dst = np.load(
        Path(f"{validation_prefix}.val-dst.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    time = np.load(
        Path(f"{validation_prefix}.val-time.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    features = np.load(
        Path(f"{validation_prefix}.val.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    if candidates.shape != (20_000, 100):
        raise ValueError(f"unexpected candidate shape: {candidates.shape}")
    if features.shape != (20_000, 100, 63):
        raise ValueError(f"unexpected feature shape: {features.shape}")
    if any(values.shape != (20_000,) for values in (src, dst, time)):
        raise ValueError("query sidecar shapes differ")
    if not np.array_equal(candidates[:, 0], dst):
        raise ValueError("candidate zero is not the positive destination")
    if not np.all(np.isfinite(features)):
        raise ValueError("validation features contain non-finite values")
    if any(np.unique(row).size != 100 for row in candidates):
        raise ValueError("validation candidate rows are not unique")

    paths = {
        "checkpoint": ROOT
        / "checkpoints/"
        "d1_time_ramp_g050_d2_gnn_short_none_e50_edges40000_"
        "setwise_w080_seed60_20260727.pkl",
        "current_setwise": ROOT
        / "result/dataset2_gnn_short_none_e50_edges40000_checkpoint_20260727/"
        "head/dataset2-gnn-short-none-e50-edges40000-setwise.npz",
        "listwise_mlp": ROOT
        / "result/dataset2_listwise_mlp_seed60_20260723/"
        "dataset2-listwise-mlp.npz",
        "listwise_two_tower": ROOT
        / "result/dataset2_two_tower_listwise_200k_seed60_20260724/"
        "candidate-model.npz",
        "short_none_validation_scores": ROOT
        / "result/dataset2_targeted_gnn_edges_seed60_20260725/"
        "artifacts/short_none.val-scores.npy",
        "validation_candidates": Path(f"{validation_prefix}.val-candidates.npy"),
        "validation_dst": Path(f"{validation_prefix}.val-dst.npy"),
        "validation_features": Path(f"{validation_prefix}.val.npy"),
        "validation_src": Path(f"{validation_prefix}.val-src.npy"),
        "validation_time": Path(f"{validation_prefix}.val-time.npy"),
    }
    expected_hashes = {
        "checkpoint": (
            "0b0a846cb6c7f5b5403a75a04ed12707340f024dee99a27cded3258bb108a7aa"
        ),
        "current_setwise": (
            "a375751630249b0ab5c77c601df22f82277d592fa7e93c51e3c07545f375badd"
        ),
        "listwise_mlp": (
            "552bbc7e0e17b27f9501b3f8fd1f3ae6fa4a625b07f7bd8b00134a21023d53fb"
        ),
        "listwise_two_tower": (
            "8e99cc6354b8576a69538b046212e87c8e7a94fac0724580087552856f7afbbf"
        ),
        "short_none_validation_scores": (
            "ba64aa07e98d72662b266110df3e1853f01b6311a20339b0e14819fc9e3690e7"
        ),
        "validation_candidates": (
            "dec159209d9c6913825591b585afa0689b7b7323912543204ca6190dad4e4a95"
        ),
        "validation_features": (
            "7c2cfb763a2803fa7b7bd754dc7f44fb40bedfa15c0015f2c1ca9bcd717ecbcf"
        ),
        "validation_src": (
            "1de31b37ad2eeaa4091fdbcbd8a59aec1ad43f03ad5b875ac75b41fa8bf18b83"
        ),
        "validation_time": (
            "b08f07610f59905ec2d1d1366c4c8c188a80a2e1621e440361d1b70ebb9c18d7"
        ),
    }
    hashes = {name: _sha256(path) for name, path in paths.items()}
    for name, expected in expected_hashes.items():
        if hashes[name] != expected:
            raise ValueError(
                f"{name} hash mismatch: {hashes[name]} != {expected}"
            )

    model_shapes = {
        name: _npz_shapes(paths[name])
        for name in ("current_setwise", "listwise_mlp", "listwise_two_tower")
    }
    listwise_feature_indices = model_shapes["listwise_mlp"].get(
        "feature_indices"
    )
    if listwise_feature_indices != [63]:
        raise ValueError(
            "listwise MLP feature-index vector must cover 63 features"
        )

    report = {
        "status": "passed",
        "metrics_read": False,
        "hashes": hashes,
        "model_shapes": model_shapes,
        "validation": {
            "candidate_shape": list(candidates.shape),
            "feature_shape": list(features.shape),
            "candidate_zero_matches_dst": True,
            "candidate_rows_unique": True,
            "feature_values_finite": True,
            "src_shape": list(src.shape),
            "dst_shape": list(dst.shape),
            "time_shape": list(time.shape),
            "time_range": [int(time.min()), int(time.max())],
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
