from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect frozen inputs for Dataset2 signal correction.",
    )
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument("--sequence-cache-dir", required=True, type=Path)
    parser.add_argument("--base-result-dir", required=True, type=Path)
    parser.add_argument("--oof-expert-logits", required=True, type=Path)
    parser.add_argument(
        "--full-validation-expert-logits",
        required=True,
        type=Path,
    )
    args = parser.parse_args()

    manifest = _read_json(args.sequence_cache_dir / "fold-manifest.json")
    prefix = str(args.train_cache_prefix)
    candidates = np.load(
        f"{prefix}.train-candidates.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    src = np.load(
        f"{prefix}.train-src.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    dst = np.load(
        f"{prefix}.train-dst.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    times = np.load(
        f"{prefix}.train-time.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    oof = np.load(args.oof_expert_logits, mmap_mode="r", allow_pickle=False)
    validation = np.load(
        args.full_validation_expert_logits,
        mmap_mode="r",
        allow_pickle=False,
    )
    expected_shape = (int(manifest["train_rows"]), int(manifest["candidate_count"]))
    if (
        candidates.shape != expected_shape
        or src.shape != expected_shape[:1]
        or dst.shape != expected_shape[:1]
        or times.shape != expected_shape[:1]
        or oof.ndim != 3
        or oof.shape[1:] != expected_shape
        or oof.shape[0] < 2
        or validation.ndim != 3
        or validation.shape[0] != oof.shape[0]
        or validation.shape[2] != expected_shape[1]
    ):
        raise ValueError("signal correction input shapes differ")
    if not np.array_equal(np.asarray(candidates[:, 0]), np.asarray(dst)):
        raise ValueError("positive candidate is not at index zero")

    folds = []
    for row in manifest["folds"]:
        start, stop = (int(value) for value in row["score_rows"])
        frozen_base = np.load(
            args.base_result_dir
            / "folds"
            / "variant-A"
            / f"fold-{int(row['index'])}"
            / "score-logits.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        oof_base = np.asarray(oof[0, start:stop])
        folds.append(
            {
                "index": int(row["index"]),
                "score_rows": [start, stop],
                "frozen_base_shape": list(frozen_base.shape),
                "oof_base_max_absolute_error": float(
                    np.max(np.abs(frozen_base - oof_base))
                ),
                "oof_base_mean_absolute_error": float(
                    np.mean(np.abs(frozen_base - oof_base))
                ),
                "strict_origin": bool(
                    int(times[start - 1]) < int(times[start])
                ),
            }
        )
    report = {
        "status": "complete",
        "train_shape": list(expected_shape),
        "oof_shape": list(oof.shape),
        "oof_dtype": str(oof.dtype),
        "validation_shape": list(validation.shape),
        "validation_dtype": str(validation.dtype),
        "expert_count": int(oof.shape[0]),
        "times_non_decreasing": bool(
            np.all(np.diff(np.asarray(times, dtype=np.int64)) >= 0)
        ),
        "folds": folds,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    raise SystemExit(main())
