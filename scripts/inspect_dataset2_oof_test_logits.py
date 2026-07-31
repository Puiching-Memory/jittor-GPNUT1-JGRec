from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect completion of a streamed OOF expert-logit array."
    )
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    logits = np.load(args.path, mmap_mode="r", allow_pickle=False)
    if logits.ndim != 3:
        raise ValueError("expert logits must be a three-dimensional array")
    written = np.any(logits != 0.0, axis=(0, 2))
    print(
        json.dumps(
            {
                "shape": list(logits.shape),
                "written_rows": int(np.count_nonzero(written)),
                "total_rows": int(logits.shape[1]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
