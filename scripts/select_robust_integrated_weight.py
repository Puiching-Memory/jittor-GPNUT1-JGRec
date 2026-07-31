from __future__ import annotations

import argparse
import json
from pathlib import Path

from jgrec.robust_weight_selection import select_rolling_origin_weight


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Select one exact integrated candidate weight using a frozen "
            "rolling-origin stability gate. This command never reads the "
            "external holdout."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    report = select_rolling_origin_weight(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
