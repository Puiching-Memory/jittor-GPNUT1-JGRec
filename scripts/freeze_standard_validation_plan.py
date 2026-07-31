from __future__ import annotations

import argparse
import json
from pathlib import Path

from jgrec.standard_validation_protocol import (
    freeze_standard_validation_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a candidate space, equal-weight rolling policy, and "
            "long-horizon external gate before any selection metric is read."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    report = freeze_standard_validation_plan(
        plan_path=args.plan,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
