from __future__ import annotations

import argparse
import json
from pathlib import Path

from jgrec.standard_validation_protocol import (
    select_standard_rolling_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Select one preregistered exact candidate using equal-weight "
            "rolling-origin fold means and cross-fold stability gates. "
            "Reserved rolling and external metrics are never read."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--plan-lock", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    report = select_standard_rolling_candidate(
        manifest_path=args.manifest,
        plan_lock_path=args.plan_lock,
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
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
