from __future__ import annotations

import argparse
import json
from pathlib import Path

from jgrec.robust_weight_selection import evaluate_locked_external


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Open one long-span external holdout for an already locked exact "
            "integrated weight. The state directory can be consumed once."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--selection-lock", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    args = parser.parse_args()

    report = evaluate_locked_external(
        manifest_path=args.manifest,
        selection_lock_path=args.selection_lock,
        state_dir=args.state_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
