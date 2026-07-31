from __future__ import annotations

import argparse
import json
from pathlib import Path

from jgrec.standard_validation_protocol import (
    evaluate_standard_external_gate,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Open one preregistered long-horizon external holdout for an "
            "already locked candidate. The state directory is one-shot."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--selection-lock", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    args = parser.parse_args()

    report = evaluate_standard_external_gate(
        manifest_path=args.manifest,
        selection_lock_path=args.selection_lock,
        state_dir=args.state_dir,
    )
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
