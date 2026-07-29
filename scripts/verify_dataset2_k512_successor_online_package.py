from __future__ import annotations

import argparse
import json
from pathlib import Path

from jgrec.cooccur_lift_online_package_contract import (
    validate_k512_online_package_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the frozen current-run K512 successor online package "
            "lineage before any test scoring."
        )
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    report = validate_k512_online_package_preflight(
        root=args.root,
        contract_path=args.contract,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(
            report,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
