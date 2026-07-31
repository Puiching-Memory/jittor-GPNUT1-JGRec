from __future__ import annotations

import argparse
import json
from pathlib import Path

from jgrec.submission import compose_submission_package


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compose exact dataset CSV members from two submission ZIPs."
    )
    parser.add_argument("--dataset1-source-zip", required=True, type=Path)
    parser.add_argument("--dataset2-source-zip", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset1-rows", required=True, type=int)
    parser.add_argument("--dataset2-rows", required=True, type=int)
    parser.add_argument("--dataset1-sha256", required=True)
    parser.add_argument("--dataset2-sha256", required=True)
    args = parser.parse_args()

    report = compose_submission_package(
        dataset1_source_zip=args.dataset1_source_zip,
        dataset2_source_zip=args.dataset2_source_zip,
        output_dir=args.output_dir,
        expected_rows={
            "dataset1": args.dataset1_rows,
            "dataset2": args.dataset2_rows,
        },
        expected_sha256={
            "dataset1": args.dataset1_sha256,
            "dataset2": args.dataset2_sha256,
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
