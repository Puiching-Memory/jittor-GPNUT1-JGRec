from __future__ import annotations

import argparse
import json
from pathlib import Path

from jgrec.partial_listwise_submission import (
    materialize_submission_member_scores,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize one verified Dataset2 submission member as an "
            "auxiliary probability matrix."
        )
    )
    parser.add_argument("--source-zip", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--expected-zip-sha256", required=True)
    parser.add_argument("--expected-member-sha256", required=True)
    args = parser.parse_args()

    if args.report.exists():
        raise FileExistsError(f"refusing to overwrite: {args.report}")
    report = materialize_submission_member_scores(
        source_zip=args.source_zip,
        member_name="dataset2.csv",
        output_path=args.output,
        expected_zip_sha256=args.expected_zip_sha256,
        expected_member_sha256=args.expected_member_sha256,
        expected_shape=(153_420, 100),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
