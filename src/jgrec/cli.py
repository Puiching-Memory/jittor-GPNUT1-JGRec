from __future__ import annotations

import argparse
from pathlib import Path

import jittor as jt

from .data import discover_datasets
from .submission import (
    build_dataset_submission,
    expected_test_rows,
    validate_submission_file,
    write_zip,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build JGRec dynamic recommendation submission files.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing dataset*/train.csv and test.csv.")
    parser.add_argument("--output-dir", type=Path, default=Path("result"), help="Directory for per-dataset CSV outputs.")
    parser.add_argument("--zip-path", type=Path, default=Path("result.zip"), help="Final zip path.")
    parser.add_argument("--recent-window", type=int, default=32, help="Number of recent destinations kept per source.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Number of test queries scored in one Jittor batch.")
    parser.add_argument("--limit-rows", type=int, default=None, help="Optional smoke-test limit per dataset.")
    parser.add_argument("--cpu", action="store_true", help="Disable CUDA for this run.")
    parser.add_argument("--skip-validate", action="store_true", help="Skip output format validation.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    jt.flags.use_cuda = 0 if args.cpu else 1

    datasets = discover_datasets(args.data_dir)
    results = []
    for dataset in datasets:
        print(f"[{dataset.name}] loading {dataset.train_path} and scoring {dataset.test_path}")
        result = build_dataset_submission(
            dataset=dataset,
            output_dir=args.output_dir,
            recent_window=args.recent_window,
            batch_size=args.batch_size,
            limit_rows=args.limit_rows,
        )
        results.append(result)

        if not args.skip_validate:
            expected_rows = None if args.limit_rows is not None else expected_test_rows(dataset)
            validate_submission_file(result.output_path, expected_rows=expected_rows)
        print(f"[{dataset.name}] wrote {result.rows} rows to {result.output_path}")

    write_zip(results, args.zip_path)
    print(f"wrote submission archive: {args.zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
