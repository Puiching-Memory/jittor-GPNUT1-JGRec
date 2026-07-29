from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def diagnose_csv(path: Path) -> dict[str, int | str]:
    rows = 0
    rows_with_ties = 0
    duplicate_adjacencies = 0
    maximum_duplicate_adjacencies = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            values = np.fromstring(line, sep=",", dtype=np.float64)
            if values.size == 0:
                raise ValueError(f"{path}:{line_number} is empty")
            ordered = np.sort(values)
            duplicates = int(np.count_nonzero(ordered[1:] == ordered[:-1]))
            rows += 1
            duplicate_adjacencies += duplicates
            maximum_duplicate_adjacencies = max(
                maximum_duplicate_adjacencies,
                duplicates,
            )
            rows_with_ties += int(duplicates > 0)
    return {
        "path": str(path),
        "rows": rows,
        "rows_with_ties": rows_with_ties,
        "duplicate_adjacencies": duplicate_adjacencies,
        "maximum_duplicate_adjacencies_per_row": (
            maximum_duplicate_adjacencies
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count exact per-query prediction score ties.",
    )
    parser.add_argument("csv", type=Path, nargs="+")
    args = parser.parse_args()
    reports = [diagnose_csv(path) for path in args.csv]
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

