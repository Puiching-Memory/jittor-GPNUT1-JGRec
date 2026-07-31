from __future__ import annotations

import argparse
from pathlib import Path

from jgrec.contest_checkpoint import compose_checkpoint_datasets


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compose a complete checkpoint with selected dataset states replaced."
    )
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--dataset2-checkpoint", required=True, type=Path)
    parser.add_argument("--output-checkpoint", required=True, type=Path)
    args = parser.parse_args()

    compose_checkpoint_datasets(
        args.output_checkpoint,
        base_checkpoint=args.base_checkpoint,
        replacements={"dataset2": args.dataset2_checkpoint},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
