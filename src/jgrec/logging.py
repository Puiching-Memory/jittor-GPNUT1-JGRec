from __future__ import annotations

from collections.abc import Iterable, Iterator

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn


console = Console()


def track(
    sequence: Iterable[int],
    *,
    description: str,
    total: int | None = None,
    enabled: bool = True,
) -> Iterator[int]:
    if not enabled:
        yield from sequence
        return

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    with progress:
        task_id = progress.add_task(description, total=total)
        for item in sequence:
            yield item
            progress.advance(task_id)


def log(message: str, *, enabled: bool = True) -> None:
    if enabled:
        console.print(message, markup=False)
