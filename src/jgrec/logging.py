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
    # 非 TTY（重定向到文件 / tee）时不渲染进度条，避免 ANSI 控制字符与 \r
    # 覆盖写入形成乱码；直接静默透传迭代（进度由上层结构化日志体现）。
    if not enabled or not console.is_terminal:
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
