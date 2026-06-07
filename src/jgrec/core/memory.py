from __future__ import annotations

import gc
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

from jgrec.logging import log

_MEMORY_LOG_PATH: Path | None = None


def configure_memory_log(path: Path | None) -> None:
    global _MEMORY_LOG_PATH
    _MEMORY_LOG_PATH = path
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"\n# memory log start {datetime.now().isoformat(timespec='seconds')} pid={os.getpid()}\n")
        f.flush()
        os.fsync(f.fileno())


def release_memory() -> None:
    """Best-effort release of Python and Jittor cached memory."""
    gc.collect()
    jt = sys.modules.get("jittor")
    if jt is None:
        return

    for name in ("gc", "clean"):
        func = getattr(jt, name, None)
        if callable(func):
            try:
                func()
            except Exception:
                pass


def memory_snapshot() -> str:
    rss_mb = _rss_mb()
    available_mb = _available_mb()
    parts = []
    if rss_mb is not None:
        parts.append(f"rss={rss_mb:.0f}MB")
    if available_mb is not None:
        parts.append(f"available={available_mb:.0f}MB")
    return " ".join(parts) if parts else "memory=unknown"


def log_memory(stage: str, enabled: bool = True) -> None:
    message = f"[memory] stage={stage} {memory_snapshot()}"
    log(message, enabled=enabled)
    _write_memory_log(message)


def log_event(message: str, enabled: bool = True) -> None:
    log(message, enabled=enabled)
    _write_memory_log(message)


def _write_memory_log(message: str) -> None:
    if _MEMORY_LOG_PATH is None:
        return
    line = f"{datetime.now().isoformat(timespec='seconds')} pid={os.getpid()} {message}\n"
    try:
        with _MEMORY_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        return


def _rss_mb() -> float | None:
    if os.name == "posix":
        try:
            with open("/proc/self/status", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024
        except OSError:
            pass

    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return usage / (1024 * 1024)
        return usage / 1024
    except Exception:
        return None


def _available_mb() -> float | None:
    if os.name != "posix":
        return None
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024
    except OSError:
        return None
    return None
