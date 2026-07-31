"""Retry SSH until the accidental full-prefix smoke process can be stopped."""

from __future__ import annotations

import time
from pathlib import Path

import paramiko

from _remote_run import _read_creds

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "remote-smoke-recovery.log"
PROCESS_MARKER = "artifacts/dataset2_two_tower_listwise_smoke_20260724"


def main() -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    creds = _read_creds()
    for attempt in range(1, 121):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                hostname=creds["IP"],
                port=int(creds["Port"]),
                username=creds["Username"],
                password=creds["Password"],
                timeout=10,
                banner_timeout=10,
                auth_timeout=10,
            )
            command = (
                "pids=$(ps -eo pid=,comm=,args= | "
                "awk '$2 ~ /^python/ && index($0,\""
                + PROCESS_MARKER
                + "\") {print $1}' | tr '\\n' ' '); "
                "if [ -n \"$pids\" ]; then kill $pids; fi; "
                "printf 'stopped=%s\\n' \"$pids\""
            )
            _, stdout, stderr = client.exec_command(command, timeout=30)
            output = stdout.read().decode(errors="replace")
            error = stderr.read().decode(errors="replace")
            LOG_PATH.write_text(
                f"attempt={attempt}\n{output}{error}",
                encoding="utf-8",
            )
            return stdout.channel.recv_exit_status()
        except Exception as exc:
            LOG_PATH.write_text(
                f"attempt={attempt}\nerror={type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )
        finally:
            client.close()
        time.sleep(10)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
