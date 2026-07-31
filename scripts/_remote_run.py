"""Local-only helper: run a command on the private server over SSH.

Reads ALL credentials from PRIVATE_SERVER_ACCESS.md (gitignored). Contains no
secrets itself. Not intended for commit; it is a throwaway operator tool.

Usage:
    python scripts/_remote_run.py "whoami; pwd"
    python scripts/_remote_run.py --file scripts/diagnose_index_memory.py  # upload only
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import socket
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("paramiko").setLevel(logging.CRITICAL)

import paramiko  # noqa: E402

ACCESS_FILE = Path(__file__).resolve().parent.parent / "PRIVATE_SERVER_ACCESS.md"


def _read_creds() -> dict[str, str]:
    text = ACCESS_FILE.read_text(encoding="utf-8")
    fields = {}
    for key in ("IP", "Port", "Username", "Password"):
        match = re.search(rf"^- {key}:\s*(.+)$", text, re.MULTILINE)
        if not match:
            raise SystemExit(f"missing {key} in {ACCESS_FILE}")
        fields[key] = match.group(1).strip()
    return fields


def _client(
    creds: dict[str, str],
    *,
    timeout: float = 180,
) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    bind_ip = os.environ.get("PRIVATE_SERVER_BIND_IP")
    bound_socket = None
    if bind_ip:
        bound_socket = socket.create_connection(
            (creds["IP"], int(creds["Port"])),
            timeout=timeout,
            source_address=(bind_ip, 0),
        )
    try:
        client.connect(
            hostname=creds["IP"],
            port=int(creds["Port"]),
            username=creds["Username"],
            password=creds["Password"],
            timeout=timeout,
            banner_timeout=timeout,
            auth_timeout=timeout,
            sock=bound_socket,
        )
    except BaseException:
        if bound_socket is not None:
            bound_socket.close()
        raise
    return client


def probe(command: str) -> int:
    creds = _read_creds()
    try:
        client = _client(creds, timeout=10)
    except Exception as exc:
        print(f"[probe] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 74
    try:
        _, stdout, stderr = client.exec_command(command, get_pty=False, timeout=10)
        out = stdout.read().decode(errors="replace")
        err = stderr.read().decode(errors="replace")
        if out:
            print(out, end="" if out.endswith("\n") else "\n")
        if err:
            print(err, file=sys.stderr, end="" if err.endswith("\n") else "\n")
        return stdout.channel.recv_exit_status()
    finally:
        client.close()


def run(command: str) -> int:
    creds = _read_creds()
    client = _client(creds)
    try:
        _stdin, stdout, stderr = client.exec_command(command, get_pty=True, timeout=None)
        for line in iter(stdout.readline, ""):
            encoding = sys.stdout.encoding or "utf-8"
            sys.stdout.write(
                line.encode(encoding, errors="replace").decode(
                    encoding,
                    errors="replace",
                )
            )
            sys.stdout.flush()
        err = stderr.read().decode(errors="replace")
        if err.strip():
            sys.stderr.write(err)
        return stdout.channel.recv_exit_status()
    finally:
        client.close()


def upload(local: str, remote: str) -> int:
    import base64  # noqa: PLC0415

    payload = base64.b64encode(Path(local).read_bytes()).decode("ascii")
    creds = _read_creds()
    client = _client(creds)
    try:
        cmd = f"mkdir -p $(dirname {remote}) && echo {payload} | base64 -d > {remote} && wc -c {remote}"
        _, stdout, stderr = client.exec_command(cmd, timeout=60)
        out = stdout.read().decode(errors="replace")
        err = stderr.read().decode(errors="replace")
        code = stdout.channel.recv_exit_status()
        print(f"[upload] {local} -> {remote}\n{out}{err}".rstrip())
        return code
    finally:
        client.close()


def upload_sftp(local: str, remote: str) -> int:
    creds = _read_creds()
    client = _client(creds)
    try:
        remote_path = Path(remote)
        parent = str(remote_path.parent).replace("\\", "/")
        _, stdout, stderr = client.exec_command(
            f"mkdir -p {shlex.quote(parent)}",
            timeout=60,
        )
        code = stdout.channel.recv_exit_status()
        if code != 0:
            error = stderr.read().decode(errors="replace")
            raise RuntimeError(error or f"remote mkdir exited {code}")
        with client.open_sftp() as sftp:
            sftp.put(local, remote)
        print(
            f"[upload-sftp] {local} -> {remote} "
            f"({Path(local).stat().st_size} bytes)"
        )
        return 0
    finally:
        client.close()


def download(remote: str, local: str) -> int:
    import base64  # noqa: PLC0415

    creds = _read_creds()
    client = _client(creds)
    try:
        cmd = f"base64 {remote}"
        _, stdout, stderr = client.exec_command(cmd, timeout=600)
        data = stdout.read()
        code = stdout.channel.recv_exit_status()
        if code != 0:
            err = stderr.read().decode(errors="replace")
            print(f"[download] error: {err}".rstrip())
            return code
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        Path(local).write_bytes(base64.b64decode(data))
        print(f"[download] {remote} -> {local} ({Path(local).stat().st_size} bytes)")
        return 0
    finally:
        client.close()


def download_sftp(remote: str, local: str) -> int:
    creds = _read_creds()
    client = _client(creds)
    local_path = Path(local)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    last_reported = -1

    def report(transferred: int, total: int) -> None:
        nonlocal last_reported
        percent = int(transferred * 100 / total) if total else 100
        bucket = percent // 10
        if bucket != last_reported:
            last_reported = bucket
            print(
                f"[download] {percent}% "
                f"({transferred}/{total} bytes)",
                flush=True,
            )

    try:
        with client.open_sftp() as sftp:
            sftp.get(remote, str(local_path), callback=report)
        print(
            f"[download] {remote} -> {local} "
            f"({local_path.stat().st_size} bytes)"
        )
        return 0
    finally:
        client.close()


def download_stream_chunk(
    remote: str,
    local: str,
    seconds: float = 45.0,
) -> int:
    creds = _read_creds()
    client = _client(creds)
    local_path = Path(local)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with client.open_sftp() as sftp:
            total = int(sftp.stat(remote).st_size)
        offset = local_path.stat().st_size if local_path.exists() else 0
        if offset > total:
            raise RuntimeError(
                f"local file is larger than remote: {offset} > {total}"
            )
        if offset == total:
            print(f"[download] already complete ({total} bytes)")
            return 0

        command = (
            f"tail -c +{offset + 1} -- {shlex.quote(remote)}"
        )
        _, stdout, stderr = client.exec_command(command, timeout=None)
        channel = stdout.channel
        deadline = time.monotonic() + seconds
        transferred = offset
        with local_path.open("ab") as handle:
            while transferred < total and time.monotonic() < deadline:
                if channel.recv_ready():
                    chunk = channel.recv(min(4 * 1024 * 1024, total - transferred))
                    if not chunk:
                        break
                    handle.write(chunk)
                    transferred += len(chunk)
                    continue
                if channel.exit_status_ready():
                    break
                time.sleep(0.01)
            handle.flush()

        if transferred == total:
            code = channel.recv_exit_status()
            if code != 0:
                error = stderr.read().decode(errors="replace")
                raise RuntimeError(error or f"remote command exited {code}")
            print(f"[download] complete {remote} -> {local} ({total} bytes)")
            return 0
        channel.close()
        percent = transferred * 100.0 / total
        print(
            f"[download] partial {percent:.2f}% "
            f"({transferred}/{total} bytes)"
        )
        return 75
    finally:
        client.close()


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == "--probe":
        raise SystemExit(probe(args[1]))
    if len(args) >= 3 and args[0] == "--upload":
        raise SystemExit(upload(args[1], args[2]))
    if len(args) >= 3 and args[0] == "--upload-sftp":
        raise SystemExit(upload_sftp(args[1], args[2]))
    if len(args) >= 3 and args[0] == "--download":
        raise SystemExit(download(args[1], args[2]))
    if len(args) >= 3 and args[0] == "--download-sftp":
        raise SystemExit(download_sftp(args[1], args[2]))
    if len(args) >= 3 and args[0] == "--download-stream-chunk":
        raise SystemExit(download_stream_chunk(args[1], args[2]))
    if not args:
        raise SystemExit(
            "usage: _remote_run.py <command> | --upload <local> <remote> "
            "| --upload-sftp <local> <remote> "
            "| --download <remote> <local> "
            "| --download-sftp <remote> <local> "
            "| --download-stream-chunk <remote> <local>"
        )
    raise SystemExit(run(args[0]))
