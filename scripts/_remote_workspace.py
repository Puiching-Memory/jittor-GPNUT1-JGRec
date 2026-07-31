"""Local operator helper for an isolated remote workspace.

Credentials are accepted only through JGREC_REMOTE_HOST, JGREC_REMOTE_PORT,
JGREC_REMOTE_USER, and JGREC_REMOTE_PASSWORD environment variables.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import sys
from pathlib import Path

import paramiko


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"missing environment variable: {name}")
    return value


def _transport() -> paramiko.Transport:
    host = _required_env("JGREC_REMOTE_HOST")
    port = int(os.environ.get("JGREC_REMOTE_PORT", "22"))
    username = _required_env("JGREC_REMOTE_USER")
    password = _required_env("JGREC_REMOTE_PASSWORD")
    transport = paramiko.Transport((host, port))
    transport.banner_timeout = 30
    transport.auth_timeout = 30
    transport.start_client(timeout=30)
    key = transport.get_remote_server_key()
    digest = base64.b64encode(
        hashlib.sha256(key.asbytes()).digest()
    ).decode("ascii").rstrip("=")
    print(
        f"[ssh] host={host}:{port} key={key.get_name()} SHA256:{digest}",
        flush=True,
    )
    transport.auth_password(username=username, password=password)
    if not transport.is_authenticated():
        raise RuntimeError("SSH password authentication failed")
    return transport


def _run(command: str) -> int:
    transport = _transport()
    try:
        channel = transport.open_session(timeout=30)
        channel.exec_command(command)
        stdout = channel.makefile("rb", -1).read().decode(errors="replace")
        stderr = channel.makefile_stderr("rb", -1).read().decode(
            errors="replace"
        )
        code = channel.recv_exit_status()
        if stdout:
            sys.stdout.write(stdout)
        if stderr:
            sys.stderr.write(stderr)
        return code
    finally:
        transport.close()


def _upload(local: Path, remote: str) -> int:
    transport = _transport()
    try:
        with paramiko.SFTPClient.from_transport(transport) as sftp:
            sftp.put(str(local), remote)
        print(f"[upload] {local} -> {remote} ({local.stat().st_size} bytes)")
        return 0
    finally:
        transport.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("command")
    subparsers.add_parser("run-env")
    upload_parser = subparsers.add_parser("upload")
    upload_parser.add_argument("local", type=Path)
    upload_parser.add_argument("remote")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "run":
        raise SystemExit(_run(args.command))
    if args.action == "run-env":
        raise SystemExit(_run(_required_env("JGREC_REMOTE_COMMAND")))
    raise SystemExit(_upload(args.local, args.remote))


if __name__ == "__main__":
    main()
