#!/usr/bin/env bash
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
exec </dev/null
exec >/dev/null 2>&1
exec bash scripts/run_dataset1_k256_static_setwise_after_cache_20260730.sh
