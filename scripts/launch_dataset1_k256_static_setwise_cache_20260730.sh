#!/usr/bin/env bash
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="$root/result/dataset1_k256_static_setwise_dual_horizon_20260730/cache"
mkdir -p "$run_dir"
exec </dev/null
exec >/dev/null 2>&1
echo "$$" > "$run_dir/build.pid"

set +e
bash "$root/scripts/run_dataset1_k256_static_setwise_cache_20260730.sh" \
  >> "$run_dir/build.log" 2>&1
code=$?
set -e
echo "$code" > "$run_dir/build.exit"
exit "$code"
