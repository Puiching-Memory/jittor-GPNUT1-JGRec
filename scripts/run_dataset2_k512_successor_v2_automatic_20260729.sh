#!/usr/bin/env bash
set -uo pipefail

ROOT="/home/edu/workspace/jittor-GPNUT1-JGRec"
RUN="$ROOT/result/dataset2_k512_cooccur_lift_successor_v2_rerun_20260729"

cd "$ROOT"
mkdir -p "$RUN"
if [[ -e "$RUN/controller-exit-code.txt" ]]; then
  echo "refusing to overwrite completed controller state" >&2
  exit 73
fi

printf '%s\n' "$$" > "$RUN/controller.pid"
date --iso-8601=seconds > "$RUN/controller-started-at.txt"

set +e
nice -n 5 ionice -c 2 -n 4 \
  "$ROOT/.deps/uv/bin/uv" run --no-sync python \
  scripts/run_dataset2_k512_successor_v2_automatic.py \
  --root "$ROOT" \
  --run-dir "$RUN" \
  --poll-seconds 30 \
  > "$RUN/controller.log" 2>&1
code=$?
set -e

printf '%s\n' "$code" > "$RUN/controller-exit-code.txt"
date --iso-8601=seconds > "$RUN/controller-finished-at.txt"
exit "$code"
