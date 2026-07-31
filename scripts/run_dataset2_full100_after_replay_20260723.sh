#!/usr/bin/env bash
set -uo pipefail

cd /home/edu/workspace/jittor-GPNUT1-JGRec

replay_exit="logs/dataset2_full100_replay_seed60_20260723.exit"
train_exit="logs/dataset2_full100_train_seed60_20260723.exit"
watch_log="logs/dataset2_full100_after_replay_seed60_20260723.log"

printf '[full100-watch] waiting for replay gate\n' >"$watch_log"
while [ ! -f "$replay_exit" ]; do
  sleep 30
done

status="$(tr -d '[:space:]' <"$replay_exit")"
if [ "$status" != "0" ]; then
  printf '[full100-watch] replay rejected status=%s; training not started\n' "$status" \
    >>"$watch_log"
  printf '98\n' >"$train_exit"
  exit 98
fi

printf '[full100-watch] replay passed; starting full-100 cache and LightGBM\n' \
  >>"$watch_log"
bash scripts/run_dataset2_full100_train_20260723.sh
status=$?
printf '[full100-watch] pipeline finished status=%s\n' "$status" >>"$watch_log"
exit "$status"
