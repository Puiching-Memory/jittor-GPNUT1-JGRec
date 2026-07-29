#!/usr/bin/env bash
set -euo pipefail

root="/home/edu/workspace/jittor-GPNUT1-JGRec"
tag="dataset1_feature_mask_loo_rolling_20260728"
run_dir="$root/result/$tag"
state_dir="$root/result/${tag}_launcher"
worker_log="$root/logs/$tag.log"
launcher_log="$root/logs/${tag}_launcher.log"

for protected in "$run_dir" "$state_dir" "$worker_log" "$launcher_log"; do
  if [[ -e "$protected" ]]; then
    echo "refusing to overwrite existing artifact: $protected" >&2
    exit 1
  fi
done

mkdir -p "$state_dir" "$root/logs"
setsid -f bash -lc "
  set -u
  printf '%s\n' \$\$ > '$state_dir/launcher.pid'
  idle_samples=0
  while (( idle_samples < 10 )); do
    compute_count=\$(
      nvidia-smi \
        --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null \
        | grep -Ec '^[[:space:]]*[0-9]+' || true
    )
    gpu_values=\$(
      nvidia-smi \
        --query-gpu=utilization.gpu,memory.free \
        --format=csv,noheader,nounits 2>/dev/null \
        | head -n 1
    )
    gpu_util=\$(printf '%s' \"\$gpu_values\" | cut -d, -f1 | tr -d ' ')
    gpu_free=\$(printf '%s' \"\$gpu_values\" | cut -d, -f2 | tr -d ' ')
    memory_available=\$(awk '/MemAvailable:/ {print int(\$2 / 1024)}' /proc/meminfo)
    load_one=\$(cut -d' ' -f1 /proc/loadavg)
    load_ok=\$(awk -v value=\"\$load_one\" 'BEGIN {print value < 8.0 ? 1 : 0}')

    if [[ \"\$compute_count\" == 0 \
      && \"\$gpu_util\" =~ ^[0-9]+$ \
      && \"\$gpu_free\" =~ ^[0-9]+$ \
      && \"\$gpu_util\" -le 10 \
      && \"\$gpu_free\" -ge 44000 \
      && \"\$memory_available\" -ge 16000 \
      && \"\$load_ok\" == 1 ]]; then
      idle_samples=\$((idle_samples + 1))
    else
      idle_samples=0
    fi
    printf '%s compute=%s util=%s free_mb=%s mem_available_mb=%s load=%s idle_samples=%s/10\n' \
      \"\$(date --iso-8601=seconds)\" \
      \"\$compute_count\" \
      \"\$gpu_util\" \
      \"\$gpu_free\" \
      \"\$memory_available\" \
      \"\$load_one\" \
      \"\$idle_samples\" \
      > '$state_dir/capacity-status.txt'
    if (( idle_samples < 10 )); then
      sleep 30
    fi
  done

  cd '$root'
  source .workspace-env.sh
  export CUDA_VISIBLE_DEVICES=0
  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4
  export OPENBLAS_NUM_THREADS=4
  export NUMEXPR_NUM_THREADS=4
  set +e
  setsid bash -lc \"
    exec nice -n 10 ionice -c2 -n7 \
      uv run --no-sync python \
      scripts/train_select_dataset1_feature_mask_loo_rolling.py \
      --rolling-manifest \
        result/dataset1_rolling_origin_setwise_20260726/manifest.json \
      --source-checkpoint \
        checkpoints/d1_time_ramp_g050_d2_gnn_short_none_e50_edges40000_setwise_w080_seed60_20260727.pkl \
      --output-dir '$run_dir'
  \" > '$worker_log' 2>&1 &
  worker_pid=\$!
  printf '%s\n' \"\$worker_pid\" > '$state_dir/worker.pid'

  collision=0
  while kill -0 \"\$worker_pid\" 2>/dev/null; do
    sleep 30
    foreign_pids=''
    while read -r gpu_pid; do
      [[ \"\$gpu_pid\" =~ ^[0-9]+$ ]] || continue
      gpu_sid=\$(ps -o sid= -p \"\$gpu_pid\" 2>/dev/null | tr -d ' ')
      if [[ -n \"\$gpu_sid\" && \"\$gpu_sid\" != \"\$worker_pid\" ]]; then
        foreign_pids=\"\$foreign_pids \$gpu_pid\"
      fi
    done < <(
      nvidia-smi \
        --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null
    )
    if [[ -n \"\$foreign_pids\" ]]; then
      printf '%s foreign_gpu_pids=%s; stopping only our worker group\n' \
        \"\$(date --iso-8601=seconds)\" \
        \"\$foreign_pids\" \
        > '$state_dir/collision.txt'
      kill -TERM -- -\"\$worker_pid\" 2>/dev/null || true
      collision=1
      break
    fi
  done
  wait \"\$worker_pid\"
  status=\$?
  if [[ \"\$collision\" == 1 ]]; then
    status=75
  fi
  printf '%s\n' \"\$status\" > '$state_dir/worker.exit'
  exit \"\$status\"
" >"$launcher_log" 2>&1 </dev/null &

for _ in {1..20}; do
  if [[ -s "$state_dir/launcher.pid" ]]; then
    echo "queued launcher_pid=$(cat "$state_dir/launcher.pid")"
    echo "capacity=$state_dir/capacity-status.txt"
    echo "launcher_log=$launcher_log"
    echo "worker_log=$worker_log"
    exit 0
  fi
  sleep 0.1
done
echo "detached launcher did not publish its pid" >&2
exit 1
