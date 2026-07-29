#!/usr/bin/env bash
set -euo pipefail

root="/home/edu/workspace/jittor-GPNUT1-JGRec"
tag="dataset2_cooccur_lift_aux_expert_v1_20260728_compact_retry2"
run_dir="$root/result/$tag"
artifact_dir="$run_dir/artifacts"
selection_dir="$run_dir/rolling-selection"
state_dir="$root/result/${tag}_launcher"
worker_log="$root/logs/$tag.log"
launcher_log="$root/logs/${tag}_launcher.log"
stage_path="$run_dir/pipeline.stage"
exit_path="$run_dir/pipeline.exit"

for protected in "$run_dir" "$state_dir" "$worker_log" "$launcher_log"; do
  if [[ -e "$protected" ]]; then
    echo "refusing to overwrite existing artifact: $protected" >&2
    exit 1
  fi
done

mkdir -p "$run_dir" "$state_dir" "$root/logs"
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
    printf '%s\n' materializing_and_training > '$stage_path'
    nice -n 10 ionice -c2 -n7 \
      uv run --no-sync python \
      scripts/train_dataset2_cooccur_lift_rolling.py \
      --frozen-config \
        docs/experiments/cooccur-lift-aux-expert-v1.frozen.json \
      --checkpoint \
        checkpoints/d1_time_ramp_g050_d2_gnn_short_none_e50_edges40000_setwise_w080_seed60_20260727.pkl \
      --train-cache-prefix \
        cache/supervised_features/dataset2_joint_recent200k_full100_seed60_20260725 \
      --train-cache-report \
        result/dataset2_joint_recent200k_full100_seed60_20260725/train-cache-report.json \
      --short-none-scores \
        result/dataset2_targeted_gnn_edges_seed60_20260725/artifacts/short_none.train-scores.npy \
      --train-csv data/dataset2/train.csv \
      --source-rolling-manifest \
        result/dataset2_listwise_mlp_exact_rolling_20260728/artifacts/rolling-manifest.json \
      --output-dir '$artifact_dir'
    producer_status=\\\$?
    if [[ \\\$producer_status -ne 0 ]]; then
      printf '%s\n' producer_failed > '$stage_path'
      printf '%s\n' \"\\\$producer_status\" > '$exit_path'
      exit \"\\\$producer_status\"
    fi

    printf '%s\n' rolling_selection > '$stage_path'
    uv run --no-sync python scripts/select_robust_integrated_weight.py \
      --manifest '$artifact_dir/rolling-manifest.json' \
      --output-dir '$selection_dir'
    selection_status=\\\$?
    if [[ \\\$selection_status -eq 0 ]]; then
      printf '%s\n' rolling_selected_external_not_opened > '$stage_path'
    elif [[ \\\$selection_status -eq 2 ]]; then
      printf '%s\n' rolling_rejected_closed_no_external > '$stage_path'
    else
      printf '%s\n' rolling_selector_failed > '$stage_path'
    fi
    printf '%s\n' \"\\\$selection_status\" > '$exit_path'
    exit \"\\\$selection_status\"
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
