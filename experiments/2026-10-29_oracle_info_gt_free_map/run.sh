#!/bin/bash
# Full 1000-step runs of oracle-info with the GT free map (planner.oracle.navigation = gt_free_map): two GPU workers (the 2-GPU limit for long jobs),
# each running two scenes one after the other. Before every launch the worker waits until its GPU has no compute processes and almost no memory in use.
#   bash experiments/2026-10-29_oracle_info_gt_free_map/run.sh <gpu A> <gpu B>   (v2 = with the opening by the agent radius; v1 was this script without it, on GPUs 1 and 2)
cd "$(dirname "$0")/../.."
V=${V:-v2}  # v3 = v2 plus a free disc around the agent; SCENES_A / SCENES_B override the scene lists
cfg=mp3d_high_resolution_oracle_info_gt_free_map.json
label="oracle-info (GT free map)"
gpu_is_free() {
  local uuid=$(nvidia-smi -i $1 --query-gpu=uuid --format=csv,noheader)
  [ "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid")" = 0 ] && \
  [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 300 ]
}
worker() {  # gpu port scene...
  local gpu=$1 port=$2; shift 2
  for scene in "$@"; do
    until gpu_is_free $gpu; do sleep 60; done
    before=$(ls -d results/*_$scene 2>/dev/null)
    log=$(mktemp)
    ROS_MASTER_URI=http://localhost:$port CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 \
      roslaunch -p $port activesplat habitat.launch config:=$PWD/config/datasets/$cfg scene_id:=$scene gpu_id:=0 step_num:=1000 \
      hide_mapper_windows:=1 hide_planner_windows:=1 save_runtime_data:=1 > "$log" 2>&1 &
    pid=$!
    for i in $(seq 1 1200); do grep -aq "mapper node finished" "$log" && break; kill -0 $pid 2>/dev/null || break; sleep 5; done
    kill -INT $pid; sleep 15; kill -KILL $pid 2>/dev/null
    run=$(comm -13 <(echo "$before") <(ls -d results/*_$scene) | tail -1)
    echo "$(date +%T) gpu $gpu $scene -> $run ($(grep -ac 'mapper node finished' "$log") finished)" >> experiments/2026-10-29_oracle_info_gt_free_map/manifest_$V.txt
    [ -n "$run" ] && echo "$label" > $run/label.txt && python scripts/oracle/coverage.py --run_dir $run --user_config config/user_config.json > "$run/coverage_log.txt" 2>&1
    cp "$log" "experiments/2026-10-29_oracle_info_gt_free_map/launch_${V}_$scene.log"; rm -f "$log"
  done
}
worker ${1:?gpu A} 11431 ${SCENES_A:-gZ6f7yhEvPG pLe4wQe7qrG} &
[ -n "${2:-}" ] && worker $2 11432 ${SCENES_B:-GdvgFV5R1Z5 YmJkqBEsHnH} &
wait
