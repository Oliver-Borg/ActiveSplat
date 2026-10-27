#!/bin/bash
# Benchmark the switchable optimisations (src/utils/optimisations.py): short runs, strictly one at a time on one GPU, interleaved over repeats so slow
# drift of the machine's load spreads over all configurations. Appends to <manifest>: order, repeat, spec, load average at start / end, result dir.
#   GPU=0 bash scripts/tools/benchmark_optimisations.sh <manifest.tsv> [steps=150] [repeats=2] [dataset_config=mp3d_high_resolution.json] [scene=gZ6f7yhEvPG]
# Check nvidia-smi and the machine's load first: other jobs make the timings meaningless.
manifest=$1; steps=${2:-150}; repeats=${3:-2}; cfg=${4:-mp3d_high_resolution.json}; scene=${5:-gZ6f7yhEvPG}
cd "$(dirname "$0")/../.."
specs=(baseline all
       -single_pass_render -prepared_params -render_cache -snapshot_cache -no_clone_cut
       single_pass_render single_pass_render,prepared_params render_cache snapshot_cache no_clone_cut)
port=13000; order=0
for rep in $(seq 1 "$repeats"); do
  for spec in "${specs[@]}"; do
    order=$((order+1)); port=$((port+1))
    load_start=$(cut -d' ' -f1 /proc/loadavg)
    log=$(mktemp)
    ACTIVESPLAT_OPTIMISATIONS=$spec ROS_MASTER_URI=http://localhost:$port CUDA_VISIBLE_DEVICES=${GPU:?set GPU} PYTHONUNBUFFERED=1 \
      roslaunch -p $port activesplat habitat.launch config:=$PWD/config/datasets/$cfg scene_id:=$scene gpu_id:=0 step_num:=$steps \
      hide_mapper_windows:=1 hide_planner_windows:=1 > "$log" 2>&1 &
    pid=$!
    for i in $(seq 1 360); do grep -aq "mapper node finished" "$log" && break; kill -0 $pid 2>/dev/null || break; sleep 5; done
    kill -INT $pid; sleep 15; kill -KILL $pid 2>/dev/null
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$order" "$rep" "$spec" "$load_start" "$(cut -d' ' -f1 /proc/loadavg)" "$(ls -dt results/*_$scene | head -1)" >> "$manifest"
    rm -f "$log"
  done
done
echo BENCHMARK DONE
