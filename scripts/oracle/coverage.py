#!/usr/bin/env python
# Exploration coverage of a run against the ground-truth mesh: replays <run>/actions.txt in the simulator, marks the surface
# samples the sensor could see after every step (OracleScene.observe) and writes <run>/coverage.csv with the fraction of the
# *observable* surface seen so far (see precompute_observable.py), plus the path length. Works for any planner, so native,
# oracle and tour runs are scored with the same definition. coverage_multi is the fraction seen from two directions at least
# OracleScene.second_look_cos apart (default 20 degrees).
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import *
import argparse
import csv

from utils import OPENCV_TO_OPENGL
from oracle import observable_cache_path, load_observable

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replay a run and score its exploration coverage.')
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--user_config', type=str, required=True)
    parser.add_argument('--cache_root', type=str, default=os.path.join(PACKAGE_PATH, 'data'))
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    out_path = os.path.join(args.run_dir, 'coverage.csv')
    if os.path.exists(out_path) and not args.force:
        print(f'{out_path} exists')
        sys.exit(0)
    with open(os.path.join(args.run_dir, 'config.json')) as f:
        config = json.load(f)
    with open(args.user_config) as f:
        user_config = json.load(f)
    actions = [int(line) for line in open(os.path.join(args.run_dir, 'actions.txt')) if line.strip()]

    dataset, dataset_config = open_dataset(config, user_config)
    scene = make_oracle_scene(dataset_config)
    far, near = float(config['dataset']['far']), float(config['dataset']['near'])
    observable = load_observable(observable_cache_path(args.cache_root, config['dataset']['format'], config['dataset']['scene_id']), scene, far)
    scene.restrict_to(observable)
    intrinsics = sensor_intrinsics(dataset_config)
    width, height = dataset_config['rgbd_width'], dataset_config['rgbd_height']
    forward_step = dataset_config['agent_forward_step_size']

    rows, path_length = [], 0.0
    for step in range(len(actions) + 1):
        if step > 0:
            dataset._sim.step(actions[step - 1])
            path_length += forward_step if actions[step - 1] == 1 else 0.0
        c2w = dataset.get_frame()['c2w'].astype(np.float64) @ OPENCV_TO_OPENGL
        scene.observe(c2w, intrinsics, width, height, near, far)
        rows.append((step, scene.coverage(), path_length, scene.multi_coverage()))
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'coverage', 'path_length_m', 'coverage_multi'])
        writer.writerows(rows)
    print(f'{args.run_dir}: {rows[-1][1] * 100:.1f} % of the observable surface seen after {len(actions)} steps ({rows[-1][3] * 100:.1f} % from two directions), {path_length:.1f} m')
    dataset.close()
