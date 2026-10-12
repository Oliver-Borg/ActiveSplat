#!/usr/bin/env python
# Sanity check of the oracle scene: replays a run's actions in the simulator and compares the depth the mesh gives at each
# pose (OracleScene.look_around_view) with the depth the simulator's sensor returned. Also times the oracle calls.
import os
import sys
PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
sys.path.append(PACKAGE_PATH)
sys.path.append(os.path.join(PACKAGE_PATH, 'src'))
import argparse
import json
import time

import numpy as np

from utils import OPENCV_TO_OPENGL
from dataloader.dataloader import get_dataset
from oracle import OracleScene, load_mesh

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare oracle (mesh) depth with the simulator sensor depth.')
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--user_config', type=str, required=True)
    parser.add_argument('--steps', type=int, default=300, help='number of actions to replay')
    parser.add_argument('--every', type=int, default=20, help='compare every n-th frame')
    args = parser.parse_args()

    with open(os.path.join(args.run_dir, 'config.json')) as f:
        config = json.load(f)
    with open(args.user_config) as f:
        user_config = json.load(f)
    actions = [int(line) for line in open(os.path.join(args.run_dir, 'actions.txt')) if line.strip()][:args.steps]

    dataset = get_dataset(config, user_config, 'Eval')
    dataset_config = dataset.setup()
    start = time.time()
    scene = OracleScene(load_mesh(dataset_config['scene_mesh_url'], dataset_config['scene_mesh_transform']))
    print(f'oracle scene: {len(scene.points)} samples, built in {time.time() - start:.1f} s')
    width, height = dataset_config['rgbd_width'], dataset_config['rgbd_height']
    intrinsics = np.array([[dataset_config['rgbd_fx'], 0, dataset_config['rgbd_cx']],
                           [0, dataset_config['rgbd_fy'], dataset_config['rgbd_cy']], [0, 0, 1]])
    near, far = dataset_config['rgbd_depth_min'], dataset_config['rgbd_depth_max']

    errors, observe_ms, look_ms = [], [], []
    for step in range(len(actions) + 1):
        if step > 0:
            dataset._sim.step(actions[step - 1])
        frame = dataset.get_frame()
        c2w = frame['c2w'].astype(np.float64) @ OPENCV_TO_OPENGL
        begin = time.time()
        scene.observe(c2w, intrinsics, width, height, near, far)
        observe_ms.append((time.time() - begin) * 1000)
        if step % args.every == 0:
            begin = time.time()
            _, depth = scene.look_around_view(c2w, intrinsics, width, height)
            look_ms.append((time.time() - begin) * 1000)
            sensor = np.asarray(frame['depth']).reshape(height, width)
            valid = (sensor > near) & (sensor < far) & (depth[..., 0] > 0)
            difference = np.abs(depth[..., 0] - sensor)[valid]
            errors.append((step, valid.mean(), np.median(difference), (difference < 0.05).mean()))
            print(f'step {step:4d}: valid pixels {valid.mean():.2f}, median |depth error| {np.median(difference) * 100:.1f} cm, within 5 cm {(difference < 0.05).mean():.2f}, seen so far {scene.coverage() * 100:.1f} % of all samples')
    print(f'mean over frames: median error {np.mean([e[2] for e in errors]) * 100:.2f} cm, within 5 cm {np.mean([e[3] for e in errors]):.3f}')
    print(f'observe(): {np.mean(observe_ms):.1f} ms per frame, look_around_view(): {np.mean(look_ms):.1f} ms for {width}x{height} rays')
    dataset.close()
