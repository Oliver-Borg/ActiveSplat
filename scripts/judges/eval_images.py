#!/usr/bin/env python
# Offline image-quality evaluation of a finished run (the visualizer now does this itself at the end of a run; see src/evaluation/image_eval.py
# for the metrics). The map is read from <run>/gaussians_data/params.npz and compared with ground truth from a replay of actions.txt
# (trajectory views) and teleports near the trajectory (novel views) in a fresh simulator.
import os
import sys
PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
SRC_PATH = os.path.abspath(os.path.join(PACKAGE_PATH, 'src'))
sys.path.append(PACKAGE_PATH)
sys.path.append(SRC_PATH)
import argparse
import json
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from utils import PROJECT_NAME
from utils.logging_utils import Log
from dataloader.dataloader import get_dataset
from evaluation.image_eval import collect_novel_views, evaluate_view_sets, load_lpips, log_summaries, make_view

def load_params(params_path:str, device:torch.device) -> Tuple[Dict[str, torch.Tensor], dict]:
    data = np.load(params_path)
    params = {key: torch.tensor(data[key], device=device).float()
              for key in ['means3D', 'rgb_colors', 'unnorm_rotations', 'logit_opacities', 'log_scales']}
    return params, {key: data[key] for key in ['intrinsics', 'cam_unnorm_rots', 'cam_trans']}

def quaternion_to_matrix(q:np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

def collect_trajectory_views(dataset, actions:List[int], stride:int) -> List[dict]:
    '''Replay the actions (as scripts/judges/eval_actions.py does) and keep every stride-th frame.'''
    views = []
    frame = dataset.get_frame()
    views.append(make_view(0, frame['c2w'], frame['rgb'], frame['depth']))
    for step, action in enumerate(actions, start=1):
        dataset._sim.step(action)
        frame = dataset.get_frame()
        if step % stride == 0:
            views.append(make_view(step, frame['c2w'], frame['rgb'], frame['depth']))
    return views

def eval_images(run_dir:str, user_config:dict, device:torch.device, trajectory_stride:int, novel_count:int,
                novel_radius:float, seed:int, example_count:int) -> Dict[str, dict]:
    '''Offline version for finished runs: replays actions.txt in a fresh simulator for the ground truth and renders from params.npz.
    Runs started after the in-run evaluation was added already have image_metrics.json from the visualizer.'''
    with open(os.path.join(run_dir, 'config.json')) as f:
        config = json.load(f)
    with open(os.path.join(run_dir, 'actions.txt')) as f:
        actions = [int(line.strip()) for line in f if line.strip()]
    params, camera_data = load_params(os.path.join(run_dir, 'gaussians_data', 'params.npz'), device)
    intrinsics = camera_data['intrinsics']
    lpips_model = load_lpips(device)

    dataset = get_dataset(config, user_config, 'Eval')
    dataset_config = dataset.setup()
    render_config = {'viz_w': dataset_config['rgbd_width'], 'viz_h': dataset_config['rgbd_height'], 'viz_near': 0.01, 'viz_far': 100.0}

    trajectory_views = collect_trajectory_views(dataset, actions, trajectory_stride)
    # NOTE: the replay must reproduce the mapped trajectory, otherwise the "ground truth" would not correspond to the mapped frames
    first_w2c = np.eye(4)
    first_w2c[:3, :3] = quaternion_to_matrix(camera_data['cam_unnorm_rots'][0][:, 0])
    first_w2c[:3, 3] = camera_data['cam_trans'][0][:, 0]
    replay_offset = np.linalg.norm(np.linalg.inv(first_w2c)[:3, 3] - np.linalg.inv(trajectory_views[0]['w2c'])[:3, 3])
    if replay_offset > 1e-3:
        Log(f'WARNING: replayed start pose is {replay_offset:.3f} m from the mapped start pose', tag=PROJECT_NAME)
    # NOTE: the replay has one more pose (after the final action) than the mapper has frames
    mapped_steps = camera_data['cam_trans'][0].shape[1]
    last_view = [view for view in trajectory_views if view['step'] < mapped_steps][-1]
    last_step = last_view['step']
    last_w2c = np.eye(4)
    last_w2c[:3, :3] = quaternion_to_matrix(camera_data['cam_unnorm_rots'][0][:, last_step])
    last_w2c[:3, 3] = camera_data['cam_trans'][0][:, last_step]
    replay_offset = np.linalg.norm(np.linalg.inv(last_w2c)[:3, 3] - np.linalg.inv(last_view['w2c'])[:3, 3])
    if replay_offset > 1e-2:
        Log(f'WARNING: replayed pose at step {last_step} is {replay_offset:.3f} m from the mapped pose', tag=PROJECT_NAME)

    novel_views = collect_novel_views(dataset, trajectory_views, novel_count, novel_radius, seed)
    dataset.close()
    settings = {'source': 'offline replay', 'trajectory_stride': trajectory_stride, 'novel_radius_m': novel_radius, 'seed': seed}
    return evaluate_view_sets(run_dir, config['dataset']['scene_id'], params, intrinsics, render_config, lpips_model,
                              {'trajectory': trajectory_views, 'novel': novel_views}, device, settings, example_count)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'{PROJECT_NAME} image-quality evaluator (final map vs simulator ground truth).')
    parser.add_argument('--run_dir', type=str, required=True, help='Result folder (needs config.json, actions.txt, gaussians_data/params.npz).')
    parser.add_argument('--user_config', type=str, required=True, help='User config url (*.json).')
    parser.add_argument('--gpu_id', type=int, required=True, help='Specify gpu id.')
    parser.add_argument('--trajectory_stride', type=int, default=5, help='Evaluate every n-th frame of the trajectory.')
    parser.add_argument('--novel_views', type=int, default=100, help='Number of held-out views to sample.')
    parser.add_argument('--novel_radius', type=float, default=1.5, help='Held-out views are within this distance (m) of the trajectory.')
    parser.add_argument('--examples', type=int, default=6, help='Comparison images to save per view set.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--force', action='store_true', help='Specify whether to force evaluation.')
    args = parser.parse_args()

    summary_path = os.path.join(args.run_dir, 'image_metrics.json')
    if os.path.exists(summary_path) and not args.force:
        print('Image evaluation already exists.')
        exit(0)
    with open(args.user_config) as f:
        user_config = json.load(f)
    device = torch.device('cuda', args.gpu_id)

    start = time.time()
    summaries = eval_images(args.run_dir, user_config, device, args.trajectory_stride, args.novel_views,
                            args.novel_radius, args.seed, args.examples)
    log_summaries(summaries)
    Log(f'Image evaluation took {time.time() - start:.0f} s', tag=PROJECT_NAME)
