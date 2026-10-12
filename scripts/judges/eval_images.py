#!/usr/bin/env python
# Image-quality evaluation of the final Gaussian map against simulator ground truth.
#
# Two view sets are rendered from <run>/gaussians_data/params.npz and compared with what the simulator renders from the textured mesh:
#   trajectory: the poses the agent visited (actions.txt replayed in the simulator). The map was optimised on these frames.
#   novel:      random poses near the trajectory that the mapper never saw, which tests the map itself.
# Metrics are on the full image, so regions the map does not cover count against it; psnr_mapped and coverage separate
# "missing" from "blurry".
import os
import sys
PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
SRC_PATH = os.path.abspath(os.path.join(PACKAGE_PATH, 'src'))
sys.path.append(PACKAGE_PATH)
sys.path.append(SRC_PATH)
import argparse
import csv
import json
import time
from typing import Dict, List, Tuple

import numpy as np
import quaternion
import torch
from PIL import Image

from utils import PROJECT_NAME, OPENCV_TO_OPENGL
from utils.logging_utils import Log
from dataloader.dataloader import get_dataset
from habitat.sims.habitat_simulator.actions import _DefaultHabitatSimActions
from mapper.splatam.splatam import render, get_rendervars
from mapper.splatam.utils.slam_external import calc_ssim

METRIC_NAMES = ['psnr', 'ssim', 'lpips', 'psnr_mapped', 'coverage', 'depth_l1_m']
# NOTE: pixels with rendered opacity above this count as covered by the map
COVERED_OPACITY = 0.5

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

def frame_to_w2c(frame:dict) -> np.ndarray:
    # NOTE: same conversion as the mapper (SplaTAM.__get_np_data): the sensor c2w is OpenGL, the renderer wants OpenCV
    return np.linalg.inv(frame['c2w'].astype(np.float64) @ OPENCV_TO_OPENGL)

def collect_trajectory_views(dataset, actions:List[int], stride:int) -> List[dict]:
    '''Replay the actions (as scripts/judges/eval_actions.py does) and keep every stride-th frame.'''
    views = []
    frame = dataset.get_frame()
    views.append({'step': 0, 'w2c': frame_to_w2c(frame), 'rgb': frame['rgb'], 'depth': frame['depth']})
    for step, action in enumerate(actions, start=1):
        dataset._sim.step(action)
        frame = dataset.get_frame()
        if step % stride == 0:
            views.append({'step': step, 'w2c': frame_to_w2c(frame), 'rgb': frame['rgb'], 'depth': frame['depth']})
    return views

def collect_novel_views(dataset, trajectory_positions:np.ndarray, count:int, radius:float, seed:int) -> List[dict]:
    '''Teleport the agent to random navigable points within radius of the trajectory, facing a random direction.'''
    rng = np.random.RandomState(seed)
    sim = dataset._sim
    views = []
    attempts = 0
    while len(views) < count and attempts < count * 50:
        attempts += 1
        centre = trajectory_positions[rng.randint(len(trajectory_positions))]
        position = sim.pathfinder.get_random_navigable_point_near(centre, radius)
        if not np.all(np.isfinite(position)):
            continue
        yaw = rng.uniform(0, 2 * np.pi)
        rotation = quaternion.from_rotation_vector(np.array([0, yaw, 0]))
        sim.set_agent_state(position.tolist(), quaternion.as_float_array(rotation)[[1, 2, 3, 0]].tolist())
        frame = dataset.get_frame()
        views.append({'step': len(views), 'w2c': frame_to_w2c(frame), 'rgb': frame['rgb'], 'depth': frame['depth']})
    if len(views) < count:
        Log(f'Only found {len(views)} of {count} novel views', tag=PROJECT_NAME)
    return views

def psnr(prediction:torch.Tensor, target:torch.Tensor) -> float:
    mse = torch.mean((prediction - target) ** 2)
    return float(-10 * torch.log10(mse.clamp_min(1e-10)))

@torch.no_grad()
def evaluate_view(params:dict, intrinsics:np.ndarray, render_config:dict, lpips_model, view:dict, device:torch.device) -> Tuple[dict, np.ndarray]:
    scene_data, scene_depth_data = get_rendervars(params, view['w2c'])
    image, depth, opacity, _ = render(view['w2c'], intrinsics, scene_data, scene_depth_data, render_config)
    image = image.clamp(0, 1)
    target = torch.tensor(view['rgb'], device=device).float().permute(2, 0, 1)
    target_depth = torch.tensor(view['depth'], device=device).float()
    covered = opacity[0] > COVERED_OPACITY
    valid_depth = target_depth > 0
    metrics = {
        'psnr': psnr(image, target),
        'ssim': float(calc_ssim(image, target)),
        'lpips': float(lpips_model(image.unsqueeze(0), target.unsqueeze(0), normalize=True)),
        'psnr_mapped': psnr(image[:, covered], target[:, covered]) if covered.any() else float('nan'),
        'coverage': float(covered.float().mean()),
        'depth_l1_m': float((depth[0] - target_depth).abs()[covered & valid_depth].mean()) if (covered & valid_depth).any() else float('nan'),
    }
    error = (image - target).abs().mean(dim=0)
    panel = np.concatenate([
        target.permute(1, 2, 0).cpu().numpy(),
        image.permute(1, 2, 0).cpu().numpy(),
        np.repeat(np.clip(error.cpu().numpy() * 4, 0, 1)[..., None], 3, axis=2)], axis=1)
    return metrics, np.uint8(panel * 255)

def summarise(rows:List[dict]) -> dict:
    summary = {'views': len(rows)}
    for name in METRIC_NAMES:
        values = np.array([row[name] for row in rows], dtype=np.float64)
        summary[name] = float(np.nanmean(values)) if np.isfinite(values).any() else float('nan')
    return summary

def eval_images(run_dir:str, user_config:dict, device:torch.device, trajectory_stride:int, novel_count:int,
                novel_radius:float, seed:int, example_count:int) -> Dict[str, dict]:
    import lpips
    with open(os.path.join(run_dir, 'config.json')) as f:
        config = json.load(f)
    with open(os.path.join(run_dir, 'actions.txt')) as f:
        actions = [int(line.strip()) for line in f if line.strip()]
    params, camera_data = load_params(os.path.join(run_dir, 'gaussians_data', 'params.npz'), device)
    intrinsics = camera_data['intrinsics']
    lpips_model = lpips.LPIPS(net='alex', verbose=False).to(device).eval()

    dataset = get_dataset(config, user_config, 'Eval')
    dataset_config = dataset.setup()
    render_config = {'viz_w': dataset_config['rgbd_width'], 'viz_h': dataset_config['rgbd_height'], 'viz_near': 0.01, 'viz_far': 100.0}

    agent_state = dataset._sim.get_agent_state()
    sensor_height = float(agent_state.sensor_states['rgb'].position[1] - agent_state.position[1])
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

    trajectory_positions = np.stack([np.linalg.inv(view['w2c'])[:3, 3] for view in trajectory_views])
    # NOTE: habitat's y axis is up; navigable points are sampled around the point on the floor below the sensor
    agent_foot = trajectory_positions - np.array([0.0, sensor_height, 0.0])
    novel_views = collect_novel_views(dataset, agent_foot, novel_count, novel_radius, seed)
    dataset.close()

    view_sets = {'trajectory': trajectory_views, 'novel': novel_views}
    examples_dir = os.path.join(run_dir, 'image_eval')
    os.makedirs(examples_dir, exist_ok=True)
    for name in os.listdir(examples_dir):
        os.remove(os.path.join(examples_dir, name))
    all_rows, summaries = [], {}
    for set_name, views in view_sets.items():
        rows = []
        example_indices = set(np.linspace(0, len(views) - 1, min(example_count, len(views))).astype(int).tolist())
        for index, view in enumerate(views):
            metrics, panel = evaluate_view(params, intrinsics, render_config, lpips_model, view, device)
            rows.append({'set': set_name, 'step': view['step'], **metrics})
            if index in example_indices:
                Image.fromarray(panel).save(os.path.join(examples_dir, f"{set_name}_{view['step']:04d}.png"))
        summaries[set_name] = summarise(rows)
        all_rows += rows
    with open(os.path.join(run_dir, 'image_metrics.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['set', 'step', *METRIC_NAMES])
        writer.writeheader()
        writer.writerows(all_rows)
    with open(os.path.join(run_dir, 'image_metrics.json'), 'w') as f:
        json.dump({'scene_id': config['dataset']['scene_id'], 'settings': {
            'trajectory_stride': trajectory_stride, 'novel_views': len(novel_views), 'novel_radius_m': novel_radius, 'seed': seed,
            'covered_opacity': COVERED_OPACITY}, **summaries}, f, indent=4)
    return summaries

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
    for set_name, summary in summaries.items():
        Log(f"{set_name} ({summary['views']} views): PSNR {summary['psnr']:.2f} dB, SSIM {summary['ssim']:.3f}, LPIPS {summary['lpips']:.3f}, "
            f"PSNR mapped {summary['psnr_mapped']:.2f} dB, coverage {summary['coverage'] * 100:.1f} %, depth L1 {summary['depth_l1_m'] * 100:.1f} cm",
            tag=PROJECT_NAME)
    Log(f'Image evaluation took {time.time() - start:.0f} s', tag=PROJECT_NAME)
