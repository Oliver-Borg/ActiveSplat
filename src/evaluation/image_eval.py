# Image-quality evaluation of a Gaussian map against simulator ground truth. Used in-run by the visualizer (at the end of a run, from the
# map in memory, so no images are written for it except a few JPEG comparison panels) and offline by scripts/judges/eval_images.py.
#
# Two view sets are rendered from the map and compared with what the simulator renders from the textured mesh:
#   trajectory: the poses the agent visited. The map was optimised on these frames.
#   novel:      random poses near the trajectory that the mapper never saw, which tests the map itself.
# Metrics are on the full image, so regions the map does not cover count against it; psnr_mapped and coverage separate
# "missing" from "blurry".
import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import quaternion
import torch
from PIL import Image

from utils import PROJECT_NAME, OPENCV_TO_OPENGL
from utils.logging_utils import Log
from mapper.splatam.splatam import render, get_rendervars
from mapper.splatam.utils.slam_external import calc_ssim

METRIC_NAMES = ['psnr', 'ssim', 'lpips', 'psnr_mapped', 'coverage', 'depth_l1_m']
# NOTE: pixels with rendered opacity above this count as covered by the map
COVERED_OPACITY = 0.5
EXAMPLE_JPEG_QUALITY = 90
MAP_KEYS = ['means3D', 'rgb_colors', 'unnorm_rotations', 'logit_opacities', 'log_scales']

def frame_to_w2c(c2w:np.ndarray) -> np.ndarray:
    # NOTE: same conversion as the mapper (SplaTAM.__get_np_data): the sensor c2w is OpenGL, the renderer wants OpenCV
    return np.linalg.inv(np.asarray(c2w, dtype=np.float64) @ OPENCV_TO_OPENGL)

def make_view(step:int, c2w:np.ndarray, rgb:np.ndarray, depth:np.ndarray) -> dict:
    '''rgb: float 0..1 or uint8 (H, W, 3); depth: (H, W) metres.'''
    return {'step': step, 'w2c': frame_to_w2c(c2w), 'rgb': rgb, 'depth': depth}

def load_lpips(device:torch.device):
    import lpips
    return lpips.LPIPS(net='alex', verbose=False).to(device).eval()

def collect_novel_views(dataset, trajectory_views:List[dict], count:int, radius:float, seed:int) -> List[dict]:
    '''Teleport the agent to random navigable points within radius of the trajectory, facing a random direction, and capture there.'''
    rng = np.random.RandomState(seed)
    sim = dataset._sim
    agent_state = sim.get_agent_state()
    sensor_height = float(agent_state.sensor_states['rgb'].position[1] - agent_state.position[1])
    # NOTE: habitat's y axis is up; navigable points are sampled around the point on the floor below the sensor
    trajectory_foot = np.stack([np.linalg.inv(view['w2c'])[:3, 3] for view in trajectory_views]) - np.array([0.0, sensor_height, 0.0])
    views = []
    attempts = 0
    while len(views) < count and attempts < count * 50:
        attempts += 1
        centre = trajectory_foot[rng.randint(len(trajectory_foot))]
        position = sim.pathfinder.get_random_navigable_point_near(centre, radius)
        if not np.all(np.isfinite(position)):
            continue
        yaw = rng.uniform(0, 2 * np.pi)
        rotation = quaternion.from_rotation_vector(np.array([0, yaw, 0]))
        sim.set_agent_state(position.tolist(), quaternion.as_float_array(rotation)[[1, 2, 3, 0]].tolist())
        frame = dataset.get_frame()
        views.append(make_view(len(views), frame['c2w'], frame['rgb'], frame['depth']))
    if len(views) < count:
        Log(f'Only found {len(views)} of {count} novel views', tag=PROJECT_NAME)
    return views

def psnr(prediction:torch.Tensor, target:torch.Tensor) -> float:
    mse = torch.mean((prediction - target) ** 2)
    return float(-10 * torch.log10(mse.clamp_min(1e-10)))

@torch.no_grad()
def evaluate_view(params:dict, intrinsics:np.ndarray, render_config:dict, lpips_model, view:dict, device:torch.device) -> Tuple[dict, np.ndarray]:
    scene_data, scene_depth_data = get_rendervars(params, view['w2c'], with_depth_sil=False)
    image, depth, opacity, _ = render(view['w2c'], intrinsics, scene_data, scene_depth_data, render_config)
    image = image.clamp(0, 1)
    target = torch.as_tensor(view['rgb'], device=device).float().permute(2, 0, 1)
    if view['rgb'].dtype == np.uint8:
        target = target / 255.0
    target_depth = torch.as_tensor(view['depth'], device=device).float()
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

def evaluate_view_sets(run_dir:str, scene_id:str, params:Dict[str, torch.Tensor], intrinsics:np.ndarray, render_config:dict, lpips_model,
                       view_sets:Dict[str, List[dict]], device:torch.device, settings:dict, example_count:int) -> Dict[str, dict]:
    '''Scores every view and writes <run_dir>/image_metrics.{csv,json} plus a few JPEG comparison panels (ground truth | map | error x4) in image_eval/.'''
    params = {key: params[key].detach().float() for key in MAP_KEYS}
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
                Image.fromarray(panel).save(os.path.join(examples_dir, f"{set_name}_{view['step']:04d}.jpg"), quality=EXAMPLE_JPEG_QUALITY)
        summaries[set_name] = summarise(rows)
        all_rows += rows
    with open(os.path.join(run_dir, 'image_metrics.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['set', 'step', *METRIC_NAMES])
        writer.writeheader()
        writer.writerows(all_rows)
    with open(os.path.join(run_dir, 'image_metrics.json'), 'w') as f:
        json.dump({'scene_id': scene_id, 'settings': {**settings, 'novel_views': len(view_sets.get('novel', [])), 'covered_opacity': COVERED_OPACITY}, **summaries}, f, indent=4)
    return summaries

def log_summaries(summaries:Dict[str, dict]) -> None:
    for set_name, summary in summaries.items():
        Log(f"{set_name} ({summary['views']} views): PSNR {summary['psnr']:.2f} dB, SSIM {summary['ssim']:.3f}, LPIPS {summary['lpips']:.3f}, "
            f"PSNR mapped {summary['psnr_mapped']:.2f} dB, coverage {summary['coverage'] * 100:.1f} %, depth L1 {summary['depth_l1_m'] * 100:.1f} cm",
            tag=PROJECT_NAME)
