#!/usr/bin/env python
# Compare the final maps of runs with different simulated odometry drift at the SAME poses. Runs diverge (the planner reacts to the map), so each
# run's own image_eval panels show different places; here every map is rendered at the reference run's true poses (replay of its actions.txt) and compared
# with the simulator's view there. The reference is the run without pose_noise.json. A map from a drifting run lives in its estimated frame, so it
# is rendered at inv(A) @ T for the rigid ATE alignment A of its own trajectory (estimated poses from params.npz vs a replay of its actions).
#   python scripts/tools/drift_same_views.py results/<clean> results/<noisy> ... --out_dir <dir> --gpu_id 0
import os
import sys
PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
sys.path.append(PACKAGE_PATH)
sys.path.append(os.path.join(PACKAGE_PATH, 'src'))
import argparse
import json

import numpy as np
import torch
from PIL import Image, ImageDraw

from dataloader.dataloader import get_dataset
from evaluation.image_eval import evaluate_view, load_lpips, make_view, summarise, METRIC_NAMES
from utils.pose_noise import PoseNoise, DEFAULTS
sys.path.append(os.path.join(PACKAGE_PATH, "scripts", "judges"))
from eval_images import load_params, quaternion_to_matrix

def replay(run_dir:str, user_config:dict, keep_steps:set) -> tuple:
    '''(true w2c of every step, {step: view with rgb/depth for keep_steps}) from a replay of the run's actions.'''
    with open(os.path.join(run_dir, 'config.json')) as f:
        config = json.load(f)
    with open(os.path.join(run_dir, 'actions.txt')) as f:
        actions = [int(line.strip()) for line in f if line.strip()]
    dataset = get_dataset(config, user_config, 'Eval')
    dataset.setup()
    w2cs, views = [], {}
    for step in range(len(actions) + 1):
        if step > 0:
            dataset._sim.step(actions[step - 1])
        frame = dataset.get_frame()
        view = make_view(step, frame['c2w'], frame['rgb'], frame['depth'])
        w2cs.append(view['w2c'])
        if step in keep_steps:
            views[step] = view
    dataset.close()
    return w2cs, views

def alignment_of(true_w2cs:list, camera_data:dict) -> np.ndarray:
    '''A with true ~= A @ estimated, from the mapper's estimated poses in params.npz and the replayed true ones.'''
    fit = PoseNoise({**DEFAULTS, 'enabled': True})
    count = min(len(true_w2cs), camera_data['cam_trans'][0].shape[1])
    for step in range(count):
        est_w2c = np.eye(4)
        est_w2c[:3, :3] = quaternion_to_matrix(camera_data['cam_unnorm_rots'][0][:, step])
        est_w2c[:3, 3] = camera_data['cam_trans'][0][:, step]
        fit.rows.append((np.linalg.inv(true_w2cs[step]), np.linalg.inv(est_w2c)))
    return fit.alignment()

def label_of(run_dir:str) -> str:
    path = os.path.join(run_dir, 'pose_noise.json')
    if not os.path.isfile(path):
        return 'no drift'
    with open(path) as f:
        return f"{json.load(f)['settings']['translation_drift'] * 100:g} % drift"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render runs with different odometry drift at the same true poses.')
    parser.add_argument('runs', nargs='+', help='the first run must be the one without drift (the reference)')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--gpu_id', type=int, required=True)
    parser.add_argument('--user_config', default=os.path.join(PACKAGE_PATH, 'config', 'user_config.json'))
    parser.add_argument('--stride', type=int, default=25, help='score every n-th pose of the reference trajectory')
    parser.add_argument('--show_steps', type=int, nargs='+', default=[100, 300, 500, 700, 900], help='steps shown in the contact sheet')
    parser.add_argument('--width', type=int, default=256)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.user_config) as f:
        user_config = json.load(f)
    device = torch.device('cuda', args.gpu_id)
    lpips_model = load_lpips(device)

    with open(os.path.join(args.runs[0], 'actions.txt')) as f:
        reference_length = sum(1 for line in f if line.strip())
    steps = sorted(set(range(0, reference_length + 1, args.stride)) | set(args.show_steps))
    _, reference_views = replay(args.runs[0], user_config, set(steps))
    render_config = None
    columns, table = [], []
    for run_dir in args.runs:
        params, camera_data = load_params(os.path.join(run_dir, 'gaussians_data', 'params.npz'), device)
        if label_of(run_dir) == 'no drift':
            alignment = np.eye(4)
        else:
            alignment = alignment_of(replay(run_dir, user_config, set())[0], camera_data)
        with open(os.path.join(run_dir, 'config.json')) as f:
            config = json.load(f)
        view0 = next(iter(reference_views.values()))
        render_config = {'viz_w': view0['rgb'].shape[1], 'viz_h': view0['rgb'].shape[0], 'viz_near': 0.01, 'viz_far': 100.0}
        rows, panels = [], {}
        for step in steps:
            view = {**reference_views[step], 'w2c': reference_views[step]['w2c'] @ alignment}
            metrics, panel = evaluate_view(params, camera_data['intrinsics'], render_config, lpips_model, view, device)
            rows.append(metrics)
            third = panel.shape[1] // 3
            panels[step] = (panel[:, :third], panel[:, third:2 * third], metrics)
        summary = summarise([{**row} for row in rows])
        table.append((label_of(run_dir), os.path.basename(run_dir.rstrip('/')), summary))
        columns.append((label_of(run_dir), panels))
        del params
        torch.cuda.empty_cache()

    lines = ['| run | drift | poses | PSNR | SSIM | LPIPS | PSNR mapped | coverage % | depth L1 cm |', '|---|---|---|---|---|---|---|---|---|']
    for label, name, s in table:
        lines.append(f"| {name} | {label} | {s['views']} | {s['psnr']:.2f} | {s['ssim']:.3f} | {s['lpips']:.3f} | {s['psnr_mapped']:.2f} | {s['coverage'] * 100:.1f} | {s['depth_l1_m'] * 100:.1f} |")
    with open(os.path.join(args.out_dir, 'same_views.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    # contact sheet: one row per shown step; simulator, then every run's map
    width = args.width
    header = 20
    height = round(width * view0['rgb'].shape[0] / view0['rgb'].shape[1])
    sheet = Image.new('RGB', (width * (1 + len(columns)), header + len(args.show_steps) * (height + header)), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 4), 'simulator', fill='black')
    for index, (label, _) in enumerate(columns):
        draw.text((width * (index + 1) + 4, 4), label, fill='black')
    for row_index, step in enumerate(args.show_steps):
        y = header + row_index * (height + header)
        draw.text((4, y + 4), f'pose at step {step}', fill='black')
        sheet.paste(Image.fromarray(columns[0][1][step][0]).resize((width, height)), (0, y + header))
        for index, (label, panels) in enumerate(columns):
            _, render, metrics = panels[step]
            draw.text((width * (index + 1) + 4, y + 4), f"PSNR {metrics['psnr']:.1f}  SSIM {metrics['ssim']:.2f}", fill='black')
            sheet.paste(Image.fromarray(render).resize((width, height)), (width * (index + 1), y + header))
    sheet.save(os.path.join(args.out_dir, 'same_views.jpg'), quality=92)
