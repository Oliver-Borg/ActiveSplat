#!/usr/bin/env python
# Compare runs of one scene made with different simulated odometry drift (pose_noise, see src/utils/pose_noise.py): a metrics table and, per scene,
# a contact sheet of the simulator's view next to the map's render at the same pose for every drift level (the panels written by the in-run image
# evaluation: simulator | map | difference; the difference is dropped here).
#   python scripts/tools/drift_sweep.py results/<run> results/<run> ... --out_dir <dir> [--view trajectory_0060]
# Runs without pose_noise.json count as drift 0.
import argparse
import json
import os
from collections import defaultdict

from PIL import Image, ImageDraw

def read_run(run_dir:str) -> dict:
    with open(os.path.join(run_dir, 'config.json')) as f:
        scene = json.load(f)['dataset']['scene_id']
    drift, ate, rot = 0.0, 0.0, 0.0
    noise_path = os.path.join(run_dir, 'pose_noise.json')
    if os.path.isfile(noise_path):
        with open(noise_path) as f:
            noise = json.load(f)
        drift = noise['settings']['translation_drift'] * 100
        ate, rot = noise['errors']['ate_rmse_m'] * 100, noise['errors']['rot_err_mean_deg']
    with open(os.path.join(run_dir, 'image_metrics.json')) as f:
        metrics = json.load(f)
    return {'run': run_dir, 'scene': scene, 'drift_percent': drift, 'ate_cm': ate, 'rot_deg': rot, 'trajectory': metrics['trajectory'], 'novel': metrics['novel']}

def contact_sheet(runs:list, view:str, path:str, width:int=384) -> None:
    rows = []
    for run in runs:
        panel_path = os.path.join(run['run'], 'image_eval', view + '.jpg')
        if not os.path.isfile(panel_path):
            continue
        panel = Image.open(panel_path)
        third = panel.width // 3
        pair = panel.crop((0, 0, 2 * third, panel.height))
        pair = pair.resize((2 * width, round(pair.height * 2 * width / pair.width)))
        label = f"drift {run['drift_percent']:g} %   PSNR {run['trajectory']['psnr']:.1f}  SSIM {run['trajectory']['ssim']:.2f}   ATE {run['ate_cm']:.1f} cm, rot {run['rot_deg']:.1f} deg"
        rows.append((label, pair))
    if not rows:
        return
    header = 22
    sheet = Image.new('RGB', (2 * width, sum(pair.height + header for _, pair in rows) + header), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 4), f'{view}:  simulator (left)  |  map render (right)', fill='black')
    y = header
    for label, pair in rows:
        draw.text((4, y + 4), label, fill='black')
        sheet.paste(pair, (0, y + header))
        y += pair.height + header
    sheet.save(path, quality=92)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Table and contact sheets of runs with different simulated odometry drift.')
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--view', default='trajectory_0060', help='panel (image_eval/<view>.jpg) shown in the contact sheets')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    by_scene = defaultdict(list)
    for run_dir in args.runs:
        by_scene[read_run(run_dir)['scene']].append(read_run(run_dir))
    lines = ['| scene | drift % | ATE cm | rot err deg | traj PSNR | traj SSIM | traj LPIPS | traj depth L1 cm | novel PSNR | novel SSIM | run |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for scene, runs in sorted(by_scene.items()):
        runs.sort(key=lambda run: run['drift_percent'])
        for run in runs:
            t, n = run['trajectory'], run['novel']
            lines.append(f"| {scene} | {run['drift_percent']:g} | {run['ate_cm']:.1f} | {run['rot_deg']:.1f} | {t['psnr']:.1f} | {t['ssim']:.2f} | {t['lpips']:.2f} | {t['depth_l1_m'] * 100:.1f} | "
                         f"{n['psnr']:.1f} | {n['ssim']:.2f} | {os.path.basename(run['run'])} |")
        contact_sheet(runs, args.view, os.path.join(args.out_dir, f'{scene}_{args.view}.jpg'))
    with open(os.path.join(args.out_dir, 'drift_sweep.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
