#!/usr/bin/env python
# Metadata and metrics table for an experiment log (experiments/<date>_<name>/README.md): the git version, and per run the directory, scene, steps,
# env config, optimisations, pose noise and image metrics. Prints markdown (and writes <out_dir>/runs.json when given).
#   python scripts/tools/experiment_log.py "label=results/<run>" "label=results/<run>" ... [--out_dir experiments/<name>]
import argparse
import json
import os
import subprocess

def git(*args:str) -> str:
    return subprocess.run(['git', *args], capture_output=True, text=True).stdout.rstrip('\n')

def read_json(path:str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)

def describe(label:str, run_dir:str) -> dict:
    config, metrics = read_json(os.path.join(run_dir, 'config.json')), read_json(os.path.join(run_dir, 'image_metrics.json'))
    noise, optimisations = read_json(os.path.join(run_dir, 'pose_noise.json')), read_json(os.path.join(run_dir, 'optimisations.json'))
    return {
        'label': label, 'run': run_dir, 'scene': config.get('dataset', {}).get('scene_id'), 'steps': config.get('dataset', {}).get('step_num'),
        'env_config': os.path.relpath(config.get('env', {}).get('config', ''), os.getcwd()) if config else None,
        'mapping_iters': config.get('mapper', {}).get('mapping_iters'),
        'optimisations': optimisations.get('label', 'before the switches (baseline)'),
        'pose_noise': noise.get('settings'), 'pose_errors': noise.get('errors'),
        'trajectory': metrics.get('trajectory'), 'novel': metrics.get('novel'), 'image_eval_settings': metrics.get('settings'),
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+', help='label=run_dir')
    parser.add_argument('--out_dir', default=None)
    args = parser.parse_args()
    runs = [describe(*item.split('=', 1)) for item in args.runs]
    meta = {'git_commit': git('rev-parse', 'HEAD').strip(), 'git_branch': git('rev-parse', '--abbrev-ref', 'HEAD').strip(),
            'git_dirty_files': [line[3:] for line in git('status', '--short').splitlines() if not line.startswith('??') and 'submodules' not in line],
            'git_untracked': [line[3:] for line in git('status', '--short').splitlines() if line.startswith('??')],
            'gpu': subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True).stdout.splitlines()[:1]}
    lines = [f"Git: `{meta['git_commit'][:10]}` on `{meta['git_branch']}`; uncommitted tracked files: {', '.join(meta['git_dirty_files']) or 'none'}", '',
             '| Run | Scene | Steps | Sensor config | Optimisations | Pose noise | Run directory |', '|---|---|---|---|---|---|---|']
    for run in runs:
        noise = run['pose_noise']
        noise_text = 'off' if not noise else f"{noise['translation_drift'] * 100:g} % / {noise['rotation_drift'] * 100:g} % (seed {noise['seed']})"
        lines.append(f"| {run['label']} | {run['scene']} | {run['steps']} | `{run['env_config']}` | {run['optimisations']} | {noise_text} | `{run['run']}` |")
    lines += ['', '| Run | Set | PSNR | SSIM | LPIPS | Depth L1 (m) | Coverage | ATE RMSE (cm) | Mean rot. err (deg) |', '|---|---|---|---|---|---|---|---|---|']
    for run in runs:
        errors = run['pose_errors'] or {}
        for name in ('trajectory', 'novel'):
            m = run[name]
            if not m:
                lines.append(f"| {run['label']} | {name} | no metrics | | | | | | |")
                continue
            ate = f"{errors['ate_rmse_m'] * 100:.1f}" if errors else '0'
            rot = f"{errors['rot_err_mean_deg']:.2f}" if errors else '0'
            lines.append(f"| {run['label']} | {name} ({m['views']} views) | {m['psnr']:.1f} | {m['ssim']:.2f} | {m['lpips']:.3f} | {m['depth_l1_m']:.2f} | {m['coverage'] * 100:.1f} % | {ate} | {rot} |")
    print('\n'.join(lines))
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, 'runs.json'), 'w') as f:
            json.dump({'meta': meta, 'runs': runs}, f, indent=2)

if __name__ == '__main__':
    main()
