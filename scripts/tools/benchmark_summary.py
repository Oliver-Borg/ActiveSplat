#!/usr/bin/env python
# Summarise a benchmark_optimisations.sh manifest: per configuration the mean (and range over repeats) of the run wall time and the per-stage mean ms
# (stages the switches do not touch, e.g. capture, are the control for machine load), whether the actions and image metrics match the baseline run,
# plus a bar chart. Prints markdown; writes <out_dir>/benchmark.csv and <out_dir>/benchmark.jpg.
#   python scripts/tools/benchmark_summary.py <manifest.tsv> --out_dir experiments/<name>
import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

STAGES = ['topdown_render', 'visibility_global', 'visibility_local', 'mapper', 'optimise', 'capture']

def read_run(run_dir:str) -> dict:
    rows = defaultdict(list)
    wall = []
    with open(os.path.join(run_dir, 'timing.csv')) as f:
        for row in csv.DictReader(f):
            rows[row['stage']].append(float(row['ms']))
            wall.append(float(row['wall_time']))
    with open(os.path.join(run_dir, 'image_metrics.json')) as f:
        metrics = json.load(f)
    with open(os.path.join(run_dir, 'actions.txt')) as f:
        actions = f.read()
    return {'wall_s': max(wall) - min(wall), 'stage_ms': {s: float(np.mean(rows[s])) for s in STAGES if rows[s]}, 'calls': {s: len(rows[s]) for s in STAGES},
            'psnr': metrics['trajectory']['psnr'], 'novel_psnr': metrics['novel']['psnr'], 'actions': actions}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest')
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()
    groups = defaultdict(list)
    order = []
    with open(args.manifest) as f:
        for _, rep, spec, load_start, load_end, run_dir in csv.reader(f, delimiter='\t'):
            if spec not in order:
                order.append(spec)
            groups[spec].append({**read_run(run_dir), 'load': float(load_start), 'run': run_dir})
    reference_actions = groups['baseline'][0]['actions']
    table = []
    for spec in order:
        runs = groups[spec]
        row = {'spec': spec, 'repeats': len(runs), 'wall_mean': np.mean([r['wall_s'] for r in runs]), 'wall_min': min(r['wall_s'] for r in runs), 'wall_max': max(r['wall_s'] for r in runs),
               'same_actions': all(r['actions'].split()[:140] == reference_actions.split()[:140] for r in runs), 'psnr': runs[0]['psnr'], 'novel_psnr': runs[0]['novel_psnr'], 'load': np.mean([r['load'] for r in runs])}
        for stage in STAGES:
            row[stage] = np.mean([r['stage_ms'][stage] for r in runs if stage in r['stage_ms']])
        table.append(row)
    base = table[0]
    lines = ['| Optimisations (spec) | Wall s (mean, range) | vs baseline | topdown_render ms | visibility_global ms | visibility_local ms | mapper ms | optimise ms | capture ms (control) | load at start | first 140 actions = baseline | traj. / novel PSNR |', '|' + '---|' * 12]
    for row in table:
        lines.append(f"| {row['spec']} | {row['wall_mean']:.0f} ({row['wall_min']:.0f}-{row['wall_max']:.0f}) | {base['wall_mean'] / row['wall_mean']:.2f}x | " +
                     ' | '.join(f"{row[s]:.1f}" for s in STAGES) + f" | {row['load']:.1f} | {'yes' if row['same_actions'] else 'NO'} | {row['psnr']:.1f} / {row['novel_psnr']:.1f} |")
    print('\n'.join(lines))
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'benchmark.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['spec', 'repeats', 'wall_mean_s', 'wall_min_s', 'wall_max_s'] + [s + '_ms' for s in STAGES] + ['mean_load_at_start', 'same_actions_as_baseline', 'trajectory_psnr', 'novel_psnr'])
        for r in table:
            writer.writerow([r['spec'], r['repeats'], f"{r['wall_mean']:.1f}", f"{r['wall_min']:.1f}", f"{r['wall_max']:.1f}"] + [f"{r[s]:.2f}" for s in STAGES] + [f"{r['load']:.2f}", r['same_actions'], f"{r['psnr']:.2f}", f"{r['novel_psnr']:.2f}"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    labels = [r['spec'].replace(',', ',\n') for r in table]
    axes[0].bar(range(len(table)), [r['wall_mean'] for r in table], yerr=[[r['wall_mean'] - r['wall_min'] for r in table], [r['wall_max'] - r['wall_mean'] for r in table]], capsize=3)
    axes[0].set_ylabel('run wall time (s)')
    axes[1].bar(range(len(table)), [r['topdown_render'] for r in table], color='tab:orange')
    axes[1].set_ylabel('topdown_render mean (ms)')
    for ax in axes:
        ax.set_xticks(range(len(table)))
        ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'benchmark.jpg'), dpi=110, pil_kwargs={'quality': 88})

if __name__ == '__main__':
    main()
