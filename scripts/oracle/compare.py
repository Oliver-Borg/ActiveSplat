#!/usr/bin/env python
# Compares planners at an equal step budget from scored runs: coverage curves, area under the curve, steps to 80 / 90 % coverage,
# path length, two-direction coverage and image metrics, per scene and averaged over scenes. Runs are grouped by the label in
# <run>/label.txt (runs without one are ignored) and by scene; repeats of the same label and scene are averaged (the spread shows
# the run-to-run noise). Needs <run>/coverage.csv (scripts/oracle/coverage.py) and, for image columns, <run>/image_metrics.json.
#   python scripts/oracle/compare.py --steps 1000 --out docs/oracle_results.md --plots_dir docs/oracle_plots
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
CHECKPOINTS = (100, 300, 500)

def read_run(run_dir:str, steps:int) -> dict:
    coverage = pd.read_csv(os.path.join(run_dir, 'coverage.csv'))
    coverage = coverage[coverage['step'] <= steps]
    curve = coverage['coverage'].to_numpy() * 100
    # NOTE: a plan that ends a few actions short of the budget keeps its final coverage
    curve = np.concatenate([curve, np.full(max(steps + 1 - len(curve), 0), curve[-1])])
    row = {'curve': curve, 'steps': int(coverage['step'].max())}
    for step in CHECKPOINTS:
        row[f'cov@{step}'] = curve[min(step, len(curve) - 1)]
    row[f'cov@{steps}'] = curve[-1]
    row['auc'] = float(curve.mean())
    for level in (80, 90):
        reached = np.nonzero(curve >= level)[0]
        row[f'steps to {level}%'] = float(reached[0]) if len(reached) else float('nan')
    row['path m'] = float(coverage['path_length_m'].iloc[-1])
    row['two directions %'] = float(coverage['coverage_multi'].iloc[-1] * 100) if 'coverage_multi' in coverage else float('nan')
    metrics_path = os.path.join(run_dir, 'image_metrics.json')
    if os.path.isfile(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        row.update({'PSNR': metrics['trajectory']['psnr'], 'SSIM': metrics['trajectory']['ssim'], 'LPIPS': metrics['trajectory']['lpips'],
                    'novel PSNR': metrics['novel']['psnr'], 'novel SSIM': metrics['novel']['ssim'], 'novel LPIPS': metrics['novel']['lpips']})
    return row

def collect(results_dir:str, steps:int) -> dict:
    '''{scene: {label: [run dict, ...]}}'''
    groups = defaultdict(lambda: defaultdict(list))
    for name in sorted(os.listdir(results_dir)):
        run_dir = os.path.join(results_dir, name)
        if not (os.path.isfile(os.path.join(run_dir, 'label.txt')) and os.path.isfile(os.path.join(run_dir, 'coverage.csv'))):
            continue
        with open(os.path.join(run_dir, 'label.txt')) as f:
            label = f.read().strip()
        with open(os.path.join(run_dir, 'config.json')) as f:
            scene = json.load(f)['dataset']['scene_id']
        row = read_run(run_dir, steps)
        if row['steps'] < steps * 0.95:
            print(f'skipping {name}: only {row["steps"]} steps')
            continue
        row['run'] = name
        groups[scene][label].append(row)
    return groups

COLUMNS = [('cov@100', '{:.1f}'), ('cov@300', '{:.1f}'), ('cov@500', '{:.1f}'), ('cov@{steps}', '{:.1f}'), ('auc', '{:.1f}'),
           ('steps to 80%', '{:.0f}'), ('steps to 90%', '{:.0f}'), ('path m', '{:.1f}'), ('two directions %', '{:.1f}'),
           ('PSNR', '{:.1f}'), ('SSIM', '{:.3f}'), ('LPIPS', '{:.3f}'),
           ('novel PSNR', '{:.1f}'), ('novel SSIM', '{:.3f}'), ('novel LPIPS', '{:.3f}')]

def summarise(runs:list, steps:int) -> list:
    cells = []
    for column, fmt in COLUMNS:
        column = column.format(steps=steps)
        values = np.array([run.get(column, np.nan) for run in runs], dtype=float)
        if np.all(np.isnan(values)):
            cells.append('–')
            continue
        text = fmt.format(np.nanmean(values))
        if len(runs) > 1 and np.sum(~np.isnan(values)) > 1:
            text += ' ± ' + fmt.format(np.nanstd(values))
        cells.append(text)
    return cells

def table(rows:dict, steps:int, order:list) -> str:
    header = '| method | runs | ' + ' | '.join(column.format(steps=steps) for column, _ in COLUMNS) + ' |'
    lines = [header, '|' + '---|' * (len(COLUMNS) + 2)]
    for label in order:
        if label in rows:
            lines.append(f'| {label} | {len(rows[label])} | ' + ' | '.join(summarise(rows[label], steps)) + ' |')
    return '\n'.join(lines)

def plot(curves:dict, title:str, path:str, steps:int, order:list) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 4.2))
    for label in order:
        if label in curves:
            mean = np.mean([run['curve'][:steps + 1] for run in curves[label]], axis=0)
            axis.plot(np.arange(len(mean)), mean, label=f'{label} ({len(curves[label])})')
    axis.set_xlabel('step')
    axis.set_ylabel('observable surface seen (%)')
    axis.set_title(title)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare planners at an equal step budget.')
    parser.add_argument('--results_dir', type=str, default=os.path.join(PACKAGE_PATH, 'results'))
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--out', type=str, default=None, help='write the markdown here (default: print)')
    parser.add_argument('--plots_dir', type=str, default=None, help='save a coverage-curve PNG per scene (and one over all scenes) here')
    parser.add_argument('--order', type=str, default='random walk,native,oracle-info,tour,NBV', help='method order in tables and plots; other labels follow')
    args = parser.parse_args()

    groups = collect(args.results_dir, args.steps)
    wanted = [label.strip() for label in args.order.split(',')]
    all_labels = sorted({label for scene in groups.values() for label in scene}, key=lambda label: (wanted.index(label) if label in wanted else len(wanted), label))
    sections = [f'Equal budget of {args.steps} steps; coverage is the percentage of the observable ground-truth surface seen, "auc" the mean of the coverage curve, '
                '"two directions" the share seen from two viewing directions at least 20° apart. Image metrics: PSNR / SSIM / LPIPS on the trajectory views, and "novel" on 100 held-out views within 1.5 m of the trajectory (these also penalise unexplored or poorly mapped areas). '
                '± is the standard deviation over repeated runs.']
    per_label = defaultdict(list)
    for scene in sorted(groups):
        sections.append(f'### {scene}\n\n' + table(groups[scene], args.steps, all_labels))
        if args.plots_dir:
            os.makedirs(args.plots_dir, exist_ok=True)
            plot(groups[scene], scene, os.path.join(args.plots_dir, f'coverage_{scene}.png'), args.steps, all_labels)
        for label, runs in groups[scene].items():
            # NOTE: a scene counts once per method (mean over its repeats), so scenes with more repeats do not dominate the average
            merged = {key: float(np.nanmean([run.get(key, np.nan) for run in runs])) for key in runs[0] if key not in ('curve', 'run', 'steps')}
            merged['curve'] = np.mean([run['curve'] for run in runs], axis=0)
            per_label[label].append(merged)
    sections.append('### Mean over scenes\n\n' + table(per_label, args.steps, all_labels))
    if args.plots_dir:
        plot(per_label, 'mean over scenes', os.path.join(args.plots_dir, 'coverage_all_scenes.png'), args.steps, all_labels)
    text = '\n\n'.join(sections) + '\n'
    if args.out:
        with open(args.out, 'w') as f:
            f.write(text)
        print(f'written {args.out}')
    else:
        print(text)
