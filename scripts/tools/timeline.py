#!/usr/bin/env python
# Timeline (Gantt / trace chart) of a run: one row per stage from timing.csv (mapper process) and planner_timing.csv (planner process).
# Each event was recorded when it ended, so its bar spans [wall_time - ms, wall_time].
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import pandas as pd
from matplotlib.figure import Figure

ROWS = [('mapper process', 'timing.csv', ['capture', 'mapper', 'optimise', 'topdown_render', 'visibility_global', 'visibility_local', 'seed_points', 'vggt_omega', 'oracle_observe']),
        ('planner process', 'planner_timing.csv', ['topdown_wait', 'opacity_wait', 'plan_topdown', 'plan_logic'])]

def load(run_dir:str) -> pd.DataFrame:
    frames = []
    for process, name, order in ROWS:
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            frame = pd.read_csv(path)
            frame['process'] = process
            frames.append(frame)
    events = pd.concat(frames, ignore_index=True)
    events['end'] = events['wall_time']
    events['start'] = events['end'] - events['ms'] / 1000
    origin = events['start'].min()
    events[['start', 'end']] -= origin
    return events

def build_figure(run_dir:str, t0:float=None, t1:float=None) -> Figure:
    '''t0, t1: optional window in seconds since the first event.'''
    events = load(run_dir)
    labels = [(process, stage) for process, _, stages in ROWS for stage in stages if ((events['process'] == process) & (events['stage'] == stage)).any()]
    if t0 is not None or t1 is not None:
        events = events[(events['end'] >= (t0 or 0)) & (events['start'] <= (t1 if t1 is not None else 1e18))]
    fig = Figure(figsize=(16, 0.55 * len(labels) + 1.5))
    ax = fig.subplots()
    colors = matplotlib.colormaps['tab10']
    for row, (process, stage) in enumerate(labels):
        part = events[(events['process'] == process) & (events['stage'] == stage)]
        ax.broken_barh(list(zip(part['start'], part['end'] - part['start'])), (row - 0.4, 0.8), facecolors=colors(row % 10), linewidth=0)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([f'{stage}  ({process.split()[0]})' for process, stage in labels])
    ax.invert_yaxis()
    ax.set_xlim(t0 or 0, t1 if t1 is not None else events['end'].max())
    ax.set_xlabel('wall-clock time since the first event (s)')
    ax.set_title(os.path.basename(os.path.normpath(run_dir)))
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    return fig

def plot(run_dir:str, out:str, t0:float=None, t1:float=None) -> None:
    build_figure(run_dir, t0, t1).savefig(out, dpi=150)
    print(out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Timeline of per-stage timings in a run.')
    parser.add_argument('run_dir')
    parser.add_argument('--out', default=None, help='default: <run_dir>/timeline.png')
    parser.add_argument('--from_s', type=float, default=None, help='zoom: start of the window (s)')
    parser.add_argument('--to_s', type=float, default=None, help='zoom: end of the window (s)')
    args = parser.parse_args()
    plot(args.run_dir, args.out or os.path.join(args.run_dir, 'timeline.png'), args.from_s, args.to_s)
