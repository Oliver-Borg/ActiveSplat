#!/usr/bin/env python
# Where the time of one capture -> map -> plan step goes, and what it means for a real drone that must settle before each capture.
# From the run traces (trace_*.jsonl): the interval between two consecutive 'capture' spans is one step; the spans inside it are summed by phase.
# Then a simple real-time model: step time = settle + motion + compute (sequential, the drone hovers while the computer works) or, as a best case,
# the compute overlapped with settle + motion. Settle time and speeds are ASSUMED, not measured.
#   python scripts/tools/step_breakdown.py "label=run_dir" "label=run_dir" ... --out_dir experiments/<name> [--forward_speed 0.5 --turn_rate 60 --tilt_rate 30]
# Runs with the same label are pooled (repeats). Prints markdown, writes breakdown.csv, step_breakdown.jpg and realtime_model.jpg.
import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

PHASES = [
    ('capture', 'capture (simulator render of the sensor frame)', ['capture']),
    ('map: other', 'mapper: frame prep, keyframes, densify', None),
    ('map: optimise', 'mapper: Gaussian optimisation', ['optimise']),
    ('render', 'top-down map render for the planner', ['topdown_render']),
    ('display', 'display image conversion (visualisation only)', ['topdown: display image conversion', 'topdown: display images + ui update']),
]
SETTLE_TIMES = [0.0, 0.1, 0.25, 0.5, 1.0]

def read_spans(run_dir:str) -> list:
    spans = []
    for name in sorted(os.listdir(run_dir)):
        if name.startswith('trace_') and name.endswith('.jsonl'):
            with open(os.path.join(run_dir, name)) as f:
                for line in f:
                    event = json.loads(line)
                    if event['ph'] == 'X':
                        spans.append(event)
    spans.sort(key=lambda e: e['ts'])
    return spans

def motion_seconds(run_dir:str, forward_speed:float, turn_rate:float, tilt_rate:float) -> tuple:
    with open(os.path.join(run_dir, 'config.json')) as f:
        env = json.load(f)['env']['config']
    with open(env) as f:
        simulator = yaml.safe_load(f)['habitat']['simulator']
    forward, turn, tilt = float(simulator['forward_step_size']), float(simulator['turn_angle']), float(simulator['tilt_angle'])
    seconds = {1: forward / forward_speed, 2: turn / turn_rate, 3: turn / turn_rate, 4: tilt / tilt_rate, 5: tilt / tilt_rate}
    with open(os.path.join(run_dir, 'actions.txt')) as f:
        actions = [int(a) for a in f.read().split()]
    return float(np.mean([seconds.get(a, 0.0) for a in actions])), {a: actions.count(a) for a in sorted(set(actions))}, (forward, turn, tilt)

def step_table(run_dir:str, skip:int=10) -> dict:
    '''Per step (interval between consecutive captures, first `skip` steps dropped): ms of each phase, the interval, and the rest.'''
    spans = read_spans(run_dir)
    captures = [e for e in spans if e['name'] == 'capture']
    starts = np.array([e['ts'] for e in captures])
    rows = []
    for i in range(skip, len(captures) - 1):
        lo, hi = starts[i], starts[i + 1]
        inside = [e for e in spans if lo <= e['ts'] < hi]
        def total(names):
            return sum(e['dur'] for e in inside if e['name'] in names) / 1000
        row = {'interval': (hi - lo) / 1000}
        row['capture'] = total(['capture'])
        mapper = total(['mapper'])
        row['map: optimise'] = total(['optimise'])
        row['map: other'] = mapper - row['map: optimise']
        row['render'] = total(['topdown_render'])
        row['display'] = total(PHASES[4][2])
        row['rest'] = row['interval'] - sum(row[p[0]] for p in PHASES)
        rows.append(row)
    return {key: np.array([r[key] for r in rows]) for key in rows[0]}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+', help='label=run_dir (same label: pooled)')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--forward_speed', type=float, default=0.5)
    parser.add_argument('--turn_rate', type=float, default=60.0)
    parser.add_argument('--tilt_rate', type=float, default=30.0)
    args = parser.parse_args()
    labels, groups, motion = [], defaultdict(list), {}
    for item in args.runs:
        label, run_dir = item.split('=', 1)
        if label not in labels:
            labels.append(label)
        groups[label].append(step_table(run_dir))
        motion[label] = motion_seconds(run_dir, args.forward_speed, args.turn_rate, args.tilt_rate)
    pooled = {label: {key: np.concatenate([t[key] for t in tables]) for key in tables[0]} for label, tables in groups.items()}
    keys = [p[0] for p in PHASES] + ['rest']
    lines = ['| Phase (ms per step) | ' + ' | '.join(f'{label} mean (median)' for label in labels) + ' |', '|---|' + '---|' * len(labels)]
    descriptions = {p[0]: p[1] for p in PHASES}
    descriptions['rest'] = 'rest: handshakes between threads, ROS messages, planning, waiting, the occasional visibility computation'
    for key in keys + ['interval']:
        lines.append(f"| {descriptions.get(key, 'whole step (capture to next capture)')} | " + ' | '.join(f"{pooled[l][key].mean():.0f} ({np.median(pooled[l][key]):.0f})" for l in labels) + ' |')
    print('\n'.join(lines))
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'breakdown.csv'), 'w') as f:
        f.write('label,phase,mean_ms,median_ms,steps\n')
        for label in labels:
            for key in keys + ['interval']:
                f.write(f'{label},{key},{pooled[label][key].mean():.1f},{np.median(pooled[label][key]):.1f},{len(pooled[label][key])}\n')
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(labels))
    for key, color in zip(keys, ['tab:gray', 'tab:blue', 'tab:green', 'tab:orange', 'tab:pink', 'lightgray']):
        values = np.array([pooled[l][key].mean() for l in labels])
        ax.bar(labels, values, bottom=bottom, label=key, color=color)
        bottom += values
    ax.set_ylabel('mean ms per step (simulator, compute only)')
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'step_breakdown.jpg'), dpi=110, pil_kwargs={'quality': 88})
    plt.close()
    print()
    forward_turn_tilt = motion[labels[0]][2]
    print(f'Motion per step from the run\'s actions {motion[labels[0]][1]} (1 forward {forward_turn_tilt[0]} m, 2/3 turn {forward_turn_tilt[1]} deg, 4/5 tilt {forward_turn_tilt[2]} deg) at '
          f'{args.forward_speed} m/s, {args.turn_rate} deg/s, {args.tilt_rate} deg/s: mean {motion[labels[0]][0]:.3f} s per step.')
    move = motion[labels[0]][0]
    compute = {label: (pooled[label]['interval'].mean() - pooled[label]['capture'].mean()) / 1000 for label in labels}
    lines = ['| Settle + capture (s) | ' + ' | '.join(f'{l}: sequential s/step' for l in labels) + ' | ' + ' | '.join(f'{l}: overlapped s/step' for l in labels) + f' | saving of {labels[-1]} vs {labels[0]}: sequential | overlapped |', '|---|' + '---|' * (2 * len(labels) + 2)]
    plot = {l: ([], []) for l in labels}
    for settle in SETTLE_TIMES:
        sequential = {l: settle + move + compute[l] for l in labels}
        overlapped = {l: max(settle + move, compute[l]) for l in labels}
        for l in labels:
            plot[l][0].append(sequential[l])
            plot[l][1].append(overlapped[l])
        lines.append(f'| {settle:g} | ' + ' | '.join(f'{sequential[l]:.2f}' for l in labels) + ' | ' + ' | '.join(f'{overlapped[l]:.2f}' for l in labels) +
                     f' | {100 * (1 - sequential[labels[-1]] / sequential[labels[0]]):.0f} % | {100 * (1 - overlapped[labels[-1]] / overlapped[labels[0]]):.0f} % |')
    print('\n'.join(lines))
    print('\nCompute per step (s, capture span excluded): ' + ', '.join(f'{l} {compute[l]:.3f}' for l in labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    for l, color in zip(labels, ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']):
        ax.plot(SETTLE_TIMES, plot[l][0], 'o-', color=color, label=f'{l}, sequential')
        ax.plot(SETTLE_TIMES, plot[l][1], 's--', color=color, label=f'{l}, compute overlapped with motion')
    ax.set_xlabel('assumed settle + capture time per step (s)')
    ax.set_ylabel('real time per step (s)')
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'realtime_model.jpg'), dpi=110, pil_kwargs={'quality': 88})

if __name__ == '__main__':
    main()
