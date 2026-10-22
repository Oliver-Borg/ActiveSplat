#!/usr/bin/env python
# Timeline (Gantt / trace chart) of a run: one row per stage from timing.csv (mapper process) and planner_timing.csv (planner process).
# Each event was recorded when it ended, so its bar spans [wall_time - ms, wall_time].
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import pandas as pd
from matplotlib.figure import Figure

# (stage, process, file, kind): 'production' stages are needed by the mapper or planner to run; 'debug' stages only produce visualisations or
# evaluation data (saved images, GUI renders, oracle scoring) and disappear when those options are off.
# NOTE: visibility_* also includes the opacity images, which are rendered only when windows are shown or runtime data is saved
STAGES = [('capture', 'mapper', 'timing.csv', 'production'), ('mapper', 'mapper', 'timing.csv', 'production'), ('optimise', 'mapper', 'timing.csv', 'production'),
          ('seed_points', 'mapper', 'timing.csv', 'production'), ('vggt_omega', 'mapper', 'timing.csv', 'production'),
          ('topdown_render', 'mapper', 'timing.csv', 'production'), ('visibility_global', 'mapper', 'timing.csv', 'production'),
          ('visibility_local', 'mapper', 'timing.csv', 'production'),
          ('topdown_wait', 'planner', 'planner_timing.csv', 'production'), ('opacity_wait', 'planner', 'planner_timing.csv', 'production'),
          ('plan_topdown', 'planner', 'planner_timing.csv', 'production'), ('plan_logic', 'planner', 'planner_timing.csv', 'production'),
          ('render_rgbd', 'mapper', 'timing.csv', 'debug'), ('topdown_render_saved', 'mapper', 'timing.csv', 'debug'),
          ('oracle_observe', 'mapper', 'timing.csv', 'debug')]
KIND_TITLES = {'production': 'PRODUCTION (needed to run)', 'debug': 'VISUALISATION / DEBUG (can be switched off)'}
PRODUCTION_COLORS = {'mapper': '#1f77b4', 'planner': '#ff7f0e'}
DEBUG_COLOR = '#737373'

def load(run_dir:str) -> pd.DataFrame:
    frames = []
    for name in sorted({stage[2] for stage in STAGES}):
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            frames.append(pd.read_csv(path))
    events = pd.concat(frames, ignore_index=True)
    events['end'] = events['wall_time']
    events['start'] = events['end'] - events['ms'] / 1000
    origin = events['start'].min()
    events[['start', 'end']] -= origin
    return events

def build_figure(run_dir:str, t0:float=None, t1:float=None) -> Figure:
    '''t0, t1: optional window in seconds since the first event.'''
    events = load(run_dir)
    rows, positions = layout(events)
    if t0 is not None or t1 is not None:
        events = events[(events['end'] >= (t0 or 0)) & (events['start'] <= (t1 if t1 is not None else 1e18))]
    fig = Figure(figsize=(16, 0.5 * (positions[-1] + 1) + 1.8))
    ax = fig.subplots()
    for (stage, process, _, kind), position in zip(rows, positions):
        part = events[events['stage'] == stage]
        waiting = stage.endswith('_wait')
        ax.broken_barh(list(zip(part['start'], part['end'] - part['start'])), (position - 0.4, 0.8),
                       facecolors=DEBUG_COLOR if kind == 'debug' else PRODUCTION_COLORS[process],
                       alpha=0.45 if waiting else 1.0, hatch='////' if kind == 'debug' else None, edgecolor='white' if kind == 'debug' else None, linewidth=0)
    for kind in KIND_TITLES:
        group = [position for (_, _, _, k), position in zip(rows, positions) if k == kind]
        if group:
            ax.axhspan(min(group) - 0.5, max(group) + 0.5, color='0.9' if kind == 'debug' else 'none', zorder=0)
            ax.text(1.002, (min(group) + max(group)) / 2, KIND_TITLES[kind], transform=ax.get_yaxis_transform(), rotation=270, va='center', ha='left', fontsize=8)
    ax.set_yticks(positions)
    ax.set_yticklabels([f'{stage}  ({process}{", waiting" if stage.endswith("_wait") else ""})' for stage, process, _, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(t0 or 0, t1 if t1 is not None else events['end'].max())
    ax.set_xlabel('wall-clock time since the first event (s)')
    ax.set_title(os.path.basename(os.path.normpath(run_dir)) + '   (blue = mapper process, orange = planner process, faded = waiting, grey hatched = visualisation/debug)', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    return fig

def layout(events:pd.DataFrame) -> tuple:
    '''Rows present in the events and their y positions, with a gap between the production and the debug group.'''
    present = set(events['stage'])
    rows = [stage for stage in STAGES if stage[0] in present]
    positions, y = [], 0.0
    for index, (_, _, _, kind) in enumerate(rows):
        if index and kind != rows[index - 1][3]:
            y += 0.8
        positions.append(y)
        y += 1.0
    return rows, positions

def build_interactive_figure(run_dir:str, t0:float=None, t1:float=None):
    '''Plotly version of the trace chart: hover for the stage, step and duration of an event, drag to zoom, double-click to reset, click a legend entry to hide a row.'''
    import plotly.graph_objects as go
    events = load(run_dir)
    rows, positions = layout(events)
    if t0 is not None or t1 is not None:
        events = events[(events['end'] >= (t0 or 0)) & (events['start'] <= (t1 if t1 is not None else 1e18))]
    figure = go.Figure()
    for (stage, process, _, kind), position in zip(rows, positions):
        part = events[events['stage'] == stage]
        waiting = stage.endswith('_wait')
        figure.add_trace(go.Bar(
            name=f'{stage}{" (waiting)" if waiting else ""}', legendgroup=kind, legendgrouptitle_text=KIND_TITLES[kind],
            orientation='h', y=[position] * len(part), x=(part['end'] - part['start']).tolist(), base=part['start'].tolist(), width=0.8,
            marker=dict(color=DEBUG_COLOR if kind == 'debug' else PRODUCTION_COLORS[process], opacity=0.45 if waiting else 1.0,
                        pattern_shape='/' if kind == 'debug' else '', line_width=0),
            customdata=list(zip(part['step'], part['ms'].round(1), part['start'].round(3))),
            hovertemplate=f'<b>{stage}</b> ({process} process)<br>step %{{customdata[0]}}<br>%{{customdata[1]}} ms<br>started at %{{customdata[2]}} s<extra></extra>'))
    shapes = [dict(type='rect', xref='paper', x0=0, x1=1, yref='y', y0=min(g) - 0.5, y1=max(g) + 0.5, fillcolor='rgba(128,128,128,0.15)', line_width=0, layer='below')
              for g in [[position for (_, _, _, k), position in zip(rows, positions) if k == 'debug']] if g]
    figure.update_layout(
        barmode='overlay', bargap=0, shapes=shapes, height=max(420, 34 * len(rows) + 160), margin=dict(l=10, r=10, t=40, b=40),
        title=dict(text=os.path.basename(os.path.normpath(run_dir)) + ' — blue: mapper process, orange: planner process, faded: waiting, grey hatched: visualisation/debug', font_size=13),
        xaxis=dict(title='wall-clock time since the first event (s)', rangeslider=dict(visible=True, thickness=0.06), range=[t0 or 0, t1] if t1 is not None else None),
        yaxis=dict(tickvals=positions, ticktext=[stage for stage, _, _, _ in rows], autorange='reversed', fixedrange=True, showgrid=False),
        legend=dict(groupclick='togglegroup'), hovermode='closest')
    return figure

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
