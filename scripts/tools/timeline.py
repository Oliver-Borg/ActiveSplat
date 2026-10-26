#!/usr/bin/env python
# Timeline (Gantt / trace chart) of a run: one row per stage from timing.csv (mapper process) and planner_timing.csv (planner process).
# Each event was recorded when it ended, so its bar spans [wall_time - ms, wall_time].
import argparse
import json
import os

import matplotlib
import matplotlib.colors
matplotlib.use('Agg')
import pandas as pd
from matplotlib.figure import Figure

def run_title(run_dir:str) -> str:
    '''Run folder name and, when recorded, its optimisation label (optimisations.json).'''
    name = os.path.basename(os.path.normpath(run_dir))
    path = os.path.join(run_dir, 'optimisations.json')
    if os.path.isfile(path):
        with open(path) as f:
            name += f" [{json.load(f)['label']}]"
    return name

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
    ax.set_title(run_title(run_dir) + '   (blue = mapper process, orange = planner process, faded = waiting, grey hatched = visualisation/debug)', fontsize=10)
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
        title=dict(text=run_title(run_dir) + ' — blue: mapper process, orange: planner process, faded: waiting, grey hatched: visualisation/debug', font_size=13),
        xaxis=dict(title='wall-clock time since the first event (s)', rangeslider=dict(visible=True, thickness=0.06), range=[t0 or 0, t1] if t1 is not None else None),
        yaxis=dict(tickvals=positions, ticktext=[stage for stage, _, _, _ in rows], autorange='reversed', fixedrange=True, showgrid=False),
        legend=dict(groupclick='togglegroup'), hovermode='closest')
    return figure

# ---- real traces: <run>/trace_<process>.jsonl written by utils.trace (spans with pid / tid; nesting by time containment) ----
DEBUG_SPANS = {'render_rgbd', 'topdown_render_saved', 'oracle_observe', 'topdown: display image conversion', 'topdown: display images + ui update',
               'opacity image save'}
# NOTE: waiting = blocked on another thread or process, not computing
WAITING_SPANS = {'topdown_wait', 'opacity_wait', 'service get_topdown', 'service get_opacity', 'wait for request', 'wait for worker',
                 'wait for main loop (render + handshake)', 'wait for main loop (global visibility)', 'wait for main loop (local visibility)',
                 'topdown: handshake with service thread', 'visibility: handshake with service thread', 'topdown: wait for gaussian lock'}
PROCESS_ORDER = ['mapper node', 'planner node', 'vggt_omega worker']

def has_trace(run_dir:str) -> bool:
    return any(name.startswith('trace_') and name.endswith('.jsonl') for name in os.listdir(run_dir))

def read_trace_events(run_dir:str) -> tuple:
    '''(raw span events, {pid: process name}, {(pid, tid): thread name}) from every trace_*.jsonl of the run.'''
    spans, processes, threads = [], {}, {}
    for name in sorted(os.listdir(run_dir)):
        if not (name.startswith('trace_') and name.endswith('.jsonl')):
            continue
        with open(os.path.join(run_dir, name)) as f:
            for line in f:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue   # a line cut off by a killed process
                if event['ph'] == 'M' and event['name'] == 'process_name':
                    processes[event['pid']] = event['args']['name']
                elif event['ph'] == 'M' and event['name'] == 'thread_name':
                    threads[(event['pid'], event['tid'])] = event['args']['name']
                elif event['ph'] == 'X':
                    spans.append(event)
    return spans, processes, threads

def load_trace(run_dir:str) -> tuple:
    '''(spans as a DataFrame with seconds since the first span, lanes, rows) where lanes = [(process, thread, first y, lane count)].'''
    spans, processes, threads = read_trace_events(run_dir)
    events = pd.DataFrame(spans)
    events['process'] = events['pid'].map(processes).fillna(events['pid'].astype(str))
    events['thread'] = [threads.get((pid, tid), str(tid)) for pid, tid in zip(events['pid'], events['tid'])]
    origin = events['ts'].min()
    events['start'] = (events['ts'] - origin) / 1e6
    events['end'] = events['start'] + events['dur'] / 1e6
    # NOTE: each thread's spans nest, so a stack of open spans gives each one's depth (a lane below its parent)
    depth = pd.Series(0, index=events.index)
    for _, group in events.groupby(['pid', 'tid']):
        stack = []
        for index in group.sort_values(['start', 'end'], ascending=[True, False]).index:
            while stack and stack[-1] <= events.at[index, 'start'] + 1e-9:
                stack.pop()
            depth[index] = len(stack)
            stack.append(events.at[index, 'end'])
    events['depth'] = depth
    order = lambda name: PROCESS_ORDER.index(name) if name in PROCESS_ORDER else len(PROCESS_ORDER)
    threads_in_order = events.groupby(['process', 'thread'])['start'].min().reset_index()
    threads_in_order['order'] = threads_in_order['process'].map(order)
    threads_in_order = threads_in_order.sort_values(['order', 'process', 'start'])
    lanes, y = [], 0.0
    for _, row in threads_in_order.iterrows():
        count = int(events[(events['process'] == row['process']) & (events['thread'] == row['thread'])]['depth'].max()) + 1
        lanes.append((row['process'], row['thread'], y, count))
        y += count + 0.5
    first_y = {(process, thread): y for process, thread, y, _ in lanes}
    events['y'] = [first_y[(process, thread)] + depth for process, thread, depth in zip(events['process'], events['thread'], events['depth'])]
    return events, lanes

def span_color(name:str, names:list) -> str:
    # NOTE: 60 distinct colours (the three tab20 maps) so each span name keeps its own
    palette = [matplotlib.colors.to_hex(color) for name_ in ('tab20', 'tab20b', 'tab20c') for color in matplotlib.colormaps[name_].colors]
    return DEBUG_COLOR if name in DEBUG_SPANS else palette[names.index(name) % len(palette)]

def lane_labels(lanes:list) -> tuple:
    return [y for _, _, y, _ in lanes], [f'{process} · {thread}' for process, thread, _, _ in lanes]

def build_trace_figure(run_dir:str, t0:float=None, t1:float=None):
    '''Plotly flame-style trace: process -> thread -> nested spans. Hover for args, drag to zoom, click a legend entry to hide a span name.'''
    import plotly.graph_objects as go
    events, lanes = load_trace(run_dir)
    if t0 is not None or t1 is not None:
        events = events[(events['end'] >= (t0 or 0)) & (events['start'] <= (t1 if t1 is not None else 1e18))]
    names = sorted(events['name'].unique())
    figure = go.Figure()
    for name in names:
        part = events[events['name'] == name]
        debug = name in DEBUG_SPANS
        figure.add_trace(go.Bar(
            name=name, legendgroup='debug' if debug else 'production', legendgrouptitle_text=KIND_TITLES['debug' if debug else 'production'],
            orientation='h', y=part['y'].tolist(), x=(part['end'] - part['start']).tolist(), base=part['start'].tolist(), width=0.85,
            marker=dict(color=span_color(name, names), opacity=0.45 if name in WAITING_SPANS else 1.0, pattern_shape='/' if debug else '', line_width=0),
            customdata=list(zip(part['dur'] / 1000, part['start'], part['process'], part['thread'], part['args'].map(lambda a: ', '.join(f'{k}={v}' for k, v in a.items())))),
            hovertemplate=f'<b>{name}</b><br>%{{customdata[0]:.1f}} ms, started at %{{customdata[1]:.3f}} s<br>%{{customdata[2]}} · %{{customdata[3]}}<br>%{{customdata[4]}}<extra></extra>'))
    shapes = []
    for process, thread, y, count in lanes:
        shapes.append(dict(type='line', xref='paper', x0=0, x1=1, yref='y', y0=y - 0.45, y1=y - 0.45, line=dict(color='rgba(128,128,128,0.5)', width=1)))
    ticks, labels = lane_labels(lanes)
    figure.update_layout(
        barmode='overlay', bargap=0, shapes=shapes, height=max(420, 26 * (lanes[-1][2] + lanes[-1][3]) + 160), margin=dict(l=10, r=10, t=40, b=40),
        title=dict(text=run_title(run_dir) + ' — rows are threads grouped by process; a span nested in another sits on the lane below it; faded: waiting; grey hatched: visualisation/debug', font_size=13),
        xaxis=dict(title='wall-clock time since the first span (s)', rangeslider=dict(visible=True, thickness=0.06), range=[t0 or 0, t1] if t1 is not None else None),
        yaxis=dict(tickvals=ticks, ticktext=labels, autorange='reversed', fixedrange=True, showgrid=False),
        legend=dict(groupclick='togglegroup'), hovermode='closest')
    return figure

def build_trace_png(run_dir:str, t0:float=None, t1:float=None) -> Figure:
    events, lanes = load_trace(run_dir)
    if t0 is not None or t1 is not None:
        events = events[(events['end'] >= (t0 or 0)) & (events['start'] <= (t1 if t1 is not None else 1e18))]
    names = sorted(events['name'].unique())
    fig = Figure(figsize=(16, 0.32 * (lanes[-1][2] + lanes[-1][3]) + 2.2))
    ax = fig.subplots()
    for name in names:
        part = events[events['name'] == name]
        debug = name in DEBUG_SPANS
        for y, group in part.groupby('y'):
            ax.broken_barh(list(zip(group['start'], group['end'] - group['start'])), (y - 0.42, 0.84), facecolors=span_color(name, names),
                           alpha=0.45 if name in WAITING_SPANS else 1.0, hatch='////' if debug else None, edgecolor='white' if debug else None, linewidth=0)
    for process, thread, y, count in lanes:
        ax.axhline(y - 0.45, color='0.6', linewidth=0.6)
    ticks, labels = lane_labels(lanes)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(t0 or 0, t1 if t1 is not None else events['end'].max())
    ax.set_xlabel('wall-clock time since the first span (s)')
    ax.set_title(run_title(run_dir) + '  (a nested span is on the lane below its parent; faded = waiting; grey hatched = visualisation/debug)', fontsize=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=span_color(name, names), alpha=0.45 if name in WAITING_SPANS else 1.0, label=name) for name in names], ncol=6, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.16))
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    return fig

def export_chrome_trace(run_dir:str, out:str) -> str:
    '''Merges the run's trace files into one Chrome Trace Event JSON; open it at https://ui.perfetto.dev or chrome://tracing.'''
    spans, processes, threads = read_trace_events(run_dir)
    events = [{'ph': 'M', 'name': 'process_name', 'pid': pid, 'args': {'name': name}} for pid, name in processes.items()]
    events += [{'ph': 'M', 'name': 'thread_name', 'pid': pid, 'tid': tid, 'args': {'name': name}} for (pid, tid), name in threads.items()]
    with open(out, 'w') as f:
        json.dump({'traceEvents': events + spans, 'displayTimeUnit': 'ms'}, f)
    return out

def plot(run_dir:str, out:str, t0:float=None, t1:float=None) -> None:
    build_figure(run_dir, t0, t1).savefig(out, dpi=150)
    print(out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Timeline of per-stage timings in a run.')
    parser.add_argument('run_dir')
    parser.add_argument('--out', default=None, help='default: <run_dir>/timeline.png')
    parser.add_argument('--from_s', type=float, default=None, help='zoom: start of the window (s)')
    parser.add_argument('--to_s', type=float, default=None, help='zoom: end of the window (s)')
    parser.add_argument('--perfetto', action='store_true', help='also write <run_dir>/trace.json for https://ui.perfetto.dev (runs with trace_*.jsonl)')
    args = parser.parse_args()
    out = args.out or os.path.join(args.run_dir, 'timeline.png')
    if has_trace(args.run_dir):
        build_trace_png(args.run_dir, args.from_s, args.to_s).savefig(out, dpi=150)
        print(out)
        if args.perfetto:
            print(export_chrome_trace(args.run_dir, os.path.join(args.run_dir, 'trace.json')))
    else:
        plot(args.run_dir, out, args.from_s, args.to_s)
