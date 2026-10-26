#!/usr/bin/env python
# Gradio dashboard for browsing ActiveSplat result folders (results/<run>).
# Runs in its own env (see scripts/tools/dashboard.sh); it only reads files, it never needs ROS or habitat.
import argparse
import datetime
import importlib.util
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from export_ply import export_ply
from timeline import build_interactive_figure, build_trace_figure, export_chrome_trace, has_trace

PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))

# NOTE: habitat-lab _DefaultHabitatSimActions
ACTION_NAMES = {0: 'stop', 1: 'move_forward', 2: 'turn_left', 3: 'turn_right', 4: 'look_up', 5: 'look_down'}
# NOTE: columns written by scripts/judges/eval_actions.py
EVAL_COLUMNS = ['comp_m', 'comp_ratio', 'comp_inf_m', 'comp_ratio_inf', 'path_length_m', 'acc_m']
RUNTIME_DIRS = ['render_rgbd', 'topdown_map', 'subregion_map', 'runtime_data']
# NOTE: (cache name, title, folder with <step>.png frames, kind). 'sensor' frames pair gaussians_data/rgb with gaussians_data/depth;
# render_rgbd is the Gaussian map rendered at each step's pose (white = not mapped yet), not the sensor image
VIDEO_SOURCES = [
    ('drone_camera', 'Drone camera: sensor RGB | depth', os.path.join('gaussians_data', 'rgb'), 'sensor'),
    ('map_render', "Gaussian map rendered at the drone's pose: RGB | depth", 'render_rgbd', 'folder'),
    ('topdown_map', 'Top-down map (planner view)', 'topdown_map', 'folder'),
    ('topdown_visibility', 'Top-down visibility map', os.path.join('runtime_data', 'topdown_visible_map'), 'folder')]
VIDEO_FPS_CHOICES = [10, 20, 30, 60]
IMAGE_SUFFIXES = ('.jpg', '.png')
# NOTE: stages written by utils.timing_recorder
TIMING_STAGES = ['capture', 'mapper', 'optimise', 'topdown_render', 'visibility_global', 'visibility_local', 'render_rgbd', 'topdown_render_saved', 'oracle_observe', 'vggt_omega', 'seed_points', 'plan_topdown', 'plan_logic', 'topdown_wait', 'opacity_wait']
# NOTE: (timeline column, label, what it is) of the compute the drone waits for after each capture
REALTIME_COMPUTE_PHASES = [
    ('optimise_s', 'hovering, optimising map', 'optimising the Gaussians'),
    ('other_s', 'hovering, mapper work', 'other mapper work (tracking, densification, seeding)'),
    ('plan_s', 'hovering, planning', 'planning (obstacle map, Voronoi graph, node selection, path)')]
# NOTE: the planner's topdown_wait / opacity_wait are not added: they are the planner blocking on the mapper's own work, which the phases above already count
TURBO_ANCHORS = np.array([
    [48, 18, 59], [70, 107, 227], [40, 187, 236], [50, 241, 152],
    [164, 252, 60], [237, 208, 58], [251, 128, 34], [210, 49, 5], [122, 4, 3]], dtype=np.float64)

def list_runs(results_dir:str) -> list:
    if not os.path.isdir(results_dir):
        return []
    runs = [name for name in os.listdir(results_dir)
            if os.path.isfile(os.path.join(results_dir, name, 'gaussians_data', 'params.npz'))]
    return sorted(runs, reverse=True)

def read_optimisations(run_dir:str) -> str:
    '''Which performance optimisations the run had (optimisations.json, written by the mapper); older runs did not record it.'''
    path = os.path.join(run_dir, 'optimisations.json')
    if not os.path.isfile(path):
        return 'optimisations not recorded'
    with open(path) as f:
        return json.load(f)['label']

def optimisations_table(run_dir:str) -> str:
    '''Markdown table of every switchable optimisation with ✅/❌ for this run (names and descriptions from src/utils/optimisations.py, which is standard library only).'''
    path = os.path.join(run_dir, 'optimisations.json')
    if not os.path.isfile(path):
        return 'Optimisations: not recorded (the run predates the switches)'
    spec = importlib.util.spec_from_file_location('activesplat_optimisations', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src', 'utils', 'optimisations.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with open(path) as f:
        enabled = set(json.load(f)['enabled'])
    rows = [f"| `{name}` | {'✅' if name in enabled else '❌'} | {description} |" for name, description in module.OPTIMISATIONS.items()]
    return '| Optimisation | On | What it does |\n|---|:-:|---|\n' + '\n'.join(rows)

def read_run_metadata(run_dir:str) -> dict:
    '''Cheap per-run facts for labels and tables: only small files and the first frame's image header are read.'''
    with open(os.path.join(run_dir, 'config.json')) as f:
        config = json.load(f)
    name = os.path.basename(run_dir)
    try:
        started = datetime.datetime.strptime(name[:19], '%Y-%m-%d_%H-%M-%S')
    except ValueError:
        started = None
    first_rgb = find_image(os.path.join(run_dir, 'gaussians_data', 'rgb'), '0000')
    resolution = '{}×{}'.format(*Image.open(first_rgb).size) if first_rgb else '?'
    actions_path = os.path.join(run_dir, 'actions.txt')
    steps = 0
    if os.path.isfile(actions_path):
        with open(actions_path) as f:
            steps = sum(1 for line in f if line.strip())
    seeding = config['mapper'].get('gaussian_seeding', {'method': 'depth'})
    if seeding['method'] == 'vggt_omega':
        vggt = seeding['vggt_omega']
        seeding_short = 'VGGT-Ω'
        seeding_detail = (f"VGGT-Ω (`{os.path.basename(vggt['checkpoint']) or 'random weights'}`, window {vggt['window_size']}, "
                          f"kNN k={vggt['knn_k']}, conf drop {vggt['conf_drop_percentile']} %)")
    else:
        seeding_short = 'sensor depth'
        seeding_detail = 'sensor depth (SplaTAM default)'
    completion = accuracy = float('nan')
    eval_path = os.path.join(run_dir, 'actions_error.txt')
    if os.path.isfile(eval_path):
        final = np.loadtxt(eval_path, ndmin=2)[-1]
        completion = final[EVAL_COLUMNS.index('comp_ratio')] * 100
        accuracy = final[EVAL_COLUMNS.index('acc_m')] * 100
    wall_min = float('nan')
    timing_path = os.path.join(run_dir, 'timing.csv')
    if os.path.isfile(timing_path):
        wall_time = pd.read_csv(timing_path, usecols=['wall_time'])['wall_time']
        if not wall_time.empty:
            wall_min = (wall_time.max() - wall_time.min()) / 60
    rot_err = trans_err = float('nan')
    pose_errors_path = os.path.join(run_dir, 'pose_errors.csv')
    if os.path.isfile(pose_errors_path):
        pose_errors = pd.read_csv(pose_errors_path)
        if not pose_errors.empty:
            rot_err = pose_errors['rot_err_deg'].median()
            trans_err = pose_errors['trans_err_m'].median() * 100
    return {
        'optimisations': read_optimisations(run_dir),
        'name': name,
        'started': started,
        'scene': config['dataset']['scene_id'],
        'format': config['dataset']['format'],
        'resolution': resolution,
        'env_config': os.path.basename(config['env']['config']),
        'mapping_iters': config['mapper']['mapping_iters'],
        'map_every': config['mapper']['map_every'],
        'mapping_window_size': config['mapper']['mapping_window_size'],
        'seeding': seeding_short,
        'seeding_detail': seeding_detail,
        'steps': steps,
        'step_budget': config['dataset']['step_num'],
        'completion': completion,
        'accuracy': accuracy,
        'wall_min': wall_min,
        'rot_err': rot_err,
        'trans_err': trans_err,
    }

def square_resolution(resolution:str) -> str:
    # NOTE: '512×512' -> '512²'; non-square sizes are left as they are
    width, _, height = resolution.partition('×')
    return f'{width}²' if width == height else resolution

def run_label(metadata:dict) -> str:
    parts = [
        metadata['scene'],
        square_resolution(metadata['resolution']),
        metadata['seeding'],
        f"{metadata['mapping_iters']} it",
        f"{metadata['steps']}/{metadata['step_budget']} steps",
        'not evaluated' if np.isnan(metadata['completion']) else f"{metadata['completion']:.1f} %",
    ]
    if metadata['started'] is not None:
        parts.append(metadata['started'].strftime('%b %d %H:%M'))
    parts.append(metadata['optimisations'])
    return ' · '.join(parts)

def run_choices(results_dir:str) -> list:
    choices = []
    for name in list_runs(results_dir):
        try:
            choices.append((run_label(read_run_metadata(os.path.join(results_dir, name))), name))
        except (OSError, ValueError, KeyError):
            choices.append((name, name))
    return choices

def coverage_choices(results_dir:str) -> list:
    '''(label, folder) of the runs with a coverage.csv (scripts/oracle/coverage.py); the label is the scene, the method in label.txt and the start time.'''
    choices = []
    for name in list_runs(results_dir):
        run_dir = os.path.join(results_dir, name)
        if not os.path.isfile(os.path.join(run_dir, 'coverage.csv')):
            continue
        try:
            with open(os.path.join(run_dir, 'config.json')) as f:
                scene = json.load(f)['dataset']['scene_id']
        except (OSError, ValueError, KeyError):
            scene = '?'
        method = 'unlabelled'
        if os.path.isfile(os.path.join(run_dir, 'label.txt')):
            with open(os.path.join(run_dir, 'label.txt')) as f:
                method = f.read().strip() or method
        choices.append((f'{scene} · {method} · {name[:16]}', name))
    return choices

def coverage_curves(results_dir:str, names:list) -> tuple:
    '''(per-step coverage of the selected runs for a line plot, one summary row per run).'''
    labels = {name: label for label, name in coverage_choices(results_dir)}
    curves, rows = [], []
    for name in names or []:
        if name not in labels:
            continue
        coverage = pd.read_csv(os.path.join(results_dir, name, 'coverage.csv'))
        label = labels[name]
        curves.append(pd.DataFrame({'step': coverage['step'], 'observable surface seen (%)': coverage['coverage'] * 100, 'run': label}))
        row = {'run': label, 'steps': int(coverage['step'].max()), 'final coverage (%)': round(float(coverage['coverage'].iloc[-1] * 100), 1),
               'mean over steps (%)': round(float(coverage['coverage'].mean() * 100), 1), 'path (m)': round(float(coverage['path_length_m'].iloc[-1]), 1)}
        if 'coverage_multi' in coverage:
            row['two directions (%)'] = round(float(coverage['coverage_multi'].iloc[-1] * 100), 1)
        for level in (80, 90):
            reached = coverage.loc[coverage['coverage'] * 100 >= level, 'step']
            row[f'steps to {level}%'] = int(reached.iloc[0]) if len(reached) else None
        metrics_path = os.path.join(results_dir, name, 'image_metrics.json')
        if os.path.isfile(metrics_path):
            with open(metrics_path) as f:
                trajectory = json.load(f)['trajectory']
            row.update({'PSNR': round(trajectory['psnr'], 2), 'SSIM': round(trajectory['ssim'], 3), 'LPIPS': round(trajectory['lpips'], 3)})
        rows.append(row)
    empty = pd.DataFrame({'step': [], 'observable surface seen (%)': [], 'run': []})
    return (pd.concat(curves, ignore_index=True) if curves else empty), pd.DataFrame(rows)

def list_images(folder:str) -> list:
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, name) for name in os.listdir(folder) if name.lower().endswith(IMAGE_SUFFIXES))

def list_frames(folder:str) -> list:
    '''Frames named <step>.jpg or <step>.png (older runs) in step order (list_images sorts as text, so 1000 would come before 2).'''
    if not os.path.isdir(folder):
        return []
    names = [name for name in os.listdir(folder) if name.lower().endswith(IMAGE_SUFFIXES) and os.path.splitext(name)[0].isdigit()]
    return [os.path.join(folder, name) for name in sorted(names, key=lambda name: int(os.path.splitext(name)[0]))]

def find_image(folder:str, stem:str):
    '''<folder>/<stem>.jpg, or .png for runs saved before JPEG output; None if neither exists.'''
    for suffix in IMAGE_SUFFIXES:
        path = os.path.join(folder, stem + suffix)
        if os.path.isfile(path):
            return path
    return None

def video_path(run_dir:str, name:str, fps:int) -> str:
    return os.path.join(run_dir, 'videos', f'{name}_{fps}fps.mp4')

def cached_video(run_dir:str, name:str, folder:str, fps:int):
    '''The encoded video if it exists and is newer than its frames, else None.'''
    path = video_path(run_dir, name, fps)
    frames_dir = os.path.join(run_dir, folder)
    if os.path.isfile(path) and os.path.isdir(frames_dir) and os.path.getmtime(path) >= os.path.getmtime(frames_dir):
        return path
    return None

def video_frames(run_dir:str, folder:str, kind:str, depth_max:float) -> list:
    '''(step, loader) pairs in step order, each loader returning an RGB PIL image.'''
    paths = list_frames(os.path.join(run_dir, folder))
    if kind == 'folder':
        return [(int(os.path.basename(path)[:-4]), lambda path=path: Image.open(path).convert('RGB')) for path in paths]
    depth_dir = os.path.join(run_dir, 'gaussians_data', 'depth')
    def sensor_frame(path:str) -> Image.Image:
        rgb = Image.open(path).convert('RGB')
        depth_path = os.path.join(depth_dir, os.path.basename(path))
        depth = Image.fromarray(colorize_depth(depth_path, 1000.0, depth_max)) if os.path.isfile(depth_path) else Image.new('RGB', rgb.size)
        canvas = Image.new('RGB', (rgb.width + depth.width, rgb.height))
        canvas.paste(rgb, (0, 0))
        canvas.paste(depth, (rgb.width, 0))
        return canvas
    return [(int(os.path.basename(path)[:-4]), lambda path=path: sensor_frame(path)) for path in paths]

def ffmpeg_command(width:int, height:int, fps:int, output_path:str) -> list:
    '''ffmpeg reading raw rgb24 frames from stdin and writing H.264.'''
    import imageio_ffmpeg
    return [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-preset', 'veryfast', '-movflags', '+faststart', output_path]

def build_video(run_dir:str, name:str, folder:str, kind:str, depth_max:float, fps:int):
    '''Encode the frames into a cached H.264 mp4 with the step number stamped on each frame.'''
    frames = video_frames(run_dir, folder, kind, depth_max)
    if not frames:
        return None
    path = cached_video(run_dir, name, folder, fps)
    if path is not None:
        return path
    import subprocess
    import tempfile
    from PIL import ImageDraw, ImageFont

    width, height = frames[0][1]().size
    # NOTE: yuv420p needs even dimensions (the top-down maps are 233 px high), so pad with black
    width, height = width + width % 2, height + height % 2
    font = ImageFont.load_default(size=max(12, height // 24))
    final_path = video_path(run_dir, name, fps)
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    tmp_video = final_path + '.part.mp4'
    command = ffmpeg_command(width, height, fps, tmp_video)
    with tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=stderr)
        try:
            for step, load in frames:
                canvas = Image.new('RGB', (width, height))
                canvas.paste(load())
                draw = ImageDraw.Draw(canvas)
                label = f'step {step}'
                box = draw.textbbox((6, 4), label, font=font)
                draw.rectangle((box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill=(0, 0, 0))
                draw.text((6, 4), label, fill=(255, 255, 255), font=font)
                process.stdin.write(canvas.tobytes())
            process.stdin.close()
        except BrokenPipeError:
            pass
        returncode = process.wait()
        stderr.seek(0)
        error = stderr.read().decode(errors='replace')
    if returncode != 0:
        if os.path.isfile(tmp_video):
            os.remove(tmp_video)
        raise RuntimeError(f'ffmpeg failed for {name}: {error.strip()[-500:]}')
    os.replace(tmp_video, final_path)
    return final_path

REALTIME_SPEEDUPS = [1, 2, 5, 10, 30]
REALTIME_PARAMETERS = {'forward_speed': 0.5, 'turn_rate': 60.0, 'tilt_rate': 30.0, 'stop_time': 0.5}

def read_motion_steps(config:dict) -> tuple:
    '''(forward step m, turn angle deg, tilt angle deg) of the habitat action space the run used.'''
    import yaml
    env_path = os.path.join(PACKAGE_PATH, config['env']['config'])
    with open(env_path) as f:
        simulator = yaml.safe_load(f)['habitat']['simulator']
    return float(simulator.get('forward_step_size', 0.25)), float(simulator.get('turn_angle', 10)), float(simulator.get('tilt_angle', 15))

def realtime_timeline(actions:np.ndarray, step_count:int, motion_steps:tuple, forward_speed:float, turn_rate:float,
                      tilt_rate:float, stop_time:float, compute_seconds:dict=None) -> pd.DataFrame:
    '''Per step: the drone stops and captures frame s, hovers while the measured compute runs (if given), then executes action s.
    compute_seconds maps the REALTIME_COMPUTE_PHASES columns to per-step arrays.'''
    forward_step, turn_angle, tilt_angle = motion_steps
    motion = {1: ('moving forward', forward_step / forward_speed), 2: ('turning left', turn_angle / turn_rate),
              3: ('turning right', turn_angle / turn_rate), 4: ('looking up', tilt_angle / tilt_rate), 5: ('looking down', tilt_angle / tilt_rate)}
    timeline = pd.DataFrame({'step': np.arange(step_count)})
    labels = [motion.get(int(actions[step]) if step < len(actions) else 0, ('idle', 0.0)) for step in range(step_count)]
    timeline['action'] = [label for label, _ in labels]
    timeline['capture_s'] = stop_time
    for column, _, _ in REALTIME_COMPUTE_PHASES:
        values = (compute_seconds or {}).get(column)
        timeline[column] = values[:step_count] if values is not None else 0.0
    timeline['move_s'] = [seconds for _, seconds in labels]
    columns = ['capture_s'] + [column for column, _, _ in REALTIME_COMPUTE_PHASES] + ['move_s']
    timeline['duration_s'] = timeline[columns].sum(axis=1)
    timeline['start_s'] = timeline['duration_s'].cumsum() - timeline['duration_s']
    return timeline

def measured_compute_seconds(timing:pd.DataFrame, step_count:int) -> dict:
    '''Per-step seconds of each REALTIME_COMPUTE_PHASES column from timing.csv (+ planner_timing.csv).'''
    per_step = timing.pivot_table(index='step', columns='stage', values='ms', aggfunc='sum').fillna(0) / 1000
    def column(stage:str) -> np.ndarray:
        return per_step[stage].reindex(range(step_count)).fillna(0).to_numpy() if stage in per_step else np.zeros(step_count)
    # NOTE: the 'mapper' stage is the whole mapper run of a step and contains 'optimise' (and the VGGT-Ω and seeding stages), so it is not added to them
    optimise = column('optimise')
    compute = {'optimise_s': optimise, 'other_s': np.maximum(column('mapper') - optimise, 0)}
    # NOTE: the planner counts accepted pose updates, so its step is one ahead of the frame index
    def planner_column(stage:str) -> np.ndarray:
        shifted = column(stage)
        return np.append(shifted[1:], 0.0)
    compute['plan_s'] = planner_column('plan_topdown') + planner_column('plan_logic')
    return compute

def format_clock(seconds:float) -> str:
    seconds = int(seconds)
    return f'{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}'

def realtime_summary(timeline:pd.DataFrame, motion_steps:tuple) -> str:
    forward_step, _, _ = motion_steps
    moving = timeline.loc[timeline['action'] == 'moving forward', 'move_s'].sum()
    rotating = timeline.loc[timeline['action'].str.startswith(('turning', 'looking')), 'move_s'].sum()
    stopped = timeline['capture_s'].sum()
    distance = (timeline['action'] == 'moving forward').sum() * forward_step
    total = timeline['duration_s'].sum()
    lines = [f'**Total real time: {format_clock(total)}** ({total / 60:.1f} min) for {len(timeline)} steps, {distance:.1f} m flown.',
             f'- moving forward: {format_clock(moving)} ({100 * moving / total:.0f} %)',
             f'- turning / tilting: {format_clock(rotating)} ({100 * rotating / total:.0f} %)',
             f'- stopped to capture: {format_clock(stopped)} ({100 * stopped / total:.0f} %)']
    for column, _, description in REALTIME_COMPUTE_PHASES:
        seconds = timeline[column].sum()
        if seconds > 0:
            lines.append(f'- hovering while {description}: {format_clock(seconds)} ({100 * seconds / total:.0f} %)')
    return '\n'.join(lines)

def build_realtime_video(run:'RunData', parameters:dict, hover:bool, speedup:int, fps:int=30):
    '''Composite video (drone camera | depth over the two top-down maps) timed as the drone would fly it. Returns (path, summary).'''
    import bisect
    import subprocess
    import tempfile
    from PIL import ImageDraw, ImageFont

    depth_max = float(run.config['dataset']['far'])
    sensor = video_frames(run.run_dir, os.path.join('gaussians_data', 'rgb'), 'sensor', depth_max)
    maps = [video_frames(run.run_dir, folder, 'folder', depth_max) for name, _, folder, _ in VIDEO_SOURCES if name in ('topdown_map', 'topdown_visibility')]
    maps = [frames for frames in maps if frames]
    if not sensor:
        return None, 'No sensor frames found.'
    motion_steps = read_motion_steps(run.config)
    compute_seconds = measured_compute_seconds(run.timing, len(sensor)) if hover and run.timing is not None else None
    timeline = realtime_timeline(run.actions, len(sensor), motion_steps, compute_seconds=compute_seconds, **parameters)
    summary = realtime_summary(timeline, motion_steps)
    summary += f'\n\nVideo plays {speedup}× real time at {fps} frames/s: {format_clock(timeline["duration_s"].sum() / speedup)}.'

    def nearest(frames:list, step:int) -> Image.Image:
        index = max(bisect.bisect_right([frame_step for frame_step, _ in frames], step) - 1, 0)
        return frames[index][1]()
    def compose(step:int) -> Image.Image:
        top = nearest(sensor, int(timeline['step'][step]))
        width = top.width + top.width % 2
        bottom_images = [nearest(frames, step) for frames in maps]
        bottom_height = int(width / sum(image.width / image.height for image in bottom_images)) if bottom_images else 0
        bottom_height += bottom_height % 2
        canvas = Image.new('RGB', (width, top.height + bottom_height))
        canvas.paste(top, (0, 0))
        x = 0
        for image in bottom_images:
            resized = image.resize((max(int(round(image.width * bottom_height / image.height)), 1), bottom_height), Image.NEAREST)
            canvas.paste(resized, (x, top.height))
            x += resized.width
        return canvas

    first = compose(0)
    font = ImageFont.load_default(size=max(14, first.height // 32))
    out_dir = os.path.join(run.run_dir, 'videos')
    os.makedirs(out_dir, exist_ok=True)
    final_path = os.path.join(out_dir, f'realtime_{speedup}x.mp4')
    tmp_video = final_path + '.part.mp4'
    starts = timeline['start_s'].to_numpy()
    total = float(timeline['duration_s'].sum())
    frame_count = int(total / speedup * fps) + 1
    phase_totals = {}
    with tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(ffmpeg_command(first.width, first.height, fps, tmp_video), stdin=subprocess.PIPE, stderr=stderr)
        cached_step, cached_canvas = -1, None
        try:
            for index in range(frame_count):
                now = index * speedup / fps
                step = min(max(int(np.searchsorted(starts, now, side='right')) - 1, 0), len(timeline) - 1)
                if step != cached_step:
                    cached_step, cached_canvas = step, compose(step)
                row = timeline.iloc[step]
                offset = now - row['start_s']
                if offset < row['capture_s']:
                    phase = 'stopped, capturing'
                else:
                    phase = None
                    elapsed = row['capture_s']
                    for column, label, _ in REALTIME_COMPUTE_PHASES:
                        elapsed += row[column]
                        if offset < elapsed:
                            phase = label
                            break
                    if phase is None:
                        phase = row['action']
                canvas = cached_canvas.copy()
                draw = ImageDraw.Draw(canvas)
                label = f'{format_clock(now)}   step {step}   {phase}'
                box = draw.textbbox((6, 4), label, font=font)
                draw.rectangle((box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill=(0, 0, 0))
                draw.text((6, 4), label, fill=(255, 255, 255), font=font)
                process.stdin.write(canvas.tobytes())
            process.stdin.close()
        except BrokenPipeError:
            pass
        returncode = process.wait()
        stderr.seek(0)
        error = stderr.read().decode(errors='replace')
    if returncode != 0:
        if os.path.isfile(tmp_video):
            os.remove(tmp_video)
        raise RuntimeError(f'ffmpeg failed for the real-time video: {error.strip()[-500:]}')
    os.replace(tmp_video, final_path)
    return final_path, summary

def build_turbo_lut(size:int=1024) -> np.ndarray:
    positions = np.linspace(0, len(TURBO_ANCHORS) - 1, size)
    return np.stack([np.interp(positions, np.arange(len(TURBO_ANCHORS)), TURBO_ANCHORS[:, channel]) for channel in range(3)], axis=1).astype(np.uint8)

TURBO_LUT = build_turbo_lut()

def colorize_depth(depth_path:str, depth_scale:float, depth_max:float) -> np.ndarray:
    depth = np.asarray(Image.open(depth_path), dtype=np.float32) / depth_scale
    # NOTE: a lookup table instead of per-pixel interpolation; this runs once per frame of the drone-camera video
    rgb = TURBO_LUT[(np.clip(depth / depth_max, 0, 1) * (len(TURBO_LUT) - 1)).astype(np.intp)]
    rgb[depth <= 0] = 0
    return rgb

def export_splat(params_path:str, splat_path:str) -> None:
    # NOTE: antimatter15 .splat layout, 32 bytes per gaussian: xyz f32, scale f32, rgba u8, quaternion (wxyz) u8
    params = np.load(params_path)
    num = params['means3D'].shape[0]
    log_scales = params['log_scales'].reshape(num, -1)
    if log_scales.shape[1] == 1:
        log_scales = np.repeat(log_scales, 3, axis=1)
    opacities = 1 / (1 + np.exp(-params['logit_opacities'].reshape(num, 1)))
    rotations = params['unnorm_rotations'] / np.linalg.norm(params['unnorm_rotations'], axis=1, keepdims=True)
    order = np.argsort(-(np.exp(log_scales.sum(axis=1)) * opacities[:, 0]))

    elements = np.empty(num, dtype=[('xyz', '<f4', 3), ('scale', '<f4', 3), ('rgba', 'u1', 4), ('rot', 'u1', 4)])
    elements['xyz'] = params['means3D'][order]
    elements['scale'] = np.exp(log_scales[order])
    elements['rgba'] = np.clip(np.concatenate([params['rgb_colors'], opacities], axis=1)[order] * 255, 0, 255)
    elements['rot'] = np.clip(rotations[order] * 128 + 128, 0, 255)
    elements.tofile(splat_path)

def get_cached_export(run_dir:str, file_name:str, exporter) -> str:
    params_path = os.path.join(run_dir, 'gaussians_data', 'params.npz')
    output_path = os.path.join(run_dir, file_name)
    if not os.path.isfile(output_path) or os.path.getmtime(output_path) < os.path.getmtime(params_path):
        exporter(params_path, output_path)
    return output_path

class RunData:

    def __init__(self, run_dir:str) -> None:
        self.run_dir = run_dir
        self.gaussians_dir = os.path.join(run_dir, 'gaussians_data')
        with open(os.path.join(run_dir, 'config.json')) as f:
            self.config = json.load(f)
        with open(os.path.join(self.gaussians_dir, 'transforms.json')) as f:
            self.transforms = json.load(f)
        self.rgb_paths = list_images(os.path.join(self.gaussians_dir, 'rgb'))
        self.depth_paths = list_images(os.path.join(self.gaussians_dir, 'depth'))
        self.keyframe_paths = {os.path.basename(path): path for path in list_images(os.path.join(self.gaussians_dir, 'keyframes'))}
        # NOTE: depth PNGs are uint16 millimetres (as read by scripts/judges/eval_actions.py);
        # transforms.json's integer_depth_scale does not describe them
        self.depth_scale = 1000.0
        actions_path = os.path.join(run_dir, 'actions.txt')
        self.actions = np.loadtxt(actions_path, dtype=int, ndmin=1) if os.path.isfile(actions_path) and os.path.getsize(actions_path) > 0 else np.zeros(0, dtype=int)
        eval_path = os.path.join(run_dir, 'actions_error.txt')
        self.eval = pd.DataFrame(np.loadtxt(eval_path, ndmin=2), columns=EVAL_COLUMNS) if os.path.isfile(eval_path) else None
        timing_path = os.path.join(run_dir, 'timing.csv')
        self.timing = pd.read_csv(timing_path) if os.path.isfile(timing_path) else None
        planner_timing_path = os.path.join(run_dir, 'planner_timing.csv')
        if self.timing is not None and os.path.isfile(planner_timing_path):
            self.timing = pd.concat([self.timing, pd.read_csv(planner_timing_path)], ignore_index=True)
        pose_errors_path = os.path.join(run_dir, 'pose_errors.csv')
        self.pose_errors = pd.read_csv(pose_errors_path) if os.path.isfile(pose_errors_path) else None

    def pose_noise_text(self) -> str:
        '''Simulator poses, or the settings and trajectory error of the simulated odometry drift (pose_noise.json).'''
        path = os.path.join(self.run_dir, 'pose_noise.json')
        if not os.path.isfile(path):
            return 'simulator (true) poses'
        with open(path) as f:
            data = json.load(f)
        settings, errors = data['settings'], data['errors']
        return (f"simulated drift: {settings['translation_drift'] * 100:g} % translation, {settings['rotation_drift'] * 100:g} % rotation, jitter {settings['jitter_m'] * 100:g} cm / {settings['jitter_deg']:g}° "
                f"(ATE RMSE {errors['ate_rmse_m'] * 100:.1f} cm, mean rotation error {errors['rot_err_mean_deg']:.1f}°)")

    def summary_markdown(self) -> str:
        dataset = self.config['dataset']
        params = np.load(os.path.join(self.gaussians_dir, 'params.npz'))
        metadata = read_run_metadata(self.run_dir)
        wall_time = 'not recorded' if np.isnan(metadata['wall_min']) else f"{metadata['wall_min']:.1f} min"
        lines = [
            f"### {dataset['format']} / {dataset['scene_id']}",
            f"| | |\n|---|---|",
            f"| Run | `{os.path.basename(self.run_dir)}` |",
            f"| Started | {metadata['started'].strftime('%Y-%m-%d %H:%M:%S') if metadata['started'] else '?'} |",
            f"| Resolution | {metadata['resolution']} (`{metadata['env_config']}`) |",
            f"| Gaussian seeding | {metadata['seeding_detail']} |",
            f"| Mapping | {metadata['mapping_iters']} optimisation iters, map every {metadata['map_every']} frames, window {metadata['mapping_window_size']} keyframes |",
            f"| Wall time | {wall_time} |",
            f"| Steps (actions taken / budget) | {len(self.actions)} / {dataset['step_num']} |",
            f"| Frames saved | {len(self.rgb_paths)} |",
            f"| Pose source | {self.pose_noise_text()} |",
            f"| Keyframes saved | {len(self.keyframe_paths)} |",
            f"| Gaussians | {params['means3D'].shape[0]:,} |",
            f"| Depth range | {dataset['near']} – {dataset['far']} m |",
        ]
        if self.eval is not None:
            final = self.eval.iloc[-1]
            lines.append(f"| Completion ratio (<5 cm) | {final['comp_ratio'] * 100:.2f} % |")
            lines.append(f"| Path length | {final['path_length_m']:.2f} m |")
        return '\n'.join(lines) + '\n\n' + optimisations_table(self.run_dir)

    def actions_frame(self) -> pd.DataFrame:
        names, counts = np.unique(self.actions, return_counts=True)
        return pd.DataFrame({
            'action': [ACTION_NAMES.get(int(name), str(name)) for name in names],
            'count': counts})

    def trajectory_frame(self) -> pd.DataFrame:
        # NOTE: transforms.json stores zero translation for every frame, so use the per-frame w2c poses in params.npz
        params = np.load(os.path.join(self.gaussians_dir, 'params.npz'))
        quaternions = params['cam_unnorm_rots'][0].T
        quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
        translations = params['cam_trans'][0].T
        w, x, y, z = quaternions.T
        rotations = np.stack([
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1)], axis=1)
        positions = -np.einsum('nji,nj->ni', rotations, translations)
        # NOTE: the map frame is the first camera's (y-down), so the floor plane is x-z
        return pd.DataFrame({'x': positions[:, 0], 'z': positions[:, 2], 'step': np.arange(len(positions))})

    def resolution(self) -> str:
        if not self.rgb_paths:
            return '?'
        width, height = Image.open(self.rgb_paths[0]).size
        return f'{width}×{height}'

    def timing_stats(self) -> pd.DataFrame:
        if self.timing is None:
            return pd.DataFrame(columns=['stage', 'steps', 'mean ms', 'median ms', 'p95 ms', 'max ms', 'total s'])
        rows = []
        for stage in TIMING_STAGES:
            ms = self.timing.loc[self.timing['stage'] == stage, 'ms']
            if ms.empty:
                continue
            rows.append({
                'stage': stage,
                'steps': len(ms),
                'mean ms': round(ms.mean(), 1),
                'median ms': round(ms.median(), 1),
                'p95 ms': round(ms.quantile(0.95), 1),
                'max ms': round(ms.max(), 1),
                'total s': round(ms.sum() / 1000, 1)})
        return pd.DataFrame(rows)

    def wall_time_s(self) -> float:
        if self.timing is None or self.timing.empty:
            return float('nan')
        return float(self.timing['wall_time'].max() - self.timing['wall_time'].min())

    def timing_breakdown(self) -> pd.DataFrame:
        if self.timing is None:
            return pd.DataFrame({'part': [], 'seconds': []})
        totals = self.timing.groupby('stage')['ms'].sum() / 1000
        capture_s = totals.get('capture', 0.0)
        mapper_s = totals.get('mapper', 0.0)
        optimise_s = min(totals.get('optimise', 0.0), mapper_s)
        # NOTE: seeding (VGGT-Omega inference + seeded point cloud / kNN scales) happens inside the mapper stage too
        seeding_s = min(totals.get('vggt_omega', 0.0) + totals.get('seed_points', 0.0), mapper_s - optimise_s)
        rest_s = max(self.wall_time_s() - capture_s - mapper_s, 0.0)
        parts = ['capture', 'training: optimisation', 'training: tracking & densification', 'planning, rendering & I/O']
        seconds = [capture_s, optimise_s, mapper_s - optimise_s - seeding_s, rest_s]
        if seeding_s > 0:
            parts.insert(2, 'seeding: VGGT-Ω + kNN')
            seconds.insert(2, seeding_s)
        return pd.DataFrame({'part': parts, 'seconds': seconds})

    def timing_steps_frame(self) -> pd.DataFrame:
        if self.timing is None:
            return pd.DataFrame({'step': [], 'ms': [], 'stage': []})
        return self.timing[['step', 'ms', 'stage']]

    def pose_error_markdown(self) -> str:
        if self.pose_errors is None or self.pose_errors.empty:
            return ('No `pose_errors.csv` in this run. Pose errors are logged for VGGT-Ω seeding runs started after pose '
                    'logging was added; sensor-depth runs use the simulator poses directly, so they have no pose error.')
        errors = self.pose_errors
        moved = errors[errors['gt_baseline_m'] > 0.05]
        def stats(values:pd.Series, factor:float=1.0) -> str:
            values = values.dropna() * factor
            if values.empty:
                return '– | – | –'
            return f'{values.mean():.2f} | {values.median():.2f} | {values.quantile(0.95):.2f}'
        return (
            "VGGT-Ω's predicted pose of each frame in its window, relative to the current frame, compared with the true "
            "relative pose. Translations are put into metres with the same scale that aligns VGGT-Ω's depth to the sensor. "
            f"{len(errors)} frame pairs over {errors['step'].nunique()} mapped frames "
            f"(window of {int(errors['window_size'].max())}).\n\n"
            '| | mean | median | p95 |\n|---|---|---|---|\n'
            f"| Rotation error (°) | {stats(errors['rot_err_deg'])} |\n"
            f"| Translation error (cm) | {stats(errors['trans_err_m'], 100)} |\n"
            f"| Translation direction error (°), pairs > 5 cm apart ({len(moved)}) | {stats(moved['trans_dir_err_deg'])} |\n"
            f"| True distance between frames (cm) | {stats(errors['gt_baseline_m'], 100)} |")
    
    def pose_error_steps_frame(self, column:str, factor:float=1.0) -> pd.DataFrame:
        if self.pose_errors is None or self.pose_errors.empty:
            return pd.DataFrame({'step': [], 'error': [], 'statistic': []})
        grouped = self.pose_errors.groupby('step')[column]
        frames = [pd.DataFrame({'step': grouped.mean().index, 'error': grouped.mean().values * factor, 'statistic': 'mean over window'}),
                  pd.DataFrame({'step': grouped.max().index, 'error': grouped.max().values * factor, 'statistic': 'max over window'})]
        return pd.concat(frames, ignore_index=True)
    
    def timing_markdown(self) -> str:
        if self.timing is None:
            return ('No `timing.csv` in this run. Timing is recorded for runs started after the timing recorder was added '
                    '(`timing_recorder` in `src/utils/__init__.py`).')
        wall_s = self.wall_time_s()
        steps = int(self.timing.loc[self.timing['stage'] == 'capture', 'step'].max()) + 1
        return (f"**Wall time:** {wall_s / 60:.1f} min for {steps} steps "
                f"({wall_s / max(steps, 1) * 1000:.0f} ms per step, {steps / max(wall_s, 1e-9):.2f} steps/s) at "
                f"{self.resolution()} with {self.config['mapper']['mapping_iters']} optimisation iterations per mapped frame.\n\n"
                '- **capture**: habitat simulator step + RGB-D sensor render\n'
                '- **mapper**: the whole SplaTAM update for a frame (tracking, adding Gaussians, optimisation)\n'
                '- **optimise**: only the gradient iterations inside the mapper\n'
                '- **vggt_omega** (VGGT-Ω seeding only): VGGT-Ω depth inference in the worker process\n'
                '- **seed_points** (VGGT-Ω seeding only): choosing pixels to seed, building the points and their kNN scales\n'
                '- **plan_topdown** (planner, `planner_timing.csv`): planner compute on each top-down map (obstacle map, Voronoi graph, subregions), excluding waits on the mapper\n'
                '- **plan_logic** (planner): the rest of the planner loop, e.g. node selection and path planning\n'
                '- **topdown_wait** / **opacity_wait** (planner): time the planner waited for the mapper to render the top-down maps / compute opacity. '
                'Planner steps are counted in accepted pose updates, so they can be off by one against the mapper\'s frames.')

    def frame_images(self, index:int) -> tuple:
        if not self.rgb_paths:
            return None, None, None
        index = int(np.clip(index, 0, len(self.rgb_paths) - 1))
        rgb_path = self.rgb_paths[index]
        depth_image = None
        if index < len(self.depth_paths):
            depth_image = colorize_depth(self.depth_paths[index], self.depth_scale, float(self.config['dataset']['far']))
        keyframe = self.keyframe_paths.get(os.path.basename(rgb_path))
        return rgb_path, depth_image, keyframe

    def eval_frame(self) -> pd.DataFrame:
        if self.eval is None:
            return pd.DataFrame({'step': [], 'completion ratio (%)': []})
        return pd.DataFrame({'step': np.arange(len(self.eval)), 'completion ratio (%)': self.eval['comp_ratio'] * 100})

def build_app(results_dir:str) -> gr.Blocks:
    def optional_image(path:str):
        return path if os.path.isfile(path) else None

    def load_run(run_name:str):
        if not run_name:
            empty = gr.update(value=None)
            return (None, f'No runs with `gaussians_data/params.npz` found in `{results_dir}`.', empty, empty,
                    gr.update(maximum=0, value=0), empty, empty, empty, empty, empty, empty, empty, *([empty] * len(RUNTIME_DIRS)), 'No run selected.',
                    'No run selected.', empty, empty, empty, 'No run selected.', empty, empty)
        run = RunData(os.path.join(results_dir, run_name))
        rgb, depth, keyframe = run.frame_images(0)
        runtime_galleries = []
        for name in RUNTIME_DIRS:
            images = list_images(os.path.join(run.run_dir, name))
            runtime_galleries.append(gr.update(value=images or None, label=f'{name} ({len(images)} images)'))
        if run.eval is None:
            eval_note = ('No `actions_error.txt` yet. Generate it with:\n\n```bash\n'
                         f'r={os.path.relpath(run.run_dir, PACKAGE_PATH)}\n'
                         'python scripts/judges/eval_actions.py --save_path $r/actions_error.txt --config $r/config.json \\\n'
                         '    --user_config config/user_config.json --actions $r/actions.txt --gpu_id 0\n```')
        else:
            final = run.eval.iloc[-1]
            eval_note = ('Metrics after the final step, from `actions_error.txt`.\n\n'
                         '| ' + ' | '.join(EVAL_COLUMNS) + ' |\n|' + '---|' * len(EVAL_COLUMNS) + '\n'
                         '| ' + ' | '.join(f'{final[column]:.4f}' for column in EVAL_COLUMNS) + ' |')
        return (
            run,
            run.summary_markdown(),
            run.actions_frame(),
            None,
            gr.update(maximum=max(len(run.rgb_paths) - 1, 0), value=0),
            rgb,
            depth,
            keyframe,
            find_image(run.run_dir, 'visited_map'),
            find_image(run.run_dir, 'topdown_free_map'),
            run.trajectory_frame(),
            run.eval_frame(),
            *runtime_galleries,
            eval_note,
            run.timing_markdown(),
            run.timing_stats(),
            run.timing_steps_frame(),
            run.timing_breakdown(),
            run.pose_error_markdown(),
            run.pose_error_steps_frame('rot_err_deg'),
            run.pose_error_steps_frame('trans_err_m', 100),
        )

    def show_frame(run:RunData, index:int):
        if run is None:
            return None, None, None
        return run.frame_images(index)

    def load_splat(run:RunData):
        if run is None:
            return None, None
        splat_path = get_cached_export(run.run_dir, 'gaussians.splat', export_splat)
        ply_path = get_cached_export(run.run_dir, 'gaussians.ply', export_ply)
        return splat_path, ply_path

    def show_videos(run:RunData, fps:int, build:bool):
        if run is None:
            return [None] * len(VIDEO_SOURCES) + ['No run selected.']
        fps = int(fps)
        depth_max = float(run.config['dataset']['far'])
        frame_counts = [len(list_frames(os.path.join(run.run_dir, folder))) for _, _, folder, _ in VIDEO_SOURCES]
        def get_video(source:tuple, frame_count:int):
            name, _, folder, kind = source
            if frame_count == 0:
                return None
            return build_video(run.run_dir, name, folder, kind, depth_max, fps) if build else cached_video(run.run_dir, name, folder, fps)
        # NOTE: encoding runs in ffmpeg subprocesses and PIL releases the GIL while decoding, so the videos build in parallel
        with ThreadPoolExecutor(max_workers=len(VIDEO_SOURCES)) as pool:
            paths = list(pool.map(get_video, VIDEO_SOURCES, frame_counts))
        notes = []
        for (_, title, folder, _), frame_count, path in zip(VIDEO_SOURCES, frame_counts, paths):
            if frame_count == 0:
                notes.append(f'- {title}: no frames in `{folder}` (run without `save_runtime_data:=1`?)')
            elif path is None:
                notes.append(f'- {title}: {frame_count} frames, not encoded yet. Open this tab or change the speed to build it.')
            else:
                notes.append(f'- {title}: {frame_count} frames, {frame_count / fps:.0f} s at {fps} frames/s')
        return paths + ['\n'.join(notes)]

    def compare_runs():
        rows = []
        for run_name in list_runs(results_dir):
            run_dir = os.path.join(results_dir, run_name)
            try:
                metadata = read_run_metadata(run_dir)
            except (OSError, ValueError, KeyError):
                continue
            timing_path = os.path.join(run_dir, 'timing.csv')
            stage_means = pd.read_csv(timing_path).groupby('stage')['ms'].mean() if os.path.isfile(timing_path) else {}
            rows.append({
                'started': metadata['started'].strftime('%Y-%m-%d %H:%M') if metadata['started'] else '?',
                'scene': metadata['scene'],
                'resolution': metadata['resolution'],
                'seeding': metadata['seeding'],
                'opt iters': metadata['mapping_iters'],
                'steps': f"{metadata['steps']}/{metadata['step_budget']}",
                'wall min': round(metadata['wall_min'], 1),
                'capture ms/step': round(stage_means.get('capture', float('nan')), 1),
                'training ms/step': round(stage_means.get('mapper', float('nan')), 1),
                'completion %': round(metadata['completion'], 2),
                'accuracy cm': round(metadata['accuracy'], 2),
                'VGGT-Ω rot err ° (median)': round(metadata['rot_err'], 2),
                'VGGT-Ω trans err cm (median)': round(metadata['trans_err'], 2),
                'run': run_name})
        return pd.DataFrame(rows)
    
    def refresh_runs(current:str):
        choices = run_choices(results_dir)
        names = [name for _, name in choices]
        return gr.update(choices=choices, value=current if current in names else (names[0] if names else None))

    choices = run_choices(results_dir)
    with gr.Blocks(title='ActiveSplat results') as app:
        run_state = gr.State(None)
        gr.Markdown(f'# ActiveSplat results\nBrowsing `{results_dir}`')
        with gr.Row():
            run_dropdown = gr.Dropdown(choices=choices, value=choices[0][1] if choices else None, scale=4, filterable=True,
                                       label='Run (scene · resolution · seeding · optimisation iters · steps · completion · started)')
            refresh_button = gr.Button('Refresh', scale=1)
        with gr.Row():
            summary = gr.Markdown()
            actions_plot = gr.BarPlot(x='action', y='count', title='Actions taken', height=260)
        with gr.Tabs():
            with gr.Tab('3D map'):
                gr.Markdown('The Gaussian map is converted to `.splat` on first view and cached in the run folder. '
                            'Drag to orbit, scroll to zoom. The `.ply` is for SuperSplat or other 3DGS viewers.')
                load_button = gr.Button('Load 3D map', variant='primary')
                model_view = gr.Model3D(label='Gaussian map', height=600, clear_color=(0, 0, 0, 1))
                ply_file = gr.File(label='Download .ply')
            with gr.Tab('Frames'):
                frame_slider = gr.Slider(0, 0, step=1, value=0, label='Frame')
                with gr.Row():
                    rgb_image = gr.Image(label='RGB', type='filepath')
                    depth_image = gr.Image(label='Depth (0 – far)', type='numpy')
                keyframe_image = gr.Image(label='Keyframe (RGB | depth), when this frame is one', type='filepath')
            with gr.Tab('Maps'):
                with gr.Row():
                    visited_image = gr.Image(label='Visited map', type='filepath')
                    free_image = gr.Image(label='Top-down free map', type='filepath')
                trajectory_plot = gr.ScatterPlot(x='x', y='z', color='step', title='Camera trajectory (x–z plane, relative to start)', height=400)
            with gr.Tab('Runtime data'):
                gr.Markdown('Only filled when the run used `save_runtime_data:=1`.')
                runtime_galleries = [gr.Gallery(label=name, columns=6, height=360) for name in RUNTIME_DIRS]
            with gr.Tab('Videos') as videos_tab:
                gr.Markdown('The runtime frames played back as looping videos, one frame per step with the step number in the corner. '
                            'Needs `save_runtime_data:=1`. Videos are encoded on first view and cached in `<run>/videos/`.')
                fps_radio = gr.Radio(VIDEO_FPS_CHOICES, value=30, label='Playback speed (steps per second)')
                videos_note = gr.Markdown()
                video_players = [gr.Video(label=title, loop=True, autoplay=True, interactive=False, show_download_button=True)
                                 for _, title, _, _ in VIDEO_SOURCES]
                gr.Markdown('### Real-time video\n'
                            'Drone camera and the two top-down maps, timed as the drone would fly it: each step stops and captures, hovers for the mapper time measured in this run, then moves or turns at the speeds below. '
                            'The step size and turn angle come from the run\'s habitat config.')
                with gr.Row():
                    forward_speed_box = gr.Number(value=REALTIME_PARAMETERS['forward_speed'], label='Forward speed (m/s)', minimum=0.01)
                    turn_rate_box = gr.Number(value=REALTIME_PARAMETERS['turn_rate'], label='Turn rate (deg/s)', minimum=1)
                    tilt_rate_box = gr.Number(value=REALTIME_PARAMETERS['tilt_rate'], label='Tilt rate (deg/s)', minimum=1)
                    stop_time_box = gr.Number(value=REALTIME_PARAMETERS['stop_time'], label='Stop and capture per step (s)', minimum=0)
                with gr.Row():
                    hover_box = gr.Checkbox(value=True, label='Hover while the mapper runs, using this run\'s measured per-step mapper time (needs timing.csv)')
                    speedup_radio = gr.Radio(REALTIME_SPEEDUPS, value=10, label='Video speed (× real time)')
                realtime_button = gr.Button('Build real-time video', variant='primary')
                realtime_note = gr.Markdown()
                realtime_player = gr.Video(label='Real-time video', loop=True, autoplay=True, interactive=False, show_download_button=True)
            with gr.Tab('Evaluation'):
                eval_markdown = gr.Markdown()
                eval_plot = gr.LinePlot(x='step', y='completion ratio (%)', title='Completion ratio over steps', height=320)
            with gr.Tab('Timing'):
                timing_markdown = gr.Markdown()
                timing_table = gr.Dataframe(label='Per-step time by stage', interactive=False)
                timing_breakdown_plot = gr.BarPlot(x='part', y='seconds', title='Where the wall time went', height=300, sort='-y')
                timing_steps_plot = gr.LinePlot(x='step', y='ms', color='stage', title='Time per step', height=360)
            with gr.Tab('Timeline') as timeline_tab:
                gr.Markdown('Trace chart. Runs with `trace_*.jsonl` show **real spans**: one row per thread, grouped by process (mapper node, planner node), so spans that '
                            'overlap in time run in parallel, and a span nested inside another sits on the lane below it. Hover for step and duration, drag to zoom, double-click to reset, '
                            'drag the slider to move the window, click a legend entry to hide a span name. Faded = waiting on another thread/process, grey hatched = visualisation/debug only. '
                            'Older runs only have `timing.csv`: one row per stage, no threads or nesting. The file below opens in [Perfetto](https://ui.perfetto.dev).')
                timeline_button = gr.Button('Draw', variant='primary')
                timeline_plot = gr.Plot(label='Timeline')
                timeline_file = gr.File(label='Perfetto / chrome://tracing file (trace.json)', interactive=False)
            with gr.Tab('Pose errors'):
                pose_error_markdown = gr.Markdown()
                pose_rot_plot = gr.LinePlot(x='step', y='error', color='statistic', title='VGGT-Ω rotation error per mapped frame (°)', height=320)
                pose_trans_plot = gr.LinePlot(x='step', y='error', color='statistic', title='VGGT-Ω translation error per mapped frame (cm)', height=320)
            with gr.Tab('Compare runs'):
                gr.Markdown('All runs side by side. Timing columns are empty for runs recorded before `timing.csv` existed.')
                compare_button = gr.Button('Build comparison')
                compare_table = gr.Dataframe(interactive=False, wrap=True)

            with gr.Tab('Coverage'):
                gr.Markdown('Exploration coverage over steps (the share of the observable ground-truth surface the sensor has seen), for runs scored with '
                            '`scripts/oracle/coverage.py`. Name a run\'s method in `<run>/label.txt`; `scripts/oracle/compare.py` builds the full tables.')
                coverage_select = gr.Dropdown(choices=coverage_choices(results_dir), multiselect=True, label='Runs to plot')
                with gr.Row():
                    coverage_button = gr.Button('Plot', variant='primary')
                    coverage_refresh = gr.Button('Refresh run list')
                coverage_plot = gr.LinePlot(x='step', y='observable surface seen (%)', color='run', title='Observable surface seen', height=420)
                coverage_table = gr.Dataframe(interactive=False, wrap=True)

        run_outputs = [run_state, summary, actions_plot, model_view, frame_slider, rgb_image, depth_image, keyframe_image,
                       visited_image, free_image, trajectory_plot, eval_plot, *runtime_galleries, eval_markdown,
                       timing_markdown, timing_table, timing_steps_plot, timing_breakdown_plot,
                       pose_error_markdown, pose_rot_plot, pose_trans_plot]
        app.load(load_run, inputs=run_dropdown, outputs=run_outputs)
        run_dropdown.change(load_run, inputs=run_dropdown, outputs=run_outputs)
        video_outputs = [*video_players, videos_note]
        # NOTE: encoding takes seconds to a minute, so only the tab and the speed selector build; switching run just shows cached videos
        videos_tab.select(lambda run, fps: show_videos(run, fps, True), inputs=[run_state, fps_radio], outputs=video_outputs)
        fps_radio.change(lambda run, fps: show_videos(run, fps, True), inputs=[run_state, fps_radio], outputs=video_outputs)
        run_state.change(lambda run, fps: show_videos(run, fps, False), inputs=[run_state, fps_radio], outputs=video_outputs)
        def make_realtime_video(run:RunData, forward_speed, turn_rate, tilt_rate, stop_time, hover, speedup):
            if run is None:
                return None, 'No run selected.'
            parameters = {'forward_speed': float(forward_speed), 'turn_rate': float(turn_rate), 'tilt_rate': float(tilt_rate), 'stop_time': float(stop_time or 0)}
            return build_realtime_video(run, parameters, bool(hover), int(speedup))
        realtime_button.click(make_realtime_video, inputs=[run_state, forward_speed_box, turn_rate_box, tilt_rate_box, stop_time_box, hover_box, speedup_radio],
                              outputs=[realtime_player, realtime_note])
        refresh_button.click(refresh_runs, inputs=run_dropdown, outputs=run_dropdown)
        frame_slider.change(show_frame, inputs=[run_state, frame_slider], outputs=[rgb_image, depth_image, keyframe_image])
        load_button.click(load_splat, inputs=run_state, outputs=[model_view, ply_file])
        compare_button.click(compare_runs, outputs=compare_table)
        coverage_button.click(lambda names: coverage_curves(results_dir, names), inputs=coverage_select, outputs=[coverage_plot, coverage_table])
        def draw_timeline(run_name:str):
            run_dir = os.path.join(results_dir, run_name or '')
            if not run_name or not os.path.isdir(run_dir):
                return None, None
            if has_trace(run_dir):
                return build_trace_figure(run_dir), export_chrome_trace(run_dir, os.path.join(run_dir, 'trace.json'))
            if os.path.isfile(os.path.join(run_dir, 'timing.csv')):
                return build_interactive_figure(run_dir), None
            return None, None
        timeline_button.click(draw_timeline, inputs=run_dropdown, outputs=[timeline_plot, timeline_file])
        timeline_tab.select(draw_timeline, inputs=run_dropdown, outputs=[timeline_plot, timeline_file])
        coverage_refresh.click(lambda: gr.update(choices=coverage_choices(results_dir)), outputs=coverage_select)
    return app

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ActiveSplat results dashboard')
    parser.add_argument('--results_dir', type=str, default=os.path.join(PACKAGE_PATH, 'results'))
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--share', action='store_true', help='create a public gradio.live link')
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    build_app(results_dir).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[results_dir])
