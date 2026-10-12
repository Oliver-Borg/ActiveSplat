#!/usr/bin/env python
# Oracle tour: an offline plan that visits viewpoints chosen from the ground-truth mesh, written as an actions.txt that ActiveSplat
# replays (roslaunch ... mode:=REPLAY actions:=<file>) with its normal online mapper.
#
#   1. candidate views: navigable positions of the start floor x yaws, scored by which observable surface samples they see
#   2. greedy (lazy) set cover picks views until the target fraction of the observable surface is covered
#   3. views at the same position are merged into one rotation sweep; the positions are ordered by a nearest-neighbour tour
#      improved with 2-opt on geodesic distance plus the sweep's turns (rotating dominates the step count at 10 deg per turn)
#   4. the tour is executed in the simulator (closed loop: turn towards the next path waypoint, step forward) to get the
#      discrete actions; coverage is tracked on the way, which gives the tour's coverage-vs-steps curve.
# This is a heuristic (set cover + TSP are NP-hard), so it is an informed reference, not a proven upper bound.
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import *
import argparse
import csv
import heapq
import time
from scipy.spatial import cKDTree
from typing import List, Tuple

import habitat_sim

from utils import OPENCV_TO_OPENGL
from oracle import observable_cache_path, load_observable

STOP, FORWARD, TURN_LEFT, TURN_RIGHT, LOOK_UP, LOOK_DOWN = 0, 1, 2, 3, 4, 5

def log(message:str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)

def wrap_angle(angle:float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi

def view_c2w(floor_position:np.ndarray, height:float, yaw:float, pitch:float=0.0) -> np.ndarray:
    '''OpenCV c2w of the sensor at a floor position: yaw about +y (0 looks along -z, as habitat agents do), then pitch up about x.'''
    c, s = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    c2w = np.eye(4)
    c2w[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    c2w[:3, 3] = floor_position + np.array([0.0, height, 0.0])
    return c2w @ OPENCV_TO_OPENGL

def agent_yaw(state) -> float:
    forward = quaternion_rotate(state.rotation, np.array([0.0, 0.0, -1.0]))
    return float(np.arctan2(-forward[0], -forward[2]))

def sensor_pitch(state) -> float:
    forward = quaternion_rotate(state.sensor_states['rgb'].rotation, np.array([0.0, 0.0, -1.0]))
    return float(np.arcsin(np.clip(forward[1], -1.0, 1.0)))

def quaternion_rotate(q, vector:np.ndarray) -> np.ndarray:
    import quaternion
    return quaternion.as_rotation_matrix(q) @ vector

def lazy_greedy_cover(visible:list, target_size:int, target_fraction:float, min_gain:int) -> list:
    '''Picks views (indices into visible) maximising newly covered samples until target_fraction is covered.'''
    covered = np.zeros(target_size, dtype=bool)
    heap = [(-len(v), index) for index, v in enumerate(visible)]
    heapq.heapify(heap)
    chosen = []
    while heap and covered.mean() < target_fraction:
        negative_gain, index = heapq.heappop(heap)
        gain = int(np.count_nonzero(~covered[visible[index]]))
        if heap and gain < -heap[0][0]:
            heapq.heappush(heap, (-gain, index))
            continue
        if gain < min_gain:
            break
        covered[visible[index]] = True
        chosen.append(index)
    return chosen

def geodesic_matrix(pathfinder, positions:np.ndarray) -> np.ndarray:
    count = len(positions)
    distances = np.zeros((count, count))
    path = habitat_sim.ShortestPath()
    for i in range(count):
        for j in range(i + 1, count):
            path.requested_start, path.requested_end = positions[i], positions[j]
            found = pathfinder.find_path(path)
            distances[i, j] = distances[j, i] = path.geodesic_distance if found else 1e4
    return distances

def order_views(start:int, cost:np.ndarray) -> list:
    '''Open tour from start over a symmetric cost: nearest neighbour, then 2-opt with O(1) move evaluation (vectorised over the second cut).'''
    count = len(cost)
    order, left = [start], np.ones(count, dtype=bool)
    left[start] = False
    for _ in range(count - 1):
        candidates = np.nonzero(left)[0]
        nearest = candidates[np.argmin(cost[order[-1], candidates])]
        order.append(int(nearest))
        left[nearest] = False
    order = np.array(order)
    improved = True
    while improved:
        improved = False
        for i in range(1, count - 1):
            # NOTE: reversing order[i..j] replaces edges (i-1, i) and (j, j+1) by (i-1, j) and (i, j+1); the tour is open, so j = last has no second edge
            a, b = order[i - 1], order[i]
            js = np.arange(i + 1, count)
            c = order[js]
            d = order[np.minimum(js + 1, count - 1)]
            has_next = js + 1 < count
            delta = cost[a, c] + np.where(has_next, cost[b, d], 0.0) - cost[a, b] - np.where(has_next, cost[c, d], 0.0)
            best = int(np.argmin(delta))
            if delta[best] < -1e-9:
                j = js[best]
                order[i:j + 1] = order[i:j + 1][::-1].copy()
                improved = True
    return order.tolist()

def sweep_order(views:list, arrival_yaw:float) -> list:
    '''
    Order of the (yaw, pitch) views at one position for a single rotation sweep: the smallest arc containing all yaws, entered
    from the end closer to the arrival heading and traversed in one direction.
    '''
    if len(views) == 1:
        return list(views)
    yaws = np.array([view[0] for view in views])
    ordered = sorted(range(len(views)), key=lambda index: yaws[index] % (2 * np.pi))
    angles = yaws[ordered] % (2 * np.pi)
    gaps = np.diff(np.append(angles, angles[0] + 2 * np.pi))
    first = (int(np.argmax(gaps)) + 1) % len(views)
    counter_clockwise = [views[ordered[(first + k) % len(views)]] for k in range(len(views))]
    clockwise = counter_clockwise[::-1]
    return min([counter_clockwise, clockwise], key=lambda sweep: abs(wrap_angle(sweep[0][0] - arrival_yaw)))

def sweep_turns(views:list, turn_angle:float) -> float:
    '''Turns of a sweep through the views (arc span only; the entry turn depends on how the position is reached).'''
    if len(views) == 1:
        return 0.0
    angles = np.sort(np.array([view[0] for view in views]) % (2 * np.pi))
    gaps = np.diff(np.append(angles, angles[0] + 2 * np.pi))
    return float((2 * np.pi - gaps.max()) / turn_angle)

def execute_tour(dataset, groups:list, order:list, args, observe) -> Tuple[List[int], List[tuple]]:
    '''Drive the simulator through the position groups in order; returns the actions and (step, coverage, path length) rows.'''
    sim = dataset._sim
    forward_step, turn_angle, tilt_angle = float(args.forward_step), np.deg2rad(args.turn_angle), np.deg2rad(args.tilt_angle)
    actions, rows, path = [], [], habitat_sim.ShortestPath()
    path_length = 0.0

    def step(action:int) -> None:
        nonlocal path_length
        before = np.asarray(sim.get_agent_state().position)
        sim.step(action)
        actions.append(action)
        if action == FORWARD:
            path_length += float(np.linalg.norm(np.asarray(sim.get_agent_state().position) - before))
        if len(actions) % args.track_every == 0:
            rows.append((len(actions),) + observe() + (path_length,))

    def face(yaw:float, pitch:float) -> None:
        while len(actions) < args.max_steps:
            error = wrap_angle(yaw - agent_yaw(sim.get_agent_state()))
            if abs(error) <= turn_angle / 2:
                break
            step(TURN_LEFT if error > 0 else TURN_RIGHT)
        for _ in range(8):
            error = pitch - sensor_pitch(sim.get_agent_state())
            if len(actions) >= args.max_steps or abs(error) <= tilt_angle / 2:
                break
            step(LOOK_UP if error > 0 else LOOK_DOWN)

    # NOTE: habitat's greedy geodesic follower picks, at each step, the action that gets closest to the goal along the navmesh,
    # which copes with corners and navmesh edges where "turn towards the next waypoint" gets stuck
    follower = sim.make_greedy_follower(0, goal_radius=forward_step, forward_key='move_forward', left_key='turn_left', right_key='turn_right')
    action_codes = {'move_forward': FORWARD, 'turn_left': TURN_LEFT, 'turn_right': TURN_RIGHT}
    rows.append((0,) + observe() + (0.0,))
    for node in order[1:]:
        goal = groups[node]['position']
        for _ in range(2000):
            if len(actions) >= args.max_steps:
                break
            try:
                action = follower.next_action_along(goal)
            except habitat_sim.errors.GreedyFollowerError:
                log(f'no path to {np.round(goal, 2)}, skipping')
                break
            if action is None:
                break
            step(action_codes[action])
        else:
            log(f'gave up reaching {np.round(goal, 2)}')
        for yaw, pitch in sweep_order(groups[node]['views'], agent_yaw(sim.get_agent_state())):
            face(yaw, pitch)
        if len(actions) >= args.max_steps:
            break
    return actions, rows

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plan an oracle viewpoint tour and write it as an actions file.')
    parser.add_argument('--config', type=str, required=True, help='dataset config (config/datasets/*.json)')
    parser.add_argument('--user_config', type=str, required=True)
    parser.add_argument('--scene_id', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default=None, help='default: data/oracle/tours/<format>_<scene>')
    parser.add_argument('--cache_root', type=str, default=os.path.join(PACKAGE_PATH, 'data'))
    parser.add_argument('--target', type=float, default=0.97, help='fraction of the observable surface the selected views should cover')
    parser.add_argument('--yaws', type=int, default=8, help='viewing directions per position')
    parser.add_argument('--pitches', type=str, default='-30,0,30', help='camera pitches (degrees, positive looks up) tried per direction')
    parser.add_argument('--position_spacing', type=float, default=0.5)
    parser.add_argument('--max_targets', type=int, default=60000, help='surface samples used for the set cover (subsampled)')
    parser.add_argument('--min_gain', type=int, default=20, help='stop when no view adds this many samples')
    parser.add_argument('--max_steps', type=int, default=100000, help='cut the tour after this many actions')
    parser.add_argument('--track_every', type=int, default=10, help='score coverage every n actions while executing (coverage.py scores every step)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    config, user_config = load_configs(args.config, args.user_config, args.scene_id)
    dataset, dataset_config = open_dataset(config, user_config)
    args.forward_step = dataset_config['agent_forward_step_size']
    args.turn_angle = dataset_config['agent_turn_angle']
    args.tilt_angle = dataset_config['agent_tilt_angle']
    near, far = float(config['dataset']['near']), float(config['dataset']['far'])
    scene = make_oracle_scene(dataset_config)
    observable = load_observable(observable_cache_path(args.cache_root, config['dataset']['format'], args.scene_id), scene, far)
    intrinsics = sensor_intrinsics(dataset_config)
    width, height = dataset_config['rgbd_width'], dataset_config['rgbd_height']
    sensor_h = sensor_height(dataset)

    # NOTE: set cover works on a subsample of the observable samples to keep the visibility lists small
    rng = np.random.RandomState(args.seed)
    observable_ids = np.nonzero(observable)[0]
    target_ids = np.sort(rng.choice(observable_ids, min(args.max_targets, len(observable_ids)), replace=False))
    target_mask = np.zeros(len(scene.points), dtype=bool)
    target_mask[target_ids] = True
    position_in_target = np.full(len(scene.points), -1, dtype=np.int64)
    position_in_target[target_ids] = np.arange(len(target_ids))

    positions = navigable_positions(dataset, args.position_spacing, args.seed)
    start_position = np.asarray(dataset._sim.get_agent_state().position)
    yaws = np.arange(args.yaws) * 2 * np.pi / args.yaws
    pitches = np.deg2rad([float(value) for value in args.pitches.split(',')])
    views, visible = [], []
    begin = time.time()
    # NOTE: a k-d tree over just the target samples keeps the per-view range query small (the full sample set is 8x larger)
    target_tree = cKDTree(scene.points[target_ids])
    for count, position in enumerate(positions):
        camera = position + np.array([0.0, sensor_h, 0.0])
        in_range = target_ids[np.asarray(target_tree.query_ball_point(camera, far * 1.05, return_sorted=False), dtype=np.int64)]
        for yaw in yaws:
            for pitch in pitches:
                seen = scene.visible_samples(view_c2w(position, sensor_h, yaw, pitch), intrinsics, width, height, near, far, candidates=in_range)
                if len(seen):
                    views.append({'position': position, 'yaw': float(yaw), 'pitch': float(pitch)})
                    visible.append(position_in_target[seen].astype(np.int32))
        if (count + 1) % 20 == 0:
            log(f'scored {count + 1}/{len(positions)} positions ({time.time() - begin:.0f} s)')
    log(f'{len(views)} candidate views from {len(positions)} positions scored in {time.time() - begin:.0f} s')

    chosen = lazy_greedy_cover(visible, len(target_ids), args.target, args.min_gain)
    covered = np.zeros(len(target_ids), dtype=bool)
    for index in chosen:
        covered[visible[index]] = True
    log(f'set cover: {len(chosen)} views cover {covered.mean() * 100:.1f} % of the observable samples')

    # NOTE: views at the same position become one rotation sweep; group 0 is the start position (no required view)
    groups = [{'position': start_position, 'views': []}]
    group_of = {tuple(np.round(start_position, 4)): 0}
    for index in chosen:
        key = tuple(np.round(views[index]['position'], 4))
        if key not in group_of:
            group_of[key] = len(groups)
            groups.append({'position': views[index]['position'], 'views': []})
        groups[group_of[key]]['views'].append((views[index]['yaw'], views[index]['pitch']))
    begin = time.time()
    geodesic = geodesic_matrix(dataset._sim.pathfinder, np.stack([group['position'] for group in groups]))
    log(f'geodesic distances between {len(groups)} positions in {time.time() - begin:.0f} s')
    turn_angle = np.deg2rad(args.turn_angle)
    sweeps = np.array([sweep_turns(group['views'], turn_angle) if group['views'] else 0.0 for group in groups])
    # NOTE: each sweep is done once wherever it is entered from, so only the walking differs between orders and the cost is symmetric
    cost = geodesic / args.forward_step + 0.5 * (sweeps[:, None] + sweeps[None, :])
    begin = time.time()
    order = order_views(0, cost)
    walking = sum(geodesic[a, b] for a, b in zip(order[:-1], order[1:])) / args.forward_step
    log(f'ordered {len(order)} positions: about {walking:.0f} forward steps + {sweeps.sum():.0f} sweep turns ({time.time() - begin:.1f} s)')

    scene.restrict_to(observable)
    def observe() -> tuple:
        c2w = dataset.get_frame()['c2w'].astype(np.float64) @ OPENCV_TO_OPENGL
        scene.observe(c2w, intrinsics, width, height, near, far)
        return (scene.coverage(),)

    actions, rows = execute_tour(dataset, groups, order, args, observe)
    rows.append((len(actions),) + observe() + (rows[-1][2],))
    out_dir = args.out_dir or os.path.join(args.cache_root, 'oracle', 'tours', f"{config['dataset']['format']}_{args.scene_id}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'actions.txt'), 'w') as f:
        f.write(''.join(f'{action}\n' for action in actions))
    with open(os.path.join(out_dir, 'tour_coverage.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'coverage', 'path_length_m'])
        writer.writerows(rows)
    with open(os.path.join(out_dir, 'tour.json'), 'w') as f:
        json.dump({'scene_id': args.scene_id, 'settings': {k: v for k, v in vars(args).items()}, 'positions': [
            {'position': groups[node]['position'].tolist(), 'views_yaw_pitch_rad': groups[node]['views']} for node in order]}, f, indent=2, default=float)
    log(f'{len(actions)} actions ({actions.count(FORWARD)} forward, {actions.count(TURN_LEFT) + actions.count(TURN_RIGHT)} turns, '
        f'{actions.count(LOOK_UP) + actions.count(LOOK_DOWN)} tilts), final observable coverage {rows[-1][1] * 100:.1f} %, path {rows[-1][2]:.1f} m -> {out_dir}')
    dataset.close()
