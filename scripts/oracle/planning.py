# Shared pieces of the oracle viewpoint planners (plan_tour.py, plan_nbv.py): the scene set-up, candidate views scored against the
# ground-truth mesh, geodesic costs, and a driver that executes viewpoint goals in the simulator and records the discrete actions.
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import *
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

def geodesic(pathfinder, start:np.ndarray, end:np.ndarray) -> float:
    path = habitat_sim.ShortestPath()
    path.requested_start, path.requested_end = start, end
    return path.geodesic_distance if pathfinder.find_path(path) else float('inf')

class OracleContext:
    '''A simulator and oracle scene for planning: candidate views (navigable positions x yaws x pitches) scored on a subsample of the observable surface.'''

    def __init__(self, args) -> None:
        self.config, user_config = load_configs(args.config, args.user_config, args.scene_id)
        self.dataset, dataset_config = open_dataset(self.config, user_config)
        self.forward_step = float(dataset_config['agent_forward_step_size'])
        self.turn_angle = float(np.deg2rad(dataset_config['agent_turn_angle']))
        self.tilt_angle = float(np.deg2rad(dataset_config['agent_tilt_angle']))
        self.near, self.far = float(self.config['dataset']['near']), float(self.config['dataset']['far'])
        self.scene = make_oracle_scene(dataset_config)
        self.observable = load_observable(observable_cache_path(args.cache_root, self.config['dataset']['format'], args.scene_id), self.scene, self.far)
        self.scene.restrict_to(self.observable)
        self.intrinsics = sensor_intrinsics(dataset_config)
        self.width, self.height = dataset_config['rgbd_width'], dataset_config['rgbd_height']
        self.sensor_height = sensor_height(self.dataset)
        self.pathfinder = self.dataset._sim.pathfinder
        self.start_position = np.asarray(self.dataset._sim.get_agent_state().position)

        # NOTE: the planners work on a subsample of the observable samples to keep the visibility lists small
        rng = np.random.RandomState(args.seed)
        observable_ids = np.nonzero(self.observable)[0]
        self.target_ids = np.sort(rng.choice(observable_ids, min(args.max_targets, len(observable_ids)), replace=False))
        self.position_in_target = np.full(len(self.scene.points), -1, dtype=np.int64)
        self.position_in_target[self.target_ids] = np.arange(len(self.target_ids))
        self.covered = np.zeros(len(self.target_ids), dtype=bool)
        self.__score_views(args)

    def __score_views(self, args) -> None:
        positions = navigable_positions(self.dataset, args.position_spacing, args.seed)
        yaws = np.arange(args.yaws) * 2 * np.pi / args.yaws
        pitches = np.deg2rad([float(value) for value in args.pitches.split(',')])
        self.positions = positions
        self.views, self.visible = [], []
        begin = time.time()
        # NOTE: a k-d tree over just the target samples keeps the per-view range query small (the full sample set is 8x larger)
        target_tree = cKDTree(self.scene.points[self.target_ids])
        for count, position in enumerate(positions):
            camera = position + np.array([0.0, self.sensor_height, 0.0])
            in_range = self.target_ids[np.asarray(target_tree.query_ball_point(camera, self.far * 1.05, return_sorted=False), dtype=np.int64)]
            for yaw in yaws:
                for pitch in pitches:
                    seen = self.scene.visible_samples(view_c2w(position, self.sensor_height, yaw, pitch), self.intrinsics, self.width, self.height,
                                                      self.near, self.far, candidates=in_range)
                    if len(seen):
                        self.views.append({'position': count, 'yaw': float(yaw), 'pitch': float(pitch)})
                        self.visible.append(self.position_in_target[seen].astype(np.int32))
            if (count + 1) % 50 == 0:
                log(f'scored {count + 1}/{len(positions)} positions ({time.time() - begin:.0f} s)')
        log(f'{len(self.views)} candidate views from {len(positions)} positions scored in {time.time() - begin:.0f} s')

    def reset(self) -> None:
        '''Back to the start pose with nothing seen.'''
        self.dataset.reset()
        self.scene.reset()
        self.covered[:] = False

    def gain(self, view:int) -> int:
        return int(np.count_nonzero(~self.covered[self.visible[view]]))

    def observe(self) -> float:
        '''Observes from the simulator's current pose: updates the oracle scene and the covered subsample; returns the observable coverage.'''
        c2w = self.dataset.get_frame()['c2w'].astype(np.float64) @ OPENCV_TO_OPENGL
        indices = self.scene.observe(c2w, self.intrinsics, self.width, self.height, self.near, self.far, return_indices=True)
        targets = self.position_in_target[indices]
        self.covered[targets[targets >= 0]] = True
        return self.scene.coverage()

    def save(self, out_dir:str, actions:list, rows:list, extra:dict) -> None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'actions.txt'), 'w') as f:
            f.write(''.join(f'{action}\n' for action in actions))
        with open(os.path.join(out_dir, 'tour_coverage.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'coverage', 'path_length_m'])
            writer.writerows(rows)
        with open(os.path.join(out_dir, 'tour.json'), 'w') as f:
            json.dump(extra, f, indent=2, default=float)

class Driver:
    '''Executes goals in the simulator, recording the discrete actions and a coverage-vs-steps curve; stops taking actions at max_steps.'''

    def __init__(self, context:OracleContext, max_steps:int, track_every:int=10) -> None:
        self.context, self.max_steps, self.track_every = context, max_steps, track_every
        self.sim = context.dataset._sim
        self.actions, self.path_length = [], 0.0
        self.rows = [(0, context.observe(), 0.0)]
        self.follower = self.sim.make_greedy_follower(0, goal_radius=context.forward_step, forward_key='move_forward', left_key='turn_left', right_key='turn_right')

    @property
    def done(self) -> bool:
        return len(self.actions) >= self.max_steps

    def step(self, action:int) -> None:
        if self.done:
            return
        before = np.asarray(self.sim.get_agent_state().position)
        self.sim.step(action)
        self.actions.append(action)
        if action == FORWARD:
            self.path_length += float(np.linalg.norm(np.asarray(self.sim.get_agent_state().position) - before))
        if len(self.actions) % self.track_every == 0:
            self.rows.append((len(self.actions), self.context.observe(), self.path_length))

    def pose(self) -> Tuple[np.ndarray, float, float]:
        state = self.sim.get_agent_state()
        return np.asarray(state.position), agent_yaw(state), sensor_pitch(state)

    def go_to(self, goal:np.ndarray) -> bool:
        # NOTE: habitat's greedy geodesic follower picks, at each step, the action that gets closest to the goal along the navmesh,
        # which copes with corners and navmesh edges where "turn towards the next waypoint" gets stuck
        codes = {'move_forward': FORWARD, 'turn_left': TURN_LEFT, 'turn_right': TURN_RIGHT}
        for _ in range(2000):
            if self.done:
                return False
            try:
                action = self.follower.next_action_along(goal)
            except habitat_sim.errors.GreedyFollowerError:
                log(f'no path to {np.round(goal, 2)}, skipping')
                return False
            if action is None:
                return True
            self.step(codes[action])
        log(f'gave up reaching {np.round(goal, 2)}')
        return False

    def face(self, yaw:float, pitch:float) -> None:
        context = self.context
        while not self.done:
            error = wrap_angle(yaw - agent_yaw(self.sim.get_agent_state()))
            if abs(error) <= context.turn_angle / 2:
                break
            self.step(TURN_LEFT if error > 0 else TURN_RIGHT)
        for _ in range(8):
            error = pitch - sensor_pitch(self.sim.get_agent_state())
            if self.done or abs(error) <= context.tilt_angle / 2:
                break
            self.step(LOOK_UP if error > 0 else LOOK_DOWN)
        context.observe()

    def finish(self) -> None:
        self.rows.append((len(self.actions), self.context.observe(), self.path_length))

def add_common_arguments(parser) -> None:
    parser.add_argument('--config', type=str, required=True, help='dataset config (config/datasets/*.json)')
    parser.add_argument('--user_config', type=str, required=True)
    parser.add_argument('--scene_id', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--cache_root', type=str, default=os.path.join(PACKAGE_PATH, 'data'))
    parser.add_argument('--yaws', type=int, default=8, help='viewing directions per position')
    parser.add_argument('--pitches', type=str, default='-30,0,30', help='camera pitches (degrees, positive looks up) tried per direction')
    parser.add_argument('--position_spacing', type=float, default=0.5)
    parser.add_argument('--max_targets', type=int, default=60000, help='surface samples used for the gains (subsampled)')
    parser.add_argument('--min_gain', type=int, default=20, help='ignore views that add fewer samples')
    parser.add_argument('--max_steps', type=int, default=100000, help='step budget: no more than this many actions')
    parser.add_argument('--track_every', type=int, default=10, help='score coverage every n actions while executing (coverage.py scores every step)')
    parser.add_argument('--seed', type=int, default=0)
