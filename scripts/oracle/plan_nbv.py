#!/usr/bin/env python
# Tier 3 oracle: greedy next-best-view with free candidate poses on the ground-truth navmesh, written as an actions.txt that
# ActiveSplat replays (roslaunch ... mode:=REPLAY actions:=<file>) with its normal online mapper.
#
# From the current pose it considers every candidate view (navigable positions x yaws x pitches), scores each by the ground-truth
# surface it would newly see per action needed to get there (geodesic walking + turns/tilts), drives to the best one with habitat's
# greedy follower, and repeats until the step budget is spent. Unlike the tour it is myopic and closed-loop, but it needs no
# global plan; unlike the native planner it picks any pose, not Voronoi graph nodes, and knows exactly what is unseen.
import argparse

from planning import *

def pick_view(context:OracleContext, driver:Driver, gain_bound:np.ndarray, args, remaining:int) -> int:
    '''Candidate view with the best (gain, capped at what the remaining budget can reach) per action; -1 if none is worth going to.'''
    position, yaw, pitch = driver.pose()
    positions = context.positions[[view['position'] for view in context.views]]
    walking_bound = np.linalg.norm(positions - position, axis=1) / context.forward_step
    turning = (np.abs(wrap_angle_array(np.array([view['yaw'] for view in context.views]) - yaw)) / context.turn_angle
               + np.abs(np.array([view['pitch'] for view in context.views]) - pitch) / context.tilt_angle)
    # NOTE: gains only shrink, so the stale gain over the cheapest possible cost bounds each view's score; views are evaluated best bound first
    # and the search stops once the next bound cannot beat the best exact score (lazy greedy)
    bound = gain_bound / np.maximum(walking_bound, 1.0)
    best, best_score = -1, 0.0
    for index in np.argsort(-bound):
        if bound[index] <= best_score:
            break
        if gain_bound[index] < args.min_gain:
            continue
        gain = context.gain(index)
        gain_bound[index] = gain
        if gain < args.min_gain:
            continue
        distance = geodesic(context.pathfinder, position, positions[index])
        if not np.isfinite(distance):
            gain_bound[index] = 0
            continue
        cost = max(distance / context.forward_step + turning[index], 1.0)
        score = gain * min(1.0, remaining / cost) / cost
        if score > best_score:
            best, best_score = int(index), score
    return best

def wrap_angle_array(angles:np.ndarray) -> np.ndarray:
    return (angles + np.pi) % (2 * np.pi) - np.pi

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plan an oracle greedy next-best-view exploration and write it as an actions file.')
    add_common_arguments(parser)
    args = parser.parse_args()

    context = OracleContext(args)
    driver = Driver(context, args.max_steps, args.track_every)
    gain_bound = np.array([len(visible) for visible in context.visible], dtype=np.float64)
    begin = time.time()
    chosen = []
    while not driver.done:
        view = pick_view(context, driver, gain_bound, args, args.max_steps - len(driver.actions))
        if view < 0:
            log(f'no view adds {args.min_gain}+ samples, stopping at step {len(driver.actions)}')
            break
        goal = context.views[view]
        driver.go_to(context.positions[goal['position']])
        driver.face(goal['yaw'], goal['pitch'])
        gain_bound[view] = 0
        chosen.append(view)
        if len(chosen) % 10 == 0:
            log(f'{len(chosen)} views, {len(driver.actions)} actions, coverage {driver.rows[-1][1] * 100:.1f} % ({time.time() - begin:.0f} s)')
    driver.finish()
    actions, rows = driver.actions, driver.rows
    out_dir = args.out_dir or os.path.join(args.cache_root, 'oracle', f'nbv_{args.max_steps}', f"{context.config['dataset']['format']}_{args.scene_id}")
    context.save(out_dir, actions, rows, {'scene_id': args.scene_id, 'settings': vars(args), 'views': [
        {'position': context.positions[context.views[v]['position']].tolist(), 'yaw': context.views[v]['yaw'], 'pitch': context.views[v]['pitch']} for v in chosen]})
    log(f'{len(chosen)} views, {len(actions)} actions ({actions.count(FORWARD)} forward, {actions.count(TURN_LEFT) + actions.count(TURN_RIGHT)} turns, '
        f'{actions.count(LOOK_UP) + actions.count(LOOK_DOWN)} tilts), final observable coverage {rows[-1][1] * 100:.1f} %, path {rows[-1][2]:.1f} m -> {out_dir}')
    context.dataset.close()
