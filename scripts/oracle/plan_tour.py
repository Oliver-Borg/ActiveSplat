#!/usr/bin/env python
# Oracle tour: an offline plan that visits viewpoints chosen from the ground-truth mesh, written as an actions.txt that ActiveSplat
# replays (roslaunch ... mode:=REPLAY actions:=<file>) with its normal online mapper.
#
#   1. candidate views: navigable positions of the start floor x yaws x pitches, scored by which observable surface samples they see
#   2. views are chosen either to reach a target coverage with the fewest views (greedy set cover; default), or, with --max_steps, to
#      maximise coverage within a step budget (greedy insertion by newly covered surface per added action, then 2-opt; see below)
#   3. views at the same position are merged into one rotation sweep; the positions are ordered by a nearest-neighbour tour
#      improved with 2-opt on geodesic distance plus the sweep's turns (rotating dominates the step count at 10 deg per turn)
#   4. the tour is executed in the simulator (closed loop, habitat's greedy geodesic follower) to get the discrete actions; coverage is
#      tracked on the way, which gives the tour's coverage-vs-steps curve.
# With a budget the plan's cost estimate (walking + sweeps) is calibrated against the executed action count and the plan is redone with
# a rescaled estimate until the executed tour just fits (a tour that still overruns is cut at the budget).
# This is a heuristic (set cover, orienteering and TSP are NP-hard), so it is an informed reference, not a proven upper bound.
import argparse

from planning import *

ENTRY_STEPS = 1.0   # estimated actions to turn into a view, on top of the sweep arc

def estimate(order:list, views_at:dict, D:np.ndarray, turn_angle:float) -> float:
    walking = sum(D[a, b] for a, b in zip(order[:-1], order[1:]))
    return float(walking + sum(sweep_turns(views_at[p], turn_angle) + ENTRY_STEPS * len(views_at[p]) for p in order if p in views_at))

def plan_within_budget(context:OracleContext, D:np.ndarray, budget:float, args) -> Tuple[list, dict]:
    '''Positions in visiting order (start first) and {position: [(yaw, pitch, view)]}: greedy insertion under an estimated action budget.'''
    start = len(context.positions) - 1
    view_position = np.array([view['position'] for view in context.views])
    order, views_at = [start], {}
    covered = np.zeros(len(context.target_ids), dtype=bool)
    gain_bound = np.array([len(visible) for visible in context.visible], dtype=np.float64)
    for _ in range(4):
        grew = False
        while True:
            tour = np.array(order)
            if len(tour) > 1:
                middle = (D[tour[:-1]] + D[tour[1:]] - D[tour[:-1], tour[1:]][:, None]).min(axis=0)
                insertion = np.minimum(middle, D[tour[-1]])
            else:
                insertion = D[tour[-1]].copy()
            insertion[tour] = 0.0
            used = estimate(order, views_at, D, context.turn_angle)
            # NOTE: extra cost of a view = walking to add its position to the tour (if new) + the longer sweep + turning into it
            extra = np.empty(len(context.views))
            for index, view in enumerate(context.views):
                p = view['position']
                arc = 0.0
                if p in views_at:
                    arc = sweep_turns([(v[0], v[1]) for v in views_at[p]] + [(view['yaw'], view['pitch'])], context.turn_angle) \
                        - sweep_turns([(v[0], v[1]) for v in views_at[p]], context.turn_angle)
                extra[index] = insertion[view_position[index]] + arc + ENTRY_STEPS
            bound = np.where(extra <= budget - used, gain_bound / extra, 0.0)
            best, best_score = -1, 0.0
            for index in np.argsort(-bound):
                if bound[index] <= best_score:
                    break
                gain = int(np.count_nonzero(~covered[context.visible[index]]))
                gain_bound[index] = gain
                if gain >= args.min_gain and gain / extra[index] > best_score:
                    best, best_score = int(index), gain / extra[index]
            if best < 0:
                break
            p = context.views[best]['position']
            if p not in views_at:
                # cheapest place in the order, or the end
                tour = np.array(order)
                options = [D[tour[i], p] + D[p, tour[i + 1]] - D[tour[i], tour[i + 1]] for i in range(len(tour) - 1)] + [D[tour[-1], p]]
                at = int(np.argmin(options))
                order.insert(at + 1, p)
                views_at[p] = []
            views_at[p].append((context.views[best]['yaw'], context.views[best]['pitch'], best))
            covered[context.visible[best]] = True
            grew = True
        # NOTE: reordering (2-opt) can free budget for more views
        groups = [p for p in order]
        sweeps = np.array([sweep_turns([(v[0], v[1]) for v in views_at[p]], context.turn_angle) + ENTRY_STEPS * len(views_at[p]) if p in views_at else 0.0 for p in groups])
        cost = D[np.ix_(groups, groups)] + 0.5 * (sweeps[:, None] + sweeps[None, :])
        new_order = [groups[i] for i in order_views(0, cost)]
        improved = estimate(new_order, views_at, D, context.turn_angle) < used - 1e-6
        order = new_order
        if not grew and not improved:
            break
    return order, views_at

def execute_plan(context:OracleContext, order:list, views_at:dict, max_steps:int, track_every:int) -> Driver:
    context.reset()
    driver = Driver(context, max_steps, track_every)
    for node in order:
        if node != order[0]:
            driver.go_to(context.positions[node])
        if node in views_at:
            for yaw, pitch in sweep_order([(v[0], v[1]) for v in views_at[node]], driver.pose()[1]):
                driver.face(yaw, pitch)
        if driver.done:
            break
    driver.finish()
    return driver

def coverage_at(rows:list, step:int) -> float:
    return [row[1] for row in rows if row[0] <= step][-1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plan an oracle viewpoint tour and write it as an actions file.')
    add_common_arguments(parser)
    parser.add_argument('--target', type=float, default=0.97, help='without a budget: fraction of the observable surface the selected views should cover')
    args = parser.parse_args()

    context = OracleContext(args)
    budgeted = args.max_steps < 100000
    begin = time.time()
    D = geodesic_matrix(context.pathfinder, context.positions) / context.forward_step
    log(f'geodesic distances between {len(context.positions)} positions in {time.time() - begin:.0f} s')

    if budgeted:
        scale, best = 1.0, None
        for iteration in range(8):
            order, views_at = plan_within_budget(context, D, args.max_steps * scale, args)
            driver = execute_plan(context, order, views_at, 10 * args.max_steps, args.track_every)
            length = len(driver.actions)
            coverage = coverage_at(driver.rows, args.max_steps)
            log(f'iteration {iteration}: estimate scale {scale:.3f}, {sum(len(v) for v in views_at.values())} views at {len(views_at)} positions, '
                f'{length} actions executed, coverage at {args.max_steps} steps {coverage * 100:.1f} %')
            if best is None or coverage > best[0]:
                best = (coverage, order, views_at, driver)
            if 0.98 <= length / args.max_steps <= 1.04:
                break
            new_scale = float(np.clip(scale * args.max_steps / length, 0.05, 3.0))
            if abs(new_scale - scale) < 1e-3:
                break
            scale = new_scale
        _, order, views_at, driver = best
        cut = args.max_steps
        actions = driver.actions[:cut]
        rows = [row for row in driver.rows if row[0] < cut] + [(len(actions), coverage_at(driver.rows, cut), [row[2] for row in driver.rows if row[0] <= cut][-1])]
    else:
        visible_ids = list(context.visible)
        chosen = lazy_greedy_cover(visible_ids, len(context.target_ids), args.target, args.min_gain)
        covered = np.zeros(len(context.target_ids), dtype=bool)
        for index in chosen:
            covered[visible_ids[index]] = True
        log(f'set cover: {len(chosen)} views cover {covered.mean() * 100:.1f} % of the observable samples')
        start = len(context.positions) - 1
        views_at = {}
        for index in chosen:
            views_at.setdefault(context.views[index]['position'], []).append((context.views[index]['yaw'], context.views[index]['pitch'], index))
        groups = [start] + [p for p in views_at if p != start]
        sweeps = np.array([sweep_turns([(v[0], v[1]) for v in views_at[p]], context.turn_angle) if p in views_at else 0.0 for p in groups])
        # NOTE: each sweep is done once wherever it is entered from, so only the walking differs between orders and the cost is symmetric
        cost = D[np.ix_(groups, groups)] + 0.5 * (sweeps[:, None] + sweeps[None, :])
        order = [groups[i] for i in order_views(0, cost)]
        driver = execute_plan(context, order, views_at, args.max_steps, args.track_every)
        actions, rows = driver.actions, driver.rows

    kind = f'tours_{args.max_steps}' if budgeted else 'tours'
    out_dir = args.out_dir or os.path.join(args.cache_root, 'oracle', kind, f"{context.config['dataset']['format']}_{args.scene_id}")
    context.save(out_dir, actions, rows, {'scene_id': args.scene_id, 'settings': vars(args), 'positions': [
        {'position': context.positions[p].tolist(), 'views_yaw_pitch_rad': [(v[0], v[1]) for v in views_at.get(p, [])]} for p in order]})
    log(f'{len(actions)} actions ({actions.count(FORWARD)} forward, {actions.count(TURN_LEFT) + actions.count(TURN_RIGHT)} turns, '
        f'{actions.count(LOOK_UP) + actions.count(LOOK_DOWN)} tilts), final observable coverage {rows[-1][1] * 100:.1f} %, path {rows[-1][2]:.1f} m -> {out_dir}')
    context.dataset.close()
