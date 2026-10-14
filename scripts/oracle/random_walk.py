#!/usr/bin/env python
# Random-walk baseline (the lower bound of the oracle comparison): a blind persistent random walk written as an actions.txt for
# ActiveSplat's replay mode (mode:=REPLAY actions:=<file>). It needs no simulator or scene: it alternates runs of turns in a random
# direction with runs of forward steps, and now and then tilts the camera. Forward steps blocked by a wall simply do nothing, as in
# the simulator. The same file can be replayed in any scene.
import argparse
import os

import numpy as np

STOP, FORWARD, TURN_LEFT, TURN_RIGHT, LOOK_UP, LOOK_DOWN = 0, 1, 2, 3, 4, 5

def random_walk(steps:int, seed:int, max_turns:int=18, forward_run:tuple=(5, 30), tilt_probability:float=0.03) -> list:
    rng = np.random.RandomState(seed)
    actions = []
    while len(actions) < steps:
        turn = TURN_LEFT if rng.rand() < 0.5 else TURN_RIGHT
        actions += [turn] * int(rng.randint(0, max_turns + 1))
        actions += [FORWARD] * int(rng.randint(forward_run[0], forward_run[1] + 1))
        if rng.rand() < tilt_probability:
            actions.append(LOOK_UP if rng.rand() < 0.5 else LOOK_DOWN)
    return actions[:steps]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write a random-walk actions file.')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default=None, help='default: data/oracle/random_walk/seed<seed>_<steps>/actions.txt')
    args = parser.parse_args()
    out = args.out or os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'data', 'oracle', 'random_walk', f'seed{args.seed}_{args.steps}', 'actions.txt')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    actions = random_walk(args.steps, args.seed)
    with open(out, 'w') as f:
        f.write(''.join(f'{action}\n' for action in actions))
    print(f'{len(actions)} actions ({actions.count(FORWARD)} forward) -> {os.path.normpath(out)}')
