#!/usr/bin/env python
# Which ground-truth surface can the agent observe at all? A sample is observable if some navigable position of the start
# floor's island, at sensor height, sees it unoccluded within the sensor's depth range. Everything else (backs of walls, other
# floors, enclosed spaces) is excluded from oracle coverage and from the oracle planner's notion of "unseen".
# Needs the simulator (navmesh) but no mapper; the result is cached under <root>/oracle/ and used by the oracle planner,
# the coverage scorer and the tour planner.
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import *
import argparse
import time

from oracle import observable_cache_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Precompute the observable ground-truth surface of a scene.')
    parser.add_argument('--config', type=str, required=True, help='dataset config (config/datasets/*.json)')
    parser.add_argument('--user_config', type=str, required=True)
    parser.add_argument('--scene_id', type=str, required=True)
    parser.add_argument('--spacing', type=float, default=0.5, help='spacing of the navigable camera positions (m)')
    parser.add_argument('--cache_root', type=str, default=os.path.join(PACKAGE_PATH, 'data'))
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    config, user_config = load_configs(args.config, args.user_config, args.scene_id)
    path = observable_cache_path(args.cache_root, config['dataset']['format'], args.scene_id)
    if os.path.exists(path) and not args.force:
        print(f'{path} exists')
        sys.exit(0)
    dataset, dataset_config = open_dataset(config, user_config)
    scene = make_oracle_scene(dataset_config)
    # NOTE: the mapper only uses depth up to dataset.far (the simulator's sensor reaches further), so that is the range that counts
    far = float(config['dataset']['far'])
    height = sensor_height(dataset)
    positions = navigable_positions(dataset, args.spacing)
    cameras = positions + np.array([0.0, height, 0.0])
    print(f'{len(scene.points)} surface samples, {len(cameras)} camera positions (sensor height {height:.2f} m, range {far} m)')
    start = time.time()
    observable = scene.observable_samples(cameras, far)
    print(f'{observable.mean() * 100:.1f} % of the samples are observable ({time.time() - start:.0f} s)')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, observable=observable, far=far, sample_spacing=scene.sample_spacing, camera_positions=cameras, sensor_height=height)
    print(f'wrote {path}')
    dataset.close()
