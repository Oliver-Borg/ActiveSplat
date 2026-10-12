# Shared helpers for the oracle scripts: dataset/simulator setup without ROS and the oracle scene of a dataset.
import json
import os
import sys
PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
sys.path.append(PACKAGE_PATH)
sys.path.append(os.path.join(PACKAGE_PATH, 'src'))
from typing import List, Tuple

import numpy as np

from dataloader.dataloader import get_dataset
from oracle import OracleScene, load_mesh

def load_configs(config_path:str, user_config_path:str, scene_id:str=None) -> Tuple[dict, dict]:
    '''A dataset config (as mapper_node loads it, env config made absolute) and the user config.'''
    with open(config_path) as f:
        config = json.load(f)
    if 'env' in config:
        config['env']['config'] = os.path.abspath(os.path.join(os.path.dirname(config_path), os.pardir, os.pardir, config['env']['config']))
    if scene_id is not None:
        config['dataset']['scene_id'] = scene_id
    with open(user_config_path) as f:
        user_config = json.load(f)
    return config, user_config

def open_dataset(config:dict, user_config:dict):
    '''(dataset with a running simulator, its dataset_config dict).'''
    dataset = get_dataset(config, user_config, 'Eval')
    return dataset, dataset.setup()

def make_oracle_scene(dataset_config:dict, **kwargs) -> OracleScene:
    return OracleScene(load_mesh(dataset_config['scene_mesh_url'], dataset_config['scene_mesh_transform']), **kwargs)

def sensor_intrinsics(dataset_config:dict) -> np.ndarray:
    return np.array([[dataset_config['rgbd_fx'], 0, dataset_config['rgbd_cx']],
                     [0, dataset_config['rgbd_fy'], dataset_config['rgbd_cy']], [0, 0, 1]])

def sensor_height(dataset) -> float:
    state = dataset._sim.get_agent_state()
    return float(state.sensor_states['rgb'].position[1] - state.position[1])

def navigable_positions(dataset, spacing:float, seed:int=0, draws:int=100000) -> np.ndarray:
    '''Navigable floor points of the start position's island, thinned to about one per spacing x spacing cell.'''
    pathfinder = dataset._sim.pathfinder
    start = np.asarray(dataset._sim.get_agent_state().position)
    island = pathfinder.get_island(start)
    pathfinder.seed(seed)
    seen_cells, points = set(), []
    for _ in range(draws):
        point = np.asarray(pathfinder.get_random_navigable_point(island_index=island))
        cell = (int(np.floor(point[0] / spacing)), int(np.floor(point[1] / 0.5)), int(np.floor(point[2] / spacing)))
        if cell not in seen_cells:
            seen_cells.add(cell)
            points.append(point)
    return np.stack(points + [start])
