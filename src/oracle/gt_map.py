# The planner's top-down free map built from the ground-truth mesh instead of the Gaussians (planner.oracle.navigation = "gt_free_map").
# Everything the mapper has not seen yet is already known, so the Voronoi graph, its subregions and its nodes cover the whole start floor
# from the first step; the planner's scoring is untouched.
from typing import Dict

import cv2
import numpy as np

from utils.gui_utils import translations_world_to_topdown

def gt_free_map(surface_points:np.ndarray, island_positions:np.ndarray, start:np.ndarray, topdown_info:Dict, height_cut:tuple,
                island_radius:float=0.6) -> np.ndarray:
    '''
    Boolean image (rows = topdown y, columns = topdown x), True where the agent can be: no ground-truth surface inside the agent's height band
    (the same cut as the Gaussian free map: keep -y between upper_limit and lower_limit), within island_radius of a navigable position of the start floor, and connected to the start.
    surface_points, island_positions and start are in the simulator's world frame; height_cut = (upper_limit, lower_limit) as in the visualizer.
    '''
    columns, rows = topdown_info['grid_map_shape']
    meter_per_pixel = topdown_info['meter_per_pixel']
    def pixels(points:np.ndarray) -> np.ndarray:
        return np.floor(translations_world_to_topdown(points, topdown_info, np.float64)).astype(np.int64)
    # NOTE: the oracle's simulator world frame is the frame of the Gaussians and of the top-down map (checked against a run's Gaussians)
    in_band = ~((-surface_points[:, 1] < height_cut[0]) | (-surface_points[:, 1] > height_cut[1]))
    occupied = np.zeros((rows, columns), dtype=np.uint8)
    xy = pixels(surface_points[in_band])
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < columns) & (xy[:, 1] >= 0) & (xy[:, 1] < rows)
    occupied[xy[inside, 1], xy[inside, 0]] = 255
    # NOTE: samples are 2 cm apart, about a pixel, so thicken the surface by a pixel to keep walls closed
    occupied = cv2.dilate(occupied, np.ones((3, 3), np.uint8))
    near_island = np.zeros_like(occupied)
    radius = int(np.ceil(island_radius / meter_per_pixel))
    for x, y in pixels(island_positions):
        if 0 <= x < columns and 0 <= y < rows:
            cv2.circle(near_island, (int(x), int(y)), radius, 255, -1)
    candidate = ((occupied == 0) & (near_island > 0)).astype(np.uint8)
    count, labels = cv2.connectedComponents(candidate, connectivity=4)
    x, y = pixels(start.reshape(1, 3))[0]
    # NOTE: the start pixel can sit on a thickened surface sample; take the nearest candidate pixel
    ys, xs = np.nonzero(candidate)
    nearest = np.argmin((xs - x) ** 2 + (ys - y) ** 2)
    return labels == labels[ys[nearest], xs[nearest]]
