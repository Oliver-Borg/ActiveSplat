'''
Oracle scene: the ground-truth mesh as a prior for viewpoint selection.

Only the planner may use it (to know which surface is still unseen); the Gaussian mapper never sees the geometry.
Everything is in the simulator's world frame, the frame of dataset.get_frame()['c2w'] and of the Gaussians. Camera poses passed in are OpenCV c2w (x right, y down, z forward), i.e.
dataset c2w @ OPENCV_TO_OPENGL, as used by the mapper's look-around renders.
'''
import os
from typing import Tuple

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

from utils import OPENCV_TO_OPENGL

MAX_SAMPLES = 4_000_000

def load_mesh(mesh_url:str, mesh_transform:np.ndarray) -> trimesh.Trimesh:
    '''
    The mesh in the simulator's world frame (the frame of dataset.get_frame()['c2w'] and of the Gaussians). The dataset's
    scene_mesh_transform gives the frame the evaluation and visualisation work in, which is that frame with y and z negated
    (the OPENCV_TO_OPENGL conjugation applied in utils.gui_utils.rgbd_to_pointcloud), so it is undone here.
    '''
    mesh = trimesh.load(mesh_url, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    mesh.apply_transform(OPENCV_TO_OPENGL @ mesh_transform)
    return mesh

def pixel_rays(width:int, height:int, intrinsics:np.ndarray) -> np.ndarray:
    '''(height, width, 3) camera-frame ray directions with z = 1, so that a ray parameter equals the z depth.'''
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    return np.stack([(u - intrinsics[0, 2]) / intrinsics[0, 0], (v - intrinsics[1, 2]) / intrinsics[1, 1], np.ones_like(u)], axis=-1).astype(np.float32)

class OracleScene:

    def __init__(self, mesh:trimesh.Trimesh, voxel_size:float=0.05, sample_spacing:float=0.02, seed:int=0) -> None:
        self.voxel_size = voxel_size
        self.sample_spacing = sample_spacing
        self.mesh = mesh
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.uint32)
        self.__raycaster = o3d.t.geometry.RaycastingScene()
        self.__raycaster.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(faces))

        # NOTE: dense surface samples are the unit of "seen"; a sample is seen once the sensor could see it (in range, in view, not occluded)
        sample_count = int(min(mesh.area / sample_spacing ** 2, MAX_SAMPLES))
        points, face_index = trimesh.sample.sample_surface(mesh, sample_count, seed=seed)
        self.points = points.astype(np.float32)
        self.normals = np.asarray(mesh.face_normals, dtype=np.float32)[face_index]
        self.__tree = cKDTree(self.points)
        self.seen = np.zeros(len(self.points), dtype=bool)

        # NOTE: voxel grid over the same surface for fast "has the surface at this hit point been seen?" lookups
        self.__grid_min = vertices.min(axis=0) - 0.25
        self.__grid_shape = np.ceil((vertices.max(axis=0) + 0.25 - self.__grid_min) / voxel_size).astype(np.int64) + 1
        self.__sample_voxel = self.__voxel_index(self.points)
        self.__voxel_seen = np.zeros(int(np.prod(self.__grid_shape)), dtype=bool)
        self.__observable = None

    @classmethod
    def from_dataset_config(cls, mesh_url:str, mesh_transform:np.ndarray, **kwargs) -> 'OracleScene':
        return cls(load_mesh(mesh_url, mesh_transform), **kwargs)

    def __voxel_index(self, points:np.ndarray) -> np.ndarray:
        '''Linear voxel index per point, -1 outside the grid.'''
        cell = np.floor((points - self.__grid_min) / self.voxel_size).astype(np.int64)
        valid = np.all((cell >= 0) & (cell < self.__grid_shape), axis=-1)
        linear = np.ravel_multi_index(np.clip(cell, 0, self.__grid_shape - 1).T, tuple(self.__grid_shape))
        return np.where(valid, linear, -1)

    def cast(self, origins:np.ndarray, directions:np.ndarray) -> np.ndarray:
        '''Hit parameter per ray (inf for none); hit point = origin + t * direction (directions need not be unit length).'''
        origins = np.broadcast_to(origins, directions.shape)
        rays = np.concatenate([origins, directions], axis=-1).astype(np.float32).reshape(-1, 6)
        return self.__raycaster.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy().reshape(directions.shape[:-1])

    def reset(self) -> None:
        self.seen[:] = False
        self.__voxel_seen[:] = False
        if self.__observable is not None:
            self.restrict_to(self.__observable)

    def visible_samples(self, c2w:np.ndarray, intrinsics:np.ndarray, width:int, height:int, near:float, far:float,
                        sample_mask:np.ndarray=None) -> np.ndarray:
        '''
        Indices of surface samples a camera at c2w (OpenCV) would see: in range, in the image and not occluded.
        sample_mask (bool per sample) restricts the test to a subset of the samples.
        '''
        position, rotation = c2w[:3, 3], c2w[:3, :3]
        candidates = np.asarray(self.__tree.query_ball_point(position, far * 1.05, return_sorted=False), dtype=np.int64)
        if sample_mask is not None:
            candidates = candidates[sample_mask[candidates]]
        if len(candidates) == 0:
            return candidates
        offsets = self.points[candidates] - position.astype(np.float32)
        camera = offsets @ rotation.astype(np.float32)
        z = camera[:, 2]
        in_range = (z > near) & (z <= far)
        z_safe = np.where(in_range, z, 1.0)
        u = intrinsics[0, 0] * camera[:, 0] / z_safe + intrinsics[0, 2]
        v = intrinsics[1, 1] * camera[:, 1] / z_safe + intrinsics[1, 2]
        in_view = in_range & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        candidates, offsets = candidates[in_view], offsets[in_view]
        if len(candidates) == 0:
            return candidates
        distance = np.linalg.norm(offsets, axis=1)
        # NOTE: the sample lies on the mesh, so an unoccluded ray reaches it at t = 1 (direction = offset); a hit clearly before is an occluder
        t_hit = self.cast(position.astype(np.float32), offsets)
        return candidates[t_hit >= 1.0 - 0.03 / np.maximum(distance, 1e-3)]

    def visible_from_position(self, position:np.ndarray, far:float) -> np.ndarray:
        '''Indices of the samples within range of a camera position that are unoccluded from it, in any viewing direction.'''
        candidates = np.asarray(self.__tree.query_ball_point(position, far, return_sorted=False), dtype=np.int64)
        if len(candidates) == 0:
            return candidates
        offsets = self.points[candidates] - position.astype(np.float32)
        distance = np.linalg.norm(offsets, axis=1)
        t_hit = self.cast(position.astype(np.float32), offsets)
        return candidates[t_hit >= 1.0 - 0.03 / np.maximum(distance, 1e-3)]

    def observable_samples(self, camera_positions:np.ndarray, far:float) -> np.ndarray:
        '''Boolean mask of the samples visible (unoccluded, within far) from at least one of the camera positions.'''
        mask = np.zeros(len(self.points), dtype=bool)
        for position in camera_positions:
            mask[self.visible_from_position(position, far)] = True
        return mask

    def restrict_to(self, observable:np.ndarray) -> None:
        '''
        Treat the samples that no camera position can ever see (backs of walls, other floors, ...) as seen, so look-around
        views and coverage only count surface the agent could actually observe.
        '''
        assert observable.shape == self.seen.shape
        self.seen |= ~observable
        voxels = self.__sample_voxel[observable]
        observable_voxels = np.zeros_like(self.__voxel_seen)
        observable_voxels[voxels[voxels >= 0]] = True
        hidden = self.__sample_voxel[~observable]
        hidden = hidden[hidden >= 0]
        # NOTE: a voxel holding both kinds counts as unseen until its observable samples are seen
        self.__voxel_seen[hidden[~observable_voxels[hidden]]] = True
        self.__observable = observable

    def observe(self, c2w:np.ndarray, intrinsics:np.ndarray, width:int, height:int, near:float, far:float) -> int:
        '''Marks the samples the sensor sees from this pose; returns how many were new.'''
        indices = self.visible_samples(c2w, intrinsics, width, height, near, far)
        new = int(np.count_nonzero(~self.seen[indices]))
        self.seen[indices] = True
        voxels = self.__sample_voxel[indices]
        self.__voxel_seen[voxels[voxels >= 0]] = True
        return new

    def look_around_view(self, c2w:np.ndarray, intrinsics:np.ndarray, width:int, height:int) -> Tuple[np.ndarray, np.ndarray]:
        '''
        What the mapper's look-around render returns, from the mesh: (seen, depth) with seen (H, W) = 1 where the first surface
        hit is already seen (or there is none), 0 where it is not; depth (H, W, 1) is the z depth of that hit (0 for none).
        '''
        directions = pixel_rays(width, height, intrinsics) @ c2w[:3, :3].T.astype(np.float32)
        t_hit = self.cast(c2w[:3, 3].astype(np.float32), directions)
        hit = np.isfinite(t_hit)
        points = c2w[:3, 3].astype(np.float32) + np.where(hit, t_hit, 0.0)[..., None] * directions
        voxels = self.__voxel_index(points.reshape(-1, 3)).reshape(hit.shape)
        seen_voxel = np.where(voxels >= 0, self.__voxel_seen[np.maximum(voxels, 0)], True)
        seen = np.where(hit, seen_voxel, True).astype(np.float32)
        depth = np.where(hit, t_hit, 0.0).astype(np.float32)[..., None]
        return seen, depth

    @property
    def seen_count(self) -> int:
        return int(self.seen.sum())

    def coverage(self, target:np.ndarray=None) -> float:
        '''Fraction of the target samples (default: the observable ones if restrict_to() was called, else all) that have been seen.'''
        if target is None:
            target = self.__observable
        if target is None:
            return float(self.seen.mean())
        return float(self.seen[target].mean()) if target.any() else float('nan')


def observable_cache_path(root:str, dataset_format:str, scene_id:str) -> str:
    return os.path.join(root, 'oracle', f'{dataset_format}_{scene_id}_observable.npz')

def load_observable(path:str, scene:'OracleScene', far:float) -> np.ndarray:
    '''The cached observable mask, checked against the scene it was computed for.'''
    data = np.load(path)
    if len(data['observable']) != len(scene.points) or abs(float(data['far']) - far) > 1e-6 or abs(float(data['sample_spacing']) - scene.sample_spacing) > 1e-9:
        raise ValueError(f'{path} does not match this oracle scene (samples, range or spacing differ); recompute it')
    return data['observable']
