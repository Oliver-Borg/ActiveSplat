import atexit
import os
import secrets
import subprocess
import threading
from multiprocessing.connection import Client
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree

from utils.logging_utils import Log

WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'worker.py')

class VGGTOmegaSeeder:
    '''
    Seeds new Gaussians from VGGT-Omega depth instead of sensor depth.
    VGGT-Omega needs python>=3.10 / torch>=2.6, so it runs in its own env as a worker process (worker.py)
    on the same GPU; this class starts it and exchanges numpy arrays with it over a local socket.
    '''

    def __init__(self, config:Dict) -> None:
        self.__window_size = int(config['window_size'])
        self.__knn_k = int(config['knn_k'])
        self.__conf_drop_percentile = float(config['conf_drop_percentile'])
        self.__min_scale_samples = 16

        python_path = os.path.expanduser(config['python'])
        checkpoint_path = os.path.expanduser(config['checkpoint']) if config['checkpoint'] else ''
        if checkpoint_path and not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f'VGGT-Omega checkpoint not found: {checkpoint_path}')
        authkey = secrets.token_hex(16)
        # NOTE: -I keeps the worker away from this env's PYTHONPATH (catkin workspace / python3.9 site-packages)
        self.__process = subprocess.Popen(
            [python_path, '-I', WORKER_PATH,
             '--checkpoint', checkpoint_path,
             '--image_resolution', str(config['image_resolution']),
             '--authkey', authkey],
            stdout=subprocess.PIPE, text=True)
        atexit.register(self.close)
        port = None
        for line in self.__process.stdout:
            if line.startswith('READY '):
                port = int(line.split()[1])
                break
        if port is None:
            raise RuntimeError(f'VGGT-Omega worker exited before becoming ready (exit code {self.__process.poll()})')
        # NOTE: keep draining the worker's stdout so it can never block on a full pipe
        threading.Thread(target=lambda: [None for _ in self.__process.stdout], daemon=True).start()
        self.__connection = Client(('127.0.0.1', port), authkey=authkey.encode())
        Log(f'VGGT-Omega worker ready (pid {self.__process.pid}, checkpoint: {checkpoint_path or "random weights"})', tag='VGGTOmega')

    def close(self) -> None:
        if self.__process is None:
            return
        try:
            self.__connection.send(None)
            self.__connection.close()
        except (OSError, AttributeError):
            pass
        try:
            self.__process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.__process.kill()
        self.__process = None

    def get_window_size(self) -> int:
        return self.__window_size

    def predict(self, images:np.ndarray) -> Dict[str, np.ndarray]:
        '''
        images: uint8 (N, H, W, 3), the frame to seed last. Returns the worker response: 'depth', 'conf' (H, W) of the
        last frame (depth up to an unknown scale), 'extrinsics' (N, 3, 4) camera-from-world for every frame in the
        same scale, and 'ms', the worker's inference time.
        '''
        self.__connection.send({'images': np.ascontiguousarray(images, dtype=np.uint8)})
        response = self.__connection.recv()
        if 'error' in response:
            raise RuntimeError(f'VGGT-Omega worker failed:\n{response["error"]}')
        return response
    
    def metric_depth(self, images:np.ndarray, sensor_depth:torch.Tensor) -> Tuple[torch.Tensor, float, Dict[str, np.ndarray]]:
        '''
        VGGT-Omega depth for the last image, scaled so its median matches the sensor depth's median over pixels
        where both are valid. Pixels in the lowest conf_drop_percentile of confidence are set to 0 (invalid).
        sensor_depth: (1, H, W) metres. Returns (1, H, W) metres, the metric scale factor (nan when it could not be
        fixed) and the raw worker response.
        '''
        response = self.predict(images)
        depth = torch.from_numpy(response['depth']).to(sensor_depth.device)
        conf = torch.from_numpy(response['conf']).to(sensor_depth.device)
        valid = (depth > 0) & torch.isfinite(depth)
        if self.__conf_drop_percentile > 0 and valid.any():
            conf_threshold = torch.quantile(conf[valid], self.__conf_drop_percentile / 100)
            valid &= conf >= conf_threshold
        both_valid = valid & (sensor_depth[0] > 0)
        if both_valid.sum() < self.__min_scale_samples:
            Log('VGGT-Omega: too few pixels with sensor depth to fix the scale, seeding nothing for this frame', tag='VGGTOmega')
            return torch.zeros_like(sensor_depth), float('nan'), response
        scale = torch.median(sensor_depth[0][both_valid]) / torch.median(depth[both_valid])
        return (depth * scale * valid)[None], float(scale), response
    
    @staticmethod
    def pose_errors(pred_extrinsics:np.ndarray, gt_w2c:np.ndarray, scale:float) -> List[Dict[str, float]]:
        '''
        Errors of VGGT-Omega's predicted poses for each window frame relative to the last (current) frame.
        pred_extrinsics: (N, 3, 4) camera-from-world in VGGT-Omega's frame and scale; gt_w2c: (N, 4, 4) true
        camera-from-world; scale: metric factor from the depth alignment (nan -> translation errors are nan).
        Relative poses don't depend on either world frame, so no alignment beyond the scale is needed.
        '''
        pred_w2c = np.tile(np.eye(4), (len(pred_extrinsics), 1, 1))
        pred_w2c[:, :3, :] = pred_extrinsics
        rows = []
        for index in range(len(pred_w2c) - 1):
            # NOTE: pose of window frame `index` expressed in the current camera's frame
            pred_relative = pred_w2c[-1] @ np.linalg.inv(pred_w2c[index])
            gt_relative = gt_w2c[-1] @ np.linalg.inv(gt_w2c[index])
            rotation_delta = pred_relative[:3, :3].T @ gt_relative[:3, :3]
            rot_err_deg = np.degrees(np.arccos(np.clip((np.trace(rotation_delta) - 1) / 2, -1.0, 1.0)))
            pred_translation = pred_relative[:3, 3] * scale
            gt_translation = gt_relative[:3, 3]
            gt_baseline = float(np.linalg.norm(gt_translation))
            trans_err_m = float(np.linalg.norm(pred_translation - gt_translation))
            trans_dir_err_deg = float('nan')
            pred_baseline = float(np.linalg.norm(pred_translation))
            # NOTE: direction is meaningless when the camera only turned in place
            if gt_baseline > 0.05 and pred_baseline > 1e-6:
                cosine = np.dot(pred_translation, gt_translation) / (pred_baseline * gt_baseline)
                trans_dir_err_deg = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
            rows.append({
                'pair_index': index,
                'rot_err_deg': float(rot_err_deg),
                'trans_err_m': trans_err_m,
                'trans_dir_err_deg': trans_dir_err_deg,
                'gt_baseline_m': gt_baseline})
        return rows
    
    def knn_mean_sq_dist(self, all_points:torch.Tensor, query_mask:torch.Tensor) -> torch.Tensor:
        '''
        Mean squared distance from each queried point to its knn_k nearest neighbours among all_points
        (the frame's whole VGGT-Omega point cloud), like 3DGS's point-cloud initialisation.
        all_points: (N, 3); query_mask: (N,) bool. Returns (num_queried,).
        '''
        points_np = all_points.detach().cpu().numpy()
        num_queried = int(query_mask.sum())
        knn_k = min(self.__knn_k, len(points_np) - 1)
        if knn_k < 1:
            return torch.full((num_queried,), 1e-4, device=all_points.device)
        tree = cKDTree(points_np)
        # NOTE: k + 1 because each queried point finds itself first
        distances, _ = tree.query(points_np[query_mask.cpu().numpy()], k=knn_k + 1, workers=-1)
        distances = distances.reshape(num_queried, knn_k + 1)
        mean_sq_dist = np.mean(distances[:, 1:] ** 2, axis=1)
        return torch.from_numpy(np.maximum(mean_sq_dist, 1e-7)).to(all_points.device).float()
