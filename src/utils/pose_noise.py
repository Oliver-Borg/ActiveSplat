# Simulated visual-odometry error: the mapper and planner are given an estimated camera pose instead of the simulator's true one.
#
# The estimate is built from the true relative motion between consecutive frames, each corrupted by noise that grows with the motion
# (drift), so errors accumulate like a real odometry/SLAM front end without loop closure:
#   est_t = est_{t-1} @ (true_{t-1}^-1 @ true_t) @ exp(noise),   noise_translation ~ N(0, (translation_drift * |dt|)^2) per axis,
#                                                                 noise_rotation    ~ N(0, (rotation_drift * |dR|)^2) per axis
# plus an independent per-frame jitter that does not accumulate (jitter_m, jitter_deg). Poses are OpenCV camera-to-world 4x4.
# Config block "pose_noise" of the dataset config, optionally overridden by the JSON in ACTIVESPLAT_POSE_NOISE, e.g.
#   ACTIVESPLAT_POSE_NOISE='{"translation_drift": 0.01, "rotation_drift": 0.01}'
import json
import os

import cv2
import numpy as np

DEFAULTS = {'enabled': False, 'translation_drift': 0.01, 'rotation_drift': 0.01, 'jitter_m': 0.0, 'jitter_deg': 0.0, 'seed': 0}
ENV_VARIABLE = 'ACTIVESPLAT_POSE_NOISE'

def resolve_settings(config:dict) -> dict:
    settings = {**DEFAULTS, **config.get('pose_noise', {})}
    override = os.environ.get(ENV_VARIABLE)
    if override:
        settings.update(json.loads(override))
        settings['enabled'] = True
    unknown = set(settings) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'pose_noise: unknown settings {sorted(unknown)}; known: {sorted(DEFAULTS)}')
    return settings

def to_matrix(rotation_vector:np.ndarray, translation:np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = cv2.Rodrigues(np.asarray(rotation_vector, dtype=np.float64).reshape(3, 1))[0]
    matrix[:3, 3] = translation
    return matrix

def rotation_angle_deg(rotation:np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(cv2.Rodrigues(rotation)[0])))

class PoseNoise:

    def __init__(self, settings:dict) -> None:
        self.settings = settings
        self.enabled = bool(settings['enabled'])
        self.__rng = np.random.RandomState(settings['seed'])
        self.__true_last = None
        self.__drifted = None
        self.rows = []   # (true c2w, estimated c2w) per call, for the error log and the alignment

    def apply(self, true_c2w:np.ndarray) -> np.ndarray:
        '''The estimated pose for the next frame (call once per captured frame, in order).'''
        true_c2w = np.asarray(true_c2w, dtype=np.float64)
        if not self.enabled:
            return true_c2w
        settings = self.__rng
        if self.__true_last is None:
            self.__drifted = true_c2w.copy()
        else:
            motion = np.linalg.inv(self.__true_last) @ true_c2w
            distance = float(np.linalg.norm(motion[:3, 3]))
            angle = rotation_angle_deg(motion[:3, :3])
            noise = to_matrix(np.radians(settings.randn(3) * self.settings['rotation_drift'] * angle),
                              settings.randn(3) * self.settings['translation_drift'] * distance)
            self.__drifted = self.__drifted @ motion @ noise
        self.__true_last = true_c2w.copy()
        jitter = to_matrix(np.radians(settings.randn(3) * self.settings['jitter_deg']), settings.randn(3) * self.settings['jitter_m'])
        estimated = self.__drifted @ jitter
        self.rows.append((true_c2w, estimated))
        return estimated

    def alignment(self) -> np.ndarray:
        '''Rigid 4x4 A with true ~= A @ estimated over the whole trajectory (the usual ATE alignment, fitted to the camera centres and a point
        along each camera axis so that rotation is determined even on a straight path). A map in the estimated frame is compared with the real world
        through it: a view at true pose T is rendered from the map at inv(A) @ T.'''
        if not self.rows:
            return np.eye(4)
        offsets = np.array([[0, 0, 0, 1], [0.5, 0, 0, 1], [0, 0.5, 0, 1], [0, 0, 0.5, 1]], dtype=np.float64).T
        estimated = np.concatenate([(e @ offsets)[:3].T for _, e in self.rows])
        true = np.concatenate([(t @ offsets)[:3].T for t, _ in self.rows])
        mean_e, mean_t = estimated.mean(axis=0), true.mean(axis=0)
        u, _, vt = np.linalg.svd((true - mean_t).T @ (estimated - mean_e))
        sign = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
        rotation = u @ sign @ vt
        alignment = np.eye(4)
        alignment[:3, :3] = rotation
        alignment[:3, 3] = mean_t - rotation @ mean_e
        return alignment

    def errors(self) -> dict:
        '''Absolute trajectory error after alignment, and the raw error at the end.'''
        if not self.rows:
            return {}
        alignment = self.alignment()
        translation = np.array([np.linalg.norm((alignment @ e)[:3, 3] - t[:3, 3]) for t, e in self.rows])
        rotation = np.array([rotation_angle_deg(t[:3, :3].T @ (alignment @ e)[:3, :3]) for t, e in self.rows])
        return {'frames': len(self.rows), 'ate_rmse_m': float(np.sqrt((translation ** 2).mean())), 'ate_max_m': float(translation.max()),
                'rot_err_mean_deg': float(rotation.mean()), 'rot_err_max_deg': float(rotation.max()),
                'final_translation_error_m': float(np.linalg.norm(self.rows[-1][1][:3, 3] - self.rows[-1][0][:3, 3]))}
