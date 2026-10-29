#!/usr/bin/env python
# Top-down figure of each method's trajectory on one scene, from finished runs (camera poses in gaussians_data/params.npz).
#   python scripts/oracle/trajectory_figure.py --out trajectories.jpg "random walk=<run>" "native=<run>" ...
# The background is the Gaussian map of the first run, cut to the height band the agent moves in; the trajectory is coloured by step.
import argparse
import csv
import os

import numpy as np

def camera_centres(params) -> np.ndarray:
    q = params['cam_unnorm_rots'][0].T
    t = params['cam_trans'][0].T
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    R = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                  2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                  2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=1).reshape(-1, 3, 3)
    return -np.einsum('nji,nj->ni', R, t)

def last_coverage(run:str) -> tuple:
    with open(os.path.join(run, 'coverage.csv')) as f:
        rows = list(csv.DictReader(f))
    return float(rows[-1]['coverage']) * 100, float(rows[-1]['path_length_m'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+', help='label=run_dir')
    parser.add_argument('--out', required=True)
    parser.add_argument('--background', type=str, default=None, help='run whose Gaussian map is drawn behind (default: the first)')
    parser.add_argument('--columns', type=int, default=3)
    parser.add_argument('--title', type=str, default='')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    runs = [item.rsplit('=', 1) for item in args.runs]
    background = np.load(os.path.join(args.background or runs[0][1], 'gaussians_data', 'params.npz'))
    centres = {label: camera_centres(np.load(os.path.join(run, 'gaussians_data', 'params.npz'))) for label, run in runs}
    height = float(np.median(centres[runs[0][0]][:, 1]))
    means = background['means3D']
    # NOTE: the band from just above the floor to just above the camera keeps walls and furniture and drops ceilings and upper floors
    band = means[(means[:, 1] > height - 1.2) & (means[:, 1] < height + 0.3)]
    all_x = np.concatenate([c[:, 0] for c in centres.values()] + [band[:, 0]])
    all_z = np.concatenate([c[:, 2] for c in centres.values()] + [band[:, 2]])
    extent = [all_x.min() - 0.3, all_x.max() + 0.3, all_z.min() - 0.3, all_z.max() + 0.3]
    density, _, _ = np.histogram2d(band[:, 2], band[:, 0], bins=[int((extent[3] - extent[2]) / 0.03), int((extent[1] - extent[0]) / 0.03)],
                                   range=[[extent[2], extent[3]], [extent[0], extent[1]]])
    image = 1 - np.clip(np.log1p(density) / np.log1p(np.percentile(density[density > 0], 98)), 0, 1)

    rows = int(np.ceil(len(runs) / args.columns))
    figure, axes = plt.subplots(rows, args.columns, figsize=(4.4 * args.columns, 3.0 * rows), constrained_layout=True, squeeze=False)
    for axis in axes.ravel():
        axis.axis('off')
    for axis, (label, run) in zip(axes.ravel(), runs):
        c = centres[label]
        axis.imshow(image, cmap='gray', vmin=0, vmax=1.15, extent=[extent[0], extent[1], extent[3], extent[2]], origin='upper', interpolation='nearest')
        axis.scatter(c[:, 0], c[:, 2], c=np.arange(len(c)), cmap='plasma', s=3, linewidths=0)
        axis.plot(c[0, 0], c[0, 2], 'o', color='lime', markeredgecolor='k', markersize=8, label='start')
        axis.plot(c[-1, 0], c[-1, 2], 's', color='cyan', markeredgecolor='k', markersize=7, label='step 1000')
        coverage, path = last_coverage(run)
        axis.set_title(f'{label}\ncoverage {coverage:.1f} %, path {path:.1f} m', fontsize=10)
        axis.set_aspect('equal')
    axes[0, 0].legend(fontsize=7, loc='lower left')
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(0, 1000))
    figure.colorbar(sm, ax=axes, shrink=0.6, label='step', pad=0.01)
    if args.title:
        figure.suptitle(args.title, fontsize=11)
    figure.savefig(args.out, dpi=110, bbox_inches='tight')
