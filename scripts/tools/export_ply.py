#!/usr/bin/env python
# Convert a SplaTAM-style params.npz (results/<run>/gaussians_data/params.npz)
# into a standard 3DGS .ply that viewers such as SuperSplat can open.
import argparse
import os

import numpy as np

SH_C0 = 0.28209479177387814

def export_ply(params_path:str, ply_path:str) -> int:
    params = np.load(params_path)
    means = params['means3D'].astype(np.float32)
    num = means.shape[0]
    f_dc = ((params['rgb_colors'] - 0.5) / SH_C0).astype(np.float32)
    opacities = params['logit_opacities'].reshape(num, 1).astype(np.float32)
    log_scales = params['log_scales'].astype(np.float32)
    if log_scales.shape[1] == 1:
        # NOTE: isotropic gaussians store a single scale
        log_scales = np.repeat(log_scales, 3, axis=1)
    rotations = params['unnorm_rotations'].astype(np.float32)
    rotations = rotations / np.linalg.norm(rotations, axis=1, keepdims=True)
    normals = np.zeros_like(means)

    attributes = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
                  'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    data = np.concatenate([means, normals, f_dc, opacities, log_scales, rotations], axis=1)
    elements = np.empty(num, dtype=[(attribute, 'f4') for attribute in attributes])
    elements[:] = list(map(tuple, data))

    header = 'ply\nformat binary_little_endian 1.0\n'
    header += f'element vertex {num}\n'
    header += ''.join(f'property float {attribute}\n' for attribute in attributes)
    header += 'end_header\n'
    with open(ply_path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(elements.tobytes())
    return num

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export ActiveSplat gaussians to a 3DGS .ply')
    parser.add_argument('result_dir', type=str, help='results/<run> folder')
    parser.add_argument('--output', type=str, default=None, help='defaults to <result_dir>/gaussians.ply')
    args = parser.parse_args()

    params_path = os.path.join(args.result_dir, 'gaussians_data', 'params.npz')
    ply_path = args.output if args.output is not None else os.path.join(args.result_dir, 'gaussians.ply')
    num = export_ply(params_path, ply_path)
    print(f'Exported {num} gaussians to {ply_path}')
