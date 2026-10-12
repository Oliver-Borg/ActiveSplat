#!/usr/bin/env python
# VGGT-Omega depth worker. Runs in its own env (python>=3.10, torch>=2.6), separate from the
# ActiveSplat env (python3.9, torch 2.0), and serves depth predictions over a local
# multiprocessing.connection socket. Started and stopped by mapper.vggt_omega.VGGTOmegaSeeder.
#
# Request:  {'images': uint8 (N, H, W, 3)}   (the frame to seed is the last one)
# Response: {'depth': float32 (H, W), 'conf': float32 (H, W)} for the last frame, resized back to the input H, W,
#           {'extrinsics': float32 (N, 3, 4)} camera-from-world (OpenCV) for every frame, in the first frame's
#           coordinates and in the same (unknown) scale as the depth, and {'ms': float}.
#           {'error': str} on failure.
import argparse
import sys
import time
import traceback
from multiprocessing.connection import Listener

import numpy as np
import torch
import torch.nn.functional as F

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.pose_enc import encoding_to_camera

def target_shape(height:int, width:int, image_resolution:int, patch_size:int) -> tuple:
    # NOTE: same token budget as vggt_omega.utils.load_fn's "balanced" mode, without its centre crop,
    # so predicted depth maps back onto the input pixels by a plain resize
    token_number = (image_resolution // patch_size) ** 2
    aspect_ratio = height / max(width, 1)
    w_patches = max(1, int(np.round(np.sqrt(token_number / aspect_ratio))))
    h_patches = max(1, int(np.round(token_number / w_patches)))
    return h_patches * patch_size, w_patches * patch_size

def main() -> None:
    parser = argparse.ArgumentParser(description='VGGT-Omega depth worker')
    parser.add_argument('--checkpoint', type=str, default='', help='empty: random weights (plumbing tests only)')
    parser.add_argument('--image_resolution', type=int, default=512)
    parser.add_argument('--authkey', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda')
    model = VGGTOmega(enable_camera=True, enable_depth=True).to(device).eval()
    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location='cpu')
        # NOTE: the text alignment head is disabled, so its weights are expected to be unused
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(f'Checkpoint is missing weights: {missing[:5]}')
        unexpected = [key for key in unexpected if not key.startswith('text_alignment_head.')]
        if unexpected:
            raise RuntimeError(f'Checkpoint has unexpected weights: {unexpected[:5]}')
    else:
        print('WARNING: no checkpoint given, using random weights', file=sys.stderr, flush=True)
    patch_size = model.aggregator.patch_embed.patch_size if hasattr(model.aggregator.patch_embed, 'patch_size') else 16
    patch_size = patch_size[0] if isinstance(patch_size, (tuple, list)) else int(patch_size)

    listener = Listener(('127.0.0.1', 0), authkey=args.authkey.encode())
    # NOTE: the parent reads this line to find the port
    print(f'READY {listener.address[1]}', flush=True)
    connection = listener.accept()
    while True:
        try:
            request = connection.recv()
        except EOFError:
            break
        if request is None:
            break
        try:
            start = time.perf_counter()
            images = torch.from_numpy(request['images']).to(device)
            num_frames, height, width = images.shape[:3]
            images = images.permute(0, 3, 1, 2).float() / 255.0
            model_height, model_width = target_shape(height, width, args.image_resolution, patch_size)
            images = F.interpolate(images, size=(model_height, model_width), mode='bicubic', align_corners=False).clamp(0, 1)
            with torch.inference_mode():
                predictions = model(images)
            extrinsics, _ = encoding_to_camera(predictions['pose_enc'], (model_height, model_width), build_intrinsics=False)
            depth = predictions['depth'].reshape(num_frames, model_height, model_width)[-1]
            conf = predictions['depth_conf'].reshape(num_frames, model_height, model_width)[-1]
            depth = F.interpolate(depth[None, None], size=(height, width), mode='bilinear', align_corners=False)[0, 0]
            conf = F.interpolate(conf[None, None], size=(height, width), mode='bilinear', align_corners=False)[0, 0]
            torch.cuda.synchronize()
            connection.send({
                'depth': depth.float().cpu().numpy(),
                'conf': conf.float().cpu().numpy(),
                'extrinsics': extrinsics.reshape(num_frames, 3, 4).float().cpu().numpy(),
                'ms': (time.perf_counter() - start) * 1000})
        except Exception:
            connection.send({'error': traceback.format_exc()})
    connection.close()
    listener.close()

if __name__ == '__main__':
    main()
