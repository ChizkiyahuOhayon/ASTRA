"""Dynamic-region PSNR (DPSNR) of a rendered Waymo test split.

PSNR is computed per test image over the pixels of its ground-truth moving-object mask
(scripts/eval/waymo_dynamic_masks.py), images without a moving object are skipped, and
the per-image values are averaged -- the protocol behind the DPSNR column of AD-GS,
StreetGS and IDSplat. Needs only the renders written by `render.py`, no GPU.

    python scripts/eval/dynamic_psnr.py -m output/waymo/scene026 -s data/waymo/scene026 \
        --masks data/waymo_dynamic_masks --iteration 30000

Writes <model>/results-dpsnr.json.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image


def test_frames(scene_dir, count):
    """nvs-75 holds out every 4th frame from index 4 (scripts/waymo/waymo.py); renders are in frame order."""
    frames = len([f for f in os.listdir(os.path.join(scene_dir, 'image')) if f.endswith(('.jpg', '.png'))])
    indices = list(range(4, frames, 4))
    if len(indices) != count:
        raise SystemExit('split mismatch: %d frames give %d test views, found %d renders' % (frames, len(indices), count))
    return indices


def resample(mask, height, width):
    """Nearest-index resize, the same indexing the data loader uses for masks."""
    if mask.shape == (height, width):
        return mask
    rows = np.linspace(0, mask.shape[0] - 1, height).astype(np.int32)
    cols = np.linspace(0, mask.shape[1] - 1, width).astype(np.int32)
    return mask[rows[:, None], cols]


def dynamic_psnr(model_dir, scene_dir, mask_root, iteration):
    root = os.path.join(model_dir, 'test', 'ours_%d' % iteration)
    names = sorted(os.listdir(os.path.join(root, 'renders')))
    mask_dir = os.path.join(mask_root, os.path.basename(os.path.normpath(scene_dir)), 'dynamic_mask')
    scores = []
    for name, frame in zip(names, test_frames(scene_dir, len(names))):
        render = np.asarray(Image.open(os.path.join(root, 'renders', name)).convert('RGB'), np.float64) / 255
        truth = np.asarray(Image.open(os.path.join(root, 'gt', name)).convert('RGB'), np.float64) / 255
        mask = resample(np.asarray(Image.open(os.path.join(mask_dir, '%06d.png' % frame))) > 0, *render.shape[:2])
        if mask.any():
            scores.append(-10 * np.log10(((render - truth) ** 2)[mask].mean()))
    return {'DPSNR': float(np.mean(scores)), 'images_scored': len(scores)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('-m', '--model_path', required=True)
    parser.add_argument('-s', '--source_path', required=True)
    parser.add_argument('--masks', default='data/waymo_dynamic_masks')
    parser.add_argument('--iteration', type=int, default=30000)
    args = parser.parse_args()
    result = dynamic_psnr(args.model_path, args.source_path, args.masks, args.iteration)
    print(json.dumps(result))
    with open(os.path.join(args.model_path, 'results-dpsnr.json'), 'w') as handle:
        json.dump(result, handle, indent=1)
