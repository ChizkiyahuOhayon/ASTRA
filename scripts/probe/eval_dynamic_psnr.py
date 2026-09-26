"""Dynamic PSNR: reconstruction quality restricted to the moving actors.

The comparison table this project targets reports a DPSNR column, and AD-GS's
own `render.py` computes only whole-frame metrics, so the number cannot be read
off an existing run.  Definition used here: PSNR over the union of the projected
ground-truth boxes of actors moving faster than MOVING_STEP, which keeps parked
cars -- static content that the background branch already handles -- out of a
column that is meant to measure dynamics.

Boxes come from `extract_metric_gt.py`; speed is the finite difference of each
track's world-frame centre between consecutive frames it appears in, in metres per
frame rather than metres per second.
"""
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from scene.env import EnvironmentMap

# Waymo records at 10 Hz, so a track's per-frame centre displacement is a tenth of
# its speed: 0.1 m/frame is 1 m/s, below which a "dynamic" actor is a parked car.
# Thresholding the raw displacement at 1.0 asks for 36 km/h and finds nobody.
MOVING_STEP = 0.1    # metres per frame
ACTOR_TYPES = (1, 2, 4)   # vehicle, pedestrian, cyclist


def track_steps(gt_dir):
    """Mean per-frame world-frame displacement of every track, keyed by box id."""
    files = sorted(f for f in os.listdir(gt_dir) if f.endswith(".npz"))
    seen = {}
    for f in files:
        d = np.load(os.path.join(gt_dir, f))
        for bid, c in zip(d["box_id"], d["box_center_world"]):
            seen.setdefault(str(bid), []).append(c)
    return {k: float(np.linalg.norm(np.diff(np.stack(v), axis=0), axis=1).mean())
            if len(v) > 1 else 0.0 for k, v in seen.items()}


def moving_mask(gt, speeds, height, width):
    """Union of the 2D boxes of the moving actors in this frame."""
    m = torch.zeros((height, width), dtype=torch.bool)
    for bid, btype, uv in zip(gt["box_id"], gt["box_type"], gt["box_uv"]):
        if int(btype) not in ACTOR_TYPES or np.isnan(uv).any():
            continue
        if speeds.get(str(bid), 0.0) <= MOVING_STEP:
            continue
        x0, y0, x1, y1 = uv
        x0, x1 = np.clip([x0, x1], 0, width - 1).astype(int)
        y0, y1 = np.clip([y0, y1], 0, height - 1).astype(int)
        m[y0:y1 + 1, x0:x1 + 1] = True
    return m


def main():
    parser = ArgumentParser()
    model = ModelParams(parser, None, sentinel=True)
    pipeline = PipelineParams(parser, None)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--gt_dir", required=True)
    args = get_combined_args(parser)
    torch.cuda.set_device(args.data_device)
    dataset, pipe = model.extract(args), pipeline.extract(args)
    speeds = track_steps(args.gt_dir)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.order_args)
        env_map = EnvironmentMap(**dataset.env_args)
        scene = Scene(dataset, gaussians, env_map, load_iteration=args.iteration, shuffle=False)

        for split, views in (("test", scene.getTestCameras()), ("train", scene.getTrainCameras())):
            se, n, frames = 0.0, 0, 0
            for view in views:
                path = os.path.join(args.gt_dir, "%06d.npz" % int(view.fid))
                if not os.path.exists(path):
                    continue
                out = render(view, gaussians, env_map, pipe)["render"].clamp(0.0, 1.0)
                gt_img = view.original_image.cuda()
                m = moving_mask(np.load(path), speeds, gt_img.shape[1], gt_img.shape[2]).cuda()
                if not m.any():
                    continue
                se += (((out - gt_img) ** 2) * m).sum().item()
                n += 3 * int(m.sum())
                frames += 1
            if n:
                print("%-6s DPSNR %.4f   frames with a moving actor %d/%d   pixels %.2f%%"
                      % (split, -10 * np.log10(se / n), frames, len(views),
                         100 * n / (3 * len(views) * gt_img.shape[1] * gt_img.shape[2])))
            else:
                print("%-6s no moving actors found" % split)


if __name__ == "__main__":
    main()
