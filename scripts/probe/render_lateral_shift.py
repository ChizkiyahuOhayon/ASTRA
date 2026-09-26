"""Render a trained model from laterally shifted viewpoints.

`probe_metric_placement.py` measures moving vehicles stretched roughly thirty-fold
along the viewing ray while their projection stays correct.  Along the recorded
trajectory that costs almost nothing, which is why the interpolated benchmark
cannot see it.  Stepping the camera sideways is the cheapest way to make the
same defect visible, and it is what closed-loop simulation actually asks for.

Shifting right by d metres in camera coordinates is exactly T -> T - [d, 0, 0],
because 3DGS stores the world-to-camera translation and the camera centre is
-R @ T.  The rotation is untouched, so this is a pure lane change.

Writes one PNG per (frame, shift): the render, and the object-branch coverage
that shows which pixels the dynamic Gaussians claim.
"""
import copy
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from scene.env import EnvironmentMap
from utils.graphics_utils import getWorld2View2


def shift_camera(view, metres):
    """A copy of the camera translated sideways by `metres`, same orientation."""
    moved = copy.deepcopy(view)
    moved.T = view.T - np.array([metres, 0.0, 0.0], dtype=view.T.dtype)
    moved.world_view_transform = torch.tensor(
        getWorld2View2(moved.R, moved.T, moved.trans, moved.scale)
    ).transpose(0, 1).cuda()
    moved.projection_matrix = moved.projection_matrix.cuda()
    moved.full_proj_transform = moved.world_view_transform.unsqueeze(0).bmm(
        moved.projection_matrix.unsqueeze(0)
    ).squeeze(0)
    moved.camera_center = moved.world_view_transform.inverse()[3, :3]
    return moved


def main():
    parser = ArgumentParser(description="Lateral-shift renders")
    model = ModelParams(parser, None, sentinel=True)
    pipeline = PipelineParams(parser, None)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--frames", type=str, nargs="+", required=True,
                        help="image names, e.g. 000053")
    parser.add_argument("--shifts", type=float, nargs="+", default=[0.0, 1.0, 2.0, 3.0],
                        help="lateral offsets in metres")
    parser.add_argument("--out_dir", type=str, required=True)
    args = get_combined_args(parser)

    args.data_device = "cuda:0" if args.data_device == "cuda" else args.data_device
    torch.cuda.set_device(args.data_device)
    dataset, pipe = model.extract(args), pipeline.extract(args)
    os.makedirs(args.out_dir, exist_ok=True)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.order_args)
        env_map = EnvironmentMap(**dataset.env_args)
        scene = Scene(dataset, gaussians, env_map, load_iteration=args.iteration, shuffle=False)
        gaussians.mip_filter = bool(getattr(dataset, "mip_filter", False))
        gaussians.compute_3D_filter(scene.getTrainCameras())

        views = {os.path.splitext(v.image_name)[0]: v
                 for v in scene.getTrainCameras() + scene.getTestCameras()}
        for name in args.frames:
            if name not in views:
                print("no such frame:", name)
                continue
            torchvision.utils.save_image(
                views[name].original_image.clamp(0, 1),
                os.path.join(args.out_dir, "{}_gt.png".format(name)),
            )
            for d in args.shifts:
                pkg = render(shift_camera(views[name], d), gaussians, env_map, pipe,
                             scene.photo, render_objmask=True)
                tag = "{}_shift{:+.1f}m".format(name, d)
                torchvision.utils.save_image(
                    pkg["render"].clamp(0, 1),
                    os.path.join(args.out_dir, tag + ".png"),
                )
                torchvision.utils.save_image(
                    pkg["img_semantic"][0].clamp(0, 1),
                    os.path.join(args.out_dir, tag + "_objmask.png"),
                )
                print(tag, "objmask coverage %.4f" % pkg["img_semantic"][0].mean().item())


if __name__ == "__main__":
    main()
