"""How much of a trained model sits in space the LiDAR says is empty?

`get_free_space_loss` is a no-op if no primitive ever lands in carved space, and a
no-op module produces a null result that looks like "no effect" rather than "not
wired".  This reports the fraction directly, per training view.
"""
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import Scene
from scene.env import EnvironmentMap


def main():
    parser = ArgumentParser()
    model = ModelParams(parser, None, sentinel=True)
    PipelineParams(parser, None)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--cells", type=int, nargs="+", default=[1, 8, 16, 32])
    args = get_combined_args(parser)
    torch.cuda.set_device(args.data_device)
    dataset = model.extract(args)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.order_args)
        env_map = EnvironmentMap(**dataset.env_args)
        scene = Scene(dataset, gaussians, env_map, load_iteration=args.iteration, shuffle=False)

        fracs, weighted = {}, {}
        for view in scene.getTrainCameras():
            path = os.path.join(args.gt_dir, "%06d.npz" % int(view.fid))
            if not os.path.exists(path):
                continue
            uvz = np.load(path)["lidar_uvz"]
            uv = torch.from_numpy(uvz[:, :2]).round().long().cuda()
            uv[:, 0].clamp_(0, view.image_width - 1)
            uv[:, 1].clamp_(0, view.image_height - 1)
            z = torch.from_numpy(uvz[:, 2]).cuda()

            xyz = gaussians.get_deformed_pkg(view.time)["xyz"]
            op = gaussians.get_opacity.squeeze(-1)
            w2v = view.world_view_transform.to(xyz.device)
            proj = view.full_proj_transform.to(xyz.device)
            p = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
            depth = (p @ w2v)[:, 2]
            clip = p @ proj
            ndc = clip[:, :2] / clip[:, 3:4].clamp(min=1e-6)
            u = ((ndc[:, 0] + 1.0) * view.image_width - 1.0) * 0.5
            v = ((ndc[:, 1] + 1.0) * view.image_height - 1.0) * 0.5
            inside = (depth > 1.0) & (u >= 0) & (v >= 0) \
                   & (u <= view.image_width - 1) & (v <= view.image_height - 1)
            if not inside.any():
                continue
            for c in args.cells:
                h = (view.image_height + c - 1) // c
                w = (view.image_width + c - 1) // c
                surface = torch.full((h * w,), float("inf"), device=xyz.device)
                surface.scatter_reduce_(0, (uv[:, 1] // c) * w + (uv[:, 0] // c), z, reduce="amin")
                hit = surface[(v[inside].long() // c) * w + (u[inside].long() // c)]
                carved = torch.isfinite(hit) & (depth[inside] < hit - args.margin)
                fracs.setdefault(c, []).append(carved.float().mean().item())
                weighted.setdefault(c, []).append(
                    op[inside][carved].sum().item() / op[inside].sum().clamp(min=1e-6).item())

        print("views: %d   margin %.1f m" % (len(next(iter(fracs.values()))), args.margin))
        print("%-8s %14s %14s" % ("cell(px)", "carved frac", "opacity share"))
        for c in args.cells:
            f, w = np.array(fracs[c]), np.array(weighted[c])
            print("%-8d %13.2f%% %13.2f%%" % (c, 100 * f.mean(), 100 * w.mean()))


if __name__ == "__main__":
    main()
