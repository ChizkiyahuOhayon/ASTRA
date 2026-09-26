"""Are the Gaussians stretched along the direction they were trained from?

Rendering the scene006 model 45 degrees off its training axis produces streaks
that cross the whole frame (experiment_logs/raw/2026-09-02-xcam).  A Gaussian
elongated along the viewing ray is invisible on-trajectory -- it projects to a
point -- and becomes a streak as soon as the camera turns.  This checks that
reading directly on the fitted covariance rather than inferring it from pixels.

For each Gaussian: its scales, its major axis in world coordinates, and the
angle between that axis and the mean direction to the cameras that saw it.
If the streaks come from ray-alignment, elongated Gaussians concentrate near
0 or 180 degrees; if the angle is uniform, the elongation is not view-induced
and a ray-aligned penalty would be treating the wrong cause.
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
from utils.general_utils import build_rotation

MIN_OPACITY = 0.05


def main():
    parser = ArgumentParser()
    model = ModelParams(parser, None, sentinel=True)
    pipeline = PipelineParams(parser, None)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--out", type=str, default=None)
    args = get_combined_args(parser)
    out = getattr(args, "out", None)
    torch.cuda.set_device(args.data_device)
    dataset, pipe = model.extract(args), pipeline.extract(args)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.order_args)
        env_map = EnvironmentMap(**dataset.env_args)
        scene = Scene(dataset, gaussians, env_map, load_iteration=args.iteration, shuffle=False)
        views = scene.getTrainCameras()

        n = gaussians.get_opacity.shape[0]
        acc = torch.zeros((n, 3), device="cuda")
        wsum = torch.zeros(n, device="cuda")
        for view in views:
            pkg = render(view, gaussians, env_map, pipe)
            w = (pkg["radii"] > 0).float()
            xyz = gaussians.get_deformed_pkg(view.time)["xyz"]
            d = view.camera_center.to(xyz.device) - xyz
            acc += w[:, None] * d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            wsum += w
        seen = wsum > 0
        mean_dir = torch.zeros_like(acc)
        mean_dir[seen] = acc[seen] / acc[seen].norm(dim=-1, keepdim=True).clamp(min=1e-6)

        scales = gaussians.get_scaling                      # N,3
        rot = build_rotation(gaussians.get_rotation)        # N,3,3, columns are axes
        order = scales.argsort(dim=1, descending=True)
        s = torch.gather(scales, 1, order)
        major = torch.gather(rot, 2, order[:, None, :1].expand(-1, 3, -1)).squeeze(-1)

        keep = seen & (gaussians.get_opacity.squeeze(-1) > MIN_OPACITY)
        cos = (major[keep] * mean_dir[keep]).sum(-1).abs().clamp(0, 1)
        ang = torch.rad2deg(torch.acos(cos)).cpu().numpy()
        sk = s[keep]
        elong = (sk[:, 0] / sk[:, 1].clamp(min=1e-8)).cpu().numpy()
        flat = (sk[:, 0] / sk[:, 2].clamp(min=1e-8)).cpu().numpy()

        print("Gaussians kept: %d of %d" % (keep.sum(), n))
        print("elongation s0/s1  p50 %.2f  p90 %.2f  p99 %.2f  max %.1f"
              % (*np.percentile(elong, [50, 90, 99]), elong.max()))
        print("flatness   s0/s2  p50 %.2f  p90 %.2f  p99 %.2f  max %.1f"
              % (*np.percentile(flat, [50, 90, 99]), flat.max()))
        print("\nangle between major axis and mean view direction (0 = along the ray)")
        print("  %-22s %8s %8s %8s" % ("group", "n", "median", "frac<30deg"))
        print("  %-22s %8d %8.1f %8.1f%%" % ("all", len(ang), np.median(ang), 100 * (ang < 30).mean()))
        for lo in (2.0, 5.0, 10.0, 20.0):
            m = elong > lo
            if m.sum() < 50:
                continue
            print("  %-22s %8d %8.1f %8.1f%%"
                  % ("elongation > %gx" % lo, m.sum(), np.median(ang[m]), 100 * (ang[m] < 30).mean()))
        # a uniformly oriented axis on the sphere has median angle 60 deg
        print("\n  uniform-orientation reference: median 60.0 deg, frac<30deg 13.4%")
        if out:
            np.savez_compressed(out, angle=ang, elong=elong, flat=flat)
            print("wrote", out)


if __name__ == "__main__":
    main()
