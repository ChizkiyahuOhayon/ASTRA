"""How much parallax did each Gaussian actually get?

The photometric loss pins a Gaussian's transverse position through its
projection, but its radial position is only ever constrained by multi-view
baseline -- and that baseline is wildly uneven across one driving log.  Static
background accumulates the whole forward drive; a vehicle travelling with the
ego is seen from nearly a single direction for the entire sequence and its depth
is close to free.

This measures the spread directly, with no new supervision.  For every training
view a Gaussian is visible in, take the unit vector from the Gaussian to that
camera.  The resultant length of those vectors,

    R = |sum_c w_c d_c| / sum_c w_c,

is 1 when every observation came from one direction and falls toward 0 as they
spread out.  R is therefore "how little evidence this Gaussian's depth has".

The prediction under test: R is bimodal, with the dynamic branch and the far
background piled up near 1.  If it is unimodal, radial evidence is uniform and
there is nothing here to exploit.
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

MIN_OPACITY = 0.05


def summarise(name, r, rng):
    if not len(r):
        print("  %-22s (empty)" % name)
        return
    q = np.percentile(r, [10, 50, 90])
    print("  %-22s n=%-8d R p10/p50/p90 = %.3f / %.3f / %.3f   frac(R>0.99)=%.1f%%  median range=%.1f m"
          % (name, len(r), q[0], q[1], q[2], 100 * (r > 0.99).mean(), np.median(rng)))


def main():
    parser = ArgumentParser()
    model = ModelParams(parser, None, sentinel=True)
    pipeline = PipelineParams(parser, None)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--out", type=str, default=None)
    args = get_combined_args(parser)
    torch.cuda.set_device(args.data_device)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.order_args)
        env_map = EnvironmentMap(**dataset.env_args)
        scene = Scene(dataset, gaussians, env_map, load_iteration=args.iteration, shuffle=False)
        obj_mask = gaussians.get_obj_mask
        n = obj_mask.shape[0]

        # resultant vector of the directions each Gaussian was seen from
        acc = torch.zeros((n, 3), device="cuda")
        wsum = torch.zeros(n, device="cuda")
        range_acc = torch.zeros(n, device="cuda")

        views = scene.getTrainCameras()
        for view in views:
            pkg = render(view, gaussians, env_map, pipe)
            vis = pkg["visibility_filter"]
            if vis.sum() == 0:
                continue
            xyz = gaussians.get_deformed_pkg(view.time)["xyz"]
            to_cam = view.camera_center.to(xyz.device) - xyz
            dist = to_cam.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            # projected radius weights the view by how much the Gaussian could
            # actually be constrained from it; an unseen Gaussian contributes 0.
            w = vis.float() * pkg["radii"].float()
            acc += w[:, None] * (to_cam / dist)
            wsum += w
            range_acc += w * dist.squeeze(-1)

        seen = wsum > 0
        r = torch.zeros(n, device="cuda")
        r[seen] = acc[seen].norm(dim=-1) / wsum[seen]
        mean_range = torch.zeros(n, device="cuda")
        mean_range[seen] = range_acc[seen] / wsum[seen]

        alpha = gaussians.get_opacity.squeeze(-1)
        keep = seen & (alpha > MIN_OPACITY)
        r_np = r[keep].cpu().numpy()
        rng_np = mean_range[keep].cpu().numpy()
        is_obj = obj_mask[keep].cpu().numpy()

        print("views: %d   Gaussians: %d   seen & opaque: %d" % (len(views), n, keep.sum()))
        print("R = 1 means every observation came from one direction (no radial evidence)")
        summarise("all", r_np, rng_np)
        summarise("static branch", r_np[~is_obj], rng_np[~is_obj])
        summarise("dynamic branch", r_np[is_obj], rng_np[is_obj])
        for lo, hi in ((0, 20), (20, 40), (40, 80), (80, 1e9)):
            m = (rng_np >= lo) & (rng_np < hi)
            summarise("static %g-%g m" % (lo, hi), r_np[~is_obj & m], rng_np[~is_obj & m])

        hist, edges = np.histogram(r_np, bins=20, range=(0.0, 1.0))
        print("\n  R histogram (all):")
        for h, e in zip(hist, edges):
            print("   %.2f %s %d" % (e, "#" * int(60 * h / max(hist.max(), 1)), h))

        if args.out:
            np.savez_compressed(args.out, r=r_np, mean_range=rng_np, is_obj=is_obj)
            print("\nwrote", args.out)


if __name__ == "__main__":
    main()
