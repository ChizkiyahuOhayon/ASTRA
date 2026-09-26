"""Dump one actor's Gaussians, its ground-truth box and its LiDAR returns.

Everything is expressed in the camera frame of the requested view, so a
bird's-eye plot of x against z shows directly how far the model's
representation of a vehicle stretches along the viewing ray compared with the
box it should occupy.  Plotting is left to the caller.
"""
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from scipy.ndimage import binary_dilation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import Scene
from scene.env import EnvironmentMap

DILATE = 4


def main():
    parser = ArgumentParser(description="Dump an actor's Gaussian cloud")
    model = ModelParams(parser, None, sentinel=True)
    PipelineParams(parser, None)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--frame", type=str, required=True)
    parser.add_argument("--track", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = get_combined_args(parser)

    args.data_device = "cuda:0" if args.data_device == "cuda" else args.data_device
    torch.cuda.set_device(args.data_device)
    dataset = model.extract(args)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.order_args)
        env_map = EnvironmentMap(**dataset.env_args)
        scene = Scene(dataset, gaussians, env_map, load_iteration=args.iteration, shuffle=False)
        views = {os.path.splitext(v.image_name)[0]: v
                 for v in scene.getTrainCameras() + scene.getTestCameras()}
        view = views[args.frame]
        gt = np.load(os.path.join(args.gt_dir, args.frame + ".npz"))
        bi = [str(b) for b in gt["box_id"]].index(args.track)

        deformed = gaussians.get_deformed_pkg(view.time)
        alpha = deformed["opacity"].squeeze(-1)
        keep = (alpha > 0.05) & gaussians.get_obj_mask
        xyz = deformed["xyz"][keep]
        p_h = torch.cat([xyz, torch.ones((xyz.shape[0], 1), device=xyz.device)], dim=1)
        p_view = (p_h @ view.world_view_transform.cuda())[:, :3]
        p_hom = p_h @ view.full_proj_transform.cuda()
        ndc = p_hom[:, :2] / p_hom[:, 3:4].clamp(min=1e-7)
        u = ((ndc[:, 0] + 1.0) * view.image_width - 1.0) * 0.5
        v = ((ndc[:, 1] + 1.0) * view.image_height - 1.0) * 0.5

        uvz = gt["lidar_uvz"][gt["lidar_box"] == bi]
        mask = np.zeros((view.image_height, view.image_width), dtype=bool)
        mask[np.clip(np.round(uvz[:, 1]).astype(int), 0, view.image_height - 1),
             np.clip(np.round(uvz[:, 0]).astype(int), 0, view.image_width - 1)] = True
        mask = binary_dilation(mask, np.ones((2 * DILATE + 1,) * 2, dtype=bool))

        ui = torch.clamp(torch.round(u), 0, view.image_width - 1).long()
        vi = torch.clamp(torch.round(v), 0, view.image_height - 1).long()
        inside = torch.from_numpy(mask).cuda()[vi, ui] & (p_view[:, 2] > 0.1)

        K = gt["K"]
        lidar_cam = np.stack(
            [(uvz[:, 0] - K[0, 2]) / K[0, 0] * uvz[:, 2],
             (uvz[:, 1] - K[1, 2]) / K[1, 1] * uvz[:, 2],
             uvz[:, 2]], axis=1)

    np.savez(
        args.out,
        gaussians_cam=p_view[inside].cpu().numpy(),
        gaussian_alpha=alpha[keep][inside].cpu().numpy(),
        lidar_cam=lidar_cam,
        box_center_cam=gt["box_center_cam"][bi],
        box_dims=gt["box_dims"][bi],
        frame=args.frame,
        track=args.track,
        model_path=dataset.model_path,
    )
    print("wrote {}: {} object Gaussians, {} LiDAR returns".format(
        args.out, int(inside.sum()), len(lidar_cam)))


if __name__ == "__main__":
    main()
