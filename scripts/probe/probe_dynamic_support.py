"""Ask how much of the dynamic branch actually sits on a moving object.

Image-space association cannot answer this: a footprint drawn from LiDAR first
returns also collects everything occluded behind it, including other vehicles.
This probe avoids association entirely -- for every Gaussian in the dynamic
branch it measures the distance to the nearest ground-truth actor box at that
timestamp, in 3D, and reports how much of the branch's rendered weight lies on
a real object and how much is spread through empty space.
"""
import csv
import json
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

MIN_OPACITY = 0.05
MARGIN = 1.0  # m; a Gaussian this close to a box surface counts as on the object


def box_frames(corners):
    """Centre, orthonormal axes and half-extents of boxes given as 8 corners each."""
    centre = corners.mean(axis=1)
    length = corners[:, 0] - corners[:, 3]
    width = corners[:, 0] - corners[:, 1]
    height = corners[:, 0] - corners[:, 4]
    half = np.stack([np.linalg.norm(a, axis=-1) for a in (length, width, height)], axis=1) / 2
    axes = np.stack([a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-6)
                     for a in (length, width, height)], axis=1)
    return centre, axes, half


def distance_to_boxes(points, centre, axes, half):
    """Shortest distance from each point to each oriented box, 0 inside."""
    delta = points[:, None, :] - centre[None, :, :]                    # N,B,3
    local = np.abs(np.einsum("nbi,bai->nba", delta, axes))             # N,B,3
    outside = np.clip(local - half[None, :, :], 0.0, None)
    return np.linalg.norm(outside, axis=-1)                            # N,B


def main():
    parser = ArgumentParser(description="Dynamic-branch support probe")
    model = ModelParams(parser, None, sentinel=True)
    PipelineParams(parser, None)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--out_prefix", type=str, required=True)
    args = get_combined_args(parser)

    args.data_device = "cuda:0" if args.data_device == "cuda" else args.data_device
    torch.cuda.set_device(args.data_device)
    dataset = model.extract(args)

    rows = []
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.order_args)
        env_map = EnvironmentMap(**dataset.env_args)
        scene = Scene(dataset, gaussians, env_map, load_iteration=args.iteration, shuffle=False)
        obj_mask = gaussians.get_obj_mask

        for view in scene.getTrainCameras() + scene.getTestCameras():
            key = os.path.splitext(view.image_name)[0]
            path = os.path.join(args.gt_dir, key + ".npz")
            if not os.path.exists(path):
                continue
            gt = np.load(path)
            corners = gt["box_corners_world"]
            if not len(corners):
                continue

            deformed = gaussians.get_deformed_pkg(view.time)
            alpha = deformed["opacity"].squeeze(-1)
            keep = (alpha > MIN_OPACITY) & obj_mask
            if keep.sum() == 0:
                continue
            xyz = deformed["xyz"][keep].cpu().numpy()
            weight = alpha[keep].cpu().numpy()

            centre, axes, half = box_frames(corners)
            nearest = distance_to_boxes(xyz, centre, axes, half).min(axis=1)
            on_object = nearest <= MARGIN
            rows.append(
                {
                    "image": key,
                    "n_boxes": len(corners),
                    "n_dynamic": int(keep.sum()),
                    "frac_on_object": round(float(on_object.mean()), 4),
                    "frac_weight_on_object": round(
                        float(weight[on_object].sum() / max(weight.sum(), 1e-9)), 4),
                    "median_dist_m": round(float(np.median(nearest)), 3),
                    "p90_dist_m": round(float(np.percentile(nearest, 90)), 3),
                }
            )

    summary = {
        "model_path": dataset.model_path,
        "iteration": scene.loaded_iter,
        "frames": len(rows),
        "margin_m": MARGIN,
        "median_frac_on_object": round(float(np.median([r["frac_on_object"] for r in rows])), 4),
        "median_frac_weight_on_object": round(
            float(np.median([r["frac_weight_on_object"] for r in rows])), 4),
        "median_dist_m": round(float(np.median([r["median_dist_m"] for r in rows])), 3),
        "median_p90_dist_m": round(float(np.median([r["p90_dist_m"] for r in rows])), 3),
        "median_dynamic_gaussians": int(np.median([r["n_dynamic"] for r in rows])),
    }
    with open(args.out_prefix + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(args.out_prefix + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
