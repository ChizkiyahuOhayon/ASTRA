"""Measure where a trained AD-GS model puts things, in metres.

The photometric loss is invariant along the direction that trades an object's
canonical size against its range, and AD-GS's only depth prior is applied
scale-and-shift-invariantly (`train.py`, `utils/depth_utils.get_scaled_shifted_depth`).
Static geometry is still pinned by the LiDAR-initialised point cloud; moving
objects are not.  This script asks whether that shows up as metric range error,
using LiDAR returns and Waymo ground-truth boxes as a yardstick the model never
saw during training.

Placement is read from the deformed Gaussian centres, not from the rendered
depth map: alpha-weighted expected depth is pulled toward the near surface by
semi-transparent Gaussians, and that bias grows with range, which would swamp
the effect being measured.  The offset is reported as two numbers -- along the
viewing ray and across it -- because only the transverse part costs photometric
error.  Prediction and ground truth are taken over the *same pixel set*, the
dilated footprint of the LiDAR returns being compared, so road and background
cannot enter one side and not the other.

Reads the .npz files produced by `extract_metric_gt.py`.  Writes one JSON summary
and one per-object CSV.  It does not train and does not modify the checkpoint.
"""
import csv
import json
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import torch
from scipy.ndimage import binary_dilation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import Scene
from scene.env import EnvironmentMap

MOVING_SPEED = 1.0     # m/s; below this a labelled vehicle is parked, not dynamic
MIN_LIDAR = 20         # LiDAR returns needed before a footprint is measured
MIN_GAUSSIANS = 20     # Gaussians needed in the footprint before a range is read
MIN_OPACITY = 0.05     # ignore Gaussians too transparent to place a surface
DILATE = 4             # px; grows the sparse LiDAR footprint into a usable mask
STATIC_WINDOW = 96     # px; grouping for the static background control
ACTOR_TYPES = (1, 2, 4)  # VEHICLE, PEDESTRIAN, CYCLIST


def project(xyz, view):
    """Replicate the rasterizer's projection: camera-space points and pixels."""
    ones = torch.ones((xyz.shape[0], 1), device=xyz.device)
    p_h = torch.cat([xyz, ones], dim=1)
    p_view = (p_h @ view.world_view_transform.to(xyz.device))[:, :3]
    p_hom = p_h @ view.full_proj_transform.to(xyz.device)
    ndc = p_hom[:, :2] / p_hom[:, 3:4].clamp(min=1e-7)
    u = ((ndc[:, 0] + 1.0) * view.image_width - 1.0) * 0.5
    v = ((ndc[:, 1] + 1.0) * view.image_height - 1.0) * 0.5
    return p_view, u, v


def footprint(uv, height, width):
    """Dilated pixel mask covering a set of projected LiDAR returns.

    Returned as a crop plus its origin: dilating the full frame for every
    object in every view is the dominant cost, and the crop is equivalent.
    """
    u = np.clip(np.round(uv[:, 0]).astype(int), 0, width - 1)
    v = np.clip(np.round(uv[:, 1]).astype(int), 0, height - 1)
    y0, y1 = max(v.min() - DILATE, 0), min(v.max() + DILATE + 1, height)
    x0, x1 = max(u.min() - DILATE, 0), min(u.max() + DILATE + 1, width)
    crop = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    crop[v - y0, u - x0] = True
    kernel = np.ones((2 * DILATE + 1, 2 * DILATE + 1), dtype=bool)
    return binary_dilation(crop, kernel), (y0, x0)


def masked_select(p_view, u, v, mask, weight=None):
    """Camera-space points of the Gaussians projecting into a pixel mask."""
    crop, (y0, x0) = mask
    h, w = crop.shape
    ui = torch.round(u).long() - x0
    vi = torch.round(v).long() - y0
    near = (p_view[:, 2] > 0.1) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    if near.sum() < MIN_GAUSSIANS:
        return None
    hit = torch.from_numpy(crop).to(u.device)[vi[near], ui[near]]
    if hit.sum() < MIN_GAUSSIANS:
        return None
    pts = p_view[near][hit].cpu().numpy()
    if weight is None:
        return pts
    return pts, weight[near][hit].cpu().numpy()


def front_surface(pts, slab):
    """The front-most slab of a point set, measured from its own near quantile.

    LiDAR returns only the first hit, but the Gaussians in the same footprint
    include everything occluded behind it, which would drag their centroid back
    along the ray.  Reducing both sides to their own front slab -- neither side
    referring to the other -- removes that asymmetry.
    """
    return pts[pts[:, 2] <= np.percentile(pts[:, 2], 10) + slab]


def ray_extent(pts, weight=None):
    """How far the point set stretches along the viewing ray, p10 to p90.

    A model that smears an object into a streak along the ray and one that
    places a compact object at the wrong range both look wrong to a
    front-surface estimator; the extent separates them.  Weighting by opacity
    asks the sharper question: how far does the part that actually renders
    stretch, as opposed to every Gaussian carrying the object's label.
    """
    z = pts[:, 2]
    if weight is None:
        return float(np.percentile(z, 90) - np.percentile(z, 10))
    order = np.argsort(z)
    z, w = z[order], weight[order]
    c = np.cumsum(w) / max(w.sum(), 1e-9)
    lo, hi = np.searchsorted(c, 0.1), min(np.searchsorted(c, 0.9), len(z) - 1)
    return float(z[hi] - z[lo])


def decompose(pred_pts, true_pts, slab):
    """Split the front-surface centroid offset into along-ray and transverse parts.

    The photometric loss constrains where a surface projects, so a transverse
    offset costs image error; sliding it along the viewing ray does not.  The
    two components therefore separate "the model is inaccurate" from "this
    direction is not identified by the training signal".
    """
    pred_pts, true_pts = front_surface(pred_pts, slab), front_surface(true_pts, slab)
    c_pred, c_true = pred_pts.mean(0), true_pts.mean(0)
    ray = c_true / max(np.linalg.norm(c_true), 1e-6)
    offset = c_pred - c_true
    along = float(offset @ ray)
    return along, float(np.linalg.norm(offset - along * ray)), float(np.linalg.norm(c_true))


def backproject(uvz, K):
    """LiDAR pixel+depth back to the camera-frame points it was projected from."""
    z = uvz[:, 2]
    x = (uvz[:, 0] - K[0, 2]) / K[0, 0] * z
    y = (uvz[:, 1] - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], axis=1)


def object_speeds(records):
    """Peak speed per box id, from ground-truth world centres across frames."""
    tracks = defaultdict(list)
    for gt in records.values():
        for i, bid in enumerate(gt["box_id"]):
            tracks[str(bid)].append((int(gt["fid"]), gt["box_center_world"][i]))
    speeds = {}
    for bid, seq in tracks.items():
        seq.sort()
        steps = [
            np.linalg.norm(seq[i + 1][1] - seq[i][1]) / max(seq[i + 1][0] - seq[i][0], 1) * 10.0
            for i in range(len(seq) - 1)
        ]
        speeds[bid] = float(np.median(steps)) if steps else 0.0
    return speeds


def summarise(rows):
    if not rows:
        return None
    along = np.abs([r["along_m"] for r in rows])
    across = np.abs([r["across_m"] for r in rows])
    rng = np.array([r["gt_range_m"] for r in rows])
    out = {
        "n_observations": len(rows),
        "n_tracks": len({r["key"] for r in rows}),
        "median_along_m": round(float(np.median(along)), 3),
        "median_across_m": round(float(np.median(across)), 3),
        "along_over_across": round(float(np.median(along) / max(np.median(across), 1e-6)), 2),
        "median_range_m": round(float(np.median(rng)), 1),
    }
    for lo, hi in [(0, 20), (20, 40), (40, 1e9)]:
        band = (rng >= lo) & (rng < hi)
        if band.sum() >= 5:
            out["band_{}_{}m".format(lo, int(min(hi, 999)))] = {
                "n": int(band.sum()),
                "median_along_m": round(float(np.median(along[band])), 3),
                "median_across_m": round(float(np.median(across[band])), 3),
            }
    return out


def main():
    parser = ArgumentParser(description="Metric placement probe")
    model = ModelParams(parser, None, sentinel=True)
    pipeline = PipelineParams(parser, None)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--out_prefix", type=str, required=True)
    parser.add_argument("--front_slab", type=float, default=3.0,
                        help="depth of the visible surface each side is reduced to, in metres")
    args = get_combined_args(parser)

    args.data_device = "cuda:0" if args.data_device == "cuda" else args.data_device
    torch.cuda.set_device(args.data_device)
    dataset = model.extract(args)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.order_args)
        env_map = EnvironmentMap(**dataset.env_args)
        scene = Scene(dataset, gaussians, env_map, load_iteration=args.iteration, shuffle=False)

        views = [(v, "train") for v in scene.getTrainCameras()]
        views += [(v, "test") for v in scene.getTestCameras()]

        records = {}
        for view, _ in views:
            key = os.path.splitext(view.image_name)[0]
            path = os.path.join(args.gt_dir, key + ".npz")
            if os.path.exists(path):
                records[key] = np.load(path)
        if not records:
            raise SystemExit("no ground truth found in " + args.gt_dir)
        speeds = object_speeds(records)

        obj_mask = gaussians.get_obj_mask
        rows, static_rows = [], []

        for view, split in views:
            key = os.path.splitext(view.image_name)[0]
            gt = records.get(key)
            if gt is None:
                continue
            deformed = gaussians.get_deformed_pkg(view.time)
            visible = deformed["opacity"].squeeze(-1) > MIN_OPACITY
            p_view, u, v = project(deformed["xyz"], view)

            alpha = deformed["opacity"].squeeze(-1)
            sel_obj = visible & obj_mask
            p_obj, u_obj, v_obj, a_obj = p_view[sel_obj], u[sel_obj], v[sel_obj], alpha[sel_obj]
            p_all, u_all, v_all = p_view[visible], u[visible], v[visible]

            uvz, owner = gt["lidar_uvz"], gt["lidar_box"]
            K = gt["K"]
            types, ids = gt["box_type"], [str(b) for b in gt["box_id"]]

            for bi in range(len(ids)):
                sel = owner == bi
                if sel.sum() < MIN_LIDAR or types[bi] not in ACTOR_TYPES:
                    continue
                mask = footprint(uvz[sel], view.image_height, view.image_width)
                true_pts = backproject(uvz[sel], K)
                pred_all = masked_select(p_all, u_all, v_all, mask)
                sel_pred_obj = masked_select(p_obj, u_obj, v_obj, mask, a_obj)
                if pred_all is None:
                    continue
                along, across, rng = decompose(pred_all, true_pts, args.front_slab)
                speed = speeds.get(ids[bi], 0.0)
                row = {
                    "image": key,
                    "split": split,
                    "key": ids[bi],
                    "type": int(types[bi]),
                    "speed_mps": round(speed, 3),
                    "moving": int(speed >= MOVING_SPEED),
                    "n_lidar": int(sel.sum()),
                    "gt_range_m": round(rng, 3),
                    "along_m": round(along, 3),
                    "across_m": round(across, 3),
                }
                row["true_extent_m"] = round(ray_extent(true_pts), 3)
                row["pred_extent_m"] = round(ray_extent(pred_all), 3)
                if sel_pred_obj is not None:
                    pred_obj, w_obj = sel_pred_obj
                    a_o, c_o, _ = decompose(pred_obj, true_pts, args.front_slab)
                    row["along_obj_m"], row["across_obj_m"] = round(a_o, 3), round(c_o, 3)
                    row["pred_extent_obj_m"] = round(ray_extent(pred_obj), 3)
                    row["pred_extent_obj_w_m"] = round(ray_extent(pred_obj, w_obj), 3)
                    row["n_obj_gauss"] = int(len(pred_obj))
                    row["obj_alpha_med"] = round(float(np.median(w_obj)), 3)
                else:
                    for k in ("along_obj_m", "across_obj_m", "pred_extent_obj_m",
                              "pred_extent_obj_w_m", "n_obj_gauss", "obj_alpha_med"):
                        row[k] = ""
                rows.append(row)

            # Static control: identical estimator, on LiDAR with no labelled actor.
            free = uvz[owner < 0]
            for y0 in range(0, view.image_height - STATIC_WINDOW, STATIC_WINDOW):
                for x0 in range(0, view.image_width - STATIC_WINDOW, STATIC_WINDOW):
                    inside = (
                        (free[:, 0] >= x0) & (free[:, 0] < x0 + STATIC_WINDOW)
                        & (free[:, 1] >= y0) & (free[:, 1] < y0 + STATIC_WINDOW)
                    )
                    if inside.sum() < MIN_LIDAR:
                        continue
                    mask = footprint(free[inside], view.image_height, view.image_width)
                    pred = masked_select(p_all, u_all, v_all, mask)
                    if pred is None:
                        continue
                    along, across, rng = decompose(
                        pred, backproject(free[inside], K), args.front_slab
                    )
                    static_rows.append(
                        {
                            "key": "{}:{},{}".format(key, x0, y0),
                            "n_lidar": int(inside.sum()),
                            "gt_range_m": round(rng, 3),
                            "along_m": round(along, 3),
                            "across_m": round(across, 3),
                        }
                    )

    moving = [r for r in rows if r["moving"]]
    parked = [r for r in rows if not r["moving"]]
    summary = {
        "model_path": dataset.model_path,
        "iteration": scene.loaded_iter,
        "images": len(records),
        "estimator": "centroid offset of Gaussian centres vs LiDAR over a shared dilated footprint, split along/across the viewing ray",
        "moving_speed_threshold_mps": MOVING_SPEED,
        "front_slab_m": args.front_slab,
        "static_background": summarise(static_rows),
        "parked_actors": summarise(parked),
        "moving_actors": summarise(moving),
    }

    with open(args.out_prefix + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    for name, table in (("", rows), ("-static", static_rows)):
        if not table:
            continue
        with open(args.out_prefix + name + ".csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
