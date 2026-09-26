"""Evaluate a FRONT-trained model on Waymo's real side cameras.

Every synthetic lateral shift we have rendered so far had no ground truth, so it
could show that the model changes but never that it is wrong.  Waymo records
FRONT_LEFT and FRONT_RIGHT at the same instants as FRONT, from a genuinely
different viewpoint, and AD-GS's preprocessing throws them away
(`--select_camera 0`).  Re-extracting them with the same tfrecord and the same
first_frame puts them in the identical world frame, so a model trained on FRONT
alone can be rendered at their poses and compared against real photographs.

That makes this an extrapolation test with ground truth, at zero training cost.
The side cameras are held out completely: they are never read during training,
only here.
"""
import copy
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from scene.env import EnvironmentMap
from utils.graphics_utils import getWorld2View2, focal2fov
from utils.image_utils import psnr
from utils.loss_utils import ssim


def retarget(view, R, T, fovx, fovy):
    """The same camera object aimed from a different pose, transforms rebuilt."""
    moved = copy.deepcopy(view)
    moved.R, moved.T, moved.FoVx, moved.FoVy = R, T, fovx, fovy
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
    parser = ArgumentParser()
    model = ModelParams(parser, None, sentinel=True)
    pipeline = PipelineParams(parser, None)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--xcam", required=True, help="dir holding the 3-camera extraction")
    parser.add_argument("--save_dir", default=None)
    # Coverage shrinks when a method fabricates less, so "PSNR over covered pixels"
    # compares two different pixel sets and is not a like-for-like number.  Freezing
    # one model's masks and scoring every method through them fixes that.
    parser.add_argument("--save_masks", default=None, help="write this run's coverage masks here")
    parser.add_argument("--mask_dir", default=None, help="score through masks written earlier")
    parser.add_argument("--save_every", type=int, default=10)
    args = get_combined_args(parser)
    save_dir = getattr(args, "save_dir", None)   # absent unless passed
    save_masks = getattr(args, "save_masks", None)
    mask_dir = getattr(args, "mask_dir", None)
    save_every = getattr(args, "save_every", 10)
    torch.cuda.set_device(args.data_device)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    meta = np.load(os.path.join(args.xcam, "cameras.npz"))
    # Waymo stores intrinsics as the flat vector [fx, fy, cx, cy, k1, k2, p1, p2, k3];
    # it is not a 3x3, and reading it as one silently substitutes a distortion
    # coefficient for fy.  The reader derives the field of view as focal2fov(f, 2c),
    # which assumes a centred principal point -- matched here so the evaluation sees
    # exactly the camera model the training saw.
    R, T, K = meta["R"], meta["T"], meta["K"]
    cam_ids, stamps = meta["camera_ids"], meta["time_stamps"]
    names = sorted(os.listdir(os.path.join(args.xcam, "image")))
    assert len(names) == len(cam_ids), "%d images vs %d poses" % (len(names), len(cam_ids))

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.order_args)
        env_map = EnvironmentMap(**dataset.env_args)
        scene = Scene(dataset, gaussians, env_map, load_iteration=args.iteration, shuffle=False)
        front = {v.fid: v for v in scene.getTrainCameras() + scene.getTestCameras()}

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        if save_masks:
            os.makedirs(save_masks, exist_ok=True)

        stats = {}
        for i, name in enumerate(names):
            cid, fid = int(cam_ids[i]), int(stamps[i])
            base = front.get(fid)
            if base is None:
                continue
            gt = torch.from_numpy(
                np.asarray(Image.open(os.path.join(args.xcam, "image", name)).convert("RGB"))
            ).permute(2, 0, 1).float().cuda() / 255.0
            h, w = gt.shape[1:]
            view = retarget(base, R[i], T[i],
                            focal2fov(K[i][0], K[i][2] * 2), focal2fov(K[i][1], K[i][3] * 2))
            pkg = render(view, gaussians, env_map, pipe)
            out = pkg["render"].clamp(0.0, 1.0)
            # A 45 deg yaw puts much of the side image outside anything FRONT ever
            # saw, and a model cannot be blamed for geometry that was never observed.
            # Accumulated opacity separates the two: pixels no Gaussian reaches are
            # unobserved, the rest is where the reconstruction can actually be wrong.
            cover = pkg["img_opacity"].squeeze(0) if "img_opacity" in pkg else None
            if out.shape[1:] != gt.shape[1:]:
                out = torch.nn.functional.interpolate(
                    out[None], size=gt.shape[1:], mode="bilinear", align_corners=False)[0]
            # Waymo exposes each camera independently, so a side view can differ from
            # FRONT by a systematic gain and offset with no geometric error at all.
            # Fitting the best per-channel affine on the covered pixels and reporting
            # the residual separates that from geometry; whatever survives is not
            # photometric.  AD-GS ships a photometric spline for the same reason.
            p, s = psnr(out, gt).mean().item(), ssim(out, gt).item()
            if cover is not None:
                m = (cover > 0.5)
                frac = m.float().mean().item()
                pc = (-10 * torch.log10(((out - gt) ** 2 * m).sum() / (3 * m.sum()).clamp(min=1))
                      ).item() if m.sum() > 0 else float("nan")
            else:
                frac, pc = float("nan"), float("nan")
            key = "cam%d_%s.npy" % (cid, os.path.splitext(name)[0])
            if save_masks and cover is not None:
                np.save(os.path.join(save_masks, key), (cover > 0.5).cpu().numpy())
            if mask_dir and os.path.exists(os.path.join(mask_dir, key)):
                cover = torch.from_numpy(np.load(os.path.join(mask_dir, key))).float().cuda()
            if cover is not None and (cover > 0.5).sum() > 100:
                m = (cover > 0.5)
                x = out.permute(1, 2, 0)[m]          # N,3
                y = gt.permute(1, 2, 0)[m]
                xm, ym = x.mean(0), y.mean(0)
                a = ((x - xm) * (y - ym)).mean(0) / ((x - xm).pow(2).mean(0) + 1e-8)
                b = ym - a * xm
                fixed = (a * x + b).clamp(0.0, 1.0)
                pa = (-10 * torch.log10(((fixed - y) ** 2).mean())).item()
                gain = a.mean().item()
            else:
                pa, gain = float("nan"), float("nan")
            stats.setdefault(cid, []).append((p, s, frac, pc, pa, gain))
            if save_dir and i % save_every == 0:
                torchvision.utils.save_image(
                    torch.cat([out, gt], dim=2),
                    os.path.join(save_dir, "cam%d_%s" % (cid, name.replace(".jpg", ".png"))))

        # R is the world-to-camera rotation (AD-GS drops 3DGS's transpose in
        # getWorld2View2), so the optical axis in world coordinates is its third row.
        axis = {int(c): R[cam_ids == c][:, 2, :] for c in np.unique(cam_ids)}
        print("\ncamera            n     PSNR     SSIM  covered  PSNR|cov  PSNR|cov+affine  gain   yaw")
        centres = {}
        for cid in sorted(stats):
            idx = np.where(cam_ids == cid)[0]
            c = np.stack([retarget(front[int(stamps[j])], R[j], T[j], 1.0, 1.0)
                          .camera_center.cpu().numpy()
                          for j in idx if int(stamps[j]) in front])
            centres[cid] = c
        for cid in sorted(stats):
            a = np.array(stats[cid])
            d = np.linalg.norm(centres[cid] - centres[0], axis=1) if cid in centres and 0 in centres else np.array([0.0])
            label = {0: "FRONT (train)", 1: "FRONT_LEFT", 2: "FRONT_RIGHT"}.get(cid, str(cid))
            ang = np.degrees(np.arccos(np.clip((axis[cid] * axis[0]).sum(1), -1, 1)))
            print("%-16s %4d  %7.3f  %6.4f  %5.1f%%  %8.3f  %13.3f  %5.3f  %5.1f deg"
                  % (label, len(a), a[:, 0].mean(), a[:, 1].mean(),
                     100 * np.nanmean(a[:, 2]), np.nanmean(a[:, 3]),
                     np.nanmean(a[:, 4]), np.nanmean(a[:, 5]), ang.mean()))


if __name__ == "__main__":
    main()
