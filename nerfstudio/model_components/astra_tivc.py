"""ASTRA: virtual-view consistency (paper Sec. 3.4; flags --pipeline.tivc_*).

Training cameras are sparse in time, and held-out views fall between them. Regions that are seen in only a few
frames (passing vehicles at the image border, overhanging foliage) are fitted per frame and break up in between.
TIVC renders a virtual camera at a random time between two neighbouring training frames of the same sensor and
supervises it with both neighbours' images, warped into the virtual view with the rendered depth (detached), the
optimized camera poses, and each object's rigid box motion. The per-pixel minimum over the two sources handles
occlusions and out-of-view regions. With tivc_consistency > 0 only pixels where both warps agree are kept, and with
tivc_spatial_prob > 0 part of the virtual cameras is moved off the trajectory (random yaw and sideways shift).
Only training images are used.
"""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from nerfstudio.cameras.camera_utils import matrix_to_rotation_6d, rotation_6d_to_matrix
from nerfstudio.cameras.cameras import Cameras


def _gl_dirs(cam: Cameras, H: int, W: int, device) -> Tensor:
    """Camera-space ray directions with unit -z (OpenGL), matching get_ray_dirs_pinhole before normalisation."""
    ys = (torch.arange(H, device=device, dtype=torch.float32) + 0.5 - cam.cy[0, 0]) / cam.fy[0, 0]
    xs = (torch.arange(W, device=device, dtype=torch.float32) + 0.5 - cam.cx[0, 0]) / cam.fx[0, 0]
    v, u = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([u, -v, -torch.ones_like(u)], dim=-1)  # H W 3


def _interp_c2w(a: Tensor, b: Tensor, w: float) -> Tensor:
    rot = rotation_6d_to_matrix((1 - w) * matrix_to_rotation_6d(a[:3, :3]) + w * matrix_to_rotation_6d(b[:3, :3]))
    return torch.cat([rot, ((1 - w) * a[:3, 3:] + w * b[:3, 3:])], dim=-1)


def _project(points: Tensor, c2w: Tensor, cam: Cameras):
    """World points (N,3) -> normalized grid coords (N,2) in [-1,1] and camera depth (N,)."""
    R, t = c2w[:3, :3], c2w[:3, 3]
    pc = (points - t) @ R  # world -> camera (GL)
    z = -pc[:, 2]
    u = cam.fx[0, 0] * pc[:, 0] / z.clamp_min(1e-6) + cam.cx[0, 0]
    v = -cam.fy[0, 0] * pc[:, 1] / z.clamp_min(1e-6) + cam.cy[0, 0]
    W, H = float(cam.width[0, 0]), float(cam.height[0, 0])
    grid = torch.stack([u / W * 2 - 1, v / H * 2 - 1], dim=-1)
    return grid, z


def _object_warp(points: Tensor, om, t_v: Tensor, t_s: Tensor, margin: float) -> Tensor:
    """Move points that lie inside an object box at t_v rigidly with that box to time t_s."""
    if om.max_id() < 0:
        return points
    b_v = om.get_boxes2world(t_v.view(1, 1))[0].squeeze(0)  # N_obj 4 4
    b_s = om.get_boxes2world(t_s.view(1, 1))[0].squeeze(0)
    half = om.actor_sizes / 2 * margin  # N_obj 3 (box x,y,z)
    out = points.clone()
    taken = torch.zeros(len(points), dtype=torch.bool, device=points.device)
    for k in range(b_v.shape[0]):
        inv = torch.linalg.inv(b_v[k])
        p_box = points @ inv[:3, :3].T + inv[:3, 3]
        inside = (p_box.abs() <= half[k]).all(-1) & ~taken
        if inside.any():
            out[inside] = p_box[inside] @ b_s[k][:3, :3].T + b_s[k][:3, 3]
            taken |= inside
    return out


def tivc_loss(
    pipeline,
    sensor: Cameras,
    scale: int = 2,
    box_margin: float = 1.25,
    w: Optional[float] = None,
    debug: bool = False,
    consistency: float = 0.0,
    min_alpha: float = 0.0,
    yaw_deg: float = 0.0,
    lateral: float = 0.0,
    spatial_prob: float = 0.0,
):
    """L1 between the virtual in-between render and its two training neighbours warped into it.

    consistency == 0 (v1): per-pixel min over the two warps. consistency > 0 (v2): keep only pixels where both warps
    are valid and agree (mean abs difference < consistency) and the render is opaque (alpha >= min_alpha); the target
    is their average. Disagreeing warps flag wrong depth, occlusion or sky, so those pixels give no supervision.
    """
    dm, model = pipeline.datamanager, pipeline.model
    i = int(sensor.metadata["cam_idx"])
    cams = dm.train_cameras
    times = cams.times.flatten().to(model.device)
    sens = cams.metadata["sensor_idxs"].flatten().to(model.device)
    same = (sens == sens[i]).nonzero().flatten()
    same = same[times[same].argsort()]
    pos = int((same == i).nonzero()[0])
    nbrs = [p for p in (pos - 1, pos + 1) if 0 <= p < len(same)]
    if not nbrs:
        return None
    j = int(same[nbrs[torch.randint(len(nbrs), (1,)).item()]])
    if w is None:
        w = float(torch.empty(1).uniform_(0.2, 0.8))

    cam_j, data_j = dm.get_train_image_from_idx(j)
    cam_i, data_i = dm.get_train_image_from_idx(i)
    with torch.no_grad():
        c2w_i = model.camera_optimizer.apply_to_camera(cam_i)[0]
        c2w_j = model.camera_optimizer.apply_to_camera(cam_j)[0]
    t_i, t_j = times[i], times[j]
    t_v = t_i + w * (t_j - t_i)

    cam_v = deepcopy(cam_i)
    c2w_v = _interp_c2w(c2w_i, c2w_j, w)
    if spatial_prob > 0 and float(torch.rand(1)) < spatial_prob:
        # off-trajectory virtual camera: yaw about the camera up axis (GL +y) and a sideways shift
        th = math.radians(float(torch.empty(1).uniform_(-yaw_deg, yaw_deg)))
        c, sn = math.cos(th), math.sin(th)
        Ry = torch.tensor([[c, 0.0, sn], [0.0, 1.0, 0.0], [-sn, 0.0, c]], device=c2w_v.device, dtype=c2w_v.dtype)
        shift = c2w_v[:3, 0] * float(torch.empty(1).uniform_(-lateral, lateral))
        c2w_v = torch.cat([c2w_v[:3, :3] @ Ry, c2w_v[:3, 3:] + shift[:, None]], dim=-1)
    cam_v.camera_to_worlds = c2w_v[None]
    cam_v.times = t_v.view(1, 1).to(cam_v.times)
    cam_v.metadata = {k: v for k, v in cam_v.metadata.items() if k != "cam_idx"}  # pose already optimized
    cam_v.rescale_output_resolution(1.0 / scale)

    prev = model.config.output_depth_during_training
    model.config.output_depth_during_training = True
    try:
        out = model(cam_v)
    finally:
        model.config.output_depth_during_training = prev
    rgb_v = out["rgb"]  # H W 3
    alpha_v = out["accumulation"].detach().reshape(-1)
    H, W = rgb_v.shape[:2]
    depth = out["depth"].detach().reshape(H, W, 1)

    with torch.no_grad():
        c2w_v = cam_v.camera_to_worlds[0]
        pts_cam = _gl_dirs(cam_v, H, W, rgb_v.device) * depth  # z-depth times unit -z dirs
        pts = (pts_cam.reshape(-1, 3) @ c2w_v[:3, :3].T) + c2w_v[:3, 3]
        errs = []
        for cam_s, data_s, c2w_s, t_s in ((cam_i, data_i, c2w_i, t_i), (cam_j, data_j, c2w_j, t_j)):
            p_s = _object_warp(pts, model.objects_manager, t_v, t_s, box_margin)
            grid, z = _project(p_s, c2w_s, cam_s)
            gt_s = model.get_gt_img(data_s["image"])[..., :3].permute(2, 0, 1)[None]
            if scale > 1:  # match the virtual view resolution to avoid aliasing in the warp target
                gt_s = F.avg_pool2d(gt_s, scale)
            samp = F.grid_sample(gt_s, grid.view(1, H, W, 2), align_corners=False, padding_mode="border")
            valid = (z > 0.5) & (grid.abs() <= 1).all(-1)
            errs.append((samp[0].permute(1, 2, 0).reshape(-1, 3), valid))
    if consistency > 0:
        (s_i, v_i), (s_j, v_j) = errs
        ok = v_i & v_j & ((s_i - s_j).abs().mean(-1) < consistency) & (alpha_v >= min_alpha)
        if debug:
            return {"kept": ok.float().mean().item(), "l1": (rgb_v.reshape(-1, 3) - (s_i + s_j) / 2).abs().mean(-1)[ok].mean().item()}
        if ok.sum() < 100:
            return None
        return (rgb_v.reshape(-1, 3)[ok] - ((s_i + s_j) / 2)[ok]).abs().mean()
    l1 = torch.stack([(rgb_v.reshape(-1, 3) - s).abs().mean(-1) for s, _ in errs], dim=0)  # 2 N
    valid = torch.stack([v for _, v in errs], dim=0)
    l1 = torch.where(valid, l1, torch.full_like(l1, float("inf")))
    best = l1.min(0).values
    ok = torch.isfinite(best)
    if debug:
        naive = (rgb_v - F.interpolate(model.get_gt_img(data_j["image"])[..., :3].permute(2, 0, 1)[None], (H, W))[0].permute(1, 2, 0)).abs().mean()
        return {"min": best[ok].mean().item(), "src_i": l1[0][valid[0]].mean().item(), "src_j": l1[1][valid[1]].mean().item(),
                "naive_j": naive.item(), "valid": ok.float().mean().item()}
    if ok.sum() < 100:
        return None
    return best[ok].mean()
