#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
from math import exp
import torch.nn as nn
from utils.flow_utils import flow_points_project
from utils.depth_utils import get_scaled_shifted_depth
from utils.general_utils import build_rotation

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def get_depth_loss(pred, gt, mask=None, space='inv', eps=1e-2):
    """Monocular-depth supervision.

    Depth-Anything is affine-invariant in INVERSE depth, so the scale/shift fit has to
    happen there. But taking the L1 there too silently throws the far field away:
    |d(1/z)| = |dz| / z^2, so the same *relative* error counts ~21x less at the far end of
    a driving scene than at the near end (measured on Waymo: background inverse depth runs
    0.038 at p10 to 0.773 at p90). The far half of the background therefore receives ~20%
    of the depth supervision while carrying ~51% of the rendering error.

    space='log' compares in log space instead, where |d log(1/z)| = |d log z| and every
    depth is weighted equally. Sky must be masked out there: it sits at inverse depth ~0,
    which is exactly where the log blows up.
    """
    pred = get_scaled_shifted_depth(pred, gt, mask)
    if mask is None:
        mask = torch.ones_like(pred)
    if space == 'log':
        pred = torch.log(pred.clamp(min=eps))
        gt = torch.log(gt.clamp(min=eps))
    elif space != 'inv':
        raise ValueError('unknown depth space: ' + space)
    loss = torch.sum(torch.abs(pred - gt) * mask) / torch.sum(mask)
    return loss

# def get_flow_loss(img_flow, flow_pkg):
#     _, K, R, T, flow, flow_vis = flow_pkg

#     flow_vis: torch.Tensor = (flow_vis > 0.5) & (flow[0] <= flow.shape[2] - 1.0) & (flow[0] >= 0.0) & (flow[1] <= flow.shape[1] - 1.0) & (flow[1] >= 0.0)
#     if not torch.nonzero(flow_vis).any():
#         return 0.0
#     loss = torch.abs(img_flow[:, flow_vis] - flow[:, flow_vis])
#     loss = torch.cat([loss[:1] / flow.shape[2], loss[1:] / flow.shape[1]], dim=0)
#     loss = torch.mean(torch.sum(loss, dim=0))
#     return loss

def get_flow_loss(img_flow, flow_pkg, img_opacity=None, dist=1e-3):
    _, K, R, T, flow, flow_vis = flow_pkg
    H, W = flow.shape[1:]

    flow_vis: torch.Tensor = (flow_vis > 0.5) & (flow[0] <= flow.shape[2] - 1.0) & (flow[0] >= 0.0) & (flow[1] <= flow.shape[1] - 1.0) & (flow[1] >= 0.0)
    selected_coord = torch.nonzero(flow_vis, as_tuple=True)
    if selected_coord[0].numel() == 0:
        return 0.0
    flow_vis = flow_vis.float()
    if img_opacity is not None:
        flow_vis = flow_vis * img_opacity
    img_flow = torch.permute(img_flow[:, selected_coord[0], selected_coord[1]], (1, 0))  # N, 3
    flow = torch.permute(flow[:, selected_coord[0], selected_coord[1]], (1, 0))  # N, 3
    flow_vis = flow_vis[selected_coord[0], selected_coord[1]]  # N,
    img_flow, mask = flow_points_project(img_flow, K, R, T, dist=dist)
    flow_vis = flow_vis * mask.float()  # N,
    
    loss = torch.abs(img_flow - flow) * flow_vis[..., None]
    loss = torch.cat([loss[..., :1] / W, loss[..., 1:] / H], dim=-1)
    loss = torch.mean(torch.sum(loss, dim=-1))
    return loss

def get_anisotropy_loss(scaling, visible, tau):
    """Charge for degenerate primitives -- ribbons that face the training camera.

    A Gaussian stretched along the viewing ray projects to a dot and buys the
    optimiser nothing, so it does not build them: measured on scene006, the more
    elongated a Gaussian is the more perpendicular its long axis sits to the
    direction it was seen from (median 79 deg at 20x elongation, against 60 deg
    for uniformly oriented axes).  What it builds instead are wide thin sheets
    facing the camera, which paint a large image area with one primitive.  The
    fitted model is full of them: longest-to-shortest axis ratio reaches 3432x at
    p90 and 566,885x at p99.  On-trajectory that is efficient; turn the camera
    forty-five degrees and the same sheet is a knife edge, which is what those
    renders are full of (EXPERIMENTS.md 32.5).

    Taken in log so the p99 ratio cannot dominate the gradient, and hinged so
    primitives that are already compact are left alone -- the one lesson that has
    survived today is that penalising what is already fine is what costs dB.
    """
    s = scaling[visible]
    ratio = s.max(dim=-1).values / s.min(dim=-1).values.clamp(min=1e-8)
    return torch.relu(torch.log(ratio) - math.log(tau)).mean()


def get_lidar_depth_loss(inv_depth, opacity, uv, inv_z, free_weight, quantile):
    """Metric depth supervision from LiDAR returns, plus free space in front of them.

    AD-GS supervises depth only through Depth-Anything's affine-invariant monocular
    prediction (`get_depth_loss` fits a scale and shift first), so nothing in the
    objective fixes where a surface sits along its ray; the LiDAR is used to seed
    the point cloud and then never again.  Returns are metric, so comparing in
    inverse depth needs no fit at all.

    The asymmetric half is the one that matters.  Rendering *closer* than the LiDAR
    return means the model has put geometry into empty space -- the failure that
    fills a 45-degree view with streaks -- so it is charged `free_weight` times
    more than rendering further, which is a miss rather than a fabrication.

    The worst residuals are dominated by LiDAR-camera desync on moving actors and
    by occlusion boundaries, not by reconstruction error, so the top 1-`quantile`
    fraction is dropped.  IDSplat guards its own LiDAR depth loss the same way.
    """
    pred = inv_depth[uv[:, 1], uv[:, 0]] / opacity[uv[:, 1], uv[:, 0]].clamp(min=1e-6)
    residual = pred - inv_z
    weight = torch.where(residual > 0, free_weight, 1.0)
    loss = weight * residual.abs()
    keep = loss < torch.quantile(loss.detach().float(), quantile)
    return (loss * keep).mean()


def get_free_space_loss(xyz, opacity, view, uv, z, margin, min_depth):
    """Charge Gaussians that sit in space the LiDAR has already shown to be empty.

    A depth loss constrains where the rendered *surface* ends up.  It cannot see a
    primitive that spans from ten metres to sixty along a ray: alpha-weighting can
    still put the rendered depth in the right place, so the residual is zero while
    the primitive is entirely fabricated.  Measured on scene006, that is exactly
    what the model builds -- the longest-to-shortest axis ratio reaches 566,885x at
    p99 and the long axis lies perpendicular to the viewing direction, i.e. sheets
    facing the camera.  They are free on-trajectory and become the streaks that
    fill a 45-degree view.

    A LiDAR return at a pixel says the ray is empty in front of it, whatever any
    view renders.  So this projects the primitives themselves, not the image, and
    pushes the opacity of anything sitting in that carved-out space toward zero --
    handing the offenders to the pruning that already runs rather than fighting the
    photometric term for them.
    """
    # the cameras keep their matrices on the CPU under lazy_load_to_gpu
    w2v = view.world_view_transform.to(xyz.device)
    proj = view.full_proj_transform.to(xyz.device)
    p = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
    depth = (p @ w2v)[:, 2]
    clip = p @ proj
    ndc = clip[:, :2] / clip[:, 3:4].clamp(min=1e-6)
    u = ((ndc[:, 0] + 1.0) * view.image_width - 1.0) * 0.5
    v = ((ndc[:, 1] + 1.0) * view.image_height - 1.0) * 0.5

    inside = (depth > min_depth) & (u >= 0) & (v >= 0) \
           & (u <= view.image_width - 1) & (v <= view.image_height - 1)
    if not inside.any():
        return torch.zeros((), device=xyz.device)

    surface = torch.zeros((view.image_height, view.image_width), device=xyz.device)
    surface[uv[:, 1], uv[:, 0]] = z
    hit = surface[v[inside].long(), u[inside].long()]
    # only where the LiDAR actually returned, and only clearly in front of it
    carved = (hit > 0) & (depth[inside] < hit - margin)
    if not carved.any():
        return torch.zeros((), device=xyz.device)
    return opacity[inside][carved].mean()
