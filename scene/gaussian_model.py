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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, fov2focal
from utils.general_utils import strip_symmetric, build_scaling_rotation, quaternion_multiply
from pytorch3d.ops import knn_points
from utils.func_utils import *
from utils.system_utils import put_color
from utils.shared_motion import SharedMotion

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int, order_args):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._scene_xyz = torch.empty(0)
        self._scene_shs_dc = torch.empty(0)
        self._scene_shs_rest = torch.empty(0)
        self._scene_scaling = torch.empty(0)
        self._scene_rotation = torch.empty(0)
        self._scene_opacity = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0.0
        self.abs_grad = False
        # Scale normalisation of the split criterion. The stock criterion is the SUM of
        # per-pixel screen-space gradients, so it grows with a Gaussian's projected area:
        # a far-field Gaussian covering a handful of pixels can never reach the threshold,
        # however wrong it is. Measured on Waymo: >50% of the static-background error sits
        # in the far half of the depth range. 'div' turns the criterion into a per-pixel
        # density; 'mul' is the opposite (Pixel-GS style); 'none' is stock 3DGS.
        self.densify_scale_norm = 'none'
        self.radii_accum = torch.empty(0)
        # 预算式 densification：每次分裂判据最高的 top-p%，而不是卡一个固定阈值。
        # 固定阈值在场景之间、判据之间都不可迁移——实测栽过三次：
        #   abs2e3 点数塌 66%、sndiv2e5 塌 12%、abs6e4 在 scene026 上 PSNR 退 0.16。
        # 分位数选择让"每步分裂多少个"与场景和判据都无关，也让等预算比较变成默认行为。
        self.densify_percent = 0.0
        # Mip-Splatting 3D 平滑滤波：每个高斯一个'最小可分辨尺度'，由它在所有训练相机中
        # 最近的一次观测距离 / 焦距决定。远处高斯因此拿到一个下界，抑制走样。
        self.mip_filter = False
        self.filter_3D = torch.empty(0)
        self.cameras_extent = 0.0
        self.scene_extent = 0.0

        self._obj_xyz = torch.empty(0)
        self._obj_shs_dc = torch.empty(0)
        self._obj_shs_rest = torch.empty(0)
        self._obj_scaling = torch.empty(0)
        self._obj_rotation = torch.empty(0)
        self._obj_opacity = torch.empty(0)

        # Per-Gaussian dynamicity logit. sigmoid(.) gates the deformation and the
        # temporal opacity window, and is what the 2D-mask loss actually supervises.
        self._scene_dyn = torch.empty(0)
        self._obj_dyn = torch.empty(0)
        self.soft_dyn = False
        self.dyn_init_logit = 4.0

        self.order_args = order_args
        self.xyz_deform_param = torch.empty(0)
        self.background_deform_param = torch.empty(0)
        self.rotation_deform_param = torch.empty(0)
        self.shs_deform_param_scene = torch.empty(0)
        self.shs_deform_param_obj = torch.empty(0)

        self.use_time_mask = None
        self.use_near_idx = False
        self.near_num = 0
        self.gs_time = torch.empty(0)
        self.gs_time_sigma = torch.empty(0)
        self.obj_near_idx = torch.empty(0)
        self.frame_gap = None
        self.shared_motion = None
        self.motion_optimizer = None

        self.setup_functions()

    @property
    def get_scaling(self):
        scaling = torch.cat([self._scene_scaling, self._obj_scaling], dim=0)
        return self.scaling_activation(scaling)
    
    @property
    def get_scene_scaling(self):
        return self.scaling_activation(self._scene_scaling)
    
    @property
    def get_obj_scaling(self):
        return self.scaling_activation(self._obj_scaling)
    
    @property
    def get_rotation(self):
        rotation = torch.cat([self._scene_rotation, self._obj_rotation], dim=0)
        return self.rotation_activation(rotation)
    
    @property
    def get_scene_rotation(self):
        return self.rotation_activation(self._scene_rotation)
    
    @property
    def get_obj_rotation(self):
        return self.rotation_activation(self._obj_rotation)

    @property
    def get_xyz(self):
        xyz = torch.cat([self._scene_xyz, self._obj_xyz], dim=0)
        return xyz
    
    @property
    def get_scene_xyz(self):
        return self._scene_xyz
    
    @property
    def get_obj_xyz(self):
        return self._obj_xyz
    
    @property
    def get_shs(self):
        shs_dc = torch.cat([self._scene_shs_dc, self._obj_shs_dc], dim=0)
        shs_rest = torch.cat([self._scene_shs_rest, self._obj_shs_rest], dim=0)
        return torch.cat((shs_dc, shs_rest), dim=1)
    
    @property
    def get_scene_shs(self):
        return torch.cat((self._scene_shs_dc, self._scene_shs_rest), dim=1)
    
    @property
    def get_obj_shs(self):
        return torch.cat((self._obj_shs_dc, self._obj_shs_rest), dim=1)

    @property
    def get_opacity(self):
        opacity = torch.cat([self._scene_opacity, self._obj_opacity], dim=0)
        return self.opacity_activation(opacity)

    @property
    def get_scene_opacity(self):
        return self.opacity_activation(self._scene_opacity)
    
    @property
    def get_obj_opacity(self):
        return self.opacity_activation(self._obj_opacity)
    
    @property
    def get_scaling_filtered(self):
        s = self.get_scaling
        if not self.mip_filter or self.filter_3D.numel() != s.shape[0]:
            return s
        return torch.sqrt(torch.square(s) + torch.square(self.filter_3D[:, None]))

    def apply_opacity_compensation(self, opacity):
        """把因为加了 3D 滤波而变大的体积补偿回去，保持积分不变。"""
        if not self.mip_filter or self.filter_3D.numel() != opacity.shape[0]:
            return opacity
        s2 = torch.square(self.get_scaling)
        det1 = s2.prod(dim=1)
        det2 = (s2 + torch.square(self.filter_3D[:, None])).prod(dim=1)
        return opacity * torch.sqrt(det1 / det2)[..., None]

    @torch.no_grad()
    def compute_3D_filter(self, cameras):
        """filter_3D = (最近观测距离 / 最大焦距) * sqrt(0.2)，Mip-Splatting 的定义。"""
        if not self.mip_filter:
            return
        xyz = self.get_xyz
        dist = torch.full((xyz.shape[0],), 1e5, dtype=torch.float32, device='cuda')
        seen = torch.zeros((xyz.shape[0],), dtype=torch.bool, device='cuda')
        focal_max = 0.0
        for cam in cameras:
            W = cam.world_view_transform.cuda()
            xyz_cam = xyz @ W[:3, :3] + W[3, :3]
            z = xyz_cam[:, 2]
            fx = fov2focal(cam.FoVx, cam.image_width)
            fy = fov2focal(cam.FoVy, cam.image_height)
            zc = z.clamp(min=1e-3)
            u = xyz_cam[:, 0] / zc * fx + cam.image_width * 0.5
            v = xyz_cam[:, 1] / zc * fy + cam.image_height * 0.5
            ok = (z > 0.2) & (u > -0.15 * cam.image_width) & (u < 1.15 * cam.image_width) \
                 & (v > -0.15 * cam.image_height) & (v < 1.15 * cam.image_height)
            dist[ok] = torch.min(dist[ok], z[ok])
            seen |= ok
            focal_max = max(focal_max, fx)
        if seen.any():
            dist[~seen] = dist[seen].max()
        else:
            dist[:] = 1.0
        self.filter_3D = dist / focal_max * (0.2 ** 0.5)
        if not getattr(self, '_mip_reported', False):
            self._mip_reported = True
            s = self.get_scaling
            grow = (self.filter_3D[:, None] > s).float().mean().item()
            q = torch.quantile(self.filter_3D.float(), torch.tensor([0.5, 0.9, 0.99], device='cuda')).tolist()
            print('[mip] filter_3D median %.4g p90 %.4g p99 %.4g | 被滤波抬高的尺度分量占比 %.1f%% | seen %.1f%%'
                  % (q[0], q[1], q[2], 100 * grow, 100 * seen.float().mean().item()))

    @property
    def get_obj_dyn(self):
        return torch.sigmoid(self._obj_dyn)

    @property
    def get_gated_xyz_deform_param(self):
        """Deformation coefficients as they actually act on the scene.

        The rigidity regulariser must see these, not the raw control points: with a gate
        of w the same physical motion needs 1/w larger coefficients, so regularising the
        raw ones scales the penalty by 1/w^2 and silently punishes gated Gaussians.
        Measured cost of getting this wrong: -0.282 dB on scene026.
"""
        if not self.soft_dyn:
            return self.xyz_deform_param
        return self.xyz_deform_param * self.get_obj_dyn[..., None]

    @property
    def get_dynamicity(self):
        """Soft static/dynamic assignment, one scalar per Gaussian, in [0, 1]."""
        return torch.sigmoid(torch.cat([self._scene_dyn, self._obj_dyn], dim=0))

    @property
    def get_obj_mask(self):
        scene_mask = torch.zeros((self.get_scene_pts_num,), dtype=torch.bool, device='cuda')
        obj_mask = torch.ones((self.get_obj_pts_num,), dtype=torch.bool, device='cuda')
        mask = torch.cat([scene_mask, obj_mask], dim=0)
        return mask
    
    @property
    def get_obj_pts_num(self):
        return self._obj_xyz.shape[0]
    
    @property
    def get_scene_pts_num(self):
        return self._scene_xyz.shape[0]
    
    @property
    def get_pts_num(self):
        return self.get_scene_pts_num + self.get_obj_pts_num
    
    def get_deformed_xyz(self, t):
        obj_xyz = self.get_obj_xyz

        # per partical deformation, gated by the learned dynamicity
        if self.shared_motion is None:
            deform = get_func_result(t, self.xyz_deform_param, self.order_args['xyz'])
        else:
            residual = (get_func_result(t, self.xyz_deform_param, self.order_args['xyz'])
                        if self.shared_motion.adgs_residual else None)
            deform = self.shared_motion(t, self.gs_time, self.xyz_deform_param[..., 0], obj_xyz,
                                        residual_displacement=residual)
        obj_xyz = obj_xyz + (self.get_obj_dyn * deform if self.soft_dyn else deform)

        scene_xyz = self.get_scene_xyz
        xyz = torch.cat([scene_xyz, obj_xyz], dim=0)

        # background
        xyz = xyz + get_func_result(t, self.background_deform_param, self.order_args['background'])  # 1, 3

        return xyz
    
    def get_deformed_rotation(self, t, bias_rot=None):
        obj_rotation = get_func_result(t, self.rotation_deform_param, self.order_args['rotation'])  # probably too slow if multiplied with self._obj_rotation
        if self.order_args['rotation'][4] == 0:
            obj_rotation = self._obj_rotation + obj_rotation

        if bias_rot is not None:
            obj_rotation = quaternion_multiply(bias_rot, obj_rotation)
        rotation = torch.cat([self._scene_rotation, obj_rotation], dim=0)
        rotation = self.rotation_activation(rotation)
        return rotation

    def get_deformed_shs(self, t):
        # color deformation should be for all Gaussians
        shs_deform_param = torch.cat([self.shs_deform_param_scene, self.shs_deform_param_obj], dim=0)
        shs_dc = torch.cat([self._scene_shs_dc, self._obj_shs_dc], dim=0)
        shs_dc = shs_dc[:, 0] + get_func_result(t, shs_deform_param, self.order_args['shs'])
        shs_rest = torch.cat([self._scene_shs_rest, self._obj_shs_rest], dim=0)
        shs = torch.cat([shs_dc[:, None], shs_rest], dim=1)
        return shs
    
    def get_time_masked_opacity(self, t):
        delta_time = t - self.gs_time
        time_sigma = torch.exp(self.gs_time_sigma)
        time_sigma = torch.where(delta_time < 0.0, time_sigma[:, :1], time_sigma[:, 1:])
        time_mask = torch.exp(-0.5 * (delta_time / time_sigma) ** 2)
        if self.soft_dyn:
            # a Gaussian that is not dynamic should not be windowed away in time
            time_mask = 1.0 - self.get_obj_dyn * (1.0 - time_mask)
        obj_opacity = self.get_obj_opacity * time_mask
        opacity = torch.cat([self.get_scene_opacity, obj_opacity], dim=0)
        return opacity

    def get_deformed_pkg(self, t):
        xyz = self.get_deformed_xyz(t)
        # The shared rigid rotation carries the Gaussian orientation too.
        bias_rot = (self.shared_motion.rotation_bias(t, self.gs_time)
                    if self.shared_motion is not None else None)
        rotation = self.get_deformed_rotation(t, bias_rot)
        shs = self.get_deformed_shs(t)
        
        if self.use_time_mask:
            opacity = self.get_time_masked_opacity(t)
        else:
            opacity = self.get_opacity

        return {
            'xyz': xyz,
            'rotation': rotation,
            'shs': shs,
            'opacity': opacity,
        }
    
    def get_flow_proj(self, flow_pkg, dist=None):
        dist = dist if dist is not None else self.scene_extent * 1e-3

        flow_time, K, R, T, _, _ = flow_pkg
        flow_xyz = self.get_deformed_xyz(flow_time)
        proj_pts = (K @ (R @ flow_xyz[..., None] + T[..., None]))[..., 0]

        mask = (torch.abs(proj_pts[..., 2:]) >= dist).float()
        sign = torch.sign(proj_pts[..., 2:])
        sign[sign == 0.0] = 1.0
        depth = sign * torch.clamp_min(torch.abs(proj_pts[..., 2:]), dist)

        proj_pts = mask * proj_pts[..., :2] / depth
        return proj_pts
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(
            self, 
            pcd: BasicPointCloud, 
            scene_extent: float, 
            cameras_extent: float, 
            frame_gap: float,
            default_order_downsample_ratio: float,
        ):
        self.scene_extent = scene_extent
        self.cameras_extent = cameras_extent
        self.object_extent = 10.0
        self.frame_gap = frame_gap
        self.order_args = set_default_param_order(self.order_args, int(1.0 / frame_gap), default_order_downsample_ratio)
        print('Parameter Arguments:', self.order_args)

        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        shs = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        shs[:, :3, 0 ] = fused_color
        shs[:, 3:, 1:] = 0.0
        shs = shs.transpose(1, 2)

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1.0

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # scene Gaussian parameters
        scene_mask = torch.tensor(pcd.obj_id[..., 0] <= 0.5, dtype=torch.bool, device='cuda')
        self._scene_xyz = nn.Parameter(fused_point_cloud[scene_mask].requires_grad_(True))
        self._scene_shs_dc = nn.Parameter(shs[scene_mask, 0:1].contiguous().requires_grad_(True))
        self._scene_shs_rest = nn.Parameter(shs[scene_mask, 1:].contiguous().requires_grad_(True))
        self._scene_scaling = nn.Parameter(scales[scene_mask].requires_grad_(True))
        self._scene_rotation = nn.Parameter(rots[scene_mask].requires_grad_(True))
        self._scene_opacity = nn.Parameter(opacities[scene_mask].requires_grad_(True))

        # object Gaussian parameters
        obj_mask = torch.logical_not(scene_mask)
        self._obj_xyz = nn.Parameter(fused_point_cloud[obj_mask].requires_grad_(True))
        self._obj_shs_dc = nn.Parameter(shs[obj_mask, 0:1].contiguous().requires_grad_(True))
        self._obj_shs_rest = nn.Parameter(shs[obj_mask, 1:].contiguous().requires_grad_(True))
        self._obj_scaling = nn.Parameter(scales[obj_mask].requires_grad_(True))
        self._obj_rotation = nn.Parameter(rots[obj_mask].requires_grad_(True))
        self._obj_opacity = nn.Parameter(opacities[obj_mask].requires_grad_(True))

        # dynamicity logits: initialised from the 2D masks, but free to move afterwards
        self._scene_dyn = nn.Parameter(torch.full((int(scene_mask.sum()), 1), -self.dyn_init_logit, dtype=torch.float32, device='cuda').requires_grad_(True))
        self._obj_dyn = nn.Parameter(torch.full((int(obj_mask.sum()), 1), self.dyn_init_logit, dtype=torch.float32, device='cuda').requires_grad_(True))

        self.max_radii2D = torch.zeros((fused_point_cloud.shape[0]), device="cuda")

        obj_pts_num = self.get_obj_xyz.shape[0]

        # deformation parameters
        xyz_deform_param = torch.rand((obj_pts_num, 3, get_param_num(self.order_args['xyz'])), device='cuda', dtype=torch.float32) * 2.0 - 1.0
        xyz_deform_param = xyz_deform_param * 1e-5
        self.xyz_deform_param = nn.Parameter(xyz_deform_param.requires_grad_(True))

        rotation_deform_param = torch.rand((obj_pts_num, 4, get_param_num(self.order_args['rotation'])), device='cuda', dtype=torch.float32) * 2.0 - 1.0
        rotation_deform_param = rotation_deform_param * 1e-5
        self.rotation_deform_param = nn.Parameter(rotation_deform_param.requires_grad_(True))

        # color deformation should be for all Gaussians
        shs_deform_param = torch.rand((fused_point_cloud.shape[0], 3, get_param_num(self.order_args['shs'])), device='cuda', dtype=torch.float32) * 2.0 - 1.0
        shs_deform_param = shs_deform_param * 1e-5
        self.shs_deform_param_scene = nn.Parameter(shs_deform_param[scene_mask].requires_grad_(True))
        self.shs_deform_param_obj = nn.Parameter(shs_deform_param[obj_mask].requires_grad_(True))

        # background deformation should be same for all Gaussians
        background_deform_param = torch.rand((1, 3, get_param_num(self.order_args['background'])), device='cuda', dtype=torch.float32) * 2.0 - 1.0
        background_deform_param = background_deform_param * 1e-5
        self.background_deform_param = nn.Parameter(background_deform_param.requires_grad_(True))

        self.gs_time = torch.tensor(pcd.time, device='cuda', dtype=torch.float32)[obj_mask]
        gs_time_sigma = torch.full((obj_pts_num, 2), fill_value=np.log(frame_gap), dtype=torch.float32, device='cuda')
        self.gs_time_sigma = nn.Parameter(gs_time_sigma.requires_grad_(True))

        print(
            "Number of points at initialization:",
            "total", fused_point_cloud.shape[0],
            'scene', self.get_scene_xyz.shape[0], 
            'object', self.get_obj_xyz.shape[0],
        )

        
    def initialize_shared_motion(self, support_gate=False, linear_fallback=False,
                                 piecewise_fallback=False, bidirectional=False,
                                 singleton_static=False, rigid_rotation=False,
                                 registration_init=False, adgs_residual=False,
                                 adgs_ungated=False):
        adgs_residual = adgs_residual or adgs_ungated
        self.shared_motion = SharedMotion.from_points(
            self.get_obj_xyz, self.gs_time, self.frame_gap, support_gate,
            linear_fallback, piecewise_fallback, bidirectional, singleton_static,
            rigid_rotation, registration_init, adgs_residual=adgs_residual,
            adgs_ungated=adgs_ungated)
        old_count = self.xyz_deform_param.numel()
        # Retain the configured AD-GS basis and its initialization when requested.
        if not adgs_residual:
            self.xyz_deform_param = nn.Parameter(torch.zeros_like(self._obj_xyz)[..., None])
            self.order_args = dict(self.order_args)
            self.order_args['xyz'] = [0, 0, 1, 0, 0, 0]
        elif adgs_ungated:
            # Supported points read coefficient 0 as a velocity, zero at birth as
            # in the translation-only model; their other coefficients stay unused.
            with torch.no_grad():
                self.xyz_deform_param[self.shared_motion.active_points()] = 0
        new_count = self.xyz_deform_param.numel() + self.shared_motion.control.numel()
        print('Translation parameters: %d -> %d' % (old_count, new_count))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.abs_grad = getattr(training_args, 'abs_grad', False)
        self.soft_dyn = getattr(training_args, 'soft_dyn', False)
        self.densify_scale_norm = getattr(training_args, 'densify_scale_norm', 'none')
        self.densify_percent = getattr(training_args, 'densify_percent', 0.0)
        pts_num = self.get_pts_num
        self.xyz_gradient_accum = torch.zeros((pts_num, 1), device="cuda")
        self.radii_accum = torch.zeros((pts_num, 1), device="cuda")
        self.denom = torch.zeros((pts_num, 1), device="cuda")
        self.object_extent = training_args.object_extent if training_args.object_extent is not None else self.object_extent
        self.cameras_extent = max(self.cameras_extent, training_args.min_camera_extent)

        l = [
            # scene Gaussian parameters
            {'params': [self._scene_xyz], 'lr': 0.0, "name": "scene_xyz"},
            {'params': [self._scene_shs_dc], 'lr': training_args.feature_lr, "name": "scene_shs_dc"},
            {'params': [self._scene_shs_rest], 'lr': training_args.feature_lr / 20.0, "name": "scene_shs_rest"},
            {'params': [self._scene_opacity], 'lr': training_args.opacity_lr, "name": "scene_opacity"},
            {'params': [self._scene_scaling], 'lr': training_args.scaling_lr, "name": "scene_scaling"},
            {'params': [self._scene_rotation], 'lr': training_args.rotation_lr, "name": "scene_rotation"},
            {'params': [self._scene_dyn], 'lr': training_args.dyn_lr, "name": "scene_dyn"},

            # object Gaussian parameters
            {'params': [self._obj_xyz], 'lr': 0.0, "name": "obj_xyz"},
            {'params': [self._obj_shs_dc], 'lr': training_args.feature_lr, "name": "obj_shs_dc"},
            {'params': [self._obj_shs_rest], 'lr': training_args.feature_lr / 20.0, "name": "obj_shs_rest"},
            {'params': [self._obj_opacity], 'lr': training_args.opacity_lr, "name": "obj_opacity"},
            {'params': [self._obj_scaling], 'lr': training_args.scaling_lr, "name": "obj_scaling"},
            {'params': [self._obj_rotation], 'lr': training_args.rotation_lr, "name": "obj_rotation"},
            {'params': [self._obj_dyn], 'lr': training_args.dyn_lr, "name": "obj_dyn"},

            # deformation parameters
            {'params': [self.rotation_deform_param], 'lr': training_args.rotation_deform_lr, "name": "deform_rotation"},
            {'params': [self.shs_deform_param_scene], 'lr': training_args.shs_deform_lr, "name": "deform_shs_scene"},
            {'params': [self.shs_deform_param_obj], 'lr': training_args.shs_deform_lr, "name": "deform_shs_obj"},
            {'params': [self.xyz_deform_param], 'lr': 0.0, "name": "deform_xyz"},
            {'params': [self.background_deform_param], 'lr': 0.0, "name": "deform_background"},
            {'params': [self.gs_time_sigma], 'lr': training_args.gs_time_sigma_lr, "name": "time_sigma"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        if self.shared_motion is not None:
            self.motion_optimizer = torch.optim.Adam(self.shared_motion.parameters(), lr=0.0, eps=1e-15)
        self.obj_xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.object_extent * training_args.obj_position_lr_scale,
            lr_final=training_args.position_lr_final * self.object_extent * training_args.obj_position_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps
        )

        self.scene_xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.cameras_extent * training_args.scene_position_lr_scale,
            lr_final=training_args.position_lr_final * self.cameras_extent * training_args.scene_position_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps
        )

        self.deform_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.scene_extent * training_args.position_deform_lr_scale,
            lr_final=training_args.position_lr_final * self.scene_extent * training_args.position_deform_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps
        )
        
        self.use_time_mask = training_args.lambda_sigma > 0.0 if self.use_time_mask is None else self.use_time_mask
        self.use_near_idx = training_args.lambda_reg > 0.0 or (training_args.lambda_sigma > 0.0 and training_args.lambda_sigma_reg > 0.0)
        self.near_num = training_args.near_num
        self.set_obj_near_idx()
        
       
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.motion_optimizer is not None:
            self.motion_optimizer.param_groups[0]['lr'] = self.deform_scheduler_args(iteration)
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in ["scene_xyz", 'deform_background']:
                lr = self.scene_xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] in ["obj_xyz"]:
                lr = self.obj_xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] in ['deform_xyz']:
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._scene_shs_dc.shape[1]*self._scene_shs_dc.shape[2]):
            l.append('shs_dc_{}'.format(i))
        for i in range(self._scene_shs_rest.shape[1]*self._scene_shs_rest.shape[2]):
            l.append('shs_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scene_scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._scene_rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('obj')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = torch.cat([self._scene_xyz, self._obj_xyz], dim=0).detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        shs_dc = torch.cat([self._scene_shs_dc, self._obj_shs_dc], dim=0).detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        shs_rest = torch.cat([self._scene_shs_rest, self._obj_shs_rest], dim=0).detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = torch.cat([self._scene_opacity, self._obj_opacity], dim=0).detach().cpu().numpy()
        scale = torch.cat([self._scene_scaling, self._obj_scaling], dim=0).detach().cpu().numpy()
        rotation = torch.cat([self._scene_rotation, self._obj_rotation], dim=0).detach().cpu().numpy()
        obj_mask = self.get_obj_mask[..., None].float().detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, shs_dc, shs_rest, opacities, scale, rotation, obj_mask), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        torch.save((
            self.xyz_deform_param, 
            self.rotation_deform_param, 
            self.shs_deform_param_scene,
            self.shs_deform_param_obj,
            self.background_deform_param,
            self.gs_time,
            self.gs_time_sigma,
            self.use_time_mask,
            self.order_args,
            self.scene_extent,
        ), os.path.join(os.path.dirname(path), "deform.pth"))

        # separate file: checkpoints written before this existed must still load
        torch.save((self.soft_dyn, self._scene_dyn, self._obj_dyn),
                   os.path.join(os.path.dirname(path), "dyn.pth"))
        if self.shared_motion is not None:
            torch.save(self.shared_motion.state_dict(),
                       os.path.join(os.path.dirname(path), 'shared_motion.pth'))

    def reset_opacity(self):
        scene_opacities_new = inverse_sigmoid(torch.min(self.get_scene_opacity, torch.ones_like(self.get_scene_opacity) * 0.01))
        self._scene_opacity = self.replace_tensor_to_optimizer(scene_opacities_new, "scene_opacity")["scene_opacity"]
        obj_opacities_new = inverse_sigmoid(torch.min(self.get_obj_opacity, torch.ones_like(self.get_obj_opacity) * 0.01))
        self._obj_opacity = self.replace_tensor_to_optimizer(obj_opacities_new, "obj_opacity")['obj_opacity']

    def load_ply(self, path, legacy_shared_motion=None):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        obj_mask = np.asarray(plydata.elements[0]['obj']) > 0.5
        scene_mask = np.logical_not(obj_mask)

        shs_dc = np.zeros((xyz.shape[0], 3, 1))
        shs_dc[:, 0, 0] = np.asarray(plydata.elements[0]["shs_dc_0"])
        shs_dc[:, 1, 0] = np.asarray(plydata.elements[0]["shs_dc_1"])
        shs_dc[:, 2, 0] = np.asarray(plydata.elements[0]["shs_dc_2"])

        extra_shs_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("shs_rest_")]
        extra_shs_names = sorted(extra_shs_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_shs_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        shs_extra = np.zeros((xyz.shape[0], len(extra_shs_names)))
        for idx, attr_name in enumerate(extra_shs_names):
            shs_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        shs_extra = shs_extra.reshape((shs_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._scene_xyz = nn.Parameter(torch.tensor(xyz[scene_mask], dtype=torch.float32, device='cuda').requires_grad_(True))
        self._scene_shs_dc = nn.Parameter(torch.tensor(shs_dc[scene_mask], dtype=torch.float32, device='cuda').transpose(1, 2).contiguous().requires_grad_(True))
        self._scene_shs_rest = nn.Parameter(torch.tensor(shs_extra[scene_mask], dtype=torch.float32, device='cuda').transpose(1, 2).contiguous().requires_grad_(True))
        self._scene_opacity = nn.Parameter(torch.tensor(opacities[scene_mask], dtype=torch.float32, device='cuda').requires_grad_(True))
        self._scene_scaling = nn.Parameter(torch.tensor(scales[scene_mask], dtype=torch.float32, device='cuda').requires_grad_(True))
        self._scene_rotation = nn.Parameter(torch.tensor(rots[scene_mask], dtype=torch.float32, device='cuda').requires_grad_(True))

        self._obj_xyz = nn.Parameter(torch.tensor(xyz[obj_mask], dtype=torch.float32, device='cuda').requires_grad_(True))
        self._obj_shs_dc = nn.Parameter(torch.tensor(shs_dc[obj_mask], dtype=torch.float32, device='cuda').transpose(1, 2).contiguous().requires_grad_(True))
        self._obj_shs_rest = nn.Parameter(torch.tensor(shs_extra[obj_mask], dtype=torch.float32, device='cuda').transpose(1, 2).contiguous().requires_grad_(True))
        self._obj_opacity = nn.Parameter(torch.tensor(opacities[obj_mask], dtype=torch.float32, device='cuda').requires_grad_(True))
        self._obj_scaling = nn.Parameter(torch.tensor(scales[obj_mask], dtype=torch.float32, device='cuda').requires_grad_(True))
        self._obj_rotation = nn.Parameter(torch.tensor(rots[obj_mask], dtype=torch.float32, device='cuda').requires_grad_(True))

        (
            xyz_deform_param, 
            rotation_deform_param, 
            shs_deform_param_scene,
            shs_deform_param_obj,
            background_deform_param,
            gs_time,
            gs_time_sigma,
            self.use_time_mask,
            self.order_args,
            self.scene_extent,
        ) = torch.load(os.path.join(os.path.dirname(path), "deform.pth"), map_location='cuda')
        assert xyz_deform_param.shape[0] == self.get_obj_pts_num
        assert xyz_deform_param.shape[-1] == get_param_num(self.order_args['xyz'])
        assert shs_deform_param_obj.shape[-1] == get_param_num(self.order_args['shs'])
        assert shs_deform_param_scene.shape[-1] == get_param_num(self.order_args['shs'])
        assert rotation_deform_param.shape[-1] == get_param_num(self.order_args['rotation'])
        assert background_deform_param.shape[-1] == get_param_num(self.order_args['background'])
        self.xyz_deform_param = nn.Parameter(xyz_deform_param.requires_grad_(True))
        self.rotation_deform_param = nn.Parameter(rotation_deform_param.requires_grad_(True))
        self.shs_deform_param_scene = nn.Parameter(shs_deform_param_scene.requires_grad_(True))
        self.shs_deform_param_obj = nn.Parameter(shs_deform_param_obj.requires_grad_(True))
        self.background_deform_param = nn.Parameter(background_deform_param.requires_grad_(True))
        self.gs_time = gs_time
        self.gs_time_sigma = nn.Parameter(gs_time_sigma.requires_grad_(True))

        motion_path = os.path.join(os.path.dirname(path), 'shared_motion.pth')
        self.shared_motion = (SharedMotion.from_state(torch.load(motion_path, map_location='cuda'), legacy_shared_motion)
                              if os.path.exists(motion_path) else None)

        dyn_path = os.path.join(os.path.dirname(path), "dyn.pth")
        if os.path.exists(dyn_path):
            self.soft_dyn, scene_dyn, obj_dyn = torch.load(dyn_path, map_location='cuda')
            self._scene_dyn = nn.Parameter(scene_dyn.requires_grad_(True))
            self._obj_dyn = nn.Parameter(obj_dyn.requires_grad_(True))
        else:
            self.soft_dyn = False
            self._scene_dyn = nn.Parameter(torch.full((self.get_scene_pts_num, 1), -self.dyn_init_logit, device='cuda'))
            self._obj_dyn = nn.Parameter(torch.full((self.get_obj_pts_num, 1), self.dyn_init_logit, device='cuda'))

        self.active_sh_degree = self.max_sh_degree
        
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, scene_mask, obj_mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ['deform_background']:
                continue
            if 'obj' in group['name'] or group['name'] in ['deform_xyz', 'deform_rotation', 'deform_shs_obj', 'time_sigma']:
                mask = obj_mask
            else:
                mask = scene_mask

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, scene_mask, obj_mask):
        valid_scene_mask = ~scene_mask
        valid_obj_mask = ~obj_mask
        optimizable_tensors = self._prune_optimizer(valid_scene_mask, valid_obj_mask)

        self._scene_xyz = optimizable_tensors["scene_xyz"]
        self._scene_shs_dc = optimizable_tensors["scene_shs_dc"]
        self._scene_shs_rest = optimizable_tensors["scene_shs_rest"]
        self._scene_opacity = optimizable_tensors["scene_opacity"]
        self._scene_scaling = optimizable_tensors["scene_scaling"]
        self._scene_rotation = optimizable_tensors["scene_rotation"]

        self._obj_xyz = optimizable_tensors["obj_xyz"]
        self._obj_shs_dc = optimizable_tensors["obj_shs_dc"]
        self._obj_shs_rest = optimizable_tensors["obj_shs_rest"]
        self._obj_opacity = optimizable_tensors["obj_opacity"]
        self._obj_scaling = optimizable_tensors["obj_scaling"]
        self._obj_rotation = optimizable_tensors["obj_rotation"]

        self.xyz_deform_param = optimizable_tensors['deform_xyz']
        self.rotation_deform_param = optimizable_tensors['deform_rotation']
        self.shs_deform_param_scene = optimizable_tensors['deform_shs_scene']
        self.shs_deform_param_obj = optimizable_tensors['deform_shs_obj']
        self.gs_time_sigma = optimizable_tensors['time_sigma']
        self._scene_dyn = optimizable_tensors['scene_dyn']
        self._obj_dyn = optimizable_tensors['obj_dyn']
        self.gs_time = self.gs_time[valid_obj_mask]
        if self.shared_motion is not None:
            self.shared_motion.prune(valid_obj_mask)

        valid_mask = torch.cat([valid_scene_mask, valid_obj_mask], dim=0)
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_mask]
        self.radii_accum = self.radii_accum[valid_mask]
        self.denom = self.denom[valid_mask]
        self.max_radii2D = self.max_radii2D[valid_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] in ['deform_background']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
            self,
            new_scene_xyz,
            new_scene_shs_dc,
            new_scene_shs_rest,
            new_scene_opacities,
            new_scene_scaling,
            new_scene_rotation,

            new_obj_xyz,
            new_obj_shs_dc,
            new_obj_shs_rest,
            new_obj_opacities,
            new_obj_scaling,
            new_obj_rotation,

            new_deform_xyz,
            new_deform_rotation,
            new_deform_shs_scene,
            new_deform_shs_obj,
            new_time,
            new_time_sigma,

            new_scene_dyn,
            new_obj_dyn,
        ):
        d = {
            "scene_xyz": new_scene_xyz,
            "scene_shs_dc": new_scene_shs_dc,
            "scene_shs_rest": new_scene_shs_rest,
            "scene_opacity": new_scene_opacities,
            "scene_scaling" : new_scene_scaling,
            "scene_rotation" : new_scene_rotation,

            "obj_xyz": new_obj_xyz,
            "obj_shs_dc": new_obj_shs_dc,
            "obj_shs_rest": new_obj_shs_rest,
            "obj_opacity": new_obj_opacities,
            "obj_scaling" : new_obj_scaling,
            "obj_rotation" : new_obj_rotation,

            "deform_xyz": new_deform_xyz,
            "deform_rotation": new_deform_rotation,
            'deform_shs_scene': new_deform_shs_scene,
            'deform_shs_obj': new_deform_shs_obj,
            'time_sigma': new_time_sigma,

            'scene_dyn': new_scene_dyn,
            'obj_dyn': new_obj_dyn,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        self._scene_xyz = optimizable_tensors["scene_xyz"]
        self._scene_shs_dc = optimizable_tensors["scene_shs_dc"]
        self._scene_shs_rest = optimizable_tensors["scene_shs_rest"]
        self._scene_opacity = optimizable_tensors["scene_opacity"]
        self._scene_scaling = optimizable_tensors["scene_scaling"]
        self._scene_rotation = optimizable_tensors["scene_rotation"]

        self._obj_xyz = optimizable_tensors["obj_xyz"]
        self._obj_shs_dc = optimizable_tensors["obj_shs_dc"]
        self._obj_shs_rest = optimizable_tensors["obj_shs_rest"]
        self._obj_opacity = optimizable_tensors["obj_opacity"]
        self._obj_scaling = optimizable_tensors["obj_scaling"]
        self._obj_rotation = optimizable_tensors["obj_rotation"]

        self.xyz_deform_param = optimizable_tensors['deform_xyz']
        self.rotation_deform_param = optimizable_tensors['deform_rotation']
        self.shs_deform_param_scene = optimizable_tensors['deform_shs_scene']
        self.shs_deform_param_obj = optimizable_tensors['deform_shs_obj']
        self.gs_time_sigma = optimizable_tensors['time_sigma']
        self._scene_dyn = optimizable_tensors['scene_dyn']
        self._obj_dyn = optimizable_tensors['obj_dyn']

        self.gs_time = torch.cat([self.gs_time, new_time], dim=0)

        pts_num = self.get_pts_num
        self.xyz_gradient_accum = torch.zeros((pts_num, 1), device="cuda")
        self.radii_accum = torch.zeros((pts_num, 1), device="cuda")
        self.denom = torch.zeros((pts_num, 1), device="cuda")
        self.max_radii2D = torch.zeros((pts_num,), device="cuda")

    def densify_and_split(self, scene_densify_mask, obj_densify_mask, N=2):
        # Extract points that satisfy the scaling condition
        scene_densify_mask = scene_densify_mask & (torch.max(self.get_scene_scaling, dim=1).values > self.scene_extent * self.percent_dense)
        obj_densify_mask = obj_densify_mask & (torch.max(self.get_obj_scaling, dim=1).values > self.object_extent * self.percent_dense)

        stds = self.get_scene_scaling[scene_densify_mask].repeat(N, 1)
        samples = torch.normal(mean=0.0, std=stds)
        rots = build_rotation(self._scene_rotation[scene_densify_mask]).repeat(N, 1, 1)
        new_scene_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_scene_xyz[scene_densify_mask].repeat(N, 1)
        new_scene_scaling = self.scaling_inverse_activation(self.get_scene_scaling[scene_densify_mask].repeat(N, 1) / (0.8 * N))
        new_scene_rotation = self._scene_rotation[scene_densify_mask].repeat(N, 1)
        new_scene_shs_dc = self._scene_shs_dc[scene_densify_mask].repeat(N, 1, 1)
        new_scene_shs_rest = self._scene_shs_rest[scene_densify_mask].repeat(N, 1, 1)
        new_scene_opacities = self._scene_opacity[scene_densify_mask].repeat(N, 1)

        # if use B-Spline quaternion curves, the rotation in densification will be disabled.
        stds = self.get_obj_scaling[obj_densify_mask].repeat(N, 1)
        samples = torch.normal(mean=0.0, std=stds)
        rots = build_rotation(self._obj_rotation[obj_densify_mask]).repeat(N, 1, 1)
        new_obj_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_obj_xyz[obj_densify_mask].repeat(N, 1)
        new_obj_scaling = self.scaling_inverse_activation(self.get_obj_scaling[obj_densify_mask].repeat(N, 1) / (0.8 * N))
        new_obj_rotation = self._obj_rotation[obj_densify_mask].repeat(N, 1)
        new_obj_shs_dc = self._obj_shs_dc[obj_densify_mask].repeat(N, 1, 1)
        new_obj_shs_rest = self._obj_shs_rest[obj_densify_mask].repeat(N, 1, 1)
        new_obj_opacities = self._obj_opacity[obj_densify_mask].repeat(N, 1)

        new_deform_shs_scene = self.shs_deform_param_scene[scene_densify_mask].repeat(N, 1, 1)
        new_deform_shs_obj = self.shs_deform_param_obj[obj_densify_mask].repeat(N, 1, 1)

        new_deform_xyz = self.xyz_deform_param[obj_densify_mask].repeat(N,1,1)
        new_deform_rotation = self.rotation_deform_param[obj_densify_mask].repeat(N,1,1)
        new_time_sigma = self.gs_time_sigma[obj_densify_mask].repeat(N, 1)
        new_time = self.gs_time[obj_densify_mask].repeat(N, 1)

        new_scene_dyn = self._scene_dyn[scene_densify_mask].repeat(N, 1)
        new_obj_dyn = self._obj_dyn[obj_densify_mask].repeat(N, 1)
        if self.shared_motion is not None:
            new_motion_ids = self.shared_motion.point_ids[obj_densify_mask].repeat(N)

        self.densification_postfix(
            new_scene_xyz,
            new_scene_shs_dc,
            new_scene_shs_rest,
            new_scene_opacities,
            new_scene_scaling,
            new_scene_rotation,

            new_obj_xyz,
            new_obj_shs_dc,
            new_obj_shs_rest,
            new_obj_opacities,
            new_obj_scaling,
            new_obj_rotation,

            new_deform_xyz,
            new_deform_rotation,
            new_deform_shs_scene,
            new_deform_shs_obj,
            new_time,
            new_time_sigma,

            new_scene_dyn,
            new_obj_dyn,
        )

        if self.shared_motion is not None:
            self.shared_motion.append(new_motion_ids)
        scene_prune_filter = torch.cat([scene_densify_mask, torch.zeros(N * scene_densify_mask.sum(), device="cuda", dtype=bool)], dim=0)
        obj_prune_filter = torch.cat([obj_densify_mask, torch.zeros(N * obj_densify_mask.sum(), device="cuda", dtype=bool)], dim=0)
        self.prune_points(scene_prune_filter, obj_prune_filter)

    def densify_and_clone(self, scene_densify_mask, obj_densify_mask):
        # Extract points that satisfy the scaling condition
        scene_densify_mask = scene_densify_mask & (torch.max(self.get_scene_scaling, dim=1).values <= self.scene_extent * self.percent_dense)
        obj_densify_mask = obj_densify_mask & (torch.max(self.get_obj_scaling, dim=1).values <= self.object_extent * self.percent_dense)

        new_scene_xyz = self._scene_xyz[scene_densify_mask]
        new_scene_shs_dc = self._scene_shs_dc[scene_densify_mask]
        new_scene_shs_rest = self._scene_shs_rest[scene_densify_mask]
        new_scene_opacities = self._scene_opacity[scene_densify_mask]
        new_scene_scaling = self._scene_scaling[scene_densify_mask]
        new_scene_rotation = self._scene_rotation[scene_densify_mask]

        new_obj_xyz = self._obj_xyz[obj_densify_mask]
        new_obj_shs_dc = self._obj_shs_dc[obj_densify_mask]
        new_obj_shs_rest = self._obj_shs_rest[obj_densify_mask]
        new_obj_opacities = self._obj_opacity[obj_densify_mask]
        new_obj_scaling = self._obj_scaling[obj_densify_mask]
        new_obj_rotation = self._obj_rotation[obj_densify_mask]

        new_deform_shs_scene = self.shs_deform_param_scene[scene_densify_mask]
        new_deform_shs_obj = self.shs_deform_param_obj[obj_densify_mask]

        new_deform_xyz = self.xyz_deform_param[obj_densify_mask]
        new_deform_rotation = self.rotation_deform_param[obj_densify_mask]
        new_time_sigma = self.gs_time_sigma[obj_densify_mask]
        new_time = self.gs_time[obj_densify_mask]

        new_scene_dyn = self._scene_dyn[scene_densify_mask]
        new_obj_dyn = self._obj_dyn[obj_densify_mask]
        if self.shared_motion is not None:
            new_motion_ids = self.shared_motion.point_ids[obj_densify_mask]

        self.densification_postfix(
            new_scene_xyz,
            new_scene_shs_dc,
            new_scene_shs_rest,
            new_scene_opacities,
            new_scene_scaling,
            new_scene_rotation,

            new_obj_xyz,
            new_obj_shs_dc,
            new_obj_shs_rest,
            new_obj_opacities,
            new_obj_scaling,
            new_obj_rotation,

            new_deform_xyz,
            new_deform_rotation,
            new_deform_shs_scene,
            new_deform_shs_obj,
            new_time,
            new_time_sigma,

            new_scene_dyn,
            new_obj_dyn,
        )

        if self.shared_motion is not None:
            self.shared_motion.append(new_motion_ids)

    def set_obj_near_idx(self, K=None):
        if not self.use_near_idx:
            return
        K = self.near_num if K is None else K
        xyz = self.get_obj_xyz
        if self.use_time_mask:
            xyz = torch.cat([xyz, self.gs_time * self.scene_extent], dim=-1)
        anchor = xyz[torch.randperm(xyz.shape[0], device='cuda')[:xyz.shape[0] // K]]
        self.obj_near_idx = knn_points(anchor[None], xyz[None], K=K).idx.squeeze()  # P, K

    def densify_and_prune(self, max_scene_grad, max_obj_grad, min_opacity, prune_big_points):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        grads = torch.norm(grads, dim=-1)

        if self.densify_scale_norm != 'none':
            radii = (self.radii_accum / self.denom).squeeze(-1)
            radii[radii.isnan()] = 0.0
            radii = radii.clamp(min=1.0)
            grads = grads / radii if self.densify_scale_norm == 'div' else grads * radii

        if os.environ.get('ADGS_GRAD_STATS'):
            q = torch.quantile(grads.float(), torch.tensor([0.5, 0.9, 0.99, 0.999], device='cuda')).tolist()
            print('[grad-stats] abs=%d norm=%s n=%d median=%.3e p90=%.3e p99=%.3e p999=%.3e' % (self.abs_grad, self.densify_scale_norm, grads.numel(), *q))

        # Extract points that satisfy the gradient condition
        obj_mask = self.get_obj_mask
        if self.densify_percent > 0.0:
            # 分位数必须在**全部高斯上算一次**，两个分支共用这一个阈值。
            # 分支内各算各的是错的（2026-08-24 实测）：object 分支本来只占 5%，
            # 让它也按自己的 top-p% 分裂 = 把它的相对分裂速率抬到和 scene 一样，
            # 结果 object 从 27,769 爆到 245,014（占 45%）、scene 从 504,006 饿到
            # 302,829，PSNR −0.39。原版用单一绝对阈值，object 梯度普遍更低，
            # 自然分裂得少——这个相对关系必须保留。
            thr = torch.quantile(grads.float(), 1.0 - self.densify_percent).item()
            max_scene_grad = thr
            max_obj_grad = thr
        scene_densify_mask = grads[~obj_mask] >= max_scene_grad
        obj_densify_mask = grads[obj_mask] >= max_obj_grad

        # split and clone
        self.densify_and_clone(scene_densify_mask, obj_densify_mask)
        scene_densify_mask = torch.cat([scene_densify_mask, torch.zeros((self.get_scene_xyz.shape[0] - scene_densify_mask.shape[0],), dtype=torch.bool, device='cuda')], dim=0)
        obj_densify_mask = torch.cat([obj_densify_mask, torch.zeros((self.get_obj_xyz.shape[0] - obj_densify_mask.shape[0],), dtype=torch.bool, device='cuda')], dim=0)
        self.densify_and_split(scene_densify_mask, obj_densify_mask)

        scene_prune_mask = (self.get_scene_opacity < min_opacity).squeeze()
        obj_prune_mask = (self.get_obj_opacity < min_opacity).squeeze()
        if prune_big_points:
            scene_big_mask = torch.max(self.get_scene_scaling, dim=1).values > self.scene_extent * 0.05
            obj_big_mask = torch.max(self.get_obj_scaling, dim=1).values > self.object_extent * 0.1
            scene_prune_mask = scene_prune_mask | scene_big_mask
            obj_prune_mask = obj_prune_mask | obj_big_mask
        self.prune_points(scene_prune_mask, obj_prune_mask)

        torch.cuda.empty_cache()
        self.set_obj_near_idx()

    def add_densification_stats(self, render_pkg):
        viewspace_point_tensor = render_pkg['viewspace_points']
        update_filter = render_pkg['visibility_filter']
        if self.abs_grad:
            # sum over pixels of |grad|, written by the abs rasterizer into the .z slot
            self.xyz_gradient_accum[update_filter] += viewspace_point_tensor.grad[update_filter, 2:3]
        else:
            self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.radii_accum[update_filter] += render_pkg['radii'][update_filter].float()[..., None]
        self.denom[update_filter] += 1
