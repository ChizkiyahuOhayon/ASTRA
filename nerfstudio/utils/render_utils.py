# Copyright 2026 the authors of IDSplat and contributors.
# Copyright 2025 the authors of IDSplat and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.poses import to4x4


def resize_image(image: torch.Tensor, d: int):
    """
    Downscale images using the same 'area' method in opencv

    :param image shape [H, W, C]
    :param d downscale factor (must be 2, 4, 8, etc.)

    return downscaled image in shape [H//d, W//d, C]
    """
    import torch.nn.functional as tf

    image = image.to(torch.float32)
    weight = (1.0 / (d * d)) * torch.ones((1, 1, d, d), dtype=torch.float32, device=image.device)
    return tf.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d).squeeze(1).permute(1, 2, 0)


def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    flip = torch.ones(1, 1, 3, device=R.device, dtype=R.dtype)
    flip[..., 1:] = -1
    R = R * flip
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.cat([R_inv, T_inv], dim=2)
    viewmat = to4x4(viewmat)
    return viewmat


def get_ray_dirs_pinhole(cameras: Cameras, width: int, height: int, c2w: torch.Tensor):
    ys = (torch.arange(height, device=cameras.device, dtype=torch.float32) + (0.5 - cameras.cy[0, 0])) / cameras.fy[
        0, 0
    ]
    xs = (torch.arange(width, device=cameras.device, dtype=torch.float32) + (0.5 - cameras.cx[0, 0])) / cameras.fx[0, 0]
    image_coords = torch.meshgrid(ys, xs, indexing="ij")
    # flip y and z to align with nerfstudio convention
    directions = torch.stack(
        [image_coords[1], -image_coords[0], -torch.ones_like(image_coords[0])], dim=-1
    )  # (h, w, 3)
    directions = directions.view(-1, 3)
    directions = torch.matmul(directions, c2w[0, :3, :3].transpose(0, 1))
    directions = directions / directions.norm(dim=-1, keepdim=True)
    directions = directions.view(height, width, 3)

    return directions
