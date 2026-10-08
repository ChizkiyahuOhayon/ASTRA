# Copyright 2026 the authors of IDSplat and contributors.
# Copyright 2025 the authors of NeuRAD and contributors.
# Copyright 2024 the authors of NeuRAD and contributors.
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

import math
from dataclasses import dataclass, field
from typing import Dict, List, Type

import torch
from torch import Tensor, nn

from nerfstudio.cameras.camera_utils import matrix_to_rotation_6d, rotation_6d_to_matrix
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.utils import poses as pose_utils
from nerfstudio.utils.poses import interpolate_trajectories_6d, interpolate_velocities
from nerfstudio.viewer.server.viewer_elements import ViewerSlider


class Object:
    def __init__(
        self,
        id: int,
        mask_id: int,
        poses: Tensor,
        size: Tensor,
        valid_timestamps: Tensor,
        linear_velocities_global: Tensor,
        angular_velocities_local: Tensor,
    ):
        self.id = id
        self.mask_id = mask_id
        self.poses = poses
        self.size = size
        self.valid_timestamps = valid_timestamps
        self.linear_velocities_global = linear_velocities_global
        self.angular_velocities_local = angular_velocities_local
        self.all_mask_ids = torch.tensor([mask_id], dtype=torch.int, device=poses.device)

    def __repr__(self):
        return f"Object(id={self.id}, mask_id={self.mask_id})"


@dataclass
class ObjectsManagerConfig(InstantiateConfig):
    """Configuration of dynamic actors."""

    _target: Type = field(default_factory=lambda: ObjectsManager)

    optimize_trajectories: bool = True
    """Whether to optimize the trajectories or not."""


class ObjectsManager(nn.Module):
    config: ObjectsManagerConfig

    def __init__(self, config: ObjectsManagerConfig, unique_timestamps: Tensor, objects: List[Object]):
        super().__init__()
        self.config = config
        self.unique_timestamps = nn.Parameter(unique_timestamps, requires_grad=False)
        self.n_times = len(unique_timestamps)

        param_dict = self.create_param_dict(objects)
        self.param_dict = nn.ParameterDict(param_dict)

        self.actor_editing = {
            "lateral": 0.0,
            "longitudinal": 0.0,
            "rotation": 0.0,
            "index": -1.0,
        }
        self.actor_lateral_shift = ViewerSlider(
            name="Actor lateral shift (m)",
            default_value=self.actor_editing["lateral"],
            min_value=-3.0,
            max_value=3.0,
            step=0.1,
            cb_hook=lambda obj: self.actor_editing.update({"lateral": obj.value}),
        )

        self.actor_longitudinal_shift = ViewerSlider(
            name="Actor longitudinal shift (m)",
            default_value=self.actor_editing["longitudinal"],
            min_value=-3.0,
            max_value=3.0,
            step=0.1,
            cb_hook=lambda obj: self.actor_editing.update({"longitudinal": obj.value}),
        )

        self.actor_rotation_shift = ViewerSlider(
            name="Actor rotation shift (m)",
            default_value=self.actor_editing["rotation"],
            min_value=-3.14,
            max_value=3.14,
            step=0.1,
            cb_hook=lambda obj: self.actor_editing.update({"rotation": obj.value}),
        )

        self.actor_index_to_edit = ViewerSlider(
            name="Actor index to edit",
            default_value=self.actor_editing["index"],
            min_value=-1.0,
            max_value=100,
            step=1.0,
            cb_hook=lambda obj: self.actor_editing.update(
                {"index": min(obj.value, int(self.actor_positions.shape[1]) - 1)}
            ),
        )

    @property
    def actor_ids(self):
        return self.param_dict["actor_ids"]

    @property
    def actor_sizes(self):
        return self.param_dict["actor_sizes"]

    @property
    def actor_positions(self):
        return self.param_dict["actor_positions"]

    @property
    def actor_rotations_6d(self):
        return self.param_dict["actor_rotations_6d"]

    @property
    def actor_present_at_time(self):
        return self.param_dict["actor_present_at_time"]

    @property
    def actor_mask_ids(self):
        return self.param_dict["actor_mask_ids"]

    @property
    def actor_vel_linear(self):
        return self.param_dict["actor_vel_linear"]

    @property
    def actor_vel_angular(self):
        return self.param_dict["actor_vel_angular"]

    def max_id(self) -> int:
        return -1 if self.actor_ids.numel() == 0 else self.actor_ids.max().item()

    def create_param_dict(
        self,
        objects: List[Object],
    ) -> Dict[str, nn.Parameter]:
        device = self.unique_timestamps.device
        actor_ids = torch.tensor([obj.id for obj in objects], device=device, dtype=torch.int)
        actor_poses = (
            torch.stack([obj.poses for obj in objects], dim=1) if len(objects) > 0 else torch.tensor([], device=device)
        )
        actor_sizes = (
            torch.stack([obj.size for obj in objects], dim=0) if len(objects) > 0 else torch.tensor([], device=device)
        )
        actor_valid_timestamps = (
            torch.stack([obj.valid_timestamps for obj in objects], dim=1)
            if len(objects) > 0
            else torch.tensor([], device=device, dtype=torch.bool)
        )
        actor_mask_ids = torch.tensor([obj.mask_id for obj in objects], device=device, dtype=torch.int)
        actor_vel_linear = (
            torch.stack([obj.linear_velocities_global for obj in objects], dim=1)
            if len(objects) > 0
            else torch.tensor([], device=device)
        )
        actor_vel_angular = (
            torch.stack([obj.angular_velocities_local for obj in objects], dim=1)
            if len(objects) > 0
            else torch.tensor([], device=device)
        )

        param_dict = {
            "actor_positions": nn.Parameter(actor_poses[..., :3, 3])
            if len(objects) > 0
            else nn.Parameter(torch.tensor([], device=device)),
            "actor_rotations_6d": nn.Parameter(matrix_to_rotation_6d(actor_poses[..., :3, :3]))
            if len(objects) > 0
            else nn.Parameter(torch.tensor([], device=device)),
            "actor_ids": nn.Parameter(actor_ids, requires_grad=False),
            "actor_sizes": nn.Parameter(actor_sizes, requires_grad=False),  # wlh
            "actor_present_at_time": nn.Parameter(actor_valid_timestamps, requires_grad=False),
            "actor_mask_ids": nn.Parameter(actor_mask_ids, requires_grad=False),
            "actor_vel_linear": nn.Parameter(actor_vel_linear, requires_grad=True),
            "actor_vel_angular": nn.Parameter(actor_vel_angular, requires_grad=True),
        }
        return param_dict

    def get_poses_3x4(self):
        rotations = rotation_6d_to_matrix(self.actor_rotations_6d)
        return torch.cat([rotations, self.actor_positions.unsqueeze(-1)], dim=-1)

    def edit_boxes2world(self, boxes2world: Tensor):
        if (
            abs(self.actor_editing["longitudinal"]) == 0.0
            and abs(self.actor_editing["lateral"]) == 0.0
            and abs(self.actor_editing["rotation"]) == 0.0
        ):
            return boxes2world
        with torch.no_grad():
            if self.actor_editing["index"] == -1.0:
                indices = self.actor_ids.unique().int()
            else:
                indices = torch.tensor([self.actor_editing["index"]], device=boxes2world.device, dtype=torch.int)

            if abs(self.actor_editing["longitudinal"]) > 0.0 or abs(self.actor_editing["lateral"]) > 0.0:
                boxes2world[:, indices, :, 3] = boxes2world[:, indices] @ torch.tensor(
                    [self.actor_editing["lateral"], self.actor_editing["longitudinal"], 0.0, 1.0],
                    device=boxes2world.device,
                )

            if abs(self.actor_editing["rotation"]) > 0.0:
                rotation_yaw = torch.tensor(
                    [
                        [math.cos(self.actor_editing["rotation"]), -math.sin(self.actor_editing["rotation"]), 0.0],
                        [math.sin(self.actor_editing["rotation"]), math.cos(self.actor_editing["rotation"]), 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    device=boxes2world.device,
                )
                boxes2world[:, indices, :3, :3] = rotation_yaw @ boxes2world[:, indices, :3, :3]
        return boxes2world

    def get_boxes2world(self, query_times: Tensor):
        poses, *extra = interpolate_trajectories_6d(
            torch.cat([self.actor_rotations_6d, self.actor_positions], dim=-1),
            self.unique_timestamps,
            query_times,
            self.actor_present_at_time.bool(),
            flatten=False,
        )
        boxes2world = torch.cat([rotation_6d_to_matrix(poses[..., :6]), poses[..., 6:].unsqueeze(-1)], dim=-1)

        if not self.training:
            boxes2world = self.edit_boxes2world(boxes2world)

        boxes2world = pose_utils.to4x4(boxes2world)
        return boxes2world, *extra

    def get_velocities(self, query_times: Tensor):
        return interpolate_velocities(
            torch.cat([self.actor_vel_linear, self.actor_vel_angular], dim=-1),
            self.unique_timestamps,
            query_times,
        )

    def get_param_groups(self, param_groups: Dict):
        if self.config.optimize_trajectories:
            for k in self.param_dict:
                if k in ["actor_positions", "actor_rotations_6d", "actor_vel_linear", "actor_vel_angular"]:
                    param_groups[k] = param_groups.get(k, []) + [self.param_dict[k]]
