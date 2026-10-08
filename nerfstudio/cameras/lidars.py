# Copyright 2026 the authors of IDSplat and contributors.
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

"""
Camera Models
"""

import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from jaxtyping import Float, Int
from matplotlib import pyplot as plt
from torch import Tensor
from torch.nn import Parameter

import nerfstudio.utils.math
import nerfstudio.utils.poses as pose_utils
from nerfstudio.cameras import camera_utils
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.lidar_elevation_mappings import (
    PANDAR64_ELEVATION_MAPPING,
    VELODYNE_128_ELEVATION_MAPPING,
    VELODYNE_HDL32E_ELEVATION_MAPPING,
    VELODYNE_VLP32C_ELEVATION_MAPPING,
    WOD64_ELEVATION_MAPPING,
    VELODYNE64E_ELEVATION_MAPPING,
)
from nerfstudio.utils.misc import strtobool, torch_compile
from nerfstudio.utils.tensor_dataclass import TensorDataclass

TORCH_DEVICE = Union[torch.device, str]  # pylint: disable=invalid-name

HORIZONTAL_BEAM_DIVERGENCE = 3.0e-3  # radians, or meters at a distance of 1m
VERTICAL_BEAM_DIVERGENCE = 1.5e-3  # radians, or meters at a distance of 1m


class LidarType(Enum):
    """Supported Lidar Types."""

    VELODYNE16 = auto()
    VELODYNE_HDL32E = auto()
    VELODYNE_VLP32C = auto()
    VELODYNE64E = auto()
    VELODYNE128 = auto()
    PANDAR64 = auto()
    WOD64 = auto()


LIDAR_MODEL_TO_TYPE = {
    "VELODYNE16": LidarType.VELODYNE16,
    "VELODYNE_HDL32E": LidarType.VELODYNE_HDL32E,
    "VELODYNE_VLP32C": LidarType.VELODYNE_VLP32C,
    "VELODYNE64E": LidarType.VELODYNE64E,
    "VELODYNE128": LidarType.VELODYNE128,
    "PANDAR64": LidarType.PANDAR64,
    "WOD64": LidarType.WOD64,
}


@dataclass(init=False)
class Lidars(TensorDataclass):
    """Dataparser outputs for the lidar dataset and the ray generator.

    If a single value is provided, it is broadcasted to all lidars.

    Args:
        lidar_to_worlds: Lidar to world matrices. Tensor of per-image l2w matrices, in [R | t] format
        lidar_type: Type of lidar sensor. This will be an int corresponding to the LidarType enum.
        assume_ego_compensated: Whether or not to assume that points are ego-compensated.
        times: Timestamps for each lidar
        n_points: Number of points in each lidar
        metadata: Additional metadata or data needed for interpolation, will mimic shape of the lidars
            and will be broadcasted to the rays generated from any derivative RaySamples we create with this
    """

    lidar_to_worlds: Float[Tensor, "*num_lidars 3 4"]
    lidar_type: Float[Tensor, "*num_lidars 1"]
    times: Optional[Int[Tensor, "*num_lidars 1"]]
    metadata: Optional[Dict]
    horizontal_beam_divergence: Float[Tensor, "*num_lidars 1"]
    vertical_beam_divergence: Float[Tensor, "*num_lidars 1"]
    azimuths: Optional[Float[Tensor, "*num_lidars n_azimuth_channels 1"]]
    elevations: Optional[Float[Tensor, "*num_lidars n_elevation_channels 1"]]

    def __init__(
        self,
        lidar_to_worlds: Float[Tensor, "*batch_l2ws 3 4"],
        lidar_type: Optional[
            Union[
                Int[Tensor, "*batch_lidar_types 1"],
                int,
                List[LidarType],
                LidarType,
            ]
        ] = LidarType.VELODYNE_VLP32C,
        assume_ego_compensated: bool = True,
        times: Optional[Float[Tensor, "*num_lidars"]] = None,
        metadata: Optional[Dict] = None,
        horizontal_beam_divergence: Optional[Union[Float, Float[Tensor, "*num_lidars 1"]]] = None,
        vertical_beam_divergence: Optional[Union[Float, Float[Tensor, "*num_lidars 1"]]] = None,
        valid_lidar_distance_threshold: float = 1e3,
        azimuths: Optional[Float[Tensor, "*num_lidars n_azimuth_channels 1"]] = None,
        elevations: Optional[Float[Tensor, "*num_lidars n_elevation_channels 1"]] = None,
    ) -> None:
        """Initializes the Lidars object.

        Note on Input Tensor Dimensions: All of these tensors have items of dimensions Shaped[Tensor, "3 4"]
        (in the case of the c2w matrices), Shaped[Tensor, "6"] (in the case of distortion params), or
        Shaped[Tensor, "1"] (in the case of the rest of the elements). The dimensions before that are
        considered the batch dimension of that tensor (batch_c2ws, batch_fxs, etc.). We will broadcast
        all the tensors to be the same batch dimension. This means you can use any combination of the
        input types in the function signature and it won't break. Your batch size for all tensors
        must be broadcastable to the same size, and the resulting number of batch dimensions will be
        the batch dimension with the largest number of dimensions.
        """

        # This will notify the tensordataclass that we have a field with more than 1 dimension
        self._field_custom_dimensions = {"lidar_to_worlds": 2, "azimuths": 2, "elevations": 2}

        self.lidar_to_worlds = lidar_to_worlds

        # @dataclass's post_init will take care of broadcasting
        self.lidar_type = self._init_get_lidar_type(lidar_type)
        self.times = self._init_get_times(times)

        self.metadata = metadata

        self.horizontal_beam_divergence = self._init_get_beam_divergence(
            horizontal_beam_divergence, HORIZONTAL_BEAM_DIVERGENCE
        )
        self.vertical_beam_divergence = self._init_get_beam_divergence(
            vertical_beam_divergence, VERTICAL_BEAM_DIVERGENCE
        )
        self.azimuths = self._init_get_azimuths(azimuths)
        self.elevations = self._init_get_elevations(elevations)

        self.__post_init__()  # This will do the dataclass post_init and broadcast all the tensors

        self._use_nerfacc = strtobool(os.environ.get("INTERSECT_WITH_NERFACC", "TRUE"))
        self.assume_ego_compensated = assume_ego_compensated

        self.valid_lidar_distance_threshold = valid_lidar_distance_threshold

    def _init_get_beam_divergence(
        self,
        beam_divergence: Union[None, Float, Float[Tensor, "*num_lidars 1"]],
        default: float,
    ) -> Float[Tensor, "*num_lidars 1"]:
        if beam_divergence is None:
            beam_divergence = torch.ones_like(self.lidar_type) * default
        elif isinstance(beam_divergence, float):
            beam_divergence = torch.ones_like(self.lidar_type) * beam_divergence
        elif isinstance(beam_divergence, torch.Tensor):
            if beam_divergence.ndim == 0 or beam_divergence.shape[-1] != 1:
                beam_divergence = beam_divergence.unsqueeze(-1)
        else:
            raise ValueError(f"beam_divergence must be None, float, or tensor, got {type(beam_divergence)}")

        return beam_divergence

    def _init_get_lidar_type(
        self,
        lidar_type: Union[
            Int[Tensor, "*batch_lidar_types 1"], Int[Tensor, "*batch_lidar_types"], int, List[LidarType], LidarType
        ],
    ) -> Int[Tensor, "*num_lidars 1"]:
        """
        Parses the __init__() argument lidar_type

        Lidar Type Calculation:
        If LidarType, convert to int and then to tensor, then broadcast to all lidars
        If List of LidarTypes, convert to ints and then to tensor, then broadcast to all lidars
        If int, first go to tensor and then broadcast to all lidars
        If tensor, broadcast to all lidars

        Args:
            lidar_type: lidar_type argument from __init__()
        """
        if isinstance(lidar_type, LidarType):
            lidar_type = torch.tensor([lidar_type.value], device=self.device)
        elif isinstance(lidar_type, List) and isinstance(lidar_type[0], LidarType):
            lidar_type = torch.tensor([[c.value] for c in lidar_type], device=self.device)
        elif isinstance(lidar_type, int):
            lidar_type = torch.tensor([lidar_type], device=self.device)
        elif isinstance(lidar_type, torch.Tensor):
            assert not torch.is_floating_point(
                lidar_type
            ), f"lidar_type tensor must be of type int, not: {lidar_type.dtype}"
            lidar_type = lidar_type.to(self.device)
            if lidar_type.ndim == 0 or lidar_type.shape[-1] != 1:
                lidar_type = lidar_type.unsqueeze(-1)
            # assert torch.all(
            #     lidar_type.view(-1)[0] == lidar_type
            # ), "Batched lidars of different lidar_types will be allowed in the future."
        else:
            raise ValueError(
                'Invalid lidar_type. Must be LidarType, List[LidarType], int, or torch.Tensor["num_lidars"]. \
                    Received: '
                + str(type(lidar_type))
            )
        return lidar_type

    def _init_get_times(self, times: Union[None, torch.Tensor]) -> Union[None, torch.Tensor]:
        if times is None:
            times = None
        elif isinstance(times, torch.Tensor):
            if times.ndim == 0 or times.shape[-1] != 1:
                times = times.unsqueeze(-1).to(self.device)
        else:
            raise ValueError(f"times must be None or a tensor, got {type(times)}")

        return times

    def _init_get_azimuths(self, azimuths: Union[None, torch.Tensor]) -> Union[None, torch.Tensor]:
        if azimuths is None:
            azimuths = None
        elif isinstance(azimuths, torch.Tensor):
            if azimuths.ndim == 0 or azimuths.shape[-1] != 1:
                azimuths = azimuths.unsqueeze(-1).to(self.device)
        else:
            raise ValueError(f"azimuths must be None or a tensor, got {type(azimuths)}")

        return azimuths

    def _init_get_elevations(self, elevations: Union[None, torch.Tensor]) -> Union[None, torch.Tensor]:
        if elevations is None:
            elevations = None
        elif isinstance(elevations, torch.Tensor):
            if elevations.ndim == 0 or elevations.shape[-1] != 1:
                elevations = elevations.unsqueeze(-1).to(self.device)
        else:
            raise ValueError(f"elevations must be None or a tensor, got {type(elevations)}")

        return elevations

    @property
    def device(self) -> TORCH_DEVICE:
        """Returns the device that the camera is on."""
        return self.lidar_to_worlds.device

    @property
    def is_jagged(self) -> bool:
        """
        Returns whether or not the lidars are "jagged" (i.e. number of point vary between lidars)
        TODO: base assumption is that all lidars have different number of points
        """
        return all(self.n_points.view(-1)[0] == self.n_points)

    @property
    def width(self) -> int:
        """Returns the width of the lidar."""
        return self.azimuths.shape[1] if self.azimuths is not None else 0

    @property
    def height(self) -> int:
        """Returns the height of the lidar."""
        return self.elevations.shape[1] if self.elevations is not None else 0


def transform_points(points, transform):
    """Transforms points by a transform."""
    points = points.clone()
    rotations = transform[..., :3, :3]
    translations = transform[..., :3, 3]
    points[:, :3] = points[:, :3] @ rotations.swapaxes(-2, -1) + translations
    return points


@torch_compile(dynamic=True, mode="reduce-overhead", backend="eager")
def transform_points_pairwise(points, transforms, with_translation=True):
    """Transform points by a pairwise transform."""
    # points: (*, 3)
    # transforms: (*, 4, 4)
    # return: (*, 3)
    rotations = transforms[..., :3, :3]
    translations = transforms[..., :3, 3]

    # Perform batch matrix multiplication
    rotated_points = torch.bmm(rotations.reshape(-1, 3, 3), points.reshape(-1, 3, 1)).reshape(*points.shape[:-1], 3)

    if with_translation:
        return rotated_points + translations
    else:
        return rotated_points


@torch_compile(dynamic=True, mode="reduce-overhead", backend="eager")
def transform_points_by_batch(points, transforms, with_translation=True):
    """Transform points by a batch of transforms."""
    # points: (N, 3)
    # transforms: (B, 4, 4)
    # return: (B, N, 3)
    batch_size = transforms.shape[0]
    points = points.clone().unsqueeze(0).repeat(batch_size, 1, 1)
    rotations = transforms[:, :3, :3]
    translations = transforms[:, :3, 3]

    if with_translation:
        return points @ rotations.permute(0, 2, 1) + translations.unsqueeze(1)
    else:
        return points @ rotations.permute(0, 2, 1)


def intensity_to_rgb(intensities: np.ndarray) -> np.ndarray:  # N -> N x 3
    """Converts intensities to RGB values."""
    # use log-scale for better visualization
    log_intensities = np.log(1 + intensities * 255) / np.log(256)
    return plt.cm.inferno(log_intensities)[:, :3]


def heights_to_rgb(heights: np.ndarray) -> np.ndarray:
    """Converts heights to RGB values."""
    return plt.cm.viridis(heights)[:, :3]


def get_lidar_elevation_mapping(lidar_type: LidarType) -> dict:
    if lidar_type == LidarType.VELODYNE16:
        raise NotImplementedError("No elevation mapping for Velodyne 16")
    elif lidar_type == LidarType.VELODYNE_HDL32E:
        return VELODYNE_HDL32E_ELEVATION_MAPPING
    elif lidar_type == LidarType.VELODYNE_VLP32C:
        return VELODYNE_VLP32C_ELEVATION_MAPPING
    elif lidar_type == LidarType.VELODYNE64E:
        return VELODYNE64E_ELEVATION_MAPPING
    elif lidar_type == LidarType.VELODYNE128:
        return VELODYNE_128_ELEVATION_MAPPING
    elif lidar_type == LidarType.PANDAR64:
        return PANDAR64_ELEVATION_MAPPING
    elif lidar_type == LidarType.WOD64:
        return WOD64_ELEVATION_MAPPING
    else:
        raise ValueError(f"Invalid lidar type: {lidar_type}")


def get_lidar_azimuth_resolution(lidar_type: LidarType) -> float:
    if lidar_type == LidarType.VELODYNE16:
        return 0.2
    elif lidar_type == LidarType.VELODYNE_HDL32E:
        return 1 / 3.0
    elif lidar_type == LidarType.VELODYNE_VLP32C:
        return 0.2
    elif lidar_type == LidarType.VELODYNE64E:
        return 0.09
    elif lidar_type == LidarType.VELODYNE128:
        return 0.2
    elif lidar_type == LidarType.PANDAR64:
        return 0.2
    elif lidar_type == LidarType.WOD64:
        return 360 / 2650.0
    else:
        raise ValueError(f"Invalid lidar type: {lidar_type}")


def get_lidar_relovution_time(lidar_type: LidarType) -> float:
    if lidar_type == LidarType.VELODYNE16:
        return 0.1
    elif lidar_type == LidarType.VELODYNE_HDL32E:
        return 0.1
    elif lidar_type == LidarType.VELODYNE_VLP32C:
        return 0.1
    elif lidar_type == LidarType.VELODYNE64E:
        return 0.1
    elif lidar_type == LidarType.VELODYNE128:
        return 0.1
    elif lidar_type == LidarType.PANDAR64:
        return 0.1
    elif lidar_type == LidarType.WOD64:
        return 0.1
    else:
        raise ValueError(f"Invalid lidar type: {lidar_type}")
