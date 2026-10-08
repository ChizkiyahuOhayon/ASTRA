# Copyright 2026 the authors of IDSplat and contributors.
# Copyright 2025 the authors of NeuRAD and contributors.
# Copyright 2024 the authors of NeuRAD and contributors.
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

import base64
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from jaxtyping import Float, Int, Shaped
from torch import Tensor
from torch.nn import Parameter

import nerfstudio.utils.math
import nerfstudio.utils.poses as pose_utils
from nerfstudio.cameras import camera_utils
from nerfstudio.data.scene_box import OrientedBox, SceneBox
from nerfstudio.utils.tensor_dataclass import TensorDataclass

TORCH_DEVICE = Union[torch.device, str]


class CameraType(Enum):
    """Supported camera types."""

    PERSPECTIVE = auto()
    FISHEYE = auto()
    EQUIRECTANGULAR = auto()
    OMNIDIRECTIONALSTEREO_L = auto()
    OMNIDIRECTIONALSTEREO_R = auto()
    VR180_L = auto()
    VR180_R = auto()
    ORTHOPHOTO = auto()
    FISHEYE624 = auto()


CAMERA_MODEL_TO_TYPE = {
    "SIMPLE_PINHOLE": CameraType.PERSPECTIVE,
    "PINHOLE": CameraType.PERSPECTIVE,
    "SIMPLE_RADIAL": CameraType.PERSPECTIVE,
    "RADIAL": CameraType.PERSPECTIVE,
    "OPENCV": CameraType.PERSPECTIVE,
    "OPENCV_FISHEYE": CameraType.FISHEYE,
    "EQUIRECTANGULAR": CameraType.EQUIRECTANGULAR,
    "OMNIDIRECTIONALSTEREO_L": CameraType.OMNIDIRECTIONALSTEREO_L,
    "OMNIDIRECTIONALSTEREO_R": CameraType.OMNIDIRECTIONALSTEREO_R,
    "VR180_L": CameraType.VR180_L,
    "VR180_R": CameraType.VR180_R,
    "ORTHOPHOTO": CameraType.ORTHOPHOTO,
    "FISHEYE624": CameraType.FISHEYE624,
}


class CameraRollingShutterReadOutDirection(Enum):
    UNKNOWN = 0
    TOP_TO_BOTTOM = 1
    LEFT_TO_RIGHT = 2
    BOTTOM_TO_TOP = 3
    RIGHT_TO_LEFT = 4
    GLOBAL_SHUTTER = 5


@dataclass(init=False)
class Cameras(TensorDataclass):
    """Dataparser outputs for the image dataset and the ray generator.

    If a single value is provided, it is broadcasted to all cameras.

    Args:
        camera_to_worlds: Camera to world matrices. Tensor of per-image c2w matrices, in [R | t] format
        fx: Focal length x
        fy: Focal length y
        cx: Principal point x
        cy: Principal point y
        width: Image width
        height: Image height
        distortion_params: distortion coefficients (OpenCV 6 radial or 6-2-4 radial, tangential, thin-prism for Fisheye624)
        camera_type: Type of camera model. This will be an int corresponding to the CameraType enum.
        times: Timestamps for each camera
        metadata: Additional metadata or data needed for interpolation, will mimic shape of the cameras
            and will be broadcasted to the rays generated from any derivative RaySamples we create with this
    """

    camera_to_worlds: Float[Tensor, "*num_cameras 3 4"]
    fx: Float[Tensor, "*num_cameras 1"]
    fy: Float[Tensor, "*num_cameras 1"]
    cx: Float[Tensor, "*num_cameras 1"]
    cy: Float[Tensor, "*num_cameras 1"]
    width: Shaped[Tensor, "*num_cameras 1"]
    height: Shaped[Tensor, "*num_cameras 1"]
    distortion_params: Optional[Float[Tensor, "*num_cameras 6"]]
    camera_type: Int[Tensor, "*num_cameras 1"]
    times: Optional[Float[Tensor, "num_cameras 1"]]
    metadata: Optional[Dict]

    def __init__(
        self,
        camera_to_worlds: Float[Tensor, "*batch_c2ws 3 4"],
        fx: Union[Float[Tensor, "*batch_fxs 1"], float],
        fy: Union[Float[Tensor, "*batch_fys 1"], float],
        cx: Union[Float[Tensor, "*batch_cxs 1"], float],
        cy: Union[Float[Tensor, "*batch_cys 1"], float],
        width: Optional[Union[Shaped[Tensor, "*batch_ws 1"], int]] = None,
        height: Optional[Union[Shaped[Tensor, "*batch_hs 1"], int]] = None,
        distortion_params: Optional[Float[Tensor, "*batch_dist_params 6"]] = None,
        camera_type: Union[
            Int[Tensor, "*batch_cam_types 1"],
            int,
            List[CameraType],
            CameraType,
        ] = CameraType.PERSPECTIVE,
        times: Optional[Float[Tensor, "num_cameras"]] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Initializes the Cameras object.

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
        self._field_custom_dimensions = {"camera_to_worlds": 2}

        self.camera_to_worlds = camera_to_worlds

        # fx fy calculation
        self.fx = self._init_get_fc_xy(fx, "fx")  # @dataclass's post_init will take care of broadcasting
        self.fy = self._init_get_fc_xy(fy, "fy")  # @dataclass's post_init will take care of broadcasting

        # cx cy calculation
        self.cx = self._init_get_fc_xy(cx, "cx")  # @dataclass's post_init will take care of broadcasting
        self.cy = self._init_get_fc_xy(cy, "cy")  # @dataclass's post_init will take care of broadcasting

        # Distortion Params Calculation:
        self.distortion_params = distortion_params  # @dataclass's post_init will take care of broadcasting

        # @dataclass's post_init will take care of broadcasting
        self.height = self._init_get_height_width(height, self.cy)
        self.width = self._init_get_height_width(width, self.cx)
        self.camera_type = self._init_get_camera_type(camera_type)
        self.times = self._init_get_times(times)

        self.metadata = metadata

        self.__post_init__()  # This will do the dataclass post_init and broadcast all the tensors

    def _init_get_fc_xy(self, fc_xy: Union[float, torch.Tensor], name: str) -> torch.Tensor:
        """
        Parses the input focal length / principle point x or y and returns a tensor of the correct shape

        Only needs to make sure that we a 1 in the last dimension if it is a tensor. If it is a float, we
        just need to make it into a tensor and it will be broadcasted later in the __post_init__ function.

        Args:
            fc_xy: The focal length / principle point x or y
            name: The name of the variable. Used for error messages
        """
        if isinstance(fc_xy, float):
            fc_xy = torch.Tensor([fc_xy], device=self.device)
        elif isinstance(fc_xy, torch.Tensor):
            if fc_xy.ndim == 0 or fc_xy.shape[-1] != 1:
                fc_xy = fc_xy.unsqueeze(-1)
            fc_xy = fc_xy.to(self.device)
        else:
            raise ValueError(f"{name} must be a float or tensor, got {type(fc_xy)}")
        return fc_xy

    def _init_get_camera_type(
        self,
        camera_type: Union[
            Int[Tensor, "*batch_cam_types 1"], Int[Tensor, "*batch_cam_types"], int, List[CameraType], CameraType
        ],
    ) -> Int[Tensor, "*num_cameras 1"]:
        """
        Parses the __init__() argument camera_type

        Camera Type Calculation:
        If CameraType, convert to int and then to tensor, then broadcast to all cameras
        If List of CameraTypes, convert to ints and then to tensor, then broadcast to all cameras
        If int, first go to tensor and then broadcast to all cameras
        If tensor, broadcast to all cameras

        Args:
            camera_type: camera_type argument from __init__()
        """
        if isinstance(camera_type, CameraType):
            camera_type = torch.tensor([camera_type.value], device=self.device)
        elif isinstance(camera_type, List) and isinstance(camera_type[0], CameraType):
            camera_type = torch.tensor([[c.value] for c in camera_type], device=self.device)
        elif isinstance(camera_type, int):
            camera_type = torch.tensor([camera_type], device=self.device)
        elif isinstance(camera_type, torch.Tensor):
            assert not torch.is_floating_point(camera_type), (
                f"camera_type tensor must be of type int, not: {camera_type.dtype}"
            )
            camera_type = camera_type.to(self.device)
            if camera_type.ndim == 0 or camera_type.shape[-1] != 1:
                camera_type = camera_type.unsqueeze(-1)
            # assert torch.all(
            #     camera_type.view(-1)[0] == camera_type
            # ), "Batched cameras of different camera_types will be allowed in the future."
        else:
            raise ValueError(
                'Invalid camera_type. Must be CameraType, List[CameraType], int, or torch.Tensor["num_cameras"]. \
                    Received: '
                + str(type(camera_type))
            )
        return camera_type

    def _init_get_height_width(
        self,
        h_w: Union[Shaped[Tensor, "*batch_hws 1"], Shaped[Tensor, "*batch_hws"], int, None],
        c_x_y: Shaped[Tensor, "*batch_cxys"],
    ) -> Shaped[Tensor, "*num_cameras 1"]:
        """
        Parses the __init__() argument for height or width

        Height/Width Calculation:
        If int, first go to tensor and then broadcast to all cameras
        If tensor, broadcast to all cameras
        If none, use cx or cy * 2
        Else raise error

        Args:
            h_w: height or width argument from __init__()
            c_x_y: cx or cy for when h_w == None
        """
        if isinstance(h_w, int):
            h_w = torch.as_tensor([h_w]).to(torch.int64).to(self.device)
        elif isinstance(h_w, torch.Tensor):
            assert not torch.is_floating_point(h_w), f"height and width tensor must be of type int, not: {h_w.dtype}"
            h_w = h_w.to(torch.int64).to(self.device)
            if h_w.ndim == 0 or h_w.shape[-1] != 1:
                h_w = h_w.unsqueeze(-1)
        # assert torch.all(h_w == h_w.view(-1)[0]), "Batched cameras of different h, w will be allowed in the future."
        elif h_w is None:
            h_w = torch.as_tensor((c_x_y * 2)).to(torch.int64).to(self.device)
        else:
            raise ValueError("Height must be an int, tensor, or None, received: " + str(type(h_w)))
        return h_w

    def _init_get_times(self, times: Union[None, torch.Tensor]) -> Union[None, torch.Tensor]:
        if times is None:
            times = None
        elif isinstance(times, torch.Tensor):
            if times.ndim == 0 or times.shape[-1] != 1:
                times = times.unsqueeze(-1).to(self.device)
        else:
            raise ValueError(f"times must be None or a tensor, got {type(times)}")

        return times

    @property
    def device(self) -> TORCH_DEVICE:
        """Returns the device that the camera is on."""
        return self.camera_to_worlds.device

    @property
    def image_height(self) -> Shaped[Tensor, "*num_cameras 1"]:
        """Returns the height of the images."""
        return self.height

    @property
    def image_width(self) -> Shaped[Tensor, "*num_cameras 1"]:
        """Returns the height of the images."""
        return self.width

    @property
    def is_jagged(self) -> bool:
        """
        Returns whether or not the cameras are "jagged" (i.e. the height and widths are different, meaning that
        you cannot concatenate the image coordinate maps together)
        """
        h_jagged = not torch.all(self.height == self.height.view(-1)[0])
        w_jagged = not torch.all(self.width == self.width.view(-1)[0])
        return h_jagged or w_jagged

    def get_image_coords(
        self, pixel_offset: float = 0.5, index: Optional[Tuple] = None
    ) -> Float[Tensor, "height width 2"]:
        """This gets the image coordinates of one of the cameras in this object.

        If no index is specified, it will return the maximum possible sized height / width image coordinate map,
        by looking at the maximum height and width of all the cameras in this object.

        Args:
            pixel_offset: Offset for each pixel. Defaults to center of pixel (0.5)
            index: Tuple of indices into the batch dimensions of the camera. Defaults to None, which returns the 0th
                flattened camera

        Returns:
            Grid of image coordinates.
        """
        if index is None:
            image_height = torch.max(self.image_height.view(-1)).item()
            image_width = torch.max(self.image_width.view(-1)).item()
            image_coords = torch.meshgrid(
                torch.arange(image_height, device=self.device),
                torch.arange(image_width, device=self.device),
                indexing="ij",
            )
            image_coords = torch.stack(image_coords, dim=-1) + pixel_offset  # stored as (y, x) coordinates
        else:
            image_height = self.image_height[index].item()
            image_width = self.image_width[index].item()
            image_coords = torch.meshgrid(
                torch.arange(image_height, device=self.device),
                torch.arange(image_width, device=self.device),
                indexing="ij",
            )
            image_coords = torch.stack(image_coords, dim=-1) + pixel_offset  # stored as (y, x) coordinates
        return image_coords

    def to_json(
        self, camera_idx: int, image: Optional[Float[Tensor, "height width 2"]] = None, max_size: Optional[int] = None
    ) -> Dict:
        """Convert a camera to a json dictionary."""
        flattened = self.flatten()
        times = flattened[camera_idx].times
        if times is not None:
            times = times.item()
        json_ = {
            "type": "PinholeCamera",
            "cx": flattened[camera_idx].cx.item(),
            "cy": flattened[camera_idx].cy.item(),
            "fx": flattened[camera_idx].fx.item(),
            "fy": flattened[camera_idx].fy.item(),
            "camera_to_world": self.camera_to_worlds[camera_idx].tolist(),
            "camera_index": camera_idx,
            "times": times,
        }
        if image is not None:
            image_uint8 = (image * 255).detach().type(torch.uint8)
            if max_size is not None:
                image_uint8 = image_uint8.permute(2, 0, 1)
                import torchvision.transforms.functional as TF
                image_uint8 = TF.resize(image_uint8, max_size, antialias=None)
                image_uint8 = image_uint8.permute(1, 2, 0)
            image_uint8 = image_uint8.cpu().numpy()
            data = cv2.imencode(".jpg", image_uint8)[1].tobytes()
            json_["image"] = str("data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"))
        return json_

    def get_intrinsics_matrices(self) -> Float[Tensor, "*num_cameras 3 3"]:
        """Returns the intrinsic matrices for each camera."""
        K = torch.zeros((*self.shape, 3, 3), dtype=torch.float32, device=self.device)
        K[..., 0, 0] = self.fx
        K[..., 1, 1] = self.fy
        K[..., 0, 2] = self.cx
        K[..., 1, 2] = self.cy
        K[..., 2, 2] = 1.0
        return K

    def rescale_output_resolution(
        self, scaling_factor: Union[Shaped[Tensor, "*num_cameras"], Shaped[Tensor, "*num_cameras 1"], float, int]
    ) -> None:
        """Rescale the output resolution of the cameras."""
        if isinstance(scaling_factor, (float, int)):
            scaling_factor = torch.full((1,), scaling_factor, device=self.device).broadcast_to((self.cx.shape))
        elif isinstance(scaling_factor, torch.Tensor) and scaling_factor.shape == self.shape:
            scaling_factor = scaling_factor.unsqueeze(-1)
        elif isinstance(scaling_factor, torch.Tensor) and scaling_factor.shape == (*self.shape, 1):
            pass
        else:
            raise ValueError(
                f"Scaling factor must be a float, int, or a tensor of shape {self.shape} or {(*self.shape, 1)}."
            )
        self.fx = self.fx * scaling_factor
        self.fy = self.fy * scaling_factor
        self.cx = self.cx * scaling_factor
        self.cy = self.cy * scaling_factor
        self.height = (self.height * scaling_factor).to(torch.int64)
        self.width = (self.width * scaling_factor).to(torch.int64)


def undistort_image(
    camera: Cameras, distortion_params: np.ndarray, data: dict, image: np.ndarray, K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Optional[torch.Tensor]]:
    mask = None
    if camera.camera_type.item() == CameraType.PERSPECTIVE.value:
        distortion_params = np.array(
            [
                distortion_params[0],
                distortion_params[1],
                distortion_params[4],
                distortion_params[5],
                distortion_params[2],
                distortion_params[3],
                0,
                0,
            ]
        )
        if np.any(distortion_params):
            newK, roi = cv2.getOptimalNewCameraMatrix(K, distortion_params, (image.shape[1], image.shape[0]), 0)
            image = cv2.undistort(image, K, distortion_params, None, newK)  # type: ignore
        else:
            newK = K
            roi = 0, 0, image.shape[1], image.shape[0]
        # crop the image and update the intrinsics accordingly
        x, y, w, h = roi
        image = image[y : y + h, x : x + w]
        if "mask" in data:
            mask = data["mask"].numpy()
            mask = mask.astype(np.uint8) * 255
            if np.any(distortion_params):
                mask = cv2.undistort(mask, K, distortion_params, None, newK)  # type: ignore
            mask = mask[y : y + h, x : x + w]
            mask = torch.from_numpy(mask).bool()
            if len(mask.shape) == 2:
                mask = mask[:, :, None]
        K = newK

    elif camera.camera_type.item() == CameraType.FISHEYE.value:
        distortion_params = np.array(
            [distortion_params[0], distortion_params[1], distortion_params[2], distortion_params[3]]
        )
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, distortion_params, (image.shape[1], image.shape[0]), np.eye(3), balance=0
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, distortion_params, np.eye(3), newK, (image.shape[1], image.shape[0]), cv2.CV_32FC1
        )
        # and then remap:
        image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
        if "mask" in data:
            mask = data["mask"].numpy()
            mask = mask.astype(np.uint8) * 255
            mask = cv2.fisheye.undistortImage(mask, K, distortion_params, None, newK)
            mask = torch.from_numpy(mask).bool()
            if len(mask.shape) == 2:
                mask = mask[:, :, None]
        K = newK
    elif camera.camera_type.item() == CameraType.FISHEYE624.value:
        fisheye624_params = torch.cat(
            [camera.fx, camera.fy, camera.cx, camera.cy, torch.from_numpy(distortion_params)], dim=0
        )
        assert fisheye624_params.shape == (16,)
        assert (
            "mask" not in data
            and camera.metadata is not None
            and "fisheye_crop_radius" in camera.metadata
            and isinstance(camera.metadata["fisheye_crop_radius"], float)
        )
        fisheye_crop_radius = camera.metadata["fisheye_crop_radius"]

        # Approximate the FOV of the unmasked region of the camera.
        upper, lower, left, right = camera_utils.fisheye624_unproject_helper(
            torch.tensor(
                [
                    [camera.cx, camera.cy - fisheye_crop_radius],
                    [camera.cx, camera.cy + fisheye_crop_radius],
                    [camera.cx - fisheye_crop_radius, camera.cy],
                    [camera.cx + fisheye_crop_radius, camera.cy],
                ],
                dtype=torch.float32,
            )[None],
            params=fisheye624_params[None],
        ).squeeze(dim=0)
        fov_radians = torch.max(
            torch.acos(torch.sum(upper * lower / torch.linalg.norm(upper) / torch.linalg.norm(lower))),
            torch.acos(torch.sum(left * right / torch.linalg.norm(left) / torch.linalg.norm(right))),
        )

        # Heuristics to determine parameters of an undistorted image.
        undist_h = int(fisheye_crop_radius * 2)
        undist_w = int(fisheye_crop_radius * 2)
        undistort_focal = undist_h / (2 * torch.tan(fov_radians / 2.0))
        undist_K = torch.eye(3)
        undist_K[0, 0] = undistort_focal  # fx
        undist_K[1, 1] = undistort_focal  # fy
        undist_K[0, 2] = (undist_w - 1) / 2.0  # cx; for a 1x1 image, center should be at (0, 0).
        undist_K[1, 2] = (undist_h - 1) / 2.0  # cy

        # Undistorted 2D coordinates -> rays -> reproject to distorted UV coordinates.
        undist_uv_homog = torch.stack(
            [
                *torch.meshgrid(
                    torch.arange(undist_w, dtype=torch.float32),
                    torch.arange(undist_h, dtype=torch.float32),
                ),
                torch.ones((undist_w, undist_h), dtype=torch.float32),
            ],
            dim=-1,
        )
        assert undist_uv_homog.shape == (undist_w, undist_h, 3)
        dist_uv = (
            camera_utils.fisheye624_project(
                xyz=(
                    torch.einsum(
                        "ij,bj->bi",
                        torch.linalg.inv(undist_K),
                        undist_uv_homog.reshape((undist_w * undist_h, 3)),
                    )[None]
                ),
                params=fisheye624_params[None, :],
            )
            .reshape((undist_w, undist_h, 2))
            .numpy()
        )
        map1 = dist_uv[..., 1]
        map2 = dist_uv[..., 0]

        # Use correspondence to undistort image.
        image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

        # Compute undistorted mask as well.
        dist_h = camera.height.item()
        dist_w = camera.width.item()
        mask = np.mgrid[:dist_h, :dist_w]
        mask[0, ...] -= dist_h // 2
        mask[1, ...] -= dist_w // 2
        mask = np.linalg.norm(mask, axis=0) < fisheye_crop_radius
        mask = torch.from_numpy(
            cv2.remap(
                mask.astype(np.uint8) * 255,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            / 255.0
        ).bool()[..., None]
        if len(mask.shape) == 2:
            mask = mask[:, :, None]
        assert mask.shape == (undist_h, undist_w, 1)
        K = undist_K.numpy()
    else:
        raise NotImplementedError("Only perspective and fisheye cameras are supported")

    return K, image, mask
