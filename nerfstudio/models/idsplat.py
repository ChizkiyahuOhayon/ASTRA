# Copyright 2026 the authors of IDSplat and contributors.
# ruff: noqa: E741
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
IDSplat: Instance-Decomposed Gaussian Splatting for autonomous driving.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
from pytorch_msssim import SSIM
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, Parameter
from typing_extensions import Literal

from nerfstudio.cameras.camera_optimizers import (
    CameraOptimizer,
    CameraOptimizerConfig,
    CameraVelocityOptimizer,
    CameraVelocityOptimizerConfig,
)
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.lie_groups import exp_map_SE3, exp_map_SO3xR3
from nerfstudio.cameras.lidars import Lidars, transform_points_pairwise
from nerfstudio.data.datamanagers.full_images_lidar_datamanager import AZIM_CHANNELS_PER_TILE, ELEV_CHANNELS_PER_TILE
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.field_components.mlp import MLP
from nerfstudio.model_components.decoders import RGBDecoderCNN
from nerfstudio.model_components.objects_manager import Object, ObjectsManager, ObjectsManagerConfig
from nerfstudio.model_components.strategy import ADMCMCStrategy
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.idsplat_utils import SeedChannels
from nerfstudio.utils.math import chamfer_distance, k_nearest_sklearn, random_quat_tensor
from nerfstudio.utils.poses import inverse as pose_inverse, to4x4
from nerfstudio.utils.render_utils import get_ray_dirs_pinhole, get_viewmat, resize_image
from nerfstudio.viewer.viewer_elements import ViewerSlider

try:
    from gsplat.rendering import lidar_rasterization, rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")


BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default


@dataclass
class IDSplatModelConfig(ModelConfig):
    """IDSplat model config."""

    _target: Type = field(default_factory=lambda: IDSplatModel)
    refinement_warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""

    background_color: Literal["random", "black", "white"] = "random"
    """Whether to randomize the background color."""
    strategy: Literal["default", "mcmc"] = "mcmc"
    """Strategy to use for the optimization"""

    stop_refine_at: int = 15000
    """stop splitting at this step"""
    use_absgrad: bool = True
    """If True, use absolute gradient for densification"""

    feature_dim: int = 13
    """Dimension of the feature vector"""
    appearance_dim: int = 8
    """Dimension of the appearance vector"""

    mcmc_cap_max: int = 4_000_000
    """Maximum number of GSs. Default to 1_000_000."""
    mcmc_noise_lr: float = 5e5
    """MCMC samping noise learning rate. Default to 5e5."""
    mcmc_min_opacity: float = 0.005
    """GSs with opacity below this value will be pruned. Default to 0.005."""
    verbose: bool = True
    """Whether to print verbose information. Default to True."""

    init_opacities: float = 0.1
    """Initial opacity of the gaussians"""
    init_scale: float = 1.0
    """Initial scale of the gaussians"""

    max_num_seed_points: int = 3_000_000
    """Maximum number of seed points to use for initialization. -1 means all seed points are used."""
    n_extra_random_points: int = 300_000
    """Number of random points to use in addition to seed points."""
    n_far_points: int = 600_000
    """Number of points to initialize outside the scene box."""

    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    mcmc_scale_reg_lambda: float = 0.001
    """weight of scale regularization loss"""
    mcmc_opacity_reg_lambda: float = 0.005
    """weight of opacity regularization loss"""
    depth_lambda: float = 0.1
    """Weight of the depth loss"""
    depth_loss_quantile_threshold: float = 0.95
    """Quantile threshold for the depth loss"""
    intensity_lambda: float = 1.0
    """Weight of the intensity loss"""
    ray_drop_lambda: float = 0.1
    """Weight of the ray drop loss"""
    line_of_sight_lambda: float = 0.1
    """Weight of the line of sight loss"""
    line_of_sight_dist: float = 0.8
    """Weight for line-of-sight lidar loss."""

    output_depth_during_training: bool = False
    """If True, output depth during training. Otherwise, only output depth during evaluation."""
    rasterize_mode: Literal["classic", "antialiased"] = "antialiased"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """

    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))
    """Config of the camera optimizer to use"""
    camera_velocity_optimizer: CameraVelocityOptimizerConfig = field(
        default_factory=lambda: CameraVelocityOptimizerConfig(enabled=True)
    )
    """Config of the camera velocity optimizer to use"""
    use_camopt_in_eval: bool = False
    """Use result of camera optimization also during evaluation. Only makes sense if trained with train_eval_split=1.0."""
    camopt_cameras_only: bool = False
    """ASTRA: apply the camera pose optimizer to camera frames only; lidar poses stay fixed as the geometric anchor."""
    camopt_interp_eval: bool = False
    """ASTRA: at eval, correct each camera pose by the time-interpolated corrections of the same sensor's train frames."""
    lens_distortion: bool = False
    """ASTRA lens-consistent rendering: rasterise a padded pinhole image and resample it into the camera's distorted
    image grid (OpenCV k1, k2, p1, p2, k3 from the dataset calibration, camera metadata `lens_distortion`) before
    decoding, so a static point lands where the real lens put it in every frame. Cameras without coefficients render
    pinhole as before."""
    gauss_dropout: float = 0.0
    """ASTRA progressive Gaussian dropout: final drop rate on camera renders in training. The rate grows linearly
    from 0 to this value over training and kept Gaussians get opacity / (1 - rate); lidar renders and eval use all
    Gaussians (0 disables)."""

    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Which implementation to use for the model."""

    compensate_rs_camera: bool = True
    """If True, compensate the camera for the RS camera"""
    compensate_rs_lidar: bool = True
    """If True, compensate the lidar for the RS camera"""

    radius_clip_pix: float = 0.0
    """Clip radius in pixels, 0.0 means no clipping"""
    radius_clip_lidar: float = 0.0
    """Clip radius in degrees, 0.0 means no clipping"""

    rgb_decoder_hidden_dim: int = 32
    rgb_decoder_kernel_size: int = 3
    rgb_decoder_num_hidden_blocks: int = 1

    only_render_present_objects: bool = True

    objects_manager: ObjectsManagerConfig = field(default_factory=ObjectsManagerConfig)

    def __post_init__(self):
        if self.strategy == "mcmc":
            self.init_opacities = 0.5
            self.init_scale = 0.2


class IDSplatModel(Model):
    """IDSplat model for autonomous driving scene reconstruction.

    Args:
        config: ADSplat configuration to instantiate model
    """

    config: IDSplatModelConfig

    def __init__(
        self,
        *args,
        seed_points,
        tracks,
        **kwargs,
    ):
        self.seed_points = seed_points
        self.last_size = (1, 1)
        self.tracks = tracks
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        super().populate_modules()
        self.collider = None
        self.downscale_factor = 1
        self.unique_timestamps = torch.nn.Parameter(
            self.kwargs["metadata"]["lidar_times"].unique(),
            requires_grad=False,
        )
        self.gt_mask_index_to_class_name = self.kwargs["metadata"]["dynamic_gt_masks_index_to_class_name"]

        seed_points = self.prune_seed_points(self.seed_points)
        self.objects_manager: ObjectsManager = self.config.objects_manager.setup(
            unique_timestamps=self.unique_timestamps,
            objects=self.tracks,
        )

        close_random_seed_points = self.create_seed_points_inside_scenebox()
        far_seed_points = self.create_seed_points_outside_scenebox()
        seed_points = torch.cat([seed_points, close_random_seed_points, far_seed_points], dim=0)

        static_gauss_params_dict = self.create_gauss_params_dict_from_seed_points(
            seed_points[seed_points[:, SeedChannels.OBJ_ID] == -1],
            scale_multiplier=self.config.init_scale,
            init_opacity=self.config.init_opacities,
            mirror_points=False,
        )
        dynamic_gauss_params_dict = self.create_gauss_params_dict_from_seed_points(
            seed_points[seed_points[:, SeedChannels.OBJ_ID] != -1],
            scale_multiplier=self.config.init_scale,
            init_opacity=0.9,
            mirror_points=True,
        )
        # merge static and dynamic gauss params
        assert set(static_gauss_params_dict.keys()) == set(dynamic_gauss_params_dict.keys())
        self.gauss_params = self.init_gauss_params([static_gauss_params_dict, dynamic_gauss_params_dict])
        self.xys_grad_norm = None
        self.max_2Dsize = None

        num_sensors = len(self.kwargs["metadata"]["sensor_idx_to_name"])
        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )
        if self.config.camopt_interp_eval:
            self.register_buffer("train_cam_times", self.kwargs["metadata"]["cam_times"].flatten().float(), False)
            self.register_buffer("train_cam_sensors", self.kwargs["metadata"]["cam_sensor_idxs"].flatten(), False)
        self.camera_velocity_optimizer: CameraVelocityOptimizer = self.config.camera_velocity_optimizer.setup(
            num_cameras=self.num_train_data,
            num_unique_cameras=num_sensors,
            device="cpu",
        )

        viewdir_dim = 3
        self.rgb_decoder = (  # torch.compile(
            RGBDecoderCNN(
                self.config.feature_dim + self.config.appearance_dim + viewdir_dim,
                hidden_dim=self.config.rgb_decoder_hidden_dim,
                kernel_size=self.config.rgb_decoder_kernel_size,
                num_hidden_blocks=self.config.rgb_decoder_num_hidden_blocks,
            )
        )

        self.appearance_embedding = torch.nn.Embedding(num_sensors, self.config.appearance_dim)
        self.fallback_sensor_idx = ViewerSlider("fallback sensor idx", 0, 0, num_sensors - 1, step=1)

        self.lidar_feature_dim = self.config.feature_dim
        self.lidar_decoder = MLP(
            in_dim=self.lidar_feature_dim
            + self.config.appearance_dim
            + viewdir_dim,  # feature + appearance + view direction
            layer_width=32,
            out_dim=2,  # (intensity, ray_drop)
            num_layers=3,
            implementation=self.config.implementation,
            out_activation=None,
        )

        self.setup_viewer_elements()

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
        self.step = 0
        self.median_l2 = lambda pred, gt: torch.median((pred - gt) ** 2)
        self.mean_rel_l2 = lambda pred, gt: torch.mean(((pred - gt) / gt) ** 2)
        self.rmse = lambda pred, gt: torch.sqrt(torch.mean((pred - gt) ** 2))
        self.chamfer_distance = lambda pred, gt: chamfer_distance(pred, gt, 1_000, True)

        # losses
        self.depth_loss = torch.nn.L1Loss(reduction="none")
        self.intensity_loss = torch.nn.MSELoss()
        self.ray_drop_loss = BCEWithLogitsLoss()

        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)

        if self.config.strategy == "mcmc":
            self.strategy = ADMCMCStrategy(
                cap_max=self.config.mcmc_cap_max,
                noise_lr=self.config.mcmc_noise_lr,
                refine_start_iter=self.config.refinement_warmup_length,
                refine_stop_iter=self.config.stop_refine_at,
                refine_every=self.config.refine_every,
                min_opacity=self.config.mcmc_min_opacity,
                verbose=self.config.verbose,
            )
            self.strategy_state = self.strategy.initialize_state()
            self.config.init_opacities = self.config.mcmc_min_opacity
        else:
            raise NotImplementedError(f"Strategy {self.config.strategy} is not implemented.")

    @property
    def means(self):
        return self.gauss_params["means"]

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @property
    def quats(self):
        return self.gauss_params["quats"]

    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    @property
    def opacities(self):
        return self.gauss_params["opacities"]

    @property
    def id(self):
        return self.gauss_params["id"]

    def setup_viewer_elements(self):
        self.setup_rs_editing()

    def setup_rs_editing(self):
        # RS sliders
        self.rs_editing = {
            "rs_time": 0.0,
            "lin_vel_x": 0.0,
            "lin_vel_y": 0.0,
            "lin_vel_z": 0.0,
            "ang_vel_x": 0.0,
            "ang_vel_y": 0.0,
            "ang_vel_z": 0.0,
        }
        self.rs_time_slider = ViewerSlider(
            name="rs time",
            default_value=self.rs_editing["rs_time"],
            min_value=0.0,
            max_value=0.2,
            step=0.001,
            cb_hook=lambda obj: self.rs_editing.update({"rs_time": obj.value}),
        )
        self.rs_lin_vel_x_slider = ViewerSlider(
            name="rs lin vel x",
            default_value=self.rs_editing["lin_vel_x"],
            min_value=-30.0,
            max_value=30.0,
            step=0.01,
            cb_hook=lambda obj: self.rs_editing.update({"lin_vel_x": obj.value}),
        )
        self.rs_lin_vel_y_slider = ViewerSlider(
            name="rs lin vel y",
            default_value=self.rs_editing["lin_vel_y"],
            min_value=-30.0,
            max_value=30.0,
            step=0.01,
            cb_hook=lambda obj: self.rs_editing.update({"lin_vel_y": obj.value}),
        )
        self.rs_lin_vel_z_slider = ViewerSlider(
            name="rs lin vel z",
            default_value=self.rs_editing["lin_vel_z"],
            min_value=-30.0,
            max_value=30.0,
            step=0.01,
            cb_hook=lambda obj: self.rs_editing.update({"lin_vel_z": obj.value}),
        )
        self.rs_ang_vel_x_slider = ViewerSlider(
            name="rs ang vel x",
            default_value=self.rs_editing["ang_vel_x"],
            min_value=-1.0,
            max_value=1.0,
            step=0.01,
            cb_hook=lambda obj: self.rs_editing.update({"ang_vel_x": obj.value}),
        )
        self.rs_ang_vel_y_slider = ViewerSlider(
            name="rs ang vel y",
            default_value=self.rs_editing["ang_vel_y"],
            min_value=-1.0,
            max_value=1.0,
            step=0.01,
            cb_hook=lambda obj: self.rs_editing.update({"ang_vel_y": obj.value}),
        )
        self.rs_ang_vel_z_slider = ViewerSlider(
            name="rs ang vel z",
            default_value=self.rs_editing["ang_vel_z"],
            min_value=-1.0,
            max_value=1.0,
            step=0.01,
            cb_hook=lambda obj: self.rs_editing.update({"ang_vel_z": obj.value}),
        )

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))

        if (
            "objects_manager.param_dict.actor_positions" in dict
            and len(dict["objects_manager.param_dict.actor_positions"]) > 0
        ):
            n_times, n_objects = dict["objects_manager.param_dict.actor_positions"].shape[:2]
            empty_objects = [
                Object(
                    id=-1,
                    mask_id=-1,
                    poses=torch.zeros((4, 4), device=self.device).repeat(n_times, 1, 1),
                    size=torch.zeros(3, device=self.device),
                    valid_timestamps=torch.zeros(n_times, dtype=torch.bool, device=self.device),
                    linear_velocities_global=torch.zeros(n_times, 3, device=self.device),
                    angular_velocities_local=torch.zeros(n_times, 3, device=self.device),
                )
                for _ in range(n_objects)
            ]
            self.objects_manager: ObjectsManager = self.config.objects_manager.setup(
                unique_timestamps=self.unique_timestamps,
                objects=empty_objects,
            )
        super().load_state_dict(dict, **kwargs)

    def create_seed_points_outside_scenebox(self) -> Tensor:
        (
            scene_width,
            scene_length,
        ) = (
            self.scene_box.aabb.diff(dim=0)[..., 0].item(),
            self.scene_box.aabb.diff(dim=0)[..., 1].item(),
        )
        random_directions = torch.rand(int(self.config.n_far_points), 3) - 0.5
        random_directions[:, -1] = torch.abs(random_directions[:, -1])
        random_directions = random_directions / random_directions.norm(dim=-1, keepdim=True)
        random_distances = torch.rand(int(self.config.n_far_points), 1)
        near = min(scene_width, scene_length) / 2
        far = 1e4
        random_distances = 1 / (1 / near * (1 - random_distances) + 1 / far * random_distances)
        far_points = random_directions * random_distances
        rgbs = torch.rand_like(far_points)
        obj_ids = torch.ones_like(far_points[:, 0], dtype=torch.int32) * -1

        seed_points = torch.zeros(int(self.config.n_far_points), SeedChannels.N_CHANNELS, device=rgbs.device)
        seed_points[:, SeedChannels.COORDS : SeedChannels.COORDS + 3] = far_points
        seed_points[:, SeedChannels.RGB : SeedChannels.RGB + 3] = rgbs
        seed_points[:, SeedChannels.OBJ_ID] = obj_ids
        return seed_points

    def create_seed_points_inside_scenebox(self) -> Tensor:
        (
            scene_width,
            scene_length,
        ) = (
            self.scene_box.aabb.diff(dim=0)[..., 0].item(),
            self.scene_box.aabb.diff(dim=0)[..., 1].item(),
        )
        # randomly sample points within the scene box
        close_points = torch.rand(int(self.config.n_extra_random_points), 3) - 0.5
        close_points = close_points * torch.tensor([scene_width, scene_length, 50]) + torch.tensor(
            [0.0, 0.0, 50 * 0.5 - 2.0]
        )
        rgbs = torch.rand_like(close_points)
        obj_ids = torch.ones_like(close_points[:, 0], dtype=torch.int32) * -1

        seed_points = torch.zeros(int(self.config.n_extra_random_points), SeedChannels.N_CHANNELS, device=rgbs.device)
        seed_points[:, SeedChannels.COORDS : SeedChannels.COORDS + 3] = close_points
        seed_points[:, SeedChannels.RGB : SeedChannels.RGB + 3] = rgbs
        seed_points[:, SeedChannels.OBJ_ID] = obj_ids
        return seed_points

    def create_gauss_params_dict_from_seed_points(
        self,
        seed_points: Tensor,
        scale_multiplier: float,
        init_opacity: float,
        mirror_points: bool = False,
    ) -> Dict[str, torch.nn.Parameter]:
        """Create a dict of parameters for the gaussians, created from a list with sets of seed points.

        seed_points_list - List of seed points. Each element in the list is a tensor of shape (N, 9).
                                                Last dimension is (x, y, z, r, g, b, time, mask ids, cam idx).

        Returns:
            Parameter dict with the learnable parameters of the gaussians.
        """

        assert seed_points is not None
        assert seed_points.shape[1] == SeedChannels.N_CHANNELS

        means = torch.nn.Parameter(seed_points[:, SeedChannels.COORDS : SeedChannels.COORDS + 3])
        num_points = means.shape[0]

        scales = torch.zeros(num_points, 3)
        for id_ in seed_points[:, SeedChannels.OBJ_ID].unique():
            curr_id_mask = (seed_points[:, SeedChannels.OBJ_ID] == id_).squeeze(-1)
            means_ = means[curr_id_mask]
            distances, _ = k_nearest_sklearn(means_.data, 3)
            distances = torch.from_numpy(distances)
            # find the average of the three nearest neighbors for each point and use that as the scale
            avg_dist = distances.mean(dim=-1, keepdim=True)
            scales[curr_id_mask] = torch.log(avg_dist.repeat(1, 3) * scale_multiplier)
        scales = torch.nn.Parameter(scales)

        quats = torch.nn.Parameter(random_quat_tensor(num_points))
        features_dc = torch.nn.Parameter(seed_points[:, SeedChannels.RGB : SeedChannels.RGB + 3])
        features_rest = torch.nn.Parameter(
            torch.randn(
                features_dc.shape[0],
                max(self.config.feature_dim, 0),
                dtype=seed_points.dtype,
                device=seed_points.device,
            )
        )

        opacities = torch.nn.Parameter(torch.logit(init_opacity * torch.ones(num_points, 1)))
        ids = torch.nn.Parameter(seed_points[:, SeedChannels.OBJ_ID].float(), requires_grad=False)

        if mirror_points:
            # Duplicate all point and flip them around the x-axis
            mirrored_means = means.clone()
            mirrored_means[:, 1] *= -1
            mirrored_quats = torch.nn.Parameter(random_quat_tensor(num_points))
            means = torch.nn.Parameter(torch.cat([means, mirrored_means], dim=0))
            scales = torch.nn.Parameter(torch.cat([scales, scales.clone()], dim=0))
            quats = torch.nn.Parameter(torch.cat([quats, mirrored_quats], dim=0))
            features_dc = torch.nn.Parameter(torch.cat([features_dc, features_dc.clone()], dim=0))
            features_rest = torch.nn.Parameter(torch.cat([features_rest, features_rest.clone()], dim=0))
            opacities = torch.nn.Parameter(torch.cat([opacities, opacities.clone()], dim=0))
            ids = torch.nn.Parameter(torch.cat([ids, ids.clone()], dim=0))

        return {
            "means": means,
            "scales": scales,
            "quats": quats,
            "features_dc": features_dc,
            "features_rest": features_rest,
            "opacities": opacities,
            "id": ids,
        }

    def init_gauss_params(self, gauss_params_dict: List[Dict[str, torch.nn.Parameter]]) -> torch.nn.ParameterDict:
        return torch.nn.ParameterDict(
            {
                key: torch.cat(
                    [param_dict[key] for param_dict in gauss_params_dict],
                    dim=0,
                )
                for key in gauss_params_dict[0].keys()
            }
        )

    def prune_seed_points(self, seed_points: Tensor):
        """Prune the seed points to the maximum number of seed points allowed.

        seed_points - Seed points tensor of shape (N, 8). The last dimension is (x, y, z, r, g, b, time, mask ids).

        Returns:
            Pruned seed points tensor.
        """
        if seed_points is None or seed_points.shape[0] == 0:
            return seed_points

        if self.config.max_num_seed_points > 0 and seed_points.shape[0] > self.config.max_num_seed_points:
            n_seed_points = seed_points.shape[0]
            perm_idx = torch.randperm(n_seed_points)
            seed_points = seed_points[perm_idx][: self.config.max_num_seed_points]
        return seed_points

    def set_background(self, background_color: Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def step_post_backward(self, step):
        assert step == self.step
        if isinstance(self.strategy, ADMCMCStrategy):
            self.strategy.step_post_backward(
                params=self.gauss_params,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=self.step,
                info=self.info,
                lr=self.optimizers["means"].param_groups[0]["lr"],
            )
        else:
            raise NotImplementedError(f"Strategy {self.config.strategy} is not implemented.")

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                self.step_cb,
                args=[training_callback_attributes.optimizers],
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.step_post_backward,
            )
        )
        return cbs

    def step_cb(self, optimizers: Optimizers, step):
        self.step = step
        self.optimizers = optimizers.optimizers

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        param_names = ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        return {name: [self.gauss_params[name]] for name in param_names}

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        param_groups = defaultdict(list)
        param_groups.update(self.get_gaussian_param_groups())
        self.camera_optimizer.get_param_groups(param_groups=param_groups)
        self.camera_velocity_optimizer.get_param_groups(param_groups=param_groups)
        self.objects_manager.get_param_groups(param_groups=param_groups)

        param_groups["fields"] = []
        param_groups["fields"] += list(self.rgb_decoder.parameters())
        param_groups["fields"] += list(self.appearance_embedding.parameters())
        param_groups["fields"] += list(self.lidar_decoder.parameters())
        return param_groups

    def _get_downscale_factor(self) -> int:
        if self.training:
            return self.downscale_factor
        else:
            return 1

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            return resize_image(image, d)
        return image

    @staticmethod
    def get_empty_outputs(width: int, height: int, background: Tensor) -> Dict[str, Union[Tensor, List]]:
        rgb = background.repeat(height, width, 1)
        depth = background.new_ones(*rgb.shape[:2], 1) * 10
        accumulation = background.new_zeros(*rgb.shape[:2], 1)
        return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}

    def _get_background_color(self):
        if self.config.background_color == "random":
            if self.training:
                background = torch.rand(3, device=self.device)
            else:
                self.background_color = self.background_color.to(self.device)
                background = self.background_color.to(self.device)
        elif self.config.background_color == "white":
            background = torch.ones(3, device=self.device)
        elif self.config.background_color == "black":
            background = torch.zeros(3, device=self.device)
        else:
            raise ValueError(f"Unknown background color {self.config.background_color}")
        return background

    def _transform_actor_means(
        self,
        means: Tensor,
        times: Tensor,
        ids: Tensor,
        is_dynamic: Tensor,
        calc_vels: bool = True,
    ):
        means_in_world = means.clone()
        boxes2world, _ = self.objects_manager.get_boxes2world(times)
        boxes2world = boxes2world.squeeze(0)
        boxes2world = boxes2world[..., :3, :]  # remove the last row of the pose matrix

        actor_idx = is_dynamic.nonzero().squeeze(-1)
        curr_ids = ids.index_select(0, actor_idx).to(torch.long)
        b2w_per_pt = boxes2world.index_select(dim=0, index=curr_ids)

        vels_in_world = None
        if calc_vels and len(boxes2world) > 0:
            lin_vel, ang_vel = self.objects_manager.get_velocities(times).split([3, 3], dim=-1)
            lin_vel = lin_vel.squeeze(0).squeeze(0)
            ang_vel = ang_vel.squeeze(0).squeeze(0)

            vels_in_world = torch.zeros_like(means)
            angle_vel_in_box = torch.cross(
                ang_vel.index_select(dim=0, index=curr_ids), means.index_select(0, actor_idx), dim=-1
            )
            vels_in_world[actor_idx] += lin_vel.index_select(dim=0, index=curr_ids) + transform_points_pairwise(
                angle_vel_in_box, b2w_per_pt, with_translation=False
            )

        means_in_world[actor_idx] = transform_points_pairwise(means_in_world.index_select(0, actor_idx), b2w_per_pt)

        return means_in_world, vels_in_world

    def forward(
        self,
        sensor: Union[Cameras, Lidars],
    ) -> Dict[str, Union[Tensor, List]]:
        """Forward pass of the model."""
        return self.get_outputs(
            sensor,
        )

    def render_image(
        self,
        means,
        quats,
        scales,
        opacities,
        colors,
        vels,
        camera,
        ray_dirs,
        viewmat,
        K,
        W,
        H,
        camera_linear_vel,
        camera_angular_vel,
        rolling_shutter_time,
        rolling_shutter_direction,
        render_mode,
        ids,
    ) -> Dict[str, Union[Tensor, List]]:
        lens = getattr(self, "_lens", None)
        rW, rH = W, H
        if lens is not None:
            pad = lens[1]
            K = K.clone()
            K[..., 0, 2] += pad
            K[..., 1, 2] += pad
            rW, rH = W + 2 * pad, H + 2 * pad
        render, alpha, self.info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            velocities=vels,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=rW,
            height=rH,
            linear_velocity=camera_linear_vel,
            angular_velocity=camera_angular_vel,
            rolling_shutter_time=rolling_shutter_time,
            rolling_shutter_direction=rolling_shutter_direction,
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.5,
            far_plane=1e10,
            radius_clip=self.config.radius_clip_pix,
            render_mode=render_mode,
            sh_degree=None,
            sparse_grad=False,
            absgrad=self.config.use_absgrad,
            rasterize_mode=self.config.rasterize_mode,
            channel_chunk=128,
            eps2d=0.3,
        )
        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
            )

        if lens is not None:
            warped = self._lens_warp(torch.cat([render, alpha], dim=-1), lens[0])
            render, alpha = warped[..., :-1], warped[..., -1:]
            if "render_medians" in self.info:
                self.info["render_medians"] = self._lens_warp(self.info["render_medians"], lens[0])

        background = self._get_background_color()

        rendered_features = render[..., :-1] if render_mode == "RGB+ED" else render

        appearance_features = self._get_appearance_embedding(camera, rendered_features)
        rendered_features = torch.cat((rendered_features, appearance_features), dim=-1)

        rgb = self.rgb_decoder(rendered_features, ray_dirs.unsqueeze(0))
        rgb = rgb + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., -1:]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        out = {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
        }  # type: ignore

        if "render_medians" in self.info and render_mode == "RGB+ED":
            rendered_median_features, rendered_median_depth = (
                self.info["render_medians"][..., :-1],
                self.info["render_medians"][..., -1:],
            )
            out["median_depth"] = rendered_median_depth.squeeze(0)

        return out

    def render_lidar(
        self,
        means,
        quats,
        scales,
        opacities,
        lidar_features,
        vels,
        lidar,
        viewmat,
        min_azimuth,
        max_azimuth,
        min_elevation,
        max_elevation,
        azimuth_resolution,
        raster_pts,
        tile_elevation_boundaries,
        lidar_linear_vel,
        lidar_angular_vel,
        rolling_shutter_time,
        optimized_lidar_to_world,
    ):
        render, alpha, alpha_sum_until_points, self.info = lidar_rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            lidar_features=lidar_features,  # [(C,) N, D]
            velocities=vels,
            viewmats=viewmat,  # [1, 4, 4]
            min_azimuth=min_azimuth,
            max_azimuth=max_azimuth,
            min_elevation=min_elevation,
            max_elevation=max_elevation,
            n_elevation_channels=raster_pts.shape[1],
            azimuth_resolution=azimuth_resolution,
            raster_pts=raster_pts,  # [C, H, W, 4]
            tile_width=AZIM_CHANNELS_PER_TILE,
            tile_height=ELEV_CHANNELS_PER_TILE,
            tile_elevation_boundaries=tile_elevation_boundaries,
            linear_velocity=lidar_linear_vel,
            angular_velocity=lidar_angular_vel,
            rolling_shutter_time=rolling_shutter_time,
            near_plane=0.2,
            far_plane=300,
            radius_clip=self.config.radius_clip_lidar,
            compute_alpha_sum_until_points=(self.config.line_of_sight_lambda > 0) and (self.training),
            compute_alpha_sum_until_points_threshold=self.config.line_of_sight_dist,
            sparse_grad=False,
            absgrad=self.config.use_absgrad,
            rasterize_mode=self.config.rasterize_mode,
            channel_chunk=128,
            eps2d=0.01718873385,
        )
        self.info["width"] = -1
        self.info["height"] = -1
        self.last_size = (self.last_size[0], self.last_size[1], -1)
        # self.xys = info["means2d"]  # [1, N, 2]
        # self.radii = info["radii"][0]  # [N]
        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
            )

        depth_im = render[:, ..., -1:]

        rendered_features = render[..., :-1]
        appearance_features = self._get_appearance_embedding(lidar, rendered_features)
        rendered_features = torch.cat((rendered_features, appearance_features), dim=-1)
        raster_pts_degrees = torch.deg2rad(raster_pts[..., :2])
        lidar_ray_dir = torch.cat(
            [
                torch.cos(raster_pts_degrees[..., 0:1])
                * torch.cos(raster_pts_degrees[..., 1:2]),  # x = cos(azimuth) * cos(elevation)
                torch.sin(raster_pts_degrees[..., 0:1])
                * torch.cos(raster_pts_degrees[..., 1:2]),  # y = sin(azimuth) * cos(elevation)
                torch.sin(raster_pts_degrees[..., 1:2]),  # z = sin(elevation)
            ],
            dim=-1,
        )
        lidar_ray_dir_in_world = (
            optimized_lidar_to_world[:, :3, :3].reshape(1, 1, 1, 3, 3) @ lidar_ray_dir.unsqueeze(-1)
        ).squeeze(-1)

        intensity, ray_drop_logits = (
            self.lidar_decoder(
                torch.cat(
                    [
                        rendered_features.reshape(-1, rendered_features.shape[-1]),
                        lidar_ray_dir_in_world.reshape(-1, lidar_ray_dir_in_world.shape[-1]),
                    ],
                    dim=-1,
                )
            )
            .reshape((*lidar_ray_dir_in_world.shape[:-1], self.lidar_decoder.out_dim))
            .split([1, 1], dim=-1)
        )

        out = {
            "depth": depth_im,  # type: ignore
            "accumulation": alpha,  # type: ignore
            "median_depth": self.info["median_depths"] + (alpha <= 0.5) * (depth_im / alpha.clamp_min(1e-10)),
        }  # type: ignore

        if intensity is not None:
            out["intensity"] = intensity.sigmoid().to(torch.float32)

        if ray_drop_logits is not None:
            out["ray_drop_logits"] = ray_drop_logits.to(torch.float32)
            out["ray_drop_prob"] = ray_drop_logits.sigmoid().to(torch.float32)

        if alpha_sum_until_points is not None:
            out["alpha_sum_until_points"] = alpha_sum_until_points
        return out

    def get_actor_adjusted_means(self, means: Tensor, ids: Tensor, times: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        is_dynamic = (ids > -1).squeeze(-1)
        is_present = torch.ones_like(is_dynamic, dtype=torch.bool, device=means.device)
        vels = None
        means_in_world = means
        if self.objects_manager.max_id() >= 0:
            if self.config.only_render_present_objects:
                t_diffs = torch.abs(self.objects_manager.unique_timestamps - times.squeeze())
                closest_idx = torch.argmin(t_diffs)
                present_at_query_time = self.objects_manager.actor_present_at_time[closest_idx, :]
                present_ids = self.objects_manager.actor_ids[present_at_query_time]
                actor_id_is_present = (ids[..., None] == present_ids[None, ...]).any(dim=-1)
                is_present = ~is_dynamic | actor_id_is_present

            means_in_world, vels = self._transform_actor_means(
                means,
                times,
                ids,
                is_dynamic & is_present,
                calc_vels=True,
            )
        return means_in_world, vels, is_present

    def get_camera_outputs(
        self,
        camera: Cameras,
    ) -> Dict[str, Union[Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            camera: The camera(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if self.training or self.config.use_camopt_in_eval:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        elif self.config.camopt_interp_eval:
            optimized_camera_to_world = self._interp_camopt(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        camera_scale_fac = self._get_downscale_factor()
        if camera_scale_fac != 1:
            camera.rescale_output_resolution(1 / camera_scale_fac)
        K = camera.get_intrinsics_matrices()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        ray_dirs = get_ray_dirs_pinhole(camera, W, H, optimized_camera_to_world)
        if camera_scale_fac != 1:
            camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        # rolling shutter
        camera_linear_vel = None
        camera_angular_vel = None
        rolling_shutter_time = None
        rolling_shutter_direction = None
        camera_times = camera.times
        if (
            camera.metadata is not None
            and "linear_velocities_local" in camera.metadata
            and "angular_velocities_local" in camera.metadata
            and self.config.compensate_rs_camera
        ):
            rolling_shutter_time = camera.metadata.get(
                "rolling_shutter_time",
                torch.zeros(
                    (
                        1,
                        1,
                    ),
                    device=self.device,
                ),
            )[0]
            time_to_center_pixel = camera.metadata.get(
                "time_to_center_pixel",
                torch.zeros(
                    (
                        1,
                        1,
                    ),
                    device=self.device,
                ),
            )
            rolling_shutter_direction = camera.metadata.get("rolling_shutter_direction", 1)
            velocities = self.camera_velocity_optimizer.apply_to_camera_velocity(
                camera, return_init_only=(not self.training) and (not self.config.use_camopt_in_eval)
            )
            camera_linear_vel, camera_angular_vel = torch.split(velocities, 3, dim=-1)
            time_to_center_pixel = (
                time_to_center_pixel + self.camera_velocity_optimizer.get_time_to_center_pixel_adjustment(camera)
            )
            optimized_camera_to_world = torch.cat(
                [
                    optimized_camera_to_world[:, :3, :3],
                    optimized_camera_to_world[:, :3, 3:4]
                    + (
                        torch.matmul(camera_linear_vel, optimized_camera_to_world[0, :3, :3].transpose(0, 1))
                        * (time_to_center_pixel)
                    )[..., None],
                ],
                dim=-1,
            )
            camera_times = camera.times + time_to_center_pixel

            flip_tensor = torch.ones(3, device=camera_linear_vel.device, dtype=camera_linear_vel.dtype)
            flip_tensor[1:] *= -1
            camera_linear_vel = camera_linear_vel * flip_tensor  # flip y and z
            camera_angular_vel = camera_angular_vel * flip_tensor  # flip y and z
        else:
            camera_linear_vel = torch.tensor(
                [[self.rs_editing["lin_vel_x"], self.rs_editing["lin_vel_y"], self.rs_editing["lin_vel_z"]]],
                device=self.device,
            )
            camera_angular_vel = torch.tensor(
                [[self.rs_editing["ang_vel_x"], self.rs_editing["ang_vel_y"], self.rs_editing["ang_vel_z"]]],
                device=self.device,
            )
            rolling_shutter_time = torch.tensor([self.rs_editing["rs_time"]], device=self.device)

        viewmat = get_viewmat(optimized_camera_to_world)

        means, vels, is_present = self.get_actor_adjusted_means(self.means, self.id, camera_times)
        colors = torch.cat((self.features_dc, self.features_rest), dim=-1)

        comp = is_present.float()
        if self.training and self.config.gauss_dropout > 0:
            # ASTRA progressive Gaussian dropout (camera renders in training only)
            p = self.config.gauss_dropout * min(1.0, self.step / 30000)
            if p > 0:
                comp = comp * (torch.rand_like(comp) >= p) / (1 - p)
        opacs = self.opacities.sigmoid().squeeze(-1)
        opacs = (opacs * comp).clamp_max(1.0)
        scales = self.scales.exp()

        # ASTRA lens-consistent rendering: padded pinhole render, resampled into the distorted image grid
        self._lens = self._lens_grid(camera, K, W, H) if self.config.lens_distortion else None
        out = self.render_image(
            means,
            self.quats,
            scales,
            opacs,
            colors,
            vels,
            camera,
            ray_dirs,
            viewmat,
            K,
            W,
            H,
            camera_linear_vel,
            camera_angular_vel,
            rolling_shutter_time,
            rolling_shutter_direction,
            render_mode,
            self.id,
        )

        return out

    def get_lidar_outputs(
        self,
        lidar: Lidars,
    ) -> Dict[str, Union[Tensor, List]]:
        """Takes in a lidar and returns a dictionary of outputs.

        Args:
            lidar: The lidar(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(lidar, Lidars):
            print("Called get_outputs with not a lidar")
            return {}
        assert (
            (lidar.azimuths is not None and lidar.elevations is not None)
            or lidar.metadata
            and "raster_pts" in lidar.metadata
        )
        if (self.training or self.config.use_camopt_in_eval) and not self.config.camopt_cameras_only:
            assert lidar.shape[0] == 1, "Only one camera at a time"
            optimized_lidar_to_world = self.camera_optimizer.apply_to_camera(lidar)
        else:
            optimized_lidar_to_world = lidar.lidar_to_worlds

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if lidar.metadata and "raster_pts" in lidar.metadata:
            raster_pts = lidar.metadata["raster_pts"][
                ..., :-1
            ]  # omit intensity channel, only needed for metric/loss computation
            tile_elevation_boundaries = lidar.metadata["elevation_boundaries"]
            min_azimuth = -180
            max_azimuth = 180
            min_elevation = tile_elevation_boundaries.min()
            max_elevation = tile_elevation_boundaries.max()
            azimuth_resolution = lidar.metadata["azimuth_resolution"]
        else:
            elevs, azims = torch.meshgrid(
                torch.rad2deg(lidar.elevations.flatten()), torch.rad2deg(lidar.azimuths.flatten())
            )
            raster_pts = torch.stack([azims, elevs, torch.ones_like(azims), torch.zeros_like(azims)], dim=-1).to(
                self.device
            )[None]
            tile_elevation_boundaries = torch.rad2deg(lidar.elevations[0, ::ELEV_CHANNELS_PER_TILE]).to(self.device)
            tile_elevation_boundaries = tile_elevation_boundaries.flatten()
            tile_elevation_boundaries = torch.cat(
                [
                    tile_elevation_boundaries,
                    torch.tensor([tile_elevation_boundaries[..., -1].item() + 1], device=self.device),
                ]
            )
            azimuth_resolution = float(torch.rad2deg((lidar.azimuths[0, 1] - lidar.azimuths[0, 0])))
            min_azimuth = -180
            max_azimuth = 180
            min_elevation = tile_elevation_boundaries.min().item()
            max_elevation = tile_elevation_boundaries.max().item() + 1e-6

        # rolling shutter
        lidar_linear_vel = torch.zeros(1, 3, device=self.device)
        lidar_angular_vel = torch.zeros(1, 3, device=self.device)
        rolling_shutter_time = torch.zeros(1, device=self.device)
        lidar_times = lidar.times
        if lidar.metadata is not None and self.config.compensate_rs_lidar:
            max_offset, min_offset = raster_pts[..., 3].max(), raster_pts[..., 3].min()
            rolling_shutter_time = (max_offset - min_offset).unsqueeze(0)
            velocities = self.camera_velocity_optimizer.apply_to_camera_velocity(
                lidar, return_init_only=(not self.training) and (not self.config.use_camopt_in_eval)
            )
            lidar_linear_vel, lidar_angular_vel = torch.split(velocities, 3, dim=-1)

            time_to_center_adjustment = (max_offset + min_offset) / 2
            optimized_lidar_to_world = torch.cat(
                [
                    optimized_lidar_to_world[:, :3, :3],
                    optimized_lidar_to_world[:, :3, 3:4]
                    + (
                        (torch.einsum("bij,bj->bi", optimized_lidar_to_world[..., :3, :3], lidar_linear_vel))
                        * time_to_center_adjustment
                    )[..., None],
                ],
                dim=-1,
            )
            lidar_times = lidar.times + time_to_center_adjustment
            raster_pts[..., 3] = raster_pts[..., 3] - time_to_center_adjustment

        means, vels, is_present = self.get_actor_adjusted_means(self.means, self.id, lidar_times)
        lidar_features = self.features_rest.unsqueeze(0)

        batch_size = raster_pts.shape[0]
        viewmat = to4x4(pose_inverse(optimized_lidar_to_world))
        if batch_size > 1:
            viewmat = viewmat.repeat(batch_size, 1, 1)
            lidar_features = lidar_features.repeat(batch_size, 1, 1)
            lidar_linear_vel = lidar_linear_vel.repeat(batch_size, 1)
            lidar_angular_vel = lidar_angular_vel.repeat(batch_size, 1)
            rolling_shutter_time = rolling_shutter_time.repeat(batch_size)

        comp = is_present.float()
        opacs = self.opacities.sigmoid().squeeze(-1)
        opacs = (opacs * comp).clamp_max(1.0)
        scales = self.scales.exp()

        out = self.render_lidar(
            means,
            self.quats,
            scales,
            opacs,
            lidar_features,
            vels,
            lidar,
            viewmat,
            min_azimuth,
            max_azimuth,
            min_elevation,
            max_elevation,
            azimuth_resolution,
            raster_pts,
            tile_elevation_boundaries,
            lidar_linear_vel,
            lidar_angular_vel,
            rolling_shutter_time,
            optimized_lidar_to_world,
        )

        return out  # type: ignore

    def _lens_grid(self, camera: Cameras, K: Tensor, W: int, H: int):
        """ASTRA: sampling grid from the distorted output image into a padded pinhole render (cached per intrinsics)."""
        d = (camera.metadata or {}).get("lens_distortion")
        if d is None:
            return None  # rectified dataset: pinhole rendering
        k1, k2, p1, p2, k3 = [float(v) for v in d.reshape(-1)[:5]]
        if max(abs(k1), abs(k2), abs(p1), abs(p2), abs(k3)) < 1e-9:
            return None
        Kc = K.reshape(-1, 3, 3)[0]
        fx, fy, cx, cy = float(Kc[0, 0]), float(Kc[1, 1]), float(Kc[0, 2]), float(Kc[1, 2])
        key = (round(fx, 3), round(fy, 3), round(cx, 3), round(cy, 3), W, H, k1, k2, p1, p2, k3)
        cache = self.__dict__.setdefault("_lens_cache", {})
        if key not in cache:
            v, u = torch.meshgrid(
                torch.arange(H, device=self.device, dtype=torch.float64) + 0.5,
                torch.arange(W, device=self.device, dtype=torch.float64) + 0.5,
                indexing="ij",
            )
            xd, yd = (u - cx) / fx, (v - cy) / fy
            x, y = xd.clone(), yd.clone()
            for _ in range(20):  # OpenCV undistortPoints fixed-point iteration
                r2 = x * x + y * y
                rad = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
                x = (xd - (2 * p1 * x * y + p2 * (r2 + 2 * x * x))) / rad
                y = (yd - (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)) / rad
            up, vp = x * fx + cx, y * fy + cy
            pad = int(max((up - u).abs().max().item(), (vp - v).abs().max().item())) + 3
            grid = torch.stack([(up + pad) / (W + 2 * pad) * 2 - 1, (vp + pad) / (H + 2 * pad) * 2 - 1], dim=-1)
            cache[key] = (grid[None].float(), pad)
        return cache[key]

    @staticmethod
    def _lens_warp(x: Tensor, grid: Tensor) -> Tensor:
        """Resample a [1, h, w, C] render onto the distorted grid [1, H, W, 2]."""
        y = torch.nn.functional.grid_sample(
            x.permute(0, 3, 1, 2), grid.to(x.dtype), mode="bilinear", padding_mode="border", align_corners=False
        )
        return y.permute(0, 2, 3, 1)

    def _interp_camopt(self, camera: Cameras) -> Tensor:
        """ASTRA: pose correction for a non-train camera, linearly interpolated (in the 6d tangent space) between the
        two train frames of the same sensor that bracket its time; clamped to the nearest one outside the train span."""
        c2w = camera.camera_to_worlds
        mode = self.config.camera_optimizer.mode
        if mode == "off" or camera.times is None or not camera.metadata or "sensor_idxs" not in camera.metadata:
            return c2w
        sel = (self.train_cam_sensors == camera.metadata["sensor_idxs"].flatten()[0]).nonzero().flatten()
        ts, order = self.train_cam_times[sel].sort()
        sel = sel[order]
        t = camera.times.flatten()[:1].float().to(ts.device)
        j = torch.searchsorted(ts, t).clamp(1, len(ts) - 1)
        t0, t1 = ts[j - 1], ts[j]
        w = ((t - t0) / (t1 - t0).clamp_min(1e-9)).clamp(0, 1)[:, None]
        pose_adj = self.camera_optimizer.pose_adjustment
        a = (1 - w) * pose_adj[sel[j - 1]] + w * pose_adj[sel[j]]
        adj = exp_map_SO3xR3(a) if mode == "SO3xR3" else exp_map_SE3(a)
        return torch.cat([torch.bmm(adj[..., :3, :3], c2w[..., :3, :3]), c2w[..., :3, 3:] + adj[..., :3, 3:]], dim=-1)

    def get_outputs(
        self,
        sensor: Union[Cameras, Lidars],
    ) -> Dict[str, Union[Tensor, List]]:
        if isinstance(sensor, Cameras):
            return self.get_camera_outputs(
                sensor,
            )
        elif isinstance(sensor, Lidars):
            return self.get_lidar_outputs(
                sensor,
            )
        else:
            raise ValueError("Unknown sensor type")

    def get_gt_img(self, image: Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return alpha * image[..., :3] + (1 - alpha) * background
        else:
            return image

    def filter_lidar_pred_and_gt(self, outputs, batch, output_point_cloud=False):
        gt_lidar = batch["raster_pts"]  # (azimuth, elev, depth, time, intensity)
        raster_pts_valid_and_did_return = batch["raster_pts_valid_depth_and_did_return"]
        raster_pts_did_return = batch["raster_pts_did_return"].flatten()
        raster_pts_valid_and_did_not_return = batch["raster_pts_valid_depth_and_did_not_return"]

        gt = {}
        gt["depth"] = gt_lidar[..., 2].flatten()[raster_pts_valid_and_did_return]
        gt["intensity"] = gt_lidar[..., 4].flatten()[raster_pts_valid_and_did_return]
        gt["ray_drop"] = ~raster_pts_did_return
        gt["valid"] = gt_lidar[..., 2].flatten() > 0

        pred = {}
        pred["depth"] = outputs["depth"].flatten()[raster_pts_valid_and_did_return]
        pred["depth_dropped"] = outputs["depth"].flatten()[raster_pts_valid_and_did_not_return]
        pred["intensity"] = outputs["intensity"].flatten()[raster_pts_valid_and_did_return]
        pred["intensity_dropped"] = outputs["intensity"].flatten()[raster_pts_valid_and_did_not_return]
        pred["ray_drop"] = outputs["ray_drop_logits"].flatten() * gt["valid"] - (~gt["valid"]) * 10_000
        pred["accumulation"] = outputs["accumulation"].flatten()[raster_pts_valid_and_did_return]
        pred["accumulation_dropped"] = outputs["accumulation"].flatten()[raster_pts_valid_and_did_not_return]
        pred["median_depth"] = outputs["median_depth"].flatten()[raster_pts_valid_and_did_return]

        if "alpha_sum_until_points" in outputs:
            pred["alpha_sum_until_points"] = outputs["alpha_sum_until_points"].flatten()[
                raster_pts_valid_and_did_return
            ]
            pred["alpha_sum_until_points_dropped"] = outputs["alpha_sum_until_points"].flatten()[
                raster_pts_valid_and_did_not_return
            ]

        if output_point_cloud:
            azimuth_angles = torch.deg2rad(gt_lidar[..., 0].flatten())
            elevation_angles = torch.deg2rad(gt_lidar[..., 1].flatten())
            directions = torch.stack(
                [
                    torch.cos(elevation_angles) * torch.cos(azimuth_angles),
                    torch.cos(elevation_angles) * torch.sin(azimuth_angles),
                    torch.sin(elevation_angles),
                ],
                dim=-1,
            )

            gt["point_cloud"] = batch["lidar"][batch["lidar_pts_did_return"].squeeze(), :3]
            pred["point_cloud"] = (
                outputs["depth"].view(-1, 1) * directions
                + batch["linear_velocities_local"] * gt_lidar[..., 3].view(-1, 1)
            )[((pred["ray_drop"].sigmoid() <= 0.5) * gt["valid"])]
            pred["median_point_cloud"] = (
                outputs["median_depth"].view(-1, 1) * directions
                + batch["linear_velocities_local"] * gt_lidar[..., 3].view(-1, 1)
            )[((pred["ray_drop"].sigmoid() <= 0.5) * gt["valid"])]

        return pred, gt

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        metrics_dict = {}
        if "image" in batch:
            gt_rgb = self.get_gt_img(batch["image"])
            if "rgb" in outputs:
                predicted_rgb = outputs["rgb"]
                metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

            metrics_dict["gaussian_count"] = self.gauss_params["id"].shape[0]
            metrics_dict["gaussian_count_static"] = (self.gauss_params["id"] == -1).sum().item()
            metrics_dict["gaussian_count_objects"] = (self.gauss_params["id"] != -1).sum().item()

        if "raster_pts" in batch and "depth" in outputs:
            pred, gt = self.filter_lidar_pred_and_gt(outputs, batch)

            metrics_dict["depth_median_l2"] = float(self.median_l2(pred["depth"], gt["depth"]))
            metrics_dict["depth_mean_rel_l2"] = float(self.mean_rel_l2(pred["depth"], gt["depth"]))
            metrics_dict["median_depth_median_l2"] = float(self.median_l2(pred["median_depth"], gt["depth"]))
            metrics_dict["median_depth_mean_rel_l2"] = float(self.mean_rel_l2(pred["median_depth"], gt["depth"]))
            metrics_dict["intensity_rmse"] = float(self.rmse(pred["intensity"], gt["intensity"]))
            metrics_dict["ray_drop_accuracy"] = (
                ((pred["ray_drop"].sigmoid() > 0.5) == gt["ray_drop"]) * gt["valid"]
            ).sum() / gt["valid"].sum()

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        self.camera_velocity_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_rgb_loss(self, pred_img, gt_img, loss_mask):
        pred_img = pred_img * loss_mask
        gt_img = gt_img * loss_mask

        l1_loss = torch.abs(gt_img - pred_img).mean()
        ssim_loss = (
            1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
            if self.config.ssim_lambda > 0
            else 0
        )
        return (1 - self.config.ssim_lambda) * l1_loss + self.config.ssim_lambda * ssim_loss

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        loss_dict = {}
        if "image" in batch:
            gt_img = self.get_gt_img(batch["image"])

            loss_mask = torch.ones(gt_img[..., 0:1].shape, dtype=torch.bool, device=self.device)
            if "rgb" in outputs:
                pred_img = outputs["rgb"]
                loss_dict["main_loss"] = self.get_rgb_loss(pred_img, gt_img, loss_mask)

        if self.config.mcmc_scale_reg_lambda and isinstance(self.strategy, ADMCMCStrategy):
            mcmc_scale_reg = torch.abs(torch.exp(self.scales).mean()) * self.config.mcmc_scale_reg_lambda
        else:
            mcmc_scale_reg = torch.zeros(1, device=self.device)
        loss_dict["mcmc_scale_reg"] = mcmc_scale_reg

        if self.config.mcmc_opacity_reg_lambda and isinstance(self.strategy, ADMCMCStrategy):
            mcmc_opacity_reg = torch.abs(torch.sigmoid(self.opacities).mean()) * self.config.mcmc_opacity_reg_lambda
        else:
            mcmc_opacity_reg = torch.zeros(1, device=self.device)
        loss_dict["mcmc_opacity_reg"] = mcmc_opacity_reg

        if self.training:
            self.camera_optimizer.get_loss_dict(loss_dict)
            self.camera_velocity_optimizer.get_loss_dict(loss_dict)

        if "raster_pts" in batch and "depth" in outputs:
            pred, gt = self.filter_lidar_pred_and_gt(outputs, batch)

            unreduced_depth_loss = self.depth_loss(pred["depth"], gt["depth"])
            quantile = torch.quantile(unreduced_depth_loss, self.config.depth_loss_quantile_threshold)
            quantile_mask = unreduced_depth_loss < quantile
            loss_dict["depth_loss"] = self.config.depth_lambda * torch.mean(unreduced_depth_loss * quantile_mask)

            loss_dict["intensity_loss"] = self.config.intensity_lambda * self.intensity_loss(
                pred["intensity"] * quantile_mask,
                gt["intensity"] * quantile_mask,
            )
            loss_dict["ray_drop_loss"] = self.config.ray_drop_lambda * self.ray_drop_loss(
                pred["ray_drop"],
                gt["ray_drop"].to(pred["ray_drop"]),
            )

            if "alpha_sum_until_points" in pred and self.config.line_of_sight_lambda > 0:
                loss_dict["alpha_sum_until_points_loss"] = self.config.line_of_sight_lambda * torch.mean(
                    pred["alpha_sum_until_points"] * quantile_mask
                )

        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera(
        self,
        camera: Union[Cameras, Lidars],
        obb_box: Optional[OrientedBox] = None,
    ) -> Dict[str, Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        outs = self.get_outputs(camera.to(self.device))
        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor], calc_chamfer: bool = False
    ) -> Tuple[Dict[str, float], Dict[str, Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        images_dict = {}
        metrics_dict = {}
        if "image" in batch:
            gt_rgb_pre = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
            predicted_rgb_pre = outputs["rgb"]

            combined_rgb = torch.cat([gt_rgb_pre, predicted_rgb_pre], dim=1).cpu().detach()
            images_dict.update({"img": combined_rgb})

            # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
            gt_rgb = torch.moveaxis(gt_rgb_pre, -1, 0)[None, ...]
            predicted_rgb = torch.moveaxis(predicted_rgb_pre, -1, 0)[None, ...]

            psnr = self.psnr(gt_rgb, predicted_rgb)
            ssim = self.ssim(gt_rgb, predicted_rgb)
            lpips = self.lpips(gt_rgb, predicted_rgb)

            # all of these metrics will be logged as scalars
            metrics_dict.update({"psnr": float(psnr), "ssim": float(ssim), "lpips": float(lpips)})  # type: ignore

            if "dynamic_gt_mask" in batch:
                dynamic_mask = batch["dynamic_gt_mask"].clone()
                assert dynamic_mask.ndim == 3
                if dynamic_mask.shape[2] == 3:
                    for i in range(dynamic_mask.shape[2]):
                        class_name = self.gt_mask_index_to_class_name[i]
                        class_mask = dynamic_mask[:, :, i].clone().to(self.device)
                        if class_mask.sum() > 0:
                            dpsnr = self.psnr(gt_rgb[:, :, class_mask], predicted_rgb[:, :, class_mask])
                            metrics_dict[f"dpsnr_{class_name}"] = float(dpsnr)

                    class_mask = dynamic_mask.any(dim=-1).to(self.device)
                    if class_mask.sum() > 0:
                        dpsnr = self.psnr(gt_rgb[:, :, class_mask], predicted_rgb[:, :, class_mask])
                        metrics_dict["dpsnr_all"] = float(dpsnr)
                elif dynamic_mask.shape[2] == 1:
                    class_mask = dynamic_mask[:, :, 0].clone().to(self.device)
                    if class_mask.sum() > 0:
                        dpsnr = self.psnr(gt_rgb[:, :, class_mask], predicted_rgb[:, :, class_mask])
                        metrics_dict["dpsnr_dynamic"] = float(dpsnr)

                # Plot masked image
                masked_gt_rgb = gt_rgb_pre.clone()
                masked_gt_rgb[~class_mask, :] = 0
                masked_predicted_rgb = predicted_rgb_pre.clone()
                masked_predicted_rgb[~class_mask, :] = 0
                combined_masked_rgb = torch.cat([masked_gt_rgb, masked_predicted_rgb], dim=1).cpu().detach()
                images_dict.update({"masked_img": combined_masked_rgb})

        if "raster_pts" in batch:
            pred, gt = self.filter_lidar_pred_and_gt(outputs, batch, output_point_cloud=True)

            metrics_dict["depth_median_l2"] = float(self.median_l2(pred["depth"], gt["depth"]))
            metrics_dict["depth_mean_rel_l2"] = float(self.mean_rel_l2(pred["depth"], gt["depth"]))
            metrics_dict["median_depth_median_l2"] = float(self.median_l2(pred["median_depth"], gt["depth"]))
            metrics_dict["median_depth_mean_rel_l2"] = float(self.mean_rel_l2(pred["median_depth"], gt["depth"]))
            metrics_dict["intensity_rmse"] = float(self.rmse(pred["intensity"], gt["intensity"]))
            metrics_dict["ray_drop_accuracy"] = float(
                (((pred["ray_drop"].sigmoid() > 0.5) == gt["ray_drop"]) * gt["valid"]).sum() / gt["valid"].sum()
            )

            if calc_chamfer and pred["point_cloud"].shape[0] > 0 and gt["point_cloud"].shape[0] > 0:
                metrics_dict["chamfer_distance"] = float(self.chamfer_distance(pred["point_cloud"], gt["point_cloud"]))

            if calc_chamfer and pred["median_point_cloud"].shape[0] > 0 and gt["point_cloud"].shape[0] > 0:
                metrics_dict["median_chamfer_distance"] = float(
                    self.chamfer_distance(pred["median_point_cloud"], gt["point_cloud"])
                )

        return metrics_dict, images_dict

    def _get_appearance_embedding(self, sensor: Union[Cameras, Lidars], features: Tensor) -> Tensor:
        metadata = sensor.metadata if sensor.metadata is not None else {}
        sensor_idx = metadata.get("sensor_idxs", None)
        if sensor_idx is None:
            assert not self.training, "Sensor sensor_idx must be present in metadata during training"
            sensor_idx = torch.full((1,), self.fallback_sensor_idx.value, device=features.device, dtype=torch.long)

        embed = self.appearance_embedding(sensor_idx).expand(*features.shape[:-1], -1)
        return embed
