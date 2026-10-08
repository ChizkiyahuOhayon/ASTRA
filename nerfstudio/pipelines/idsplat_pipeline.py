# Copyright 2026 the authors of IDSplat and contributors.
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
IDSplat training pipeline with preprocessing for object tracking and registration.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.lidars import Lidars, transform_points
from nerfstudio.data.datamanagers.idsplat_datamanager import IDSplatDatamanager, IDSplatDatamanagerConfig
from nerfstudio.data.utils.data_utils import MaskClassIndex
from nerfstudio.model_components.astra_tivc import tivc_loss
from nerfstudio.models.idsplat import IDSplatModel, IDSplatModelConfig
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.idsplat_utils import (
    SeedChannels,
    add_velocities_to_tracks,
    create_tracks,
    get_eroded_masks,
    get_random_seed_points_inside_bbox,
    get_seed_points_from_tracks,
    merge_instance_ids_by_overlapping_lidar_points,
    recenter_local_coordinate_system,
    remove_static_tracks,
    run_registration,
    set_valid_timestamps_from_mask_times,
    smooth_tracks,
    stack_poses,
    world_to_image,
)


@dataclass
class IDSplatPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: IDSplatPipeline)
    """target class to instantiate"""
    datamanager: IDSplatDatamanagerConfig = field(default_factory=IDSplatDatamanagerConfig)
    """specifies the datamanager config"""
    model: IDSplatModelConfig = field(default_factory=IDSplatModelConfig)
    """specifies the model config"""
    calc_chamfer_steps: Tuple[int, ...] = (99999999,)  # NOTE: must also be an eval step for this to work
    """steps to calculate Chamfer distance"""
    num_downscales: int = 2
    """the number of downscales in the static phase"""
    resolution_schedule: int = 3000
    """the resolution schedule"""

    min_lidar_points_per_mask: int = 5
    max_object_size: float = 10.0

    feature_descriptor: Literal["sam2", "dinov3", "rgb"] = "dinov3"
    sam2_checkpoint: str = "checkpoints/sam2.1_hiera_large.pt"
    """Path to SAM2 checkpoint file"""
    sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    """Path to SAM2 model config file"""
    dino_layer_idx: int = 7
    dino_use_pca: bool = False
    dino_pca_dim: int = 3
    anyup_upsampling_factor: int = 4
    use_anyup: bool = False

    ransac_num_iterations: int = 100_000
    ransac_distance_threshold: float = 0.20
    ransac_mutual: bool = False
    ransac_batch_size: int = 100_000
    ransac_refine: bool = True
    min_registration_fitness_for_pose_update: float = 0.5
    min_registration_fitness_for_adding_points: float = 0.5
    min_registration_fitness_for_measurement: float = 0.5

    smooth_poses_at_init: bool = True
    smooth_poses_w_abs_mult: float = 1.0
    run_iterative_smoothing: bool = True

    merge_instance_ids_by_overlapping_lidar_points: bool = True
    dist_threshold_for_removing_static_tracks: float = 0.25
    set_valid_timestamps_from_mask_times: bool = True

    add_random_dynamic_seed_points: bool = True
    n_random_dynamic_seed_points: int = 1000

    use_eroded_masks: bool = True
    n_pixels_to_shrink: int = 3

    use_dbscan_filtering: bool = True

    tivc_lambda: float = 0.0
    """ASTRA virtual-view consistency: loss weight (0 disables it)."""
    tivc_every: int = 2
    """ASTRA virtual-view consistency: apply on every n-th camera training step."""
    tivc_start: int = 6000
    """ASTRA virtual-view consistency: first step (after the resolution schedule)."""
    tivc_scale: int = 2
    """ASTRA virtual-view consistency: the virtual view is rendered at 1/scale resolution."""
    tivc_consistency: float = 0.0
    """ASTRA virtual-view consistency: >0 keeps only pixels where the two warped neighbours agree (max mean abs RGB
    difference); 0 uses the per-pixel minimum over the two warps."""
    tivc_min_alpha: float = 0.0
    """ASTRA virtual-view consistency: with the agreement check, ignore pixels with accumulation below this (sky)."""
    tivc_yaw: float = 0.0
    """ASTRA off-trajectory virtual views: max |yaw| (deg) about the camera up axis."""
    tivc_lateral: float = 0.0
    """ASTRA off-trajectory virtual views: max |sideways shift| (m) along the camera right axis."""
    tivc_spatial_prob: float = 0.0
    """ASTRA off-trajectory virtual views: fraction of virtual views moved off the trajectory."""


class IDSplatPipeline(VanillaPipeline):
    """The pipeline class for IDSplat."""

    def __init__(
        self,
        config: IDSplatPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        Pipeline.__init__(self)
        self.config: IDSplatPipelineConfig = config
        self.test_mode = test_mode
        self.datamanager: IDSplatDatamanager = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
        )
        if test_mode in ("inference", "test"):
            # if test mode is inference we assume that we load everything from checkpoint
            self._model: IDSplatModel = config.model.setup(
                scene_box=self.datamanager.train_dataset.scene_box,
                num_train_data=self.datamanager.get_num_train_data(),
                metadata=self.datamanager.train_dataset.metadata,
                device=device,
                grad_scaler=grad_scaler,
                seed_points=torch.empty((0, 7)).cpu(),
                tracks=[],
            )
            self.model.to(device)
        else:
            seed_points = self.get_seed_points()
            seed_points = torch.cat(seed_points, dim=-1).to(device)
            seed_points[:, SeedChannels.RGB : SeedChannels.RGB + 3] = (
                seed_points[:, SeedChannels.RGB : SeedChannels.RGB + 3] / 255.0
            )
            seed_points = seed_points[
                (seed_points[:, SeedChannels.OBJ_ID] == -1).squeeze(-1)
            ]  # only keep static points
            self.datamanager.to(device)
            assert self.datamanager.train_dataset is not None, "Missing input dataset"
            unique_timestamps = self.datamanager.train_dataset.metadata["lidar_times"].unique().to(device)

            feature_descriptor_func = self._setup_feature_descriptor(device)
            initial_object_points = self._create_initial_object_points(feature_descriptor_func)
            self._teardown_feature_descriptor()

            # ASTRA: a scene where no mask gathers enough lidar points has no objects to register; train it static-only
            # instead of crashing in merge_instance_ids_by_overlapping_lidar_points (KeyError: mask_ids).
            if "mask_ids" not in initial_object_points or initial_object_points["mask_ids"].numel() == 0:
                print("[ASTRA] no object points found: training without dynamic objects")
                tracks, reg_results_per_id = [], None
            else:
                if self.config.merge_instance_ids_by_overlapping_lidar_points:
                    merge_instance_ids_by_overlapping_lidar_points(initial_object_points)

                reg_results_per_id = run_registration(
                    initial_object_points=initial_object_points,
                    device=device,
                    ransac_num_iterations=self.config.ransac_num_iterations,
                    ransac_distance_threshold=self.config.ransac_distance_threshold,
                    ransac_mutual=self.config.ransac_mutual,
                    ransac_batch_size=self.config.ransac_batch_size,
                    ransac_refine=self.config.ransac_refine,
                    min_fitness_for_pose_update=self.config.min_registration_fitness_for_pose_update,
                )

                seed_points = remove_static_tracks(
                    reg_results_per_id=reg_results_per_id,
                    seed_points=seed_points,
                    dist_threshold=self.config.dist_threshold_for_removing_static_tracks,
                )

                reg_results_per_id = stack_poses(reg_results_per_id)
                tracks = create_tracks(reg_results_per_id, unique_timestamps)
                if self.config.smooth_poses_at_init:
                    tracks = smooth_tracks(
                        tracks,
                        unique_timestamps,
                        reg_results_per_id,
                        w_abs_mult=self.config.smooth_poses_w_abs_mult,
                        min_fitness_for_measurement=self.config.min_registration_fitness_for_measurement,
                        iterative=self.config.run_iterative_smoothing,
                    )

                recenter_local_coordinate_system(tracks, reg_results_per_id)

                # velocities for rolling shutter compensation in idsplat
                add_velocities_to_tracks(tracks, unique_timestamps)

                if self.config.set_valid_timestamps_from_mask_times:
                    start_time_per_mask_id, end_time_per_mask_id = self._get_start_and_end_time_per_mask_id(
                        mask_ids=[t.all_mask_ids for t in tracks],
                        max_gap=3.0,
                    )
                    set_valid_timestamps_from_mask_times(
                        tracks, unique_timestamps, start_time_per_mask_id, end_time_per_mask_id, max_pred_time=3.0
                    )

                dynamic_seed_points = get_seed_points_from_tracks(
                    tracks, reg_results_per_id, self.config.min_registration_fitness_for_adding_points
                )
                if self.config.add_random_dynamic_seed_points:
                    random_dynamic_seed_points = get_random_seed_points_inside_bbox(
                        tracks, self.config.n_random_dynamic_seed_points
                    )
                    dynamic_seed_points = torch.cat([dynamic_seed_points, random_dynamic_seed_points], dim=0)

                seed_points = torch.cat([seed_points, dynamic_seed_points.to(seed_points.device)], dim=0)

            del initial_object_points
            del reg_results_per_id
            torch.cuda.empty_cache()

            self._model: IDSplatModel = config.model.setup(
                scene_box=self.datamanager.train_dataset.scene_box,
                num_train_data=self.datamanager.get_num_train_data(),
                metadata=self.datamanager.train_dataset.metadata,
                device=device,
                grad_scaler=grad_scaler,
                seed_points=seed_points.cpu(),
                tracks=tracks,
            )
            self.model.to(device)


    def get_seed_points(self):
        dp_meta = self.datamanager.train_dataparser_outputs.metadata
        assert (
            hasattr(self.datamanager, "train_dataparser_outputs")
            and "point_clouds" in dp_meta
            and "lidars" in dp_meta
            and "point_clouds_rgb" in dp_meta
            and "point_clouds_is_dynamic" in dp_meta
        )
        points_in_world = []
        returning_masks = []
        for l2w, pc in zip(
            dp_meta["lidars"].lidar_to_worlds,
            dp_meta["point_clouds"],
        ):
            returning = pc[:, :3].norm(dim=-1) < dp_meta["lidars"].valid_lidar_distance_threshold
            returning_masks.append(returning)
            points_in_world.append(transform_points(pc[returning, :3], l2w))
        points_in_world = torch.cat([pc_[:, :3] for pc_ in points_in_world], dim=0)
        points_in_world_rgb = torch.cat([c_[r_] for c_, r_ in zip(dp_meta["point_clouds_rgb"], returning_masks)])
        points_in_world_is_dynamic = torch.cat(dp_meta["point_clouds_is_dynamic"], dim=0).unsqueeze(-1)
        points_in_world_obj_id = points_in_world_is_dynamic.int() - 1  # offset to give static points id -1

        seed_pts = (
            points_in_world,
            points_in_world_rgb,
            points_in_world_obj_id,
        )
        return seed_pts

    def _create_initial_object_points(self, feature_descriptor_func: Callable):
        object_points = {}
        prev_max_actor_id = -1
        for cam_idx in tqdm(
            range(len(self.datamanager.train_cameras)),
            desc="Creating dynamic gaussians",
        ):
            seed_cam, seed_batch = self.datamanager.get_train_image_from_idx(cam_idx)
            lidar_idx = int(torch.argmin((self.datamanager.train_lidars.times - seed_cam.times).abs().squeeze()).item())
            closest_lidar = self.datamanager.train_lidars[lidar_idx]
            closest_lidar_pc = self.datamanager.cached_lidar_train[lidar_idx]["lidar"]
            closest_lidar_is_dynamic = self.datamanager.cached_lidar_train[lidar_idx]["is_dynamic"].squeeze(-1)

            pts_from_cam = self._get_object_points_from_masks(
                seed_batch["image"],
                seed_batch["mask"] * (seed_batch["mask_classes"] == MaskClassIndex.VEHICLE),
                seed_cam[0],
                closest_lidar,
                closest_lidar_pc[closest_lidar_is_dynamic],
                lidar_idx,
                prev_max_actor_id,
                feature_descriptor_func,
            )
            if len(pts_from_cam["actor_ids"]) > 0:
                prev_max_actor_id = max(prev_max_actor_id, pts_from_cam["actor_ids"].max().item())

            for k in pts_from_cam:
                if len(pts_from_cam[k]) > 0:
                    if k in object_points:
                        object_points[k].append(pts_from_cam[k])
                    else:
                        object_points[k] = [pts_from_cam[k]]

        for k, v in object_points.items():
            object_points[k] = torch.cat(v, dim=0)

        return object_points

    @profiler.time_function
    def get_train_loss_dict(
        self, step: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Union[float, int]]]:
        """Get training loss dictionary."""
        sensor, batch = self.datamanager.next_train(step)
        self.model.downscale_factor = self._get_downscale_factor(step)
        model_outputs = self._model(sensor)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        metrics_dict["downscale_factor"] = self.model.downscale_factor
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        if (
            self.config.tivc_lambda > 0
            and isinstance(sensor, Cameras)
            and step >= self.config.tivc_start
            and step % self.config.tivc_every == 0
        ):
            tivc = tivc_loss(
                self,
                sensor,
                scale=self.config.tivc_scale,
                consistency=self.config.tivc_consistency,
                min_alpha=self.config.tivc_min_alpha,
                yaw_deg=self.config.tivc_yaw,
                lateral=self.config.tivc_lateral,
                spatial_prob=self.config.tivc_spatial_prob,
            )
            if tivc is not None:
                loss_dict["tivc_loss"] = self.config.tivc_lambda * tivc

        return model_outputs, loss_dict, metrics_dict

    def _get_downscale_factor(self, step: int) -> int:
        if self.training:
            return 2 ** max(
                (self.config.num_downscales - step // self.config.resolution_schedule),
                0,
            )
        else:
            return 1

    def _get_object_points_from_masks(
        self,
        image: torch.Tensor,
        masks: torch.Tensor,
        cam: Cameras,
        lidar: Lidars,
        pc: torch.Tensor,
        lidar_idx: int,
        prev_max_actor_id: int,
        feature_descriptor_func: Callable,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        assert (cam.width == masks.shape[1]) and (cam.height == masks.shape[0])

        # helps remove some bleeding from misalignment between lidar and camera
        if self.config.use_eroded_masks:
            eroded_masks = get_eroded_masks(
                masks.squeeze(-1), n_pixels_to_shrink=self.config.n_pixels_to_shrink
            ).unsqueeze(-1)
        else:
            eroded_masks = masks.clone().squeeze(-1)

        with torch.no_grad():
            image_features = feature_descriptor_func(image)
        if image_features.shape[0] != cam.height or image_features.shape[1] != cam.width:
            image_features = (
                torch.nn.functional.interpolate(
                    image_features.permute(2, 0, 1).unsqueeze(0),
                    size=(cam.height, cam.width),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .permute(1, 2, 0)
            )

        # project lidar points to image and extract corresponding pixels from depth image
        is_close = (pc[:, :3].norm(dim=-1) >= 0.1) & (pc[:, :3].norm(dim=-1) <= 80.0)
        pc_in_world = transform_points(pc[is_close, :3], lidar.lidar_to_worlds)
        pc_in_image, depths, valid_points = world_to_image(pc_in_world, cam)
        pc_in_image = pc_in_image[valid_points]  # W, H
        pc_mask_indices = eroded_masks[pc_in_image[:, 1], pc_in_image[:, 0]]

        gaussians = {
            "coords": [],
            "rgbs": [],
            "mask_ids": [],
            "actor_ids": [],
            "cam_idx": [],
            "lidar_idx": [],
            "lidar_point_ids": [],
            "desc_feats": [],
        }
        new_actor_id = prev_max_actor_id + 1
        for mask_id in eroded_masks.unique():
            if mask_id == 0:
                continue

            pc_mask = (pc_mask_indices == mask_id).squeeze()
            if pc_mask.sum() < self.config.min_lidar_points_per_mask:
                continue

            curr_pc_in_world = pc_in_world[valid_points][pc_mask]

            if self.config.use_dbscan_filtering:
                # filter out outliers using clustering (assumes that the largest cluster is the object)
                dbscan = DBSCAN(eps=0.5, min_samples=10)
                labels = dbscan.fit_predict(curr_pc_in_world.cpu().numpy())
                unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
                if len(unique_labels) == 0:
                    continue
                largest_cluster_label = unique_labels[np.argmax(counts)]
                keep_mask = labels == largest_cluster_label
            else:
                keep_mask = torch.ones_like(curr_pc_in_world[:, 0], dtype=torch.bool)

            if keep_mask.sum() < self.config.min_lidar_points_per_mask:
                continue

            obj_points = curr_pc_in_world[keep_mask]
            size = obj_points.max(dim=0).values - obj_points.min(dim=0).values
            if (size > self.config.max_object_size).any():
                continue

            obj_points_img_coords = pc_in_image[pc_mask][keep_mask]
            obj_rgbs = image[obj_points_img_coords[:, 1], obj_points_img_coords[:, 0]] / 255.0
            obj_features = image_features[obj_points_img_coords[:, 1], obj_points_img_coords[:, 0]]

            gaussians["coords"].append(obj_points)
            gaussians["rgbs"].append(obj_rgbs)
            gaussians["mask_ids"].append(mask_id * torch.ones_like(obj_points[:, :1], dtype=torch.int32))
            gaussians["actor_ids"].append(new_actor_id * torch.ones_like(obj_points[:, :1], dtype=torch.int32))
            gaussians["cam_idx"].append(cam.metadata["cam_idx"] * torch.ones_like(obj_points[:, :1], dtype=torch.int32))
            gaussians["lidar_idx"].append(lidar_idx * torch.ones_like(obj_points[:, :1], dtype=torch.int32))
            gaussians["lidar_point_ids"].append(valid_points.nonzero()[pc_mask][keep_mask])
            gaussians["desc_feats"].append(obj_features)
            new_actor_id += 1

        for k, v in gaussians.items():
            if len(v) > 0:
                gaussians[k] = torch.cat(v, dim=0)

        return gaussians

    @profiler.time_function
    def get_eval_loss_dict(
        self, step: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Union[float, int]]]:
        """Get evaluation loss dictionary."""
        self.eval()
        camera, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(camera)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_image_metrics_and_images(
        self, step: int
    ) -> Tuple[Dict[str, Union[float, int]], Dict[str, torch.Tensor]]:
        """Get evaluation image metrics and images."""
        self.eval()
        camera, batch = self.datamanager.next_eval_image(step)
        outputs = self.model.get_outputs_for_camera(camera)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(
            outputs, batch, calc_chamfer=step in self.config.calc_chamfer_steps
        )
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = (camera.height * camera.width * camera.size).item()

        lidar, batch = self.datamanager.next_eval_lidar(step)
        outputs = self.model.get_lidar_outputs(lidar)
        lidar_metrics_dict, lidar_images_dict = self.model.get_image_metrics_and_images(
            outputs, batch, calc_chamfer=step in self.config.calc_chamfer_steps
        )
        images_dict.update(lidar_images_dict)
        assert not set(lidar_metrics_dict.keys()).intersection(metrics_dict.keys())
        metrics_dict.update(lidar_metrics_dict)
        self.train()
        return metrics_dict, images_dict

    @profiler.time_function
    def get_average_eval_image_metrics(
        self,
        step: Optional[int] = None,
        output_path: Optional[Path] = None,
        get_std: bool = False,
        dump_img_to_disk: bool = False,
    ) -> Dict[str, Union[float, int]]:
        """Get average evaluation image metrics."""
        self.eval()
        metrics_dict_list = []
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        num_lidar = len(self.datamanager.fixed_indices_eval_lidar_dataloader)
        assert isinstance(self.datamanager, IDSplatDatamanager)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for camera, batch in self.datamanager.fixed_indices_eval_dataloader:
                inner_start = time.time()
                outputs = self.model.get_outputs_for_camera(camera=camera)
                torch.cuda.synchronize()
                inference_time_camera = time.time() - inner_start
                height, width = camera.height, camera.width
                num_camera_rays = height * width
                metrics_dict, _ = self.model.get_image_metrics_and_images(
                    outputs, batch, calc_chamfer=step in self.config.calc_chamfer_steps
                )
                pred_height, pred_width = outputs["rgb"].shape[:2]
                batch["image"] = batch["image"][:pred_height, :pred_width]
                metrics_dict["num_camera_rays_per_sec"] = (num_camera_rays / inference_time_camera).item()
                metrics_dict["fps"] = (metrics_dict["num_camera_rays_per_sec"] / (height * width)).item()
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)

            task = progress.add_task("[green]Evaluating all eval point clouds...", total=num_lidar)
            for lidar, batch in self.datamanager.fixed_indices_eval_lidar_dataloader:
                inner_start = time.time()
                outputs = self.model.get_lidar_outputs(lidar)
                inference_time_lidar = time.time() - inner_start
                metrics_dict, _ = self.model.get_image_metrics_and_images(
                    outputs, batch, calc_chamfer=step in self.config.calc_chamfer_steps
                )
                num_lidar_rays = (batch["raster_pts"][..., 2] > 0).sum()
                metrics_dict["num_lidar_rays_per_sec"] = (num_lidar_rays / inference_time_lidar).item()
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)

        metrics_dict = {}
        keys = {key for metrics_dict in metrics_dict_list for key in metrics_dict.keys()}
        for key in keys:
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list if key in metrics_dict])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(
                        torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list if key in metrics_dict])
                    )
                )
        self.train()
        return metrics_dict

    def _setup_feature_descriptor(self, device):
        feature_descriptor_func = None
        if self.config.feature_descriptor == "sam2":
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            self.sam2_model = build_sam2(self.config.sam2_model_cfg, self.config.sam2_checkpoint, device=device)
            self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

            @torch.no_grad()
            def extract_features(img: torch.Tensor) -> torch.Tensor:
                assert img.ndim == 3 and img.shape[2] == 3, "Image input is expected to be [H, W, 3]"
                img_size = img.shape[:2]
                self.sam2_predictor.set_image(img.cpu().numpy())
                image_embeddings = self.sam2_predictor._features["image_embed"]
                high_res_features = self.sam2_predictor._features["high_res_feats"]
                extracted_features = torch.cat(
                    [
                        torch.nn.functional.interpolate(f, size=img_size, mode="bilinear", align_corners=False)
                        .squeeze(0)
                        .permute(1, 2, 0)
                        for f in [image_embeddings, *high_res_features]
                    ],
                    dim=-1,
                )
                assert extracted_features.shape[:2] == (img_size[0], img_size[1])
                return extracted_features

            feature_descriptor_func = extract_features

        elif self.config.feature_descriptor == "dinov3":
            from transformers import AutoImageProcessor, AutoModel

            dino_model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
            self.dino_processor = AutoImageProcessor.from_pretrained(dino_model_name, do_resize=False, do_rescale=True)
            self.dino_model = AutoModel.from_pretrained(dino_model_name, device_map=None).eval()
            self.dino_model.to(device)

            self.upsampler = None
            if self.config.use_anyup:
                self.upsampler = torch.hub.load("wimmerth/anyup", "anyup")
                self.upsampler.to(device)

            @torch.no_grad()
            def extract_features(img: torch.Tensor) -> torch.Tensor:
                assert img.ndim == 3 and img.shape[2] == 3, "Image input is expected to be [H, W, 3]"
                img_size = img.shape[:2]
                img = img.permute(2, 0, 1).unsqueeze(0)
                h_feat, w_feat = img.shape[2] // 16, img.shape[3] // 16
                inputs = self.dino_processor(img, return_tensors="pt")
                outputs = self.dino_model(**inputs, output_hidden_states=True)
                sel_feats = outputs.hidden_states[self.config.dino_layer_idx][:, 5:, :]
                feats = sel_feats.squeeze(0)  # squeeze batch dim

                if self.config.dino_use_pca:
                    mc_feats = feats - feats.mean(dim=0, keepdim=True)
                    _, _, v = torch.pca_lowrank(
                        mc_feats, q=self.config.dino_pca_dim + 5
                    )  # 5 is a small oversampling factor
                    feats = mc_feats @ v[:, : self.config.dino_pca_dim]

                if self.upsampler is not None:
                    feat_dim = feats.shape[-1]
                    feats = feats.permute(1, 0).view(1, feat_dim, h_feat, w_feat)  # [1, C, h, w]
                    feats = self.upsampler(
                        inputs["pixel_values"],
                        feats,
                        output_size=(
                            h_feat * self.config.anyup_upsampling_factor,
                            w_feat * self.config.anyup_upsampling_factor,
                        ),
                        q_chunk_size=1024,
                    )
                    feats = feats.permute(0, 2, 3, 1).squeeze(0)
                    feats = feats.reshape(
                        h_feat * self.config.anyup_upsampling_factor,
                        w_feat * self.config.anyup_upsampling_factor,
                        self.config.dino_pca_dim,
                    )
                else:
                    feats = feats.reshape(h_feat, w_feat, feats.shape[-1])

                feats = feats.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
                feats = torch.nn.functional.interpolate(feats, size=img_size, mode="bilinear", align_corners=False)
                return feats.squeeze(0).permute(1, 2, 0)  # [H, W, 3]

            feature_descriptor_func = extract_features

        elif self.config.feature_descriptor == "rgb":

            @torch.no_grad()
            def extract_features(img: torch.Tensor) -> torch.Tensor:
                if img.dtype == torch.uint8:
                    img = img.float() / 255.0
                assert img.dtype == torch.float32, "Image input is expected to be in float32"
                assert img.ndim == 3 and img.shape[2] == 3, "Image input is expected to be [H, W, 3]"
                assert img.max() <= 1.0 and img.min() >= 0.0, "Image input is expected to be in [0, 1]"
                return img

            feature_descriptor_func = extract_features

        else:
            raise ValueError(f"Invalid feature descriptor: {self.config.feature_descriptor}")

        return feature_descriptor_func

    def _teardown_feature_descriptor(self):
        if self.config.feature_descriptor == "sam2":
            del self.sam2_model
            del self.sam2_predictor
            torch.cuda.empty_cache()
        elif self.config.feature_descriptor == "dinov3":
            del self.dino_model
            del self.dino_processor
            if hasattr(self, "upsampler") and self.upsampler is not None:
                del self.upsampler
            torch.cuda.empty_cache()

    def _get_start_and_end_time_per_mask_id(
        self, mask_ids: List[torch.Tensor], max_gap: float = 3.0
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        start_time_per_mask_id = {}
        end_time_per_mask_id = {}
        times_per_mask_id = defaultdict(list)
        for cam_idx in range(len(self.datamanager.train_cameras)):
            cam, batch = self.datamanager.get_train_image_from_idx(cam_idx)
            time = cam.times.item()
            for mask_id in mask_ids:
                for unique_id in mask_id.tolist():
                    if unique_id in batch["mask"].unique().tolist():
                        times_per_mask_id[unique_id].append(time)

        for mask_id, times in times_per_mask_id.items():
            times = torch.tensor(sorted(times))
            gaps = torch.diff(times)
            large_gaps = gaps > max_gap
            split_indices = large_gaps.nonzero().flatten()
            clusters = []
            start_idx = 0
            for split_idx in split_indices:
                clusters.append(times[start_idx:split_idx])
                start_idx = split_idx + 1
            clusters.append(times[start_idx:])
            largest_cluster = max(clusters, key=len)
            start_time_per_mask_id[mask_id] = largest_cluster[0].item()
            end_time_per_mask_id[mask_id] = largest_cluster[-1].item()

        return start_time_per_mask_id, end_time_per_mask_id
