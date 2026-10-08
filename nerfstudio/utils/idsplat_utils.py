# Copyright 2026 the authors of IDSplat and contributors.
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
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.lidars import transform_points
from nerfstudio.model_components.objects_manager import Object
from nerfstudio.utils.ct_smoothing_utils import get_smoothed_poses
from nerfstudio.utils.poses import interpolate_trajectories, inverse as pose_inverse, rotation_difference, to4x4
from nerfstudio.utils.ransac_utils_torch_batched import (
    compute_registration_fitness,
    ransac_rigid_registration_torch_batched,
)


@dataclass
class SeedChannels:
    COORDS: int = 0
    RGB: int = 3
    OBJ_ID: int = 6
    N_CHANNELS: int = 7


class UnionFind:
    def __init__(self, ids: Tensor):
        self.parent = {int(i): int(i) for i in ids.unique()}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a

    def merge(self, mapping_list: Tensor):
        for a, b in mapping_list.tolist():
            self.union(int(a), int(b))


def world_to_image(pc_in_world: Tensor, cam: Cameras, filter_depth: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
    pc_in_camera = transform_points(pc_in_world, pose_inverse(cam.camera_to_worlds.squeeze(0)))
    # Flip the y and z axis because of nerfstudio conventions
    pc_in_camera[:, 1] = -pc_in_camera[:, 1]
    pc_in_camera[:, 2] = -pc_in_camera[:, 2]
    depths = pc_in_camera[:, 2]
    # Only paint points in front of the camera
    valid_points = depths > 0
    # Normalize the points
    pc_in_camera = pc_in_camera / pc_in_camera[:, 2:]

    intrinsics = cam.get_intrinsics_matrices().squeeze(0)
    pc_in_image = (torch.matmul(intrinsics, pc_in_camera[:, :3].T).T).to(torch.int64)

    # Only paint points that are within the image
    valid_points = (
        valid_points
        & (pc_in_image[:, 0] >= 0)
        & (pc_in_image[:, 0] < cam.width)
        & (pc_in_image[:, 1] >= 0)
        & (pc_in_image[:, 1] < cam.height)
    )

    if filter_depth:
        # Filter out points based on distance
        unique_pixels, counts = torch.unique(pc_in_image[valid_points], return_counts=True, dim=0)
        if len(unique_pixels) > 0 and counts.max() > 1:
            indices_to_check = (counts > 1).nonzero().flatten()
            for idx in indices_to_check:
                pixel = unique_pixels[idx]
                # Get all points that map to this pixel
                points_in_pixel = (pc_in_image == pixel).all(dim=1)
                point_indices = points_in_pixel.nonzero().flatten()
                # Get the closest point in depth
                closest_point_idx = point_indices[torch.argmin(depths[points_in_pixel])]
                # Set all other points in this pixel to invalid
                valid_points[point_indices] = False
                valid_points[closest_point_idx] = True

    return pc_in_image, depths, valid_points


def image_to_world(cam: Cameras, img_coords: Tensor, depths: Tensor) -> Tensor:
    K = cam.get_intrinsics_matrices().squeeze(0)

    # Extract intrinsic parameters
    fx, fy = K[0, 0], K[1, 1]  # Focal lengths
    cx, cy = K[0, 2], K[1, 2]  # Principal point

    # Compute normalized coordinates
    x_c = (img_coords[:, 0] - cx) * depths / fx
    y_c = (img_coords[:, 1] - cy) * depths / fy
    z_c = depths

    # Stack into a 3D points
    cam_coords = torch.stack([x_c, y_c, z_c], dim=1)
    # Flip the y and z axis because of nerfstudio conventions
    cam_coords[:, 1] = -cam_coords[:, 1]
    cam_coords[:, 2] = -cam_coords[:, 2]

    # Transform to world coordinates
    world_coords = transform_points(cam_coords, cam.camera_to_worlds.squeeze(0))
    return world_coords


def get_eroded_masks(orig_masks: Tensor, n_pixels_to_shrink: int = 10) -> Tensor:
    """Erodes the masks by a fixed number of pixels. Input masks are of type integer and have shape (H, W).
    Integer values represent different objects."""
    eroded_masks = torch.zeros_like(orig_masks)
    for obj_idx in orig_masks.unique():  # range(1, orig_masks.max() + 1):
        if obj_idx == 0:  # 0 is treated as background
            continue
        mask = orig_masks == obj_idx

        n_pixels_to_shrink = max(int((mask.sum() * 0.20) / (mask.sum().sqrt() * 4)), 1)
        kernel_size = 2 * n_pixels_to_shrink + 1
        # Apply 2D max pooling, which acts as an erosion in this case
        eroded_mask = -torch.nn.functional.max_pool2d(
            -mask.view(1, 1, *orig_masks.shape).float(),
            kernel_size,
            stride=1,
            padding=n_pixels_to_shrink,
        )
        eroded_mask = eroded_mask.squeeze().bool()
        eroded_masks[eroded_mask] = obj_idx

    return eroded_masks


def voxel_based_relabeling(points: torch.Tensor, labels: torch.Tensor, voxel_size: float) -> torch.Tensor:
    voxel_coords = torch.floor(points / voxel_size).int()
    voxel_keys, inverse_indices = torch.unique(voxel_coords, return_inverse=True, dim=0)

    num_voxels = voxel_keys.shape[0]
    num_classes = int(labels.max().item() + 1)

    votes = torch.zeros((num_voxels, num_classes), device=points.device, dtype=torch.int32)
    votes.index_add_(0, inverse_indices, torch.nn.functional.one_hot(labels, num_classes).int())

    majority_labels = votes.argmax(dim=1)
    relabeled = majority_labels[inverse_indices]
    return relabeled


def compute_mask_overlap_matrix(mask_ids: Tensor, lidar_indices: Tensor, point_ids: Tensor) -> Tuple[Tensor, Tensor]:
    """
    mask_point_pairs: (N, 2)
        column 0: mask_id  (0..num_masks-1)
        column 1: lidar_idx
        column 2: point_id in lidar
    num_masks: int
        Total number of masks.

    Returns:
        (num_masks, num_masks) where entry [i, j] is the overlap count.
    """
    assert len(lidar_indices.shape) == 1
    assert len(point_ids.shape) == 1
    assert len(mask_ids.shape) == 1
    # Sort by point_id to group same points
    sort_idx = torch.argsort(lidar_indices)
    mask_ids = mask_ids[sort_idx]
    lidar_indices = lidar_indices[sort_idx]
    point_ids = point_ids[sort_idx]

    mask_ids, mask_id_remapping, original_unique_mask_ids = remap_ids_to_contiguous(mask_ids)
    num_masks = int(mask_ids.max() + 1)

    # Result matrix
    overlap_mat = torch.zeros((num_masks, num_masks), dtype=mask_ids.dtype, device=mask_ids.device)

    for lidar_idx in lidar_indices.unique():
        lidar_mask = lidar_indices == lidar_idx
        curr_mask_ids = mask_ids[lidar_mask]
        curr_point_ids = point_ids[lidar_mask]
        curr_point_ids, _, _ = remap_ids_to_contiguous(curr_point_ids)
        num_points = int(curr_point_ids.max() + 1)
        M = torch.zeros((num_masks, num_points), dtype=torch.float32, device=mask_ids.device)
        M[curr_mask_ids, curr_point_ids] = 1.0
        overlap_f = M @ M.T
        overlap_mat += torch.round(overlap_f).to(dtype=overlap_mat.dtype)

    return overlap_mat, original_unique_mask_ids


def remap_ids_to_contiguous(ids: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Remap the ids to be contiguous.

    Args:
        ids: the ids to remap

    Returns:
        remapped_ids: the remapped ids
        id_remapping: the mapping from the original ids to the remapped ids
    """
    assert ids.numel() > 0, "No ids to remap"
    unique_ids = ids.unique()
    id_remapping = torch.ones(unique_ids.max() + 1, dtype=unique_ids.dtype, device=unique_ids.device) * -1
    id_remapping[unique_ids] = torch.arange(len(unique_ids), device=unique_ids.device, dtype=unique_ids.dtype)
    remapped_ids = id_remapping[ids]
    return remapped_ids, id_remapping, unique_ids


def _get_pose_and_size_from_coords(coords):
    min_coords, max_coords = (
        coords.min(dim=0).values,
        coords.max(dim=0).values,
    )
    size = max_coords - min_coords
    center = min_coords + size / 2.0
    pose = torch.eye(4, device=center.device)
    pose[:3, 3] = center
    return pose, size


def merge_instance_ids_by_overlapping_lidar_points(initial_object_points: Dict[str, Tensor]):
    mask_instance_ids = initial_object_points["mask_ids"].squeeze(-1).int()
    mask_overlap_matrix, original_unique_mask_ids = compute_mask_overlap_matrix(
        mask_ids=mask_instance_ids,
        lidar_indices=initial_object_points["lidar_idx"].squeeze(-1).int(),
        point_ids=initial_object_points["lidar_point_ids"].squeeze(-1).int(),
    )
    overlapping_masks = (torch.triu(mask_overlap_matrix, diagonal=1) > 50).nonzero()
    overlapping_masks = original_unique_mask_ids[overlapping_masks]
    uf = UnionFind(initial_object_points["mask_ids"].squeeze(-1).int())
    uf.merge(overlapping_masks)
    merged_mask_instance_ids = torch.tensor(
        [uf.find(int(x)) for x in mask_instance_ids],
        device=mask_instance_ids.device,
    )
    initial_object_points["mask_ids"] = merged_mask_instance_ids.unsqueeze(-1)


@dataclass
class RegistrationResults:
    poses: List[Tensor]
    size: Tensor
    lidar_indices: List[int]
    lidar_points: List[Tensor]
    feats: List[Tensor]
    rgbs: List[Tensor]
    fitness: List[float]


def get_relative_pose(
    src_pts: Tensor,
    tgt_pts: Tensor,
    src_feats: Tensor,
    tgt_feats: Tensor,
    ransac_num_iterations: int = 100_000,
    ransac_distance_threshold: float = 0.10,
    ransac_mutual: bool = False,
    ransac_batch_size: int = 100_000,
    ransac_refine: bool = True,
    device: torch.device = torch.device("cuda"),
) -> Tuple[Tensor, float]:
    rot, trans, _ = ransac_rigid_registration_torch_batched(
        src_pts,
        tgt_pts,
        src_feats,
        tgt_feats,
        num_hypotheses=ransac_num_iterations,
        batch_size=ransac_batch_size,
        distance_threshold=ransac_distance_threshold,
        mutual=ransac_mutual,
        refine=ransac_refine,
        device=device,
    )
    if rot is not None and trans is not None:
        fitness = compute_registration_fitness(
            src_pts,
            tgt_pts,
            rot,
            trans,
            threshold=0.1,
            mode="tgt_coverage",
        )
    else:
        fitness = 0.0

    if rot is not None and trans is not None:
        pose_src_to_tgt = to4x4(torch.cat([rot, trans.unsqueeze(-1)], dim=-1))
    else:
        pose_src_to_tgt = torch.eye(4, device=src_pts.device, dtype=src_pts.dtype)

    return pose_src_to_tgt, fitness


def run_registration(
    initial_object_points: Dict[str, Tensor],
    device: torch.device,
    ransac_num_iterations: int = 100_000,
    ransac_distance_threshold: float = 0.10,
    ransac_mutual: bool = False,
    ransac_batch_size: int = 100_000,
    ransac_refine: bool = True,
    min_fitness_for_pose_update: float = 0.1,
) -> Dict[int, RegistrationResults]:
    reg_results_per_id: Dict[int, RegistrationResults] = {}
    for id_ in tqdm(initial_object_points["mask_ids"].unique(), desc="Running registration"):
        id_ = int(id_.item())
        curr_id_mask = (initial_object_points["mask_ids"] == id_).squeeze(-1)
        lidar_indices, n_lidar_points = initial_object_points["lidar_idx"][curr_id_mask].unique(return_counts=True)
        sort_indices = torch.argsort(n_lidar_points, descending=True)
        lidar_indices = lidar_indices[sort_indices]
        for lidar_idx in lidar_indices:
            curr_lidar_mask = (initial_object_points["lidar_idx"][curr_id_mask] == lidar_idx).squeeze(-1)
            new_pts = initial_object_points["coords"][curr_id_mask][curr_lidar_mask]
            new_feats = torch.nn.functional.normalize(
                initial_object_points["desc_feats"][curr_id_mask][curr_lidar_mask]
            )
            new_rgbs = initial_object_points["rgbs"][curr_id_mask][curr_lidar_mask]

            if id_ not in reg_results_per_id:
                # initialize track
                pose, size = _get_pose_and_size_from_coords(new_pts)
                reg_results_per_id[id_] = RegistrationResults(
                    poses=[pose],
                    size=size,
                    lidar_indices=[lidar_idx.item()],
                    lidar_points=[new_pts],
                    feats=[new_feats],
                    rgbs=[new_rgbs],
                    fitness=[1.0],
                )
            else:
                # update track with new pose
                prev_pose = reg_results_per_id[id_].poses[0]
                prev_pts = reg_results_per_id[id_].lidar_points[0]
                prev_feats = reg_results_per_id[id_].feats[0]
                prev_rgbs = reg_results_per_id[id_].rgbs[0]

                max_points_for_pose_estimation = 5000
                if prev_pts.shape[0] > max_points_for_pose_estimation:
                    rand_indices = torch.randperm(prev_pts.shape[0])[:max_points_for_pose_estimation]
                    src_pts = prev_pts[rand_indices]
                    src_feats = prev_feats[rand_indices]
                else:
                    src_pts = prev_pts
                    src_feats = prev_feats

                if new_pts.shape[0] > max_points_for_pose_estimation:
                    rand_indices = torch.randperm(new_pts.shape[0])[:max_points_for_pose_estimation]
                    tgt_pts = new_pts[rand_indices]
                    tgt_feats = new_feats[rand_indices]
                else:
                    tgt_pts = new_pts
                    tgt_feats = new_feats

                pose_prev_to_new, fitness = get_relative_pose(
                    src_pts,
                    tgt_pts,
                    src_feats,
                    tgt_feats,
                    ransac_num_iterations,
                    ransac_distance_threshold,
                    ransac_mutual,
                    ransac_batch_size,
                    ransac_refine,
                    device,
                )
                if fitness > min_fitness_for_pose_update:
                    new_pose = pose_prev_to_new @ prev_pose
                    reg_results_per_id[id_].lidar_indices.append(lidar_idx.item())
                    reg_results_per_id[id_].lidar_points.append(new_pts)
                    reg_results_per_id[id_].feats.append(new_feats)
                    reg_results_per_id[id_].rgbs.append(new_rgbs)
                    reg_results_per_id[id_].fitness.append(fitness)
                    reg_results_per_id[id_].poses.append(new_pose)

                    new_pts_in_prev_frame = transform_points(new_pts, pose_inverse(pose_prev_to_new))
                    reg_results_per_id[id_].lidar_points[0] = torch.cat([prev_pts, new_pts_in_prev_frame], dim=0)
                    reg_results_per_id[id_].feats[0] = torch.cat([prev_feats, new_feats], dim=0)
                    reg_results_per_id[id_].rgbs[0] = torch.cat([prev_rgbs, new_rgbs], dim=0)

    return reg_results_per_id


def remove_static_tracks(
    reg_results_per_id: Dict[int, RegistrationResults],
    seed_points: Tensor,
    dist_threshold: float = 1.0,
) -> Tensor:
    # remove static tracks
    ids = list(reg_results_per_id.keys())
    for id_ in ids:
        is_dynamic = (torch.stack(reg_results_per_id[id_].poses)[:, :3, 3].std(axis=0) > dist_threshold).any()
        if not is_dynamic:
            pts = torch.cat(reg_results_per_id[id_].lidar_points, dim=0)
            rgb = torch.cat(reg_results_per_id[id_].rgbs, dim=0)
            new_gauss_ids = -1 * torch.ones_like(pts[:, :1])
            new_seed_points = torch.cat([pts, rgb, new_gauss_ids], dim=-1)
            seed_points = torch.cat([seed_points, new_seed_points], dim=0)
            reg_results_per_id.pop(id_)

    return seed_points


def stack_poses(reg_results_per_id: Dict[int, RegistrationResults]) -> Dict[int, RegistrationResults]:
    for id_ in reg_results_per_id:
        reg_results_per_id[id_].poses = torch.stack(reg_results_per_id[id_].poses, dim=0)
    return reg_results_per_id


def create_tracks(reg_results_per_id: Dict[int, RegistrationResults], unique_timestamps: Tensor) -> List[Object]:
    # create tracks
    tracks: List[Object] = []
    new_track_id = 0
    for id_ in reg_results_per_id:
        # fill in missing poses
        lidar_indices = torch.tensor(reg_results_per_id[id_].lidar_indices, device=unique_timestamps.device)
        sort_indices = torch.argsort(lidar_indices)
        lidar_indices = lidar_indices[sort_indices]
        poses = reg_results_per_id[id_].poses[sort_indices]
        pose_times = unique_timestamps[lidar_indices]
        interpolated_poses, _, _ = interpolate_trajectories(
            poses.unsqueeze(1),
            pose_times,
            unique_timestamps,
            clamp_frac=True,
        )
        interpolated_poses = to4x4(interpolated_poses)
        tracks.append(
            Object(
                id=new_track_id,
                mask_id=id_,
                poses=interpolated_poses,
                size=reg_results_per_id[id_].size,
                valid_timestamps=torch.isin(unique_timestamps, pose_times),
                linear_velocities_global=torch.zeros(interpolated_poses.shape[0], 3, device=unique_timestamps.device),
                angular_velocities_local=torch.zeros(interpolated_poses.shape[0], 3, device=unique_timestamps.device),
            )
        )
        new_track_id += 1

    return tracks


def fitness_to_weights(fitness: np.ndarray, min_fitness_for_measurement: float = 0.1) -> np.ndarray:
    weights = np.copy(fitness)
    weights[weights <= min_fitness_for_measurement] = 0.0
    return weights


def smooth_tracks(
    tracks: List[Object],
    unique_timestamps: Tensor,
    reg_results_per_id: Dict[int, RegistrationResults],
    w_abs_mult: float = 1.0,
    min_fitness_for_measurement: float = 0.1,
    iterative: bool = True,
) -> List[Object]:
    for track in tqdm(tracks, desc="Smoothing tracks"):
        Z_abs = track.poses.cpu().numpy()
        dt = unique_timestamps.diff().cpu().numpy()
        track_mask_ids = track.all_mask_ids.cpu().numpy()
        active_times = track.valid_timestamps.nonzero().squeeze(-1).cpu().numpy()
        fitness_scores = np.zeros_like(active_times, dtype=np.float32)
        lidar_indices = np.zeros_like(active_times, dtype=np.int32)
        for mask in track_mask_ids:
            if mask not in reg_results_per_id:
                continue
            for idx, time_idx in enumerate(active_times):
                if time_idx in reg_results_per_id[mask].lidar_indices:
                    lidar_idx = reg_results_per_id[mask].lidar_indices.index(time_idx)
                    fitness_scores[idx] = reg_results_per_id[mask].fitness[lidar_idx]
                    lidar_indices[idx] = time_idx
        assert len(fitness_scores) == len(lidar_indices)
        assert len(fitness_scores) == len(active_times)

        w_abs = np.zeros(unique_timestamps.shape[0])
        w_abs[lidar_indices] = fitness_to_weights(np.array(fitness_scores), min_fitness_for_measurement)
        w_abs = w_abs * w_abs_mult
        smoothed_poses = get_smoothed_poses(
            Z_abs=Z_abs,
            dt=dt,
            w_abs=w_abs,
            iterative=iterative,
        )
        track.poses = torch.from_numpy(smoothed_poses).to(track.poses.device).float()

    for track in tracks:
        valid_indices = track.valid_timestamps.nonzero().squeeze(-1)
        start_idx = max(0, valid_indices[0].item() - 1)
        end_idx = min(len(unique_timestamps) - 1, valid_indices[-1].item() + 1)
        track.valid_timestamps = torch.zeros_like(unique_timestamps, dtype=torch.bool)
        track.valid_timestamps[start_idx : (end_idx + 1)] = True
    return tracks


def add_velocities_to_tracks(tracks: List[Object], unique_timestamps: Tensor):
    for track in tracks:
        poses = track.poses
        times = unique_timestamps

        linear_velocities_global = (poses[1:, :3, 3] - poses[:-1, :3, 3]) / (times[1:] - times[:-1]).reshape(-1, 1)
        linear_velocities_global = torch.cat((linear_velocities_global, linear_velocities_global[-1:]), 0)
        angular_velocities_local = rotation_difference(poses[:-1, :3, :3], poses[1:, :3, :3]) / (
            times[1:] - times[:-1]
        ).reshape(-1, 1)
        angular_velocities_local = torch.cat((angular_velocities_local, angular_velocities_local[-1:]), 0)
        track.linear_velocities_global = linear_velocities_global
        track.angular_velocities_local = angular_velocities_local


def set_valid_timestamps_from_mask_times(
    tracks: List[Object],
    unique_timestamps: Tensor,
    start_time_per_mask_id: Dict[int, float],
    end_time_per_mask_id: Dict[int, float],
    max_pred_time: float = 3.0,
):
    for track in tracks:
        track_valid_times = unique_timestamps[track.valid_timestamps]
        track_start_time = track_valid_times.min()
        track_end_time = track_valid_times.max()

        min_start_time_per_mask_id = min([start_time_per_mask_id[mask_id.item()] for mask_id in track.all_mask_ids])
        max_end_time_per_mask_id = max([end_time_per_mask_id[mask_id.item()] for mask_id in track.all_mask_ids])
        new_start_time = max(track_start_time - max_pred_time, min_start_time_per_mask_id)
        start_idx = torch.searchsorted(unique_timestamps, new_start_time) - 3
        start_idx = max(0, start_idx)

        new_end_time = min(track_end_time + max_pred_time, max_end_time_per_mask_id)
        end_idx = torch.searchsorted(unique_timestamps, new_end_time)
        end_idx += 1
        end_idx = min(len(unique_timestamps) - 1, end_idx)

        track.valid_timestamps[start_idx : (end_idx + 1)] = True


def recenter_local_coordinate_system(tracks: List[Object], reg_results_per_id: Dict[int, RegistrationResults]):
    for track in tracks:
        canonical_pts = reg_results_per_id[track.mask_id].lidar_points[0].clone()  # canonical points
        canonical_idx = reg_results_per_id[track.mask_id].lidar_indices[0]
        canonical2local = pose_inverse(track.poses[canonical_idx])
        local_pts = transform_points(canonical_pts, canonical2local)

        min_coords, max_coords = (
            local_pts.min(dim=0).values,
            local_pts.max(dim=0).values,
        )
        size = max_coords - min_coords
        new_center = min_coords + size / 2.0

        recentering_transform = torch.eye(4, device=local_pts.device)
        recentering_transform[:3, 3] = new_center
        track.poses = track.poses @ recentering_transform
        track.size = size


def get_seed_points_from_tracks(
    tracks: List[Object], reg_results_per_id: Dict[int, RegistrationResults], min_fitness_for_adding_points: float
) -> Tensor:
    seed_points = []
    for track in tracks:
        canonical_pts = reg_results_per_id[track.mask_id].lidar_points[0].clone()  # canonical points
        canonical_rgbs = reg_results_per_id[track.mask_id].rgbs[0]  # canonical rgbs
        canonical_idx = reg_results_per_id[track.mask_id].lidar_indices[0]

        canonical2local = pose_inverse(track.poses[canonical_idx])
        local_pts = transform_points(canonical_pts, canonical2local)
        seed_points.append(torch.cat([local_pts, canonical_rgbs, track.id * torch.ones_like(local_pts[:, :1])], dim=-1))
    if len(seed_points) > 0:
        return torch.cat(seed_points, dim=0)
    else:
        return torch.empty((0, 7))


def get_random_seed_points_inside_bbox(tracks: List[Object], n_points: int) -> Tensor:
    seed_points = []
    for track in tracks:
        random_points = (torch.rand((n_points, 3), device=track.poses.device) - 0.5) * track.size
        random_colors = torch.rand((n_points, 3), device=track.poses.device)
        obj_ids = track.id * torch.ones((n_points, 1), device=track.poses.device)
        seed_points.append(torch.cat([random_points, random_colors, obj_ids], dim=-1))
    if len(seed_points) > 0:
        return torch.cat(seed_points, dim=0)
    else:
        return torch.empty((0, 7))
