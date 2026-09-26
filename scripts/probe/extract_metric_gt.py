"""Extract per-image LiDAR depth and ground-truth 3D boxes, in camera coordinates.

AD-GS is trained with scale-and-shift-invariant monocular depth only, so neither
signal enters the model.  They are used here purely as a metric yardstick for
`probe_metric_placement.py`.

Camera convention matches `scripts/waymo/waymo.py`: a point in the vehicle frame
maps to the OpenCV camera frame by inv(OPENCV2DATASET) @ inv(extrinsic).  The
world frame therefore never appears, and the readout is independent of which
frame `waymo.py` chose as the origin.

Run in the `waymo-prep` environment.
"""
import argparse
import os

import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils

OPENCV2DATASET = np.array(
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64
)


def box_corners(center, dims, heading):
    """8 corners of an upright box, vehicle frame. dims is (length, width, height)."""
    l, w, h = dims
    x = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * l / 2
    y = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * w / 2
    z = np.array([1, 1, 1, 1, -1, -1, -1, -1]) * h / 2
    c, s = np.cos(heading), np.sin(heading)
    return np.stack([c * x - s * y, s * x + c * y, z], axis=1) + center


def project(points_cam, K):
    """Pinhole projection; returns uv and depth for points in front of the camera."""
    z = points_cam[:, 2]
    front = z > 1e-3
    uv = (K @ points_cam[front].T).T
    return uv[:, :2] / uv[:, 2:], z[front], front


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="tfrecord path")
    ap.add_argument("dst", help="output directory")
    ap.add_argument("--first_frame", type=int, required=True)
    ap.add_argument("--last_frame", type=int, required=True)
    ap.add_argument("--select_camera", type=int, nargs="+", default=[0])
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    image_id = 0
    ego_0 = None

    for fid, data in enumerate(tf.data.TFRecordDataset(args.src, compression_type="")):
        if fid < args.first_frame or fid > args.last_frame:
            continue
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        ego_to_world = np.array(frame.pose.transform).reshape(4, 4)
        if ego_0 is None:
            ego_0 = np.linalg.inv(ego_to_world)
        ego_to_world = ego_0 @ ego_to_world

        range_images, camera_projections, _, range_image_top_pose = (
            frame_utils.parse_range_image_and_camera_projection(frame)
        )
        points, _ = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose, ri_index=0
        )
        points_vehicle = np.concatenate(points, axis=0)

        labels = [l for l in frame.laser_labels if l.num_lidar_points_in_box > 0]
        centers = np.array(
            [[l.box.center_x, l.box.center_y, l.box.center_z] for l in labels]
        ).reshape(-1, 3)
        dims = np.array(
            [[l.box.length, l.box.width, l.box.height] for l in labels]
        ).reshape(-1, 3)
        headings = np.array([l.box.heading for l in labels]).reshape(-1)
        centers_world = (ego_to_world[:3, :3] @ centers.T + ego_to_world[:3, 3:]).T
        corners_world = np.stack(
            [
                (ego_to_world[:3, :3] @ box_corners(centers[i], dims[i], headings[i]).T
                 + ego_to_world[:3, 3:]).T
                for i in range(len(labels))
            ]
        ).reshape(-1, 8, 3)

        for img in frame.images:
            if img.name - 1 not in args.select_camera:
                continue
            cam = [c for c in frame.context.camera_calibrations if c.name == img.name][0]
            K = np.array(
                [
                    [cam.intrinsic[0], 0.0, cam.intrinsic[2]],
                    [0.0, cam.intrinsic[1], cam.intrinsic[3]],
                    [0.0, 0.0, 1.0],
                ]
            )
            vehicle_to_cam = np.linalg.inv(
                np.array(cam.extrinsic.transform).reshape(4, 4) @ OPENCV2DATASET
            )
            R, t = vehicle_to_cam[:3, :3], vehicle_to_cam[:3, 3:]
            H, W = cam.height, cam.width

            pts_cam = (R @ points_vehicle.T + t).T
            uv, z, front = project(pts_cam, K)
            inside = (
                (uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1)
            )
            lidar_uvz = np.concatenate([uv[inside], z[inside, None]], axis=1).astype(np.float32)

            # Which box, if any, each retained LiDAR point falls in (-1 = none).
            kept = np.where(front)[0][inside]
            owner = np.full(len(kept), -1, dtype=np.int32)
            for bi in range(len(labels)):
                c, hd = centers[bi], headings[bi]
                cs, sn = np.cos(-hd), np.sin(-hd)
                d = points_vehicle[kept] - c
                local = np.stack([cs * d[:, 0] - sn * d[:, 1], sn * d[:, 0] + cs * d[:, 1], d[:, 2]], 1)
                hit = np.all(np.abs(local) <= dims[bi] / 2 + 0.1, axis=1)
                owner[hit] = bi

            box_center_cam = (
                (R @ centers.T + t).T if len(labels) else np.zeros((0, 3))
            )
            box_uv = np.zeros((len(labels), 4), dtype=np.float32)
            for bi in range(len(labels)):
                corners_cam = (R @ box_corners(centers[bi], dims[bi], headings[bi]).T + t).T
                cuv, _, cfront = project(corners_cam, K)
                box_uv[bi] = (
                    [cuv[:, 0].min(), cuv[:, 1].min(), cuv[:, 0].max(), cuv[:, 1].max()]
                    if cfront.all()
                    else [np.nan] * 4
                )

            np.savez_compressed(
                os.path.join(args.dst, "{:06d}.npz".format(image_id)),
                fid=fid,
                K=K.astype(np.float32),
                image_hw=np.array([H, W], dtype=np.int32),
                lidar_uvz=lidar_uvz,
                lidar_box=owner,
                box_id=np.array([l.id for l in labels], dtype="U64"),
                box_type=np.array([l.type for l in labels], dtype=np.int32),
                box_npts=np.array([l.num_lidar_points_in_box for l in labels], dtype=np.int32),
                box_center_cam=box_center_cam.astype(np.float32),
                box_center_world=centers_world.astype(np.float32),
                box_corners_world=corners_world.astype(np.float32),
                box_dims=dims.astype(np.float32),
                box_uv=box_uv,
            )
            image_id += 1

    print("wrote {} files to {}".format(image_id, args.dst))


if __name__ == "__main__":
    main()
