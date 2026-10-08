"""Eval-only dynamic GT masks for the IDSplat DPSNR metric, in a TF 2.12 env where waymo's camera op loads.

    make_gt_masks.py <tfrecord> <sequence> <start> <end> <out_dir> [<height> <width>]

Line-for-line the logic of WoDParser.save_dynamic_mask: 2D boxes of projected upright laser boxes, max speed per
class channel, threshold 1 m/s. Written to <out_dir>/<sequence>/gt_masks/<frame:03d>_<cam>.npz, which the training
env then loads instead of recomputing. Never read by training.
"""
import os
import sys

import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2, label_pb2
from waymo_open_dataset.utils import box_utils
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops

OBJECT_ID_TO_MASK_INDEX = {label_pb2.Label.TYPE_VEHICLE: 0, label_pb2.Label.TYPE_PEDESTRIAN: 1,
                           label_pb2.Label.TYPE_CYCLIST: 2}
NUM_LIDAR_POINT_THRESH = 10


def project_vehicle_to_image(vehicle_pose, calibration, points):
    pose_matrix = np.array(vehicle_pose.transform).reshape(4, 4)
    world_points = np.zeros_like(points)
    for i, point in enumerate(points):
        cx, cy, cz, _ = np.matmul(pose_matrix, [*point, 1])
        world_points[i] = (cx, cy, cz)
    extrinsic = tf.reshape(tf.constant(list(calibration.extrinsic.transform), dtype=tf.float32), [4, 4])
    intrinsic = tf.constant(list(calibration.intrinsic), dtype=tf.float32)
    metadata = tf.constant([calibration.width, calibration.height, dataset_pb2.CameraCalibration.GLOBAL_SHUTTER],
                           dtype=tf.int32)
    camera_image_metadata = list(vehicle_pose.transform) + [0.0] * 10
    return py_camera_model_ops.world_to_image(extrinsic, intrinsic, metadata, camera_image_metadata,
                                              world_points).numpy()


def main():
    tfrecord, sequence, start, end, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    # the parser sizes every mask like img_filenames[0], i.e. the first FRONT image
    shape = (int(sys.argv[6]), int(sys.argv[7]), 3) if len(sys.argv) > 7 else (1280, 1920, 3)
    folder = os.path.join(out, sequence, "gt_masks")
    os.makedirs(folder, exist_ok=True)
    count = 0
    for frame_idx, data in enumerate(tf.data.TFRecordDataset(tfrecord, compression_type="")):
        if frame_idx < start or frame_idx >= end:
            continue
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytes(data.numpy()))
        filter_available = any(l.num_top_lidar_points_in_box > NUM_LIDAR_POINT_THRESH for l in frame.laser_labels)
        for img in frame.images:
            path = os.path.join(folder, f"{str(frame_idx).zfill(3)}_{str(img.name - 1)}.npz")
            if os.path.exists(path):
                continue
            dynamic_mask = np.zeros(shape, dtype=np.float32)
            calibration = next(cc for cc in frame.context.camera_calibrations if cc.name == img.name)
            for label in frame.laser_labels:
                box = label.box
                label_type = np.uint32(label.type)
                if label_type not in OBJECT_ID_TO_MASK_INDEX:
                    continue
                label_index = OBJECT_ID_TO_MASK_INDEX[label_type]
                speed = np.linalg.norm([label.metadata.speed_x, label.metadata.speed_y])
                if not box.ByteSize():
                    continue
                if (filter_available and not label.num_top_lidar_points_in_box) or (
                        not filter_available and not label.num_lidar_points_in_box):
                    continue
                box_coords = np.array([[box.center_x, box.center_y, box.center_z, box.length, box.width, box.height,
                                        box.heading]])
                corners = box_utils.get_upright_3d_box_corners(box_coords)[0].numpy()
                u, v, ok = project_vehicle_to_image(frame.pose, calibration, corners).transpose()
                ok = ok.astype(bool)
                if not all(ok):
                    continue
                u = np.clip(u[ok], 0, calibration.width)
                v = np.clip(v[ok], 0, calibration.height)
                if u.max() - u.min() == 0 or v.max() - v.min() == 0:
                    continue
                xy = (u.min(), v.min())
                width, height = u.max() - u.min(), v.max() - v.min()
                region = dynamic_mask[int(xy[1]): int(xy[1] + height), int(xy[0]): int(xy[0] + width), label_index]
                dynamic_mask[int(xy[1]): int(xy[1] + height), int(xy[0]): int(xy[0] + width), label_index] = \
                    np.maximum(region, speed)
            dynamic_mask = np.clip((dynamic_mask > 1.0) * 255, 0, 255).astype(np.uint8)
            np.savez_compressed(path, dynamic_mask=dynamic_mask)
            count += 1
    print("done", sequence, count, "masks")


if __name__ == "__main__":
    main()
