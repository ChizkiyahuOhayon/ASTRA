"""Ground-truth moving-object masks for the eight Waymo scenes (evaluation only).

Follows the StreetGS / EmerNeRF protocol used to report dynamic-region PSNR (DPSNR):
a laser label is moving when its horizontal speed is at least 1 m/s; its 3D box is
projected into the FRONT camera and its six faces are filled, without dilation.
Image i of a scene is tfrecord frame `first_frame + i`, matching
scripts/waymo/prepare-waymo.sh. Ground truth is never used for training.

    python scripts/eval/waymo_dynamic_masks.py --raw_dir <waymo tfrecords> --out data/waymo_dynamic_masks

Writes <out>/<scene>/dynamic_mask/<index>.png. Reads frames with the same
`waymo_open_dataset` + tensorflow environment as scripts/waymo/waymo.py, or with
`simple_waymo_open_dataset_reader` when that is what is installed.
"""
import argparse
import math
import os

import cv2
import numpy as np

SCENES = {  # name: (segment, first_frame, last_frame), as in scripts/waymo/prepare-waymo.sh
    'scene006': ('10448102132863604198_472_000_492_000', 0, 85),
    'scene026': ('12374656037744638388_1412_711_1432_711', 0, 100),
    'scene090': ('17612470202990834368_2800_000_2820_000', 0, 102),
    'scene105': ('1906113358876584689_1359_560_1379_560', 20, 186),
    'scene108': ('2094681306939952000_2972_300_2992_300', 20, 115),
    'scene134': ('4246537812751004276_1560_000_1580_000', 106, 198),
    'scene150': ('5372281728627437618_2005_000_2025_000', 96, 197),
    'scene181': ('8398516118967750070_3958_000_3978_000', 0, 160),
}
FRONT = 1  # dataset_pb2.CameraName.FRONT
MIN_SPEED = 1.0  # m/s, EmerNeRF / StreetGS threshold
# Waymo camera frame (x forward, y left, z up) -> OpenCV (x right, y down, z forward).
OPENCV_TO_CAMERA = np.array([[0., 0., 1., 0.], [-1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., 0., 1.]])
# Box faces over the corner order of `box_corners`, as in StreetGS get_bound_2d_mask.
FACES = ([0, 1, 3, 2, 0], [4, 5, 7, 6, 5], [0, 1, 5, 4, 0], [2, 3, 7, 6, 2], [0, 2, 6, 4, 0], [1, 3, 7, 5, 1])


def read_frames(path):
    try:
        import tensorflow as tf
        from waymo_open_dataset import dataset_pb2
    except ImportError:
        from simple_waymo_open_dataset_reader import WaymoDataFileReader
        yield from WaymoDataFileReader(path)
        return
    for record in tf.data.TFRecordDataset(path, compression_type=''):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(record.numpy()))
        yield frame


def box_corners(box):
    """The eight corners of an upright box in vehicle coordinates, x slowest and z fastest."""
    half = np.array([box.length, box.width, box.height]) / 2
    signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1], indexing='ij')).reshape(3, -1).T
    c, s = math.cos(box.heading), math.sin(box.heading)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return (signs * half) @ rotation.T + [box.center_x, box.center_y, box.center_z]


def box_mask(box, calibration):
    camera_to_vehicle = np.array(calibration.extrinsic.transform).reshape(4, 4) @ OPENCV_TO_CAMERA
    vehicle_to_camera = np.linalg.inv(camera_to_vehicle)
    points = box_corners(box) @ vehicle_to_camera[:3, :3].T + vehicle_to_camera[:3, 3]
    fx, fy, cx, cy = calibration.intrinsic[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    points[:, 2] = np.where(points[:, 2] > 1e-3, points[:, 2], np.nan)
    pixels = points @ K.T
    pixels = pixels[:, :2] / pixels[:, 2:]
    inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < calibration.width)
              & (pixels[:, 1] >= 0) & (pixels[:, 1] < calibration.height))
    if not inside.any():  # no corner in front of the camera and inside the image
        return None
    points[:, 2] = np.nan_to_num(points[:, 2], nan=1e-3)
    pixels = points @ K.T
    pixels = np.round(pixels[:, :2] / pixels[:, 2:]).astype(np.int64)
    mask = np.zeros((calibration.height, calibration.width), np.uint8)
    for face in FACES:
        cv2.fillPoly(mask, [pixels[face].astype(np.int32)], 1)
    return mask


def write_scene(raw_dir, out_dir, name):
    segment, first, last = SCENES[name]
    path = os.path.join(raw_dir, 'individual_files_validation_segment-%s_with_camera_labels.tfrecord' % segment)
    target = os.path.join(out_dir, name, 'dynamic_mask')
    os.makedirs(target, exist_ok=True)
    written = 0
    for index, frame in enumerate(read_frames(path)):
        if index < first:
            continue
        if index > last:
            break
        calibration = next(c for c in frame.context.camera_calibrations if c.name == FRONT)
        mask = np.zeros((calibration.height, calibration.width), np.uint8)
        for label in frame.laser_labels:
            if math.hypot(label.metadata.speed_x, label.metadata.speed_y) < MIN_SPEED:
                continue
            projected = box_mask(label.box, calibration)
            if projected is not None:
                mask |= projected
        cv2.imwrite(os.path.join(target, '%06d.png' % (index - first)), mask * 255)
        written += 1
    if written != last - first + 1:
        raise RuntimeError('%s: wrote %d masks, expected %d' % (name, written, last - first + 1))
    print(name, written, 'masks')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--raw_dir', required=True, help='folder with the eight Waymo v1.4.1 tfrecords')
    parser.add_argument('--out', default='data/waymo_dynamic_masks')
    parser.add_argument('--scenes', nargs='+', default=sorted(SCENES))
    args = parser.parse_args()
    for scene in args.scenes:
        write_scene(args.raw_dir, args.out, scene)
