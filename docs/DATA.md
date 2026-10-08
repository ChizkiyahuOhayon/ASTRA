# Data preparation

Everything below assumes one data root, `data/`, next to the repository. Paths are passed explicitly to every
script, so the root can live anywhere.

```
data/
├── wod/validation/          Waymo Open Dataset v2 parquet (8 segments)
├── wod/images/              image cache, written on the first run
├── wod/v1/                  Waymo v1.4 tfrecords of the same segments (DPSNR only)
├── wod_gt_masks/            eval-only dynamic masks for DPSNR
├── kitti/training/          KITTI-MOT
├── nuscenes/                nuScenes (6 scenes)
├── pandaset/                PandaSet (10 sequences)   <- the path must contain "data/pandaset/"
└── masks/{waymo,kitti,nuscenes,pandaset}/
```

No step of training reads annotations. Ground-truth boxes are used only to build the DPSNR evaluation masks.

## 1. Waymo Open Dataset (v2)

1. Register at [waymo.com/open](https://waymo.com/open) and install `gcloud`.
2. Download the components below for the eight validation segments listed in
   `scripts/arrays/wod_id_to_seq_adgs_split.txt` (scene ids 800 to 807):

   ```bash
   SEGS="10448102132863604198_472_000_492_000 12374656037744638388_1412_711_1432_711 \
         1906113358876584689_1359_560_1379_560 2094681306939952000_2972_300_2992_300 \
         4246537812751004276_1560_000_1580_000 5372281728627437618_2005_000_2025_000 \
         8398516118967750070_3958_000_3978_000 17612470202990834368_2800_000_2820_000"
   for c in camera_image camera_calibration lidar lidar_calibration lidar_pose vehicle_pose lidar_box; do
     mkdir -p data/wod/validation/$c
     for s in $SEGS; do
       gcloud storage cp gs://waymo_open_dataset_v_2_0_1/validation/$c/$s.parquet data/wod/validation/$c/
     done
   done
   ```
   The eight segments take about 4.1 GB.
3. **Instance masks** (Grounding DINO + SAM 2, tracked ids; about 10 min per segment on an RTX 4090):
   ```bash
   for id in 800 801 802 803 804 805 806 807; do
     IFS=, read -r sid seg s e _ <<< "$(grep "^$id," scripts/arrays/wod_id_to_seq_adgs_split.txt)"
     seq=$(ls data/wod/validation/camera_image | grep "^${seg#seg}"); seq=${seq%.parquet}
     uv run python tools/make_masks_wod.py data/wod/validation $seq $sid $s $e data/masks/waymo \
         --sam2_ckpt checkpoints/sam2.1_hiera_large.pt
   done
   ```
4. **DPSNR masks (evaluation only).** IDSplat projects the ground-truth boxes from the v1 tfrecords of the same
   segments (Waymo v1.4, `individual_files/validation/segment-<seq>_with_camera_labels.tfrecord`). Its own
   `uv.lock` pairs TensorFlow 2.19 with `waymo-open-dataset-tf-2-12-0`, whose camera op fails to load, so we build
   the masks once in a separate environment and the parser then reads them:
   ```bash
   python3.10 -m venv gtenv
   gtenv/bin/pip install tensorflow==2.14.1 numpy==1.26.4 protobuf==4.25.9
   gtenv/bin/pip install waymo-open-dataset-tf-2-12-0==1.6.5 --no-deps
   # one call per segment: <v1 tfrecord> <sequence> <start> <end> <out_dir>
   gtenv/bin/python tools/make_gt_masks.py \
       data/wod/v1/individual_files_validation_segment-<seq>_with_camera_labels.tfrecord <seq> <start> <end> data/wod_gt_masks
   ```
   Skip this step if you do not need DPSNR; PSNR / SSIM / LPIPS do not use it.

The Waymo licence forbids redistributing the data and anything derived from it, so we cannot host Waymo masks
or checkpoints.

## 2. KITTI-MOT

Download the tracking benchmark (left and right colour images, Velodyne, calibration, OXTS) from
[cvlibs.net](https://www.cvlibs.net/datasets/kitti/eval_tracking.php) and keep sequences 0001, 0002 and 0006:

```
data/kitti/training/{image_02,image_03,velodyne,calib,oxts}/<seq>...
```

Masks: download `masks_kitti.tar.gz` from the [Drive folder](https://drive.google.com/drive/folders/1ZmVEwQ7Zw_ymfHw06QawHFZSbmk0pUk0) and extract it to
`data/masks/kitti/`, or regenerate them:

```bash
for seq in 0001 0002 0006; do for cam in image_02 image_03; do
  out=data/masks/kitti/$seq/$cam; mkdir -p $out
  ls data/kitti/training/$cam/$seq/*.png | sort > $out/list.txt
  uv run python tools/make_masks_generic.py $out/list.txt $out --sam2_ckpt checkpoints/sam2.1_hiera_large.pt
done; done
```

## 3. nuScenes

Download nuScenes v1.0-trainval (metadata, maps, and the `samples` / `sweeps` of CAM_FRONT, CAM_FRONT_LEFT,
CAM_FRONT_RIGHT and LIDAR_TOP) from [nuscenes.org](https://www.nuscenes.org/nuscenes). Only the six AD-GS scenes
are used: scene-0230, scene-0242, scene-0255, scene-0295, scene-0518, scene-0749. Install the devkit with
`uv pip install nuscenes-devkit --no-deps`.

Masks: `masks_nuscenes.tar.gz` from the [Drive folder](https://drive.google.com/drive/folders/1ZmVEwQ7Zw_ymfHw06QawHFZSbmk0pUk0) to `data/masks/nuscenes/`, or
regenerate them per scene and camera with `tools/make_masks_generic.py` on the time-ordered image list of that
camera (output folder `data/masks/nuscenes/<scene>/CAM_<NAME>/`).

## 4. PandaSet

Download PandaSet from [Hugging Face](https://huggingface.co/datasets/georghess/pandaset) (or the official site)
and keep sequences 001, 011, 016, 028, 053, 063, 084, 106, 123, 158 under `data/pandaset/`. The devkit globs
`*.pkl.gz`; if your copy stores uncompressed `.pkl` files, run `gzip -k` on them.

Masks: `masks_pandaset_<part>.tar.gz` from the [Drive folder](https://drive.google.com/drive/folders/1ZmVEwQ7Zw_ymfHw06QawHFZSbmk0pUk0) to `data/masks/pandaset/`, or
regenerate them per sequence and camera (`<seq>/<cam>_camera/`) with `tools/make_masks_generic.py`.

## Checksums

Every archive in the Drive folder has a `.sha256` file next to it:

```bash
sha256sum -c masks_kitti.tar.gz.sha256
```
