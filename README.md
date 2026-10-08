<h1 align="center">ASTRA</h1>

<p align="center">
  <b>Driving-Scene Gaussian Splatting that Holds Up off the Recorded Trajectory</b>
</p>

<p align="center">
  <a href="https://github.com/zenseact/idsplat">
    <img src="https://img.shields.io/badge/Built%20on-IDSplat-181717?style=for-the-badge&logo=github&logoColor=white">
  </a>
  <img src="https://img.shields.io/badge/Datasets-Waymo%20%7C%20KITTI%20%7C%20nuScenes%20%7C%20PandaSet-4c8eda?style=for-the-badge">
  <a href="https://drive.google.com/drive/folders/1ZmVEwQ7Zw_ymfHw06QawHFZSbmk0pUk0">
    <img src="https://img.shields.io/badge/Data-Google%20Drive-34a853?style=for-the-badge&logo=googledrive&logoColor=white">
  </a>
  <img src="https://img.shields.io/badge/License-Apache%202.0-orange?style=for-the-badge">
</p>

<p align="center">
  <img src="docs/assets/teaser.jpg" width="96%"><br>
  <em>Left: a held-out frame on the recorded trajectory. Right: the same timestamp rendered at a camera that no model
  saw in training. IDSplat fits the recorded path but falls apart off it; ASTRA keeps the scene solid.</em>
</p>

## 📜 News

🔥 **[2026/10]** Code, benchmark scripts, the held-out-camera benchmark, pseudo-label masks for KITTI / nuScenes /
PandaSet and all our result files are released.

## ✒️ Contents

- [Overview](#-overview)
- [Results](#-results)
- [Installation](#%EF%B8%8F-installation)
- [Data](#-data)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Reproducing the paper tables](#-reproducing-the-paper-tables)
- [Checkpoints](#-checkpoints)
- [Repository layout](#%EF%B8%8F-repository-layout)
- [License](#-license)
- [Citation](#-citation)
- [Acknowledgment](#-acknowledgment)

## 👀 Overview

Driving-scene reconstructions are trained and evaluated on one trajectory: every fourth frame of the drive is held
out, and the model is scored there. A held-out frame lies between two training frames about a meter apart, so the
benchmark rewards memorising the recorded path. A simulator needs the opposite.

We measure the gap without any new data. Waymo records five cameras and the benchmark trains on the front camera
only, so we render the trained model at the **real front-left and front-right cameras** of the same timestamps.
IDSplat drops from 34.7 dB on the standard held-out frames to 19.8 dB on these views, even on the pixels that the
front camera observed and after compensating the exposure.

We trace this trajectory overfitting to three sources and remove each one, using only the training images:

| Source | ASTRA component | Flag(s) |
|---|---|---|
| Logged poses are slightly wrong; the scene absorbs the error at training frames, and held-out frames keep it | **Trajectory-consistent pose refinement**: learn a correction per training camera and carry it to any timestamp by interpolating the bracketing corrections | `--pipeline.model.camopt_interp_eval True` (+ `camera_optimizer.mode SO3xR3`, `camopt_cameras_only`) |
| The Waymo pipeline renders a pinhole image and compares it with the raw, distorted image (up to 30 px off in the corners) | **Lens-consistent rendering**: render a padded pinhole feature image and resample it through the calibrated Brown–Conrady model before the decoder | `--pipeline.model.lens_distortion True` |
| All supervision comes from the recorded path | **Virtual-view consistency**: render virtual cameras between neighbouring frames, half of them yawed (±20°) and shifted sideways (±1 m), and supervise them with the two real neighbours, warped by rendered depth and actor boxes and kept only where the two warps agree | `--pipeline.tivc_*` |
| | **Progressive Gaussian dropout**: drop each Gaussian with a rate that grows to 0.2, so no stack of primitives can memorise a view | `--pipeline.model.gauss_dropout 0.2` |

<p align="center">
  <img src="docs/assets/pipeline.jpg" width="96%"><br>
  <em>ASTRA keeps the IDSplat scene representation (grey) and changes how it is posed, rendered and supervised (orange).</em>
</p>

Nothing is annotated, no generative prior or extra network is used, and inference renders exactly the IDSplat
representation (10.5 vs. 10.2 FPS at 1920×1280).

## 📊 Results

Every number below comes from the scripts in this repository, one run per scene. Rows marked "reproduced" were
trained with the official IDSplat code on the same machine; rows without the mark are copied from the AD-GS and
IDSplat papers. † uses annotated 3D boxes.

### Waymo: 8 sequences, front camera, every 4th frame held out

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DPSNR ↑ |
|---|---|---|---|---|
| StreetGS† | 33.97 | 0.926 | 0.227 | 28.50 |
| 4DGF† | 34.64 | 0.940 | 0.244 | 29.77 |
| SplatAD† | 34.24 | 0.925 | 0.246 | 29.68 |
| PVG | 29.54 | 0.895 | 0.266 | 21.56 |
| EmerNeRF | 31.32 | 0.881 | 0.301 | 21.80 |
| Grid4D | 32.19 | 0.921 | 0.253 | 22.77 |
| AD-GS | 33.91 | 0.927 | 0.228 | 27.41 |
| IDSplat | 34.59 | 0.929 | 0.235 | 29.63 |
| IDSplat (reproduced) | 34.68 | 0.931 | 0.232 | 30.24 |
| **ASTRA** | **36.31** | **0.948** | **0.224** | **30.70** |

Our DPSNR uses ground-truth boxes of actors faster than 1 m/s (IDSplat code); the published rows use the
class-agnostic StreetGS masks.

### Held-out cameras (new benchmark): Waymo front-left / front-right, never trained on

PSNR on co-visible pixels after a per-image affine colour fit; masks are computed once from the IDSplat depth and
shared by all methods.

| Method | PSNR (full) ↑ | PSNR (co-visible) ↑ | + affine ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|---|
| IDSplat (reproduced) | 13.98 | 17.48 | 19.76 | 0.662 | 0.217 |
| **ASTRA** | **14.94** | **18.69** | **20.81** | **0.714** | **0.201** |

### KITTI, nuScenes and PandaSet (same hyperparameters)

| Dataset | Protocol | Metric | AD-GS | IDSplat (reproduced) | **ASTRA** |
|---|---|---|---|---|---|
| KITTI | 75% train | PSNR / SSIM / LPIPS-Alex | 29.16 / 0.920 / 0.033 | 27.35 / 0.874 / 0.057 | **29.21** / 0.911 / 0.040 |
| KITTI | 50% train | PSNR / SSIM / LPIPS-Alex | 28.51 / 0.912 / 0.035 | 25.70 / 0.829 / 0.068 | 28.18 / 0.881 / 0.044 |
| KITTI | 25% train | PSNR / SSIM / LPIPS-Alex | 24.12 / 0.868 / 0.065 | 23.46 / 0.777 / 0.090 | **25.40** / 0.847 / **0.058** |
| nuScenes | 6 scenes, 3 cams | PSNR / SSIM / LPIPS-VGG | 31.06 / 0.925 / 0.164 | 31.41 / 0.919 / 0.207 | **32.44** / **0.935** / 0.193 |
| PandaSet | 10 seqs, 6 cams | PSNR / SSIM / LPIPS-Alex | – | 26.79 / 0.812 / 0.172 | **27.44** / **0.830** / **0.160** |

> KITTI, nuScenes and PandaSet rows currently report ASTRA without the off-trajectory virtual views (pose refinement,
> temporal virtual views); the full-model runs on these datasets are in progress and will replace them.

Full tables, ablations and per-scene numbers: [`docs/RESULTS.md`](docs/RESULTS.md).

<p align="center">
  <img src="docs/assets/qual_waymo.jpg" width="96%"><br>
  <em>Held-out Waymo frames (full frame + zoom-in) for IDSplat, ASTRA and the ground truth.</em>
</p>

## 🛠️ Installation

Tested on Ubuntu 22.04 with an RTX 4090 (48 GB) and CUDA 11.8 / 12.4. The environment is exactly IDSplat's.

```bash
git clone https://github.com/ChizkiyahuOhayon/ASTRA.git
cd ASTRA
# install uv once: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                 # creates .venv with torch, the SplatAD gsplat fork, SAM 2, ...
uv run python -c "import gsplat, torch; print(torch.cuda.is_available())"
```

- **GPU / CUDA.** `pyproject.toml` ships IDSplat's CUDA 12.8 index (RTX 50xx). For older GPUs swap the two
  commented `[tool.uv]` blocks to the CUDA 12.4 index before `uv sync`. On drivers that only support CUDA 11.8 we
  used `https://download.pytorch.org/whl/cu118` with `TORCH_CUDA_ARCH_LIST=8.9`.
- **DINOv3** (`facebook/dinov3-vitb16-pretrain-lvd1689m`) is gated on Hugging Face: accept the licence, then
  `huggingface-cli login` with your own token. After the first download, `export HF_HUB_OFFLINE=1`.
- **SAM 2.1 checkpoint** for the mask tools: download `sam2.1_hiera_large.pt` from the
  [SAM 2 release](https://github.com/facebookresearch/sam2#model-description) into `checkpoints/`.

## 📦 Data

See [`docs/DATA.md`](docs/DATA.md) for the full step-by-step guide. In short, everything lives under one root:

```
data/
├── wod/validation/{camera_image,camera_calibration,lidar,lidar_calibration,vehicle_pose,...}   # Waymo v2 parquet
├── wod_gt_masks/<sequence>/gt_masks/*.npz        # eval-only DPSNR masks (tools/make_gt_masks.py)
├── kitti/training/{image_02,image_03,velodyne,calib,oxts}
├── nuscenes/{samples,sweeps,v1.0-trainval,maps}
├── pandaset/<seq>/{camera,lidar,meta}
└── masks/{waymo,kitti,nuscenes,pandaset}/        # Grounded-SAM-2 tracked instance masks
```

| What | Where |
|---|---|
| Waymo Open Dataset v2 (8 validation segments) | official download, [waymo.com/open](https://waymo.com/open) (licence forbids redistribution) |
| Waymo instance masks | generate with `tools/make_masks_wod.py` (about 10 min per segment) |
| KITTI-MOT sequences 0001 / 0002 / 0006 | [cvlibs.net](https://www.cvlibs.net/datasets/kitti/eval_tracking.php) |
| nuScenes (6 AD-GS scenes) | [nuscenes.org](https://www.nuscenes.org/nuscenes) |
| PandaSet (10 IDSplat sequences) | [Hugging Face](https://huggingface.co/datasets/georghess/pandaset) |
| **Our masks for KITTI / nuScenes / PandaSet** | [Google Drive](https://drive.google.com/drive/folders/1ZmVEwQ7Zw_ymfHw06QawHFZSbmk0pUk0) |

## 🚀 Training

All scripts take the method as `--method` (`astra`, `idsplat`, or an ablation such as `astra-nopose`,
`astra-novvc`, `astra-nolens`, `astra-nodrop`, `astra-temporal`; see `scripts/astra/flags.sh`).

```bash
# Waymo, scene ids 800..807 (scripts/arrays/wod_id_to_seq_adgs_split.txt)
bash scripts/astra/train_waymo.sh --scene 802 --data data/wod --masks_dir data/masks/waymo \
     --gt_masks_dir data/wod_gt_masks --tfrecord_dir data/wod/v1 --method astra

# KITTI, --split 75 | 50 | 25
bash scripts/astra/train_kitti.sh --sequence 0001 --split 75 --data data/kitti --masks_dir data/masks/kitti

# nuScenes
bash scripts/astra/train_nuscenes.sh --scene scene-0230 --data data/nuscenes --masks_dir data/masks/nuscenes

# PandaSet (needs the GPU alone)
bash scripts/astra/train_pandaset.sh --sequence 001 --data data/pandaset --masks_dir data/masks/pandaset
```

IDSplat trains one Waymo scene in about 1.5 h on an RTX 4090 when it has the GPU to itself; ASTRA adds one
half-resolution virtual render every second step after step 6k. KITTI and nuScenes scenes are faster.

## 📏 Evaluation

```bash
# Waymo held-out frames: PSNR / SSIM / LPIPS / DPSNR
bash scripts/astra/eval_waymo.sh --config outputs/astra-802/idsplat-wod/<time>/config.yml \
     --data data/wod --gt_masks_dir data/wod_gt_masks --out results/astra-802.json

# KITTI / nuScenes / PandaSet held-out frames
bash scripts/astra/eval_render.sh --config outputs/astra-kitti0001-75/idsplat/<time>/config.yml \
     --out results/astra-kitti0001-75

# Held-out cameras: run the IDSplat model of a scene first, it defines the shared co-visibility masks
bash scripts/astra/eval_xcam.sh --config outputs/idsplat-802/idsplat-wod/<time>/config.yml \
     --mask_dir results/xcam/masks --out results/xcam/idsplat-802.json
bash scripts/astra/eval_xcam.sh --config outputs/astra-802/idsplat-wod/<time>/config.yml \
     --mask_dir results/xcam/masks --out results/xcam/astra-802.json
python tools/xcam_table.py results/xcam idsplat astra
```

## 🔁 Reproducing the paper tables

[`docs/REPRODUCE.md`](docs/REPRODUCE.md) lists every run behind every table, as copy-paste loops. The raw result
files of our runs (`results/*.json`, one per run) are in the Drive folder; `tools/make_tables.py` turns a results
folder into the paper tables.

## 💾 Checkpoints

Trained ASTRA checkpoints for KITTI, nuScenes and PandaSet (pipeline state only, loadable with
`ns-eval` / `scripts/astra/eval_render.sh`) will be added to the [Drive folder](https://drive.google.com/drive/folders/1ZmVEwQ7Zw_ymfHw06QawHFZSbmk0pUk0) under
`checkpoints/` as the final runs finish.

Waymo checkpoints are not distributed: a Gaussian-splatting checkpoint is a reconstruction of the Waymo data,
which the Waymo licence does not allow us to share. Each Waymo scene can be retrained with the command above.

## 🗂️ Repository layout

```
nerfstudio/                       IDSplat code base (Nerfstudio) with the ASTRA components
  models/idsplat.py               pose interpolation (_interp_camopt), lens rendering (_lens_grid/_lens_warp), dropout
  model_components/astra_tivc.py  virtual-view consistency
  pipelines/idsplat_pipeline.py   loss wiring (--pipeline.tivc_*)
  data/dataparsers/               + KITTI-MOT and nuScenes parsers, AD-GS EVERY_N split
scripts/astra/                    training / evaluation entry points and the component flags
tools/                            mask generation, DPSNR GT masks, render dumps, metrics, held-out-camera benchmark
docs/                             data guide, reproduction guide, full results
```

## 📄 License

Apache 2.0, as the IDSplat code it builds on (see `LICENSE` and `NOTICE`). Datasets keep their own licences; the
released masks for KITTI (CC BY-NC-SA 3.0), nuScenes (CC BY-NC-SA 4.0) and PandaSet (CC BY 4.0) are derived from
those datasets and inherit their terms.

## 📌 Citation

The ASTRA paper is under review. Until it is public, please cite IDSplat, on which this code is built:

```bibtex
@inproceedings{lindstrom2026idsplat,
  title     = {{IDSplat}: Instance-Decomposed {3D} {Gaussian} Splatting for Driving Scenes},
  author    = {Lindstr{\"o}m, Carl and Rafidashti, Mahan and Fatemi, Maryam and Hammarstrand, Lars and Oswald, Martin R. and Svensson, Lennart},
  booktitle = {CVPR},
  year      = {2026}
}
```

## 🙏 Acknowledgment

ASTRA is built on [IDSplat](https://github.com/zenseact/idsplat) (Zenseact), which builds on
[SplatAD](https://github.com/carlinds/splatad), [NeuRAD-studio](https://github.com/georghess/neurad-studio) and
[Nerfstudio](https://github.com/nerfstudio-project/nerfstudio). The KITTI and nuScenes parsers are ported from
NeuRAD-studio. Masks are produced with [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) and
[SAM 2](https://github.com/facebookresearch/sam2). We thank the authors of AD-GS for their evaluation protocol.
