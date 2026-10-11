"""Point a downloaded ASTRA run at your local folders, so that it can be evaluated or rendered.

The released checkpoints keep the config.yml written at training time, which holds the paths of our machine.
This rewrites the output folder, the run name, the dataset root and the mask folder in place.

usage:
  uv run python tools/relocate_run.py outputs/astra-kitti0001-75 --data data/kitti --masks_dir data/masks/kitti
"""
import argparse
from pathlib import Path

import yaml

ap = argparse.ArgumentParser()
ap.add_argument("run_dir", type=Path, help="unpacked run folder, e.g. outputs/astra-kitti0001-75")
ap.add_argument("--data", type=Path, required=True, help="dataset root (KITTI, nuScenes or PandaSet folder)")
ap.add_argument("--masks_dir", type=Path, required=True, help="instance-mask folder of that dataset")
a = ap.parse_args()

run = a.run_dir.resolve()
cfgs = sorted(run.glob("*/*/config.yml"))
assert cfgs, f"no config.yml under {run}"
for cfg_path in cfgs:
    cfg = yaml.load(cfg_path.read_text(), Loader=yaml.Loader)
    cfg.output_dir = run.parent
    cfg.experiment_name = run.name
    dp = cfg.pipeline.datamanager.dataparser
    dp.data = a.data.resolve()
    dp.object_masks_dir = str(a.masks_dir.resolve())
    cfg_path.write_text(yaml.dump(cfg), "utf8")
    print("updated", cfg_path)
