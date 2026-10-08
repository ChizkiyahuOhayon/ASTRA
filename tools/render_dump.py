"""Dump IDSplat renders (test views + every k-th train view) with GT and dynamic masks for error analysis.

usage: render_dump.py <config.yml> <out_dir> <wod_root> <gt_masks_dir> [train_every]
"""
import sys
from pathlib import Path

import numpy as np
import torch

from nerfstudio.utils.eval_utils import eval_setup

cfg, out, wod, gtdir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
train_every = int(sys.argv[5]) if len(sys.argv) > 5 else 4
out.mkdir(parents=True, exist_ok=True)


def update(config):
    if str(wod) != "-":  # WOD configs need the data root and GT-mask dir restored; other datasets keep theirs
        config.pipeline.datamanager.dataparser.data = wod
        config.pipeline.datamanager.dataparser.dynamic_gt_masks_dir = gtdir
    return config


_, pipeline, _, _ = eval_setup(cfg, update_config_callback=update)
model, dm = pipeline.model, pipeline.datamanager


def to_u8(x):
    return (x.clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()


@torch.no_grad()
def dump(split, camera, batch, idx):
    outputs = model.get_outputs_for_camera(camera=camera)
    gt = model.composite_with_background(model.get_gt_img(batch["image"].to(model.device)), outputs["background"])
    h, w = outputs["rgb"].shape[:2]
    rec = dict(pred=to_u8(outputs["rgb"]), gt=to_u8(gt[:h, :w]), time=float(camera.times.flatten()[0]))
    if "depth" in outputs:
        rec["depth"] = outputs["depth"].squeeze(-1).half().cpu().numpy()
    if "dynamic_gt_mask" in batch:
        m = batch["dynamic_gt_mask"]
        rec["dyn"] = (m.any(dim=-1) if m.ndim == 3 else m).cpu().numpy()[:h, :w]
    np.savez_compressed(out / f"{split}_{idx:04d}.npz", **rec)


model.eval()
for i, (camera, batch) in enumerate(dm.fixed_indices_eval_dataloader):
    dump("test", camera, batch, i)
for i in range(0, len(dm.train_dataset), train_every):
    camera, batch = dm.get_train_image_from_idx(i)
    dump("train", camera, batch, i)
print("DUMP_DONE", out)
