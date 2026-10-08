"""PSNR / SSIM / LPIPS(vgg) / LPIPS(alex) over the test renders of a dump dir (same metric code as IDSplat).

usage: metrics_dump.py <dump_dir> [<dump_dir> ...]   -> one JSON line per dir
"""
import json
import sys
from glob import glob

import numpy as np
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

dev = "cuda"
psnr = PeakSignalNoiseRatio(data_range=1.0).to(dev)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(dev)
lp = {n: LearnedPerceptualImagePatchSimilarity(net_type=n, normalize=True).to(dev) for n in ("vgg", "alex")}
for d in sys.argv[1:]:
    vals = {"psnr": [], "ssim": [], "lpips_vgg": [], "lpips_alex": []}
    for f in sorted(glob(f"{d}/test_*.npz")):
        z = np.load(f)
        p = torch.from_numpy(z["pred"]).to(dev).permute(2, 0, 1)[None].float() / 255
        g = torch.from_numpy(z["gt"]).to(dev).permute(2, 0, 1)[None].float() / 255
        with torch.no_grad():
            vals["psnr"].append(psnr(p, g).item())
            vals["ssim"].append(ssim(p, g).item())
            for n, m in lp.items():
                vals[f"lpips_{n}"].append(m(p, g).item())
    print(json.dumps({"dir": d, "n": len(vals["psnr"]), **{k: float(np.mean(v)) for k, v in vals.items()}}), flush=True)
