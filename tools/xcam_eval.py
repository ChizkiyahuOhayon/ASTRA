"""ASTRA held-out-camera extrapolation eval: render a FRONT-trained IDSplat model at the real FRONT_LEFT / FRONT_RIGHT
poses of the same frames (never seen in training) and score against the real images.

usage: xcam_eval.py <config.yml> <out_json> <mask_dir> [every]
  mask_dir: co-visibility masks shared by all methods. If <mask_dir>/<sid>.npz is missing, this model's depth defines
            them (run the reference model first). A side pixel is covered when its back-projected point falls inside
            some FRONT train image (0.5 m < z < 300 m).
Metrics per camera: full-image PSNR/SSIM/LPIPS-VGG, and on covered pixels PSNR, SSIM, LPIPS (uncovered pixels pasted
from GT), plus PSNR after a per-image per-channel affine colour fit on covered pixels (Waymo exposes cameras
independently and the appearance embedding is FRONT's).
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.utils.eval_utils import eval_setup

cfg_path, out_json, mask_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
every = int(sys.argv[4]) if len(sys.argv) > 4 else 2
SIDE = ("FRONT_LEFT", "FRONT_RIGHT")
DS = 8  # co-visibility mask resolution divisor

config, pipeline, _, _ = eval_setup(cfg_path)
model, dm = pipeline.model, pipeline.datamanager
dev = model.device
model.eval()

# ---- re-parse the same segment with the side cameras (same lidar-defined world frame)
pc = copy.deepcopy(config.pipeline.datamanager.dataparser)
pc.cameras = ("FRONT",) + SIDE
pc.train_eval_split_type = type(pc.train_eval_split_type).LINSPACE
pc.train_split_fraction = 1.0
pc.dynamic_gt_masks_dir = None
outs = pc.setup().get_dataparser_outputs("train")
cams = outs.cameras
sid_of = cams.metadata["sensor_idxs"].flatten()  # 0 FRONT, 1 FRONT_LEFT, 2 FRONT_RIGHT

# align time origin + verify world frame against the trained FRONT train cameras
tr = dm.train_dataset.cameras
tf = cams.times.flatten()[sid_of == 0].double()
cf = cams.camera_to_worlds[sid_of == 0]
tt = tr.times.flatten().double()
nn_idx = (tf[:, None] - tt[None]).abs().argmin(0)  # for each train frame, nearest new FRONT frame
dt = (tf[nn_idx] - tt).median().item()
pose_err = (cf[nn_idx, :3, 3] - tr.camera_to_worlds[:, :3, 3]).norm(dim=-1).max().item()
print(f"time offset {dt*1e3:.3f} ms, FRONT position mismatch {pose_err:.2e} m", flush=True)
assert pose_err < 1e-2, "world frame differs from training"
cams.times = (cams.times.double() - dt).float()

lp = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(dev)

# FRONT train cameras for co-visibility (raw poses; corrections are mm-level here)
tr_c2w = tr.camera_to_worlds.to(dev)


def Kmat(c):
    f = lambda a: a.flatten().float()
    K = torch.zeros(len(f(c.fx)), 3, 3)
    K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = f(c.fx), f(c.fy), f(c.cx), f(c.cy), 1.0
    return K.to(dev)


tr_K = Kmat(tr)
trW, trH = int(tr.width.flatten()[0]), int(tr.height.flatten()[0])

mask_file = mask_dir / f"{cfg_path.parts[-4].split('-')[-1]}.npz"
masks = dict(np.load(mask_file)) if mask_file.exists() else {}
make_masks = not masks


def covis(depth, cam):
    """depth HxW (rendered) -> bool covered mask at 1/DS resolution."""
    H, W = depth.shape
    d = F.avg_pool2d(depth[None, None], DS)[0, 0]
    h, w = d.shape
    K = Kmat(cam)[0]
    v, u = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")
    u = (u.float() + 0.5) * DS
    v = (v.float() + 0.5) * DS
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    # nerfstudio camera convention: x right, y up, z back (OpenGL)
    p_cam = torch.stack([x * d, -y * d, -d], -1).reshape(-1, 3)
    c2w = cam.camera_to_worlds[0].to(dev)
    p_w = p_cam @ c2w[:3, :3].T + c2w[:3, 3]
    R, t = tr_c2w[:, :3, :3], tr_c2w[:, :3, 3]
    q = torch.einsum("nji,nmj->nmi", R, p_w[None] - t[:, None])  # world -> each train cam
    z = -q[..., 2]
    uu = tr_K[:, None, 0, 0] * q[..., 0] / z.clamp_min(1e-6) + tr_K[:, None, 0, 2]
    vv = -tr_K[:, None, 1, 1] * q[..., 1] / z.clamp_min(1e-6) + tr_K[:, None, 1, 2]
    ok = (z > 0.5) & (z < 300) & (uu >= 0) & (uu < trW) & (vv >= 0) & (vv < trH)
    return ok.any(0).reshape(h, w)


def affine_fit(p, g, m):
    """per-channel least squares g ~ a p + b on mask m; p,g 3xHxW."""
    out = torch.empty_like(p)
    for c in range(3):
        x, y = p[c][m], g[c][m]
        A = torch.stack([x, torch.ones_like(x)], -1)
        sol = torch.linalg.lstsq(A, y[:, None]).solution.flatten()
        out[c] = (p[c] * sol[0] + sol[1]).clamp(0, 1)
    return out


def psnr_on(p, g, m=None):
    se = (p - g) ** 2
    mse = se.mean() if m is None else se[:, m].mean()
    return (-10 * torch.log10(mse.clamp_min(1e-12))).item()


rows = {c: [] for c in SIDE}
fig_dir = out_json.with_suffix("")
fig_dir.mkdir(parents=True, exist_ok=True)
side_idx = [i for i in range(len(cams)) if int(sid_of[i]) != 0]
seen = {}
with torch.no_grad():
    for k, i in enumerate(side_idx):
        name = SIDE[int(sid_of[i]) - 1]
        seen[name] = seen.get(name, -1) + 1
        if seen[name] % every:
            continue
        cam = cams[i : i + 1].to(dev)
        cam.metadata = dict(cam.metadata)
        cam.metadata["sensor_idxs"] = torch.zeros_like(cam.metadata["sensor_idxs"])  # FRONT appearance + camopt
        cam.metadata["cam_idx"] = i
        o = model.get_outputs_for_camera(camera=cam)
        p = o["rgb"].clamp(0, 1).permute(2, 0, 1)
        H, W = p.shape[1:]
        g = torch.from_numpy(np.array(Image.open(outs.image_filenames[i]).convert("RGB"))).to(dev)
        g = g.permute(2, 0, 1).float()[:, :H, :W] / 255
        key = f"{i:05d}"
        if make_masks:
            masks[key] = np.packbits(covis(o["depth"].squeeze(-1).float(), cam).cpu().numpy())
        mshape = (H // DS, W // DS)
        m_lo = torch.from_numpy(np.unpackbits(masks[key])[: mshape[0] * mshape[1]].reshape(mshape)).to(dev)
        m = F.interpolate(m_lo[None, None].float(), size=(H, W), mode="nearest")[0, 0] > 0.5
        s_full, s_map = ssim_fn(p[None], g[None], data_range=1.0, return_full_image=True)
        pc_ = torch.where(m[None], p, g)
        r = dict(
            i=i,
            cov=m.float().mean().item(),
            psnr=psnr_on(p, g),
            ssim=s_full.item() if s_full.ndim == 0 else s_full.mean().item(),
            lpips=lp(p[None], g[None]).item(),
        )
        if m.any():
            pa = affine_fit(p, g, m)
            r.update(
                psnr_cov=psnr_on(p, g, m),
                psnr_cov_aff=psnr_on(pa, g, m),
                ssim_cov=s_map[0][:, m].mean().item(),
                lpips_cov=lp(pc_[None], g[None]).item(),
            )
        rows[name].append(r)
        if len(rows[name]) % 10 == 1:  # a few frames for figures
            Image.fromarray((p.permute(1, 2, 0) * 255 + 0.5).byte().cpu().numpy()).save(fig_dir / f"{name}_{i:05d}.jpg", quality=92)
        if k % 40 == 0:
            print(name, i, {a: round(b, 3) for a, b in r.items() if a != "i"}, flush=True)

if make_masks:
    mask_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(mask_file, **masks)
summary = {}
for c, rs in rows.items():
    keys = [k for k in rs[0] if k != "i"]
    summary[c] = {k: float(np.mean([r[k] for r in rs if k in r])) for k in keys} | {"n": len(rs)}
summary["mean"] = {k: float(np.mean([summary[c][k] for c in SIDE])) for k in summary[SIDE[0]] if k != "n"}
json.dump({"config": str(cfg_path), "time_offset_s": dt, "summary": summary, "rows": rows}, open(out_json, "w"), indent=1)
print("XCAM_DONE", json.dumps(summary["mean"]))
