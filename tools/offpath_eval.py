"""Off-path tests scored against real measurements (paper Sec. 4.3, Tab. 5): cameras the model never saw.

usage: offpath_eval.py <config.yml> <out_json> <mode>
  side    Waymo FRONT-trained model rendered at the real FRONT_LEFT / FRONT_RIGHT poses; depth scored against the
          LiDAR points of the same instant (geometry of the held-out views; no image or colour fit involved).
  shift   Waymo FRONT cameras moved sideways by 0 / 0.5 / 1 / 2 / 3 m (alternating left and right); rendered depth
          scored against the same instant's LiDAR points projected into the moved camera.
  stereo  KITTI model trained on the left camera only, rendered at the real right camera (0.54 m to the side, same
          orientation) at every frame: PSNR / SSIM / LPIPS against the real right image, plus LiDAR depth error.
LiDAR points are projected with the camera model the method itself renders with (OpenCV lens model when the model
renders lens distortion, pinhole otherwise), and a point is kept only if no other point lies clearly in front of it
in a 9x9 pixel window, so points hidden from the new viewpoint are not scored.
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

cfg_path, out_json, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
SHIFTS = (0.0, 0.5, 1.0, 2.0, 3.0)
config, pipeline, _, _ = eval_setup(cfg_path)
model, dm = pipeline.model, pipeline.datamanager
dev = model.device
model.eval()
lens_on = bool(getattr(model.config, "lens_distortion", False))

pc = copy.deepcopy(config.pipeline.datamanager.dataparser)
if mode == "stereo":
    ref, eval_names = "image_02", ("image_03",)
    pc.cameras = ("image_02", "image_03")
else:
    ref = "FRONT"
    eval_names = ("FRONT_LEFT", "FRONT_RIGHT") if mode == "side" else ("FRONT",)
    pc.cameras = ("FRONT", "FRONT_LEFT", "FRONT_RIGHT") if mode == "side" else ("FRONT",)
    pc.remove_points_outside_camera_frustum = False  # keep LiDAR returns outside the training camera's view
    if hasattr(pc, "dynamic_gt_masks_dir"):
        pc.dynamic_gt_masks_dir = None
if hasattr(pc, "add_missing_points"):
    pc.add_missing_points = False  # rays without a return are stored as fake far points; score real returns only
pc.train_eval_split_type = type(pc.train_eval_split_type).LINSPACE
pc.train_split_fraction = 1.0
outs = pc.setup().get_dataparser_outputs("train")
cams = outs.cameras
sid_of = cams.metadata["sensor_idxs"].flatten()

# align time origin and check the world frame against the trained reference-camera train frames
tr = dm.train_dataset.cameras
tr = tr[tr.metadata["sensor_idxs"].flatten() == 0]
tf = cams.times.flatten()[sid_of == 0].double()
cf = cams.camera_to_worlds[sid_of == 0]
tt = tr.times.flatten().double()
nn_idx = (tf[:, None] - tt[None]).abs().argmin(0)
dt = (tf[nn_idx] - tt).median().item()
pose_err = (cf[nn_idx, :3, 3] - tr.camera_to_worlds[:, :3, 3]).norm(dim=-1).max().item()
print(f"time offset {dt*1e3:.3f} ms, {ref} position mismatch {pose_err:.2e} m, lens {lens_on}", flush=True)
assert pose_err < 1e-2, "world frame differs from training"
cams.times = (cams.times.double() - dt).float()
train_times = tt.float()

lid = outs.metadata["lidars"]
l2w = lid.lidar_to_worlds.float()
ltimes = (lid.times.flatten().double() - dt).float()
pcs = outs.metadata["point_clouds"]


def lidar_world(t):
    """Real LiDAR returns of the scan(s) at time t in world coordinates (WOD returns are already motion-compensated)."""
    sel = ((ltimes - t).abs() < 0.06).nonzero().flatten().tolist()
    pts = [pcs[k][:, :3].float() @ l2w[k][:3, :3].T + l2w[k][:3, 3] for k in sel]
    return torch.cat(pts).to(dev) if pts else torch.zeros(0, 3, device=dev)


tr_c2w = tr.camera_to_worlds.to(dev).float()
tr_f = lambda a: a.flatten().float().to(dev)
trK = (tr_f(tr.fx), tr_f(tr.fy), tr_f(tr.cx), tr_f(tr.cy))
trW, trH = float(tr.width.flatten()[0]), float(tr.height.flatten()[0])


def covered(pw):
    """True for points that fall inside at least one training image of the reference camera (0.5 m < z < 300 m).
    Uses LiDAR geometry only, so the mask does not depend on any of the compared models."""
    out = torch.zeros(len(pw), dtype=torch.bool, device=dev)
    for a in range(0, len(tr_c2w), 32):
        R, t = tr_c2w[a : a + 32, :3, :3], tr_c2w[a : a + 32, :3, 3]
        q = torch.einsum("nji,nmj->nmi", R, pw[None] - t[:, None])
        z = -q[..., 2]
        fx, fy, cx, cy = (k[a : a + 32, None] for k in trK)
        uu = fx * q[..., 0] / z.clamp_min(1e-6) + cx
        vv = -fy * q[..., 1] / z.clamp_min(1e-6) + cy
        out |= ((z > 0.5) & (z < 300) & (uu >= 0) & (uu < trW) & (vv >= 0) & (vv < trH)).any(0)
    return out


# Occlusion-aware version: a point counts as seen only if it is in front of (or on) the LiDAR surface that the
# training camera saw at that instant. Built from each training frame's own LiDAR scan, at quarter resolution with a
# 5x5 min filter to close the gaps between beams. Still LiDAR only, so it does not depend on any compared model.
QS = 4
qW, qH = int(trW) // QS, int(trH) // QS
seen_zb = None


def _proj_train(pw, a, b):
    R, t = tr_c2w[a:b, :3, :3], tr_c2w[a:b, :3, 3]
    q = torch.einsum("nji,nmj->nmi", R, pw[None] - t[:, None])
    z = -q[..., 2]
    fx, fy, cx, cy = (k[a:b, None] / QS for k in trK)
    uu = fx * q[..., 0] / z.clamp_min(1e-6) + cx
    vv = -fy * q[..., 1] / z.clamp_min(1e-6) + cy
    ok = (z > 0.5) & (z < 300) & (uu >= 0) & (uu < qW) & (vv >= 0) & (vv < qH)
    return uu.long().clamp(0, qW - 1), vv.long().clamp(0, qH - 1), z, ok


def build_seen_zb():
    zbs = []
    for j in range(len(tr_c2w)):
        pj = lidar_world(float(train_times[j]))
        zb = torch.full((qH * qW,), 1e9, device=dev)
        if len(pj):
            u, v, z, ok = _proj_train(pj, j, j + 1)
            u, v, z = u[0][ok[0]], v[0][ok[0]], z[0][ok[0]]
            zb.scatter_reduce_(0, v * qW + u, z, reduce="amin")
        zbs.append(-F.max_pool2d(-zb.view(1, 1, qH, qW), 5, 1, 2).view(-1))
    return torch.stack(zbs)


def seen_mask(pw):
    global seen_zb
    if seen_zb is None:
        seen_zb = build_seen_zb()
    out = torch.zeros(len(pw), dtype=torch.bool, device=dev)
    for a in range(0, len(tr_c2w), 32):
        b = min(a + 32, len(tr_c2w))
        u, v, z, ok = _proj_train(pw, a, b)
        zref = torch.gather(seen_zb[a:b], 1, v * qW + u)
        out |= (ok & (z <= zref * 1.05 + 0.3)).any(0)
    return out


def project(pw, c2w, cam, H, W):
    """World points -> pixel (u, v) and z, with the rendering camera model of this method."""
    R, t = c2w[:3, :3], c2w[:3, 3]
    q = (pw - t) @ R
    z = -q[:, 2]
    x, y = q[:, 0] / z.clamp_min(1e-6), -q[:, 1] / z.clamp_min(1e-6)
    fx, fy = float(cam.fx.flatten()[0]), float(cam.fy.flatten()[0])
    cx, cy = float(cam.cx.flatten()[0]), float(cam.cy.flatten()[0])
    # the lens polynomial folds back outside the field of view: only points whose pinhole projection is near the
    # image are kept
    inside = (x * fx + cx > -0.1 * W) & (x * fx + cx < 1.1 * W) & (y * fy + cy > -0.1 * H) & (y * fy + cy < 1.1 * H)
    z = torch.where(inside, z, torch.zeros_like(z))
    d = (cam.metadata or {}).get("lens_distortion") if lens_on else None
    if d is not None:
        k1, k2, p1, p2, k3 = [float(v) for v in d.reshape(-1)[:5]]
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        x, y = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    u = fx * x + cx
    v = fy * y + cy
    return u, v, z


def depth_metrics(depth, pw, c2w, cam, cov=None, sn=None):
    """LiDAR depth error of a rendered depth map; with `cov`, also restricted to LiDAR-covered points.
    Returns (metrics, pixel index of covered visible points) for photometric scoring on the same pixels."""
    H, W = depth.shape
    u, v, z = project(pw, c2w, cam, H, W)
    ok = (z > 1.0) & (z < 80.0) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    cv = cov[ok] if cov is not None else None
    sv_ = sn[ok] if sn is not None else None
    u, v, z = u[ok].long(), v[ok].long(), z[ok]
    if len(z) < 50:
        return None, None
    zb = torch.full((H * W,), 1e9, device=dev)
    zb.scatter_reduce_(0, v * W + u, z, reduce="amin")
    near = -F.max_pool2d(-zb.view(1, 1, H, W), 9, 1, 4).view(-1)
    vis = z <= near[v * W + u] * 1.05 + 0.1
    res = {}
    for tag, m in (("", vis), ("_cov", vis & cv if cv is not None else None),
                   ("_seen", vis & sv_ if sv_ is not None else None)):
        if m is None or m.sum() < 50:
            continue
        g, p = z[m], depth[v[m], u[m]].nan_to_num(80.0).clamp(1e-3, 80.0)  # standard 80 m cap; empty pixel = far
        rel = (p - g).abs() / g
        ratio = torch.maximum(p / g.clamp_min(1e-6), g / p.clamp_min(1e-6))
        res.update({f"absrel{tag}": rel.mean().item(), f"rmse{tag}": ((p - g) ** 2).mean().sqrt().item(),
                    f"d1{tag}": (ratio < 1.25).float().mean().item(), f"npts{tag}": int(m.sum())})
    pix = {}
    for tag, c in (("cov", cv), ("seen", sv_)):
        if c is not None:
            m = vis & c
            pix[tag] = torch.unique(v[m] * W + u[m])
    return res, pix


def psnr(p, g):
    return (-10 * torch.log10(((p - g) ** 2).mean().clamp_min(1e-12))).item()


def render(cam, i):
    cam.metadata = dict(cam.metadata)
    cam.metadata["sensor_idxs"] = torch.zeros_like(cam.metadata["sensor_idxs"])  # reference appearance + camopt
    cam.metadata["cam_idx"] = i
    o = model.get_outputs_for_camera(camera=cam)
    return o["rgb"].clamp(0, 1), o["median_depth"].squeeze(-1).float()  # depth where opacity reaches 0.5 = the surface


def eval_c2w(cam):
    """Pose the model actually renders with (interpolated per-frame correction when the model has one)."""
    if getattr(model.config, "camopt_interp_eval", False):
        return model._interp_camopt(cam)[0].to(dev)
    return cam.camera_to_worlds[0].to(dev)


fig_dir = out_json.with_suffix("")
fig_dir.mkdir(parents=True, exist_ok=True)
rows = []
if mode == "stereo":
    lp = {n: LearnedPerceptualImagePatchSimilarity(net_type=n, normalize=True).to(dev) for n in ("alex", "vgg")}
idx = [i for i in range(len(cams)) if cams.metadata["sensor_idxs"].flatten()[i] in
       [pc.cameras.index(n) for n in eval_names]]
seen = {}
with torch.no_grad():
    for k, i in enumerate(idx):
        nm_ = int(sid_of[i]); seen[nm_] = seen.get(nm_, -1) + 1
        if mode in ("side", "shift") and seen[nm_] % 2:
            continue
        base = cams[i : i + 1].to(dev)
        t = float(base.times.flatten()[0])
        pw = lidar_world(t)
        is_train_t = bool((train_times - t).abs().min() < 1e-3)
        name = pc.cameras[int(sid_of[i])]
        if mode in ("side", "stereo"):
            cam = copy.deepcopy(base)
            rgb, dep = render(cam, i)
            c2w = eval_c2w(cam)
            r = dict(i=i, cam=name, train_time=is_train_t)
            cov = covered(pw) if mode == "side" else None
            sn = seen_mask(pw) if mode == "side" else None
            dm_, pix = depth_metrics(dep, pw, c2w, cam, cov, sn)
            if dm_:
                r.update(dm_)
            p = rgb.permute(2, 0, 1)
            H, W = p.shape[1:]
            g = torch.from_numpy(np.array(Image.open(outs.image_filenames[i]).convert("RGB"))).to(dev)
            g = g.permute(2, 0, 1).float()[:, :H, :W] / 255
            for tag, px in (pix or {}).items():
                if mode == "side" and len(px) > 100:
                    pp, gg = p.reshape(3, -1)[:, px], g.reshape(3, -1)[:, px]
                    r[f"psnr_l{tag}"] = (-10 * torch.log10(((pp - gg) ** 2).mean().clamp_min(1e-12))).item()
                    r[f"l{tag}_frac"] = len(px) / float(dm_.get("npts", 1))
            if mode == "stereo":
                r.update(psnr=psnr(p, g), ssim=ssim_fn(p[None], g[None], data_range=1.0).item(),
                         lpips_alex=lp["alex"](p[None], g[None]).item(), lpips_vgg=lp["vgg"](p[None], g[None]).item())
                if k % 8 == 0:
                    Image.fromarray((rgb * 255 + 0.5).byte().cpu().numpy()).save(fig_dir / f"{name}_{i:05d}.jpg", quality=95)
            rows.append(r)
        else:
            sign = 1.0 if (seen[nm_] // 2) % 2 == 0 else -1.0
            for s in SHIFTS:
                cam = copy.deepcopy(base)
                c2w0 = cam.camera_to_worlds[0]
                cam.camera_to_worlds = torch.cat([c2w0[:3, :3], c2w0[:3, 3:] + sign * s * c2w0[:3, 0:1]], -1)[None]
                rgb, dep = render(cam, i)
                r = dict(i=i, shift=s, sign=sign, train_time=is_train_t)
                dm_, _ = depth_metrics(dep, pw, eval_c2w(cam), cam)
                if dm_:
                    r.update(dm_)
                rows.append(r)
                if seen[nm_] % 40 == 0:
                    Image.fromarray((rgb * 255 + 0.5).byte().cpu().numpy()).save(
                        fig_dir / f"shift{s:.1f}_{'L' if sign < 0 else 'R'}_{i:05d}.jpg", quality=95)
        if k % 20 == 0:
            print(k, len(idx), {a: (round(b, 4) if isinstance(b, float) else b) for a, b in rows[-1].items()}, flush=True)


def mean_of(rs):
    keys = [k for k in ("absrel", "rmse", "d1", "npts", "absrel_cov", "rmse_cov", "d1_cov", "npts_cov", "psnr_lcov",
                        "lcov_frac", "absrel_seen", "rmse_seen", "d1_seen", "npts_seen", "psnr_lseen", "lseen_frac", "psnr", "ssim", "lpips_alex", "lpips_vgg") if any(k in r for r in rs)]
    return {k: float(np.mean([r[k] for r in rs if k in r])) for k in keys} | {"n": len(rs)}


summary = {}
if mode == "shift":
    for s in SHIFTS:
        summary[f"{s:.1f}"] = mean_of([r for r in rows if r["shift"] == s])
else:
    summary["all"] = mean_of(rows)
    summary["train_time"] = mean_of([r for r in rows if r["train_time"]])
    summary["test_time"] = mean_of([r for r in rows if not r["train_time"]])
    for n in eval_names:
        summary[n] = mean_of([r for r in rows if r["cam"] == n])
json.dump({"config": str(cfg_path), "mode": mode, "lens": lens_on, "summary": summary, "rows": rows}, open(out_json, "w"), indent=1)
print("V3_DONE", json.dumps(summary))
