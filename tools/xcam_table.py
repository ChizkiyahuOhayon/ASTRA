"""Tabulate held-out-camera results: <xcam_dir>/<method>-<scene>.json -> per-method means + per-scene psnr_cov_aff.
usage: xcam_table.py <xcam_dir> [main methods...]   (default: idsplat astra)
Per-camera values are nanmeans over frames (a frame with an empty fit gives NaN); the camera mean averages the two side
cameras. "scenes in common" is the intersection over the main methods only; other methods are shown on their own scenes."""
import json, glob, collections, sys
import numpy as np
R = sys.argv[1] if len(sys.argv) > 1 else "results/xcam"
SIDE = ("FRONT_LEFT", "FRONT_RIGHT")
keys = ["cov", "psnr", "ssim", "lpips", "psnr_cov", "psnr_cov_aff", "ssim_cov", "lpips_cov"]
d = collections.defaultdict(dict)
for f in glob.glob(f"{R}/*.json"):
    m, sid = f.split("/")[-1][:-5].rsplit("-", 1)
    rows = json.load(open(f))["rows"]
    d[m][sid] = {k: float(np.mean([np.nanmean([r.get(k, np.nan) for r in rows[c]]) for c in SIDE])) for k in keys}
main = sys.argv[2:] or ["idsplat", "astra"]
main = [m for m in main if m in d]
sids = sorted(set.intersection(*[set(d[m]) for m in main])) if main else []
print("scenes in common:", sids)
print("method".ljust(12), *[k.rjust(12) for k in keys])
for m in main:
    print(m.ljust(12), *[f"{np.mean([d[m][s][k] for s in sids]):12.4f}" for k in keys])
print("other methods (own scenes):")
for m in sorted(set(d) - set(main)):
    ss = sorted(d[m])
    print(m.ljust(12), ss, *[f"{k}={np.mean([d[m][s][k] for s in ss]):.3f}" for k in ("psnr_cov_aff", "lpips_cov")])
print("per-scene psnr_cov_aff:")
for s in sorted(set().union(*[set(v) for v in d.values()])):
    print(s, {m: round(d[m][s]["psnr_cov_aff"], 2) for m in main + sorted(set(d) - set(main)) if s in d[m]})
