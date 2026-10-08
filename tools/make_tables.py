"""Build every paper table from the raw result files.

usage: python3 tools/make_tables.py [results_dir] [paper_dir]
writes <paper_dir>/tables/*.tex, <paper_dir>/numbers.tex and ../results/PAPER_TABLES.md (next to the paper dir).
Our method row uses the final configuration ("stivc") wherever all scenes of a benchmark are finished, and the
pose + temporal-VVC configuration ("astra") otherwise; the table builder prints which one it used.
"""
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
R = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data" / "results"
P = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "paper"
(P / "tables").mkdir(parents=True, exist_ok=True)
pub = json.load(open(R / "published.json"))
WAYMO = list(range(800, 808))
SIDE = ("FRONT_LEFT", "FRONT_RIGHT")
LOG = []


def nmean(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(x)) if x else float("nan")


# ---------------------------------------------------------------- loaders
def waymo_run(exp, sid, suffix=""):
    f = R / f"{exp}-{sid}{suffix}.json"
    if not f.exists():
        return None
    r = json.load(open(f))["results"]
    return dict(psnr=r["psnr"], ssim=r["ssim"], lpips=r["lpips"], dpsnr=r.get("dpsnr_all", float("nan")), fps=r.get("fps"))


def waymo_mean(exp, sids=WAYMO, suffix=""):
    rows = [waymo_run(exp, s, suffix) for s in sids]
    if any(r is None for r in rows):
        return None
    return {k: nmean([r[k] for r in rows]) for k in ("psnr", "ssim", "lpips", "dpsnr")}


def misc_runs(exp, pat):
    fs = sorted(glob.glob(str(R / f"{exp}-{pat}.metrics.json")))
    return {Path(f).name[len(exp) + 1:].replace(".metrics.json", ""): json.load(open(f)) for f in fs}


def misc_mean(exp, pat, n_expected, lp):
    rs = misc_runs(exp, pat)
    if len(rs) < n_expected:
        return None
    rs = list(rs.values())
    return dict(psnr=nmean([r["psnr"] for r in rs]), ssim=nmean([r["ssim"] for r in rs]), lpips=nmean([r[lp] for r in rs]))


def xcam(exp, sid):
    f = R / "xcam" / f"{exp}-{sid}.json"
    if not f.exists():
        return None
    rows = json.load(open(f))["rows"]
    keys = ("psnr", "psnr_cov", "psnr_cov_aff", "ssim_cov", "lpips_cov", "cov")
    return {k: float(np.mean([nmean([r.get(k, float("nan")) for r in rows[c]]) for c in SIDE])) for k in keys}


def xcam_mean(exp, sids=WAYMO):
    rows = [xcam(exp, s) for s in sids]
    if any(r is None for r in rows):
        return None
    return {k: nmean([r[k] for r in rows]) for k in rows[0]}


def ours(waymo=True, pat=None, n=None, lp=None):
    """final config when complete, else the pose + temporal-VVC config."""
    for exp in ("stivc", "astra"):
        m = waymo_mean(exp) if waymo else misc_mean(exp, pat, n, lp)
        if m:
            LOG.append(f"ours[{'waymo' if waymo else pat}] = {exp}")
            return m, exp
    return None, None


# ---------------------------------------------------------------- formatting helpers
def rank_marks(vals, higher):
    """returns per-index cell colour: 1st/2nd/3rd (IDSplat paper convention)."""
    idx = [i for i, v in enumerate(vals) if v is not None and not math.isnan(v)]
    order = sorted(idx, key=lambda i: -vals[i] if higher else vals[i])
    marks = [""] * len(vals)
    for r, i in enumerate(order[:3]):
        marks[i] = ["\\cellcolor{tabfirst}", "\\cellcolor{tabsecond}", "\\cellcolor{tabthird}"][r]
    return marks


def fmt(v, d):
    return "--" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{d}f}"


def tex_table(rows, ncols, higher, digits, mark=True):
    cols = range(ncols)
    """rows: [(name, [vals], is_ours)] -> body lines with rank colours per column."""
    body = []
    marks = [rank_marks([r[1][c] for r in rows], higher[c]) if mark else [""] * len(rows) for c in range(len(cols))]
    for i, (name, vals, is_ours) in enumerate(rows):
        cells = [f"{marks[c][i]}{'\\textbf{' if is_ours else ''}{fmt(vals[c], digits[c])}{'}' if is_ours else ''}" for c in range(len(cols))]
        body.append(f"{name} & " + " & ".join(cells) + " \\\\")
    return body


NAMES = {"StreetGS*": "StreetGS$^\\dagger$~\\cite{yan2024streetgs}", "4DGF*": "4DGF$^\\dagger$~\\cite{4dgf}",
         "PVG": "PVG~\\cite{chen2026periodic}", "PVG*": "PVG$^\\dagger$~\\cite{chen2026periodic}",
         "EmerNeRF": "EmerNeRF~\\cite{emernerf}", "Grid4D": "Grid4D~\\cite{grid4d}",
         "SplatAD*": "SplatAD$^\\dagger$~\\cite{hess2025splatad}", "AD-GS": "AD-GS~\\cite{xu2025ad}",
         "IDSplat": "IDSplat~\\cite{lindstrom2026idsplat}", "ML-NSG*": "ML-NSG$^\\dagger$~\\cite{fischer2024multi}",
         "SUDS": "SUDS~\\cite{turki2023suds}"}
NUM = {}

# ---------------------------------------------------------------- Table: Waymo main
rep = waymo_mean("repro")
our, our_exp = ours()
rows = [(NAMES[k], v, False) for k, v in pub["waymo"].items()]
rows.append(("IDSplat (reproduced)", [rep["psnr"], rep["ssim"], rep["lpips"], rep["dpsnr"]], False))
if our:
    rows.append(("\\textbf{ASTRA (ours)}", [our["psnr"], our["ssim"], our["lpips"], our["dpsnr"]], True))
body = tex_table(rows, 4, [True, True, False, True], [2, 3, 3, 2])
open(P / "tables" / "waymo.tex", "w").write("\n".join(body) + "\n")
NUM.update(WaymoOursPSNR=fmt(our["psnr"], 2), WaymoOursSSIM=fmt(our["ssim"], 3), WaymoOursLPIPS=fmt(our["lpips"], 3),
           WaymoOursDPSNR=fmt(our["dpsnr"], 2), WaymoRepPSNR=fmt(rep["psnr"], 2), WaymoRepDPSNR=fmt(rep["dpsnr"], 2),
           WaymoGainPSNR=fmt(our["psnr"] - rep["psnr"], 2), WaymoGainDPSNR=fmt(our["dpsnr"] - rep["dpsnr"], 2),
           WaymoGainPub=fmt(our["psnr"] - pub["waymo"]["IDSplat"][0], 2),
           WaymoGainADGS=fmt(our["psnr"] - pub["waymo"]["AD-GS"][0], 2))

# ---------------------------------------------------------------- Tables: KITTI / nuScenes / PandaSet
other = {}
for key, pat, n, lp, label in (("kitti75", "kitti*-75", 3, "lpips_alex", "KITTISeventyFive"), ("kitti50", "kitti*-50", 3, "lpips_alex", "KITTIFifty"),
                               ("kitti25", "kitti*-25", 3, "lpips_alex", "KITTITwentyFive"), ("nuscenes", "nusscene-*", 6, "lpips_vgg", "NUS"),
                               ("pandaset", "panda*", 10, "lpips_alex", "PANDA")):
    rep_m = misc_mean("repro", pat, n, lp)
    our_m, _ = ours(False, pat, n, lp)
    other[key] = (rep_m, our_m)
    if our_m:
        NUM[f"{label}OursPSNR"] = fmt(our_m["psnr"], 2)
        NUM[f"{label}GainPSNR"] = fmt(our_m["psnr"] - rep_m["psnr"], 2)
        NUM[f"{label}OursSSIM"] = fmt(our_m["ssim"], 3)
        NUM[f"{label}OursLPIPS"] = fmt(our_m["lpips"], 3)

# KITTI: three splits side by side
methods = list(pub["kitti75"].keys())
krows = []
for m in methods:
    krows.append((NAMES[m], sum([pub[s][m] for s in ("kitti75", "kitti50", "kitti25")], []), False))
krows.append(("IDSplat (reproduced)", sum([[other[s][0][k] for k in ("psnr", "ssim", "lpips")] for s in ("kitti75", "kitti50", "kitti25")], []), False))
if all(other[s][1] for s in ("kitti75", "kitti50", "kitti25")):
    krows.append(("\\textbf{ASTRA (ours)}", sum([[other[s][1][k] for k in ("psnr", "ssim", "lpips")] for s in ("kitti75", "kitti50", "kitti25")], []), True))
body = tex_table(krows, 9, [True, True, False] * 3, [2, 3, 3] * 3)
open(P / "tables" / "kitti.tex", "w").write("\n".join(body) + "\n")

for key, fn in (("nuscenes", "nuscenes"), ("pandaset", "pandaset")):
    rws = [(NAMES[m], v, False) for m, v in pub[key].items()]
    rep_m, our_m = other[key]
    rws.append(("IDSplat (reproduced)", [rep_m[k] for k in ("psnr", "ssim", "lpips")], False))
    if our_m:
        rws.append(("\\textbf{ASTRA (ours)}", [our_m[k] for k in ("psnr", "ssim", "lpips")], True))
    open(P / "tables" / f"{fn}.tex", "w").write("\n".join(tex_table(rws, 3, [True, True, False], [2, 3, 3])) + "\n")

# ---------------------------------------------------------------- Table: ablation (Waymo, all 8 scenes, cumulative)
abl = [("IDSplat~\\cite{lindstrom2026idsplat}", "repro", ""), ("+ pose refinement, raw test poses", "camopt", ".nointerp"),
       ("+ traj.-consistent pose refinement", "camopt", ""), ("+ temporal virtual views", "astra", ""),
       ("+ lens + dropout", "gdl", ""), ("+ off-trajectory views (full)", "stivc", "")]
arows = []
for name, exp, suf in abl:
    m = waymo_mean(exp, suffix=suf)
    x = xcam_mean(exp) if not suf else None
    arows.append((name, [None if m is None else m["psnr"], None if m is None else m["ssim"], None if m is None else m["lpips"],
                         None if m is None else m["dpsnr"], None if x is None else x["psnr_cov_aff"]], exp == "stivc"))
lines = []
for i, (name, vals, is_ours) in enumerate(arows):
    cells = [fmt(v, d) for v, d in zip(vals, (2, 3, 3, 2, 2))]
    if is_ours:
        cells = [f"\\textbf{{{c}}}" for c in cells]
    lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
open(P / "tables" / "ablation.tex", "w").write("\n".join(lines) + "\n")
for name, exp, suf in abl:
    m = waymo_mean(exp, suffix=suf)
    if m:
        NUM[f"Abl{exp.capitalize()}{'Raw' if suf else ''}PSNR"] = fmt(m["psnr"], 2)

# leave-one-out on 802 / 806 (full model without the spatial term)
loo = [("Full (temporal VVC)", "gdl"), ("w/o pose refinement", "gdlnopose"), ("w/o virtual views", "gdlnotivc"),
       ("w/o lens-consistent rendering", "gdrop"), ("w/o Gaussian dropout", "lens")]
ll = []
for name, exp in loo:
    rs = [waymo_run(exp, s) for s in (802, 806)]
    xs = [xcam(exp, s) for s in (802, 806)]
    vals = [r["psnr"] if r else None for r in rs] + [nmean([r["psnr"] for r in rs]) if all(rs) else None,
                                                    nmean([x["psnr_cov_aff"] for x in xs]) if all(xs) else None]
    ll.append(f"{name} & " + " & ".join(fmt(v, 2) for v in vals) + " \\\\")
open(P / "tables" / "loo.tex", "w").write("\n".join(ll) + "\n")

# ---------------------------------------------------------------- Table: held-out camera extrapolation
xm = [("IDSplat~\\cite{lindstrom2026idsplat}", "repro"), ("+ pose refinement", "camopt"), ("+ temporal virtual views", "astra"),
      ("+ lens + dropout", "gdl"), ("\\textbf{ASTRA (full)}", "stivc")]
xs_avail = [s for s in WAYMO if all(xcam(e, s) for _, e in xm)]
xl = []
xvals = []
for name, exp in xm:
    m = xcam_mean(exp, xs_avail)
    xvals.append((name, [m["psnr"], m["psnr_cov"], m["psnr_cov_aff"], m["ssim_cov"], m["lpips_cov"]], exp == "stivc"))
open(P / "tables" / "xcam.tex", "w").write("\n".join(tex_table(xvals, 5, [True, True, True, True, False], [2, 2, 2, 3, 3])) + "\n")
NUM["XcamScenes"] = str(len(xs_avail))
NUM["XcamOurs"] = fmt(xvals[-1][1][2], 2)
NUM["XcamRep"] = fmt(xvals[0][1][2], 2)
NUM["XcamGain"] = fmt(xvals[-1][1][2] - xvals[0][1][2], 2)
NUM["XcamWins"] = str(sum(xcam("stivc", s)["psnr_cov_aff"] > xcam("repro", s)["psnr_cov_aff"] for s in xs_avail))
NUM["XcamCov"] = fmt(100 * xcam_mean("repro", xs_avail)["cov"], 0)
NUM["XcamCamoptDrop"] = fmt(abs(xvals[1][1][2] - xvals[0][1][2]), 2)
NUM["XcamTivcGain"] = fmt(xvals[2][1][2] - xvals[1][1][2], 2)

# ---------------------------------------------------------------- per-scene Waymo (appendix)
ps = []
for s in WAYMO:
    a, b = waymo_run("repro", s), waymo_run(our_exp, s)
    xa, xb = xcam("repro", s), xcam(our_exp, s)
    ps.append(f"{s} & {fmt(a['psnr'],2)} & \\textbf{{{fmt(b['psnr'],2)}}} & {fmt(a['dpsnr'],2)} & \\textbf{{{fmt(b['dpsnr'],2)}}} & "
              f"{fmt(xa['psnr_cov_aff'] if xa else None,2)} & \\textbf{{{fmt(xb['psnr_cov_aff'] if xb else None,2)}}} \\\\")
open(P / "tables" / "perscene.tex", "w").write("\n".join(ps) + "\n")
NUM["WaymoWins"] = str(sum(waymo_run(our_exp, s)["psnr"] > waymo_run("repro", s)["psnr"] for s in WAYMO))

# ---------------------------------------------------------------- numbers.tex + PAPER_TABLES.md
with open(P / "numbers.tex", "w") as fh:
    for k, v in NUM.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


def md(rows, cols):
    out = ["| method | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, vals, _ in rows:
        clean = name.replace("\\textbf{", "").replace("}", "").split("~\\cite")[0].replace("$^\\dagger$", "*")
        out.append(f"| {clean} | " + " | ".join(fmt(v, 3 if c in ("SSIM", "LPIPS", "SSIM cov", "LPIPS cov") else 2) for v, c in zip(vals, cols)) + " |")
    return "\n".join(out)


with open(HERE.parent / "results" / "PAPER_TABLES.md" if False else HERE / "results" / "PAPER_TABLES.md", "w") as fh:
    fh.write("# ASTRA paper tables (generated by tools/make_tables.py)\n\n")
    fh.write("\n".join(f"- {l}" for l in LOG) + "\n\n")
    fh.write("## Waymo (8 scenes, AD-GS split, FRONT camera)\n\n" + md(rows, ["PSNR", "SSIM", "LPIPS", "DPSNR"]) + "\n\n")
    fh.write("## KITTI (nvs-75 | nvs-50 | nvs-25; LPIPS-Alex)\n\n" + md(krows, ["PSNR75", "SSIM", "LPIPS", "PSNR50", "SSIM", "LPIPS", "PSNR25", "SSIM", "LPIPS"]) + "\n\n")
    for key in ("nuscenes", "pandaset"):
        rws = [(m, v, False) for m, v in pub[key].items()] + [("IDSplat (reproduced)", [other[key][0][k] for k in ("psnr", "ssim", "lpips")], False)]
        if other[key][1]:
            rws.append(("ASTRA (ours)", [other[key][1][k] for k in ("psnr", "ssim", "lpips")], True))
        fh.write(f"## {key}\n\n" + md(rws, ["PSNR", "SSIM", "LPIPS"]) + "\n\n")
    fh.write("## Ablation (Waymo, 8 scenes; last column = held-out camera PSNR, covered pixels, affine)\n\n" +
             md(arows, ["PSNR", "SSIM", "LPIPS", "DPSNR", "XCam"]) + "\n\n")
    fh.write(f"## Held-out camera extrapolation (Waymo, {len(xs_avail)} scenes, FRONT_LEFT/FRONT_RIGHT)\n\n" +
             md(xvals, ["PSNR full", "PSNR cov", "PSNR cov+aff", "SSIM cov", "LPIPS cov"]) + "\n")
print("\n".join(LOG))
print(json.dumps(NUM, indent=0))
