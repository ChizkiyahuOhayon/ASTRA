"""Build every paper table from the raw result files.

usage: python3 tools/make_tables.py [results_dir] [paper_dir]
writes <paper_dir>/tables/*.tex, <paper_dir>/numbers.tex and ../results/PAPER_TABLES.md (next to the paper dir).
Our method row is the final configuration "sv" (pose refinement + lens-consistent rendering + temporal and
off-trajectory virtual views), with one set of hyperparameters on every dataset.
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
    """the final configuration (sv)."""
    for exp in ("sv",):
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


def pair(a, b, d, higher=True):
    """two cells, the better one in bold."""
    if a is None or b is None:
        return f"{fmt(a, d)} & {fmt(b, d)}"
    ra, rb = round(a, d), round(b, d)
    better_b = rb > ra if higher else rb < ra
    better_a = ra > rb if higher else ra < rb
    A, B = fmt(a, d), fmt(b, d)
    return f"{'\\textbf{' + A + '}' if better_a else A} & {'\\textbf{' + B + '}' if better_b else B}"


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
    rows.append(("\\textbf{\\ours{} (ours)}", [our["psnr"], our["ssim"], our["lpips"], our["dpsnr"]], True))
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
    krows.append(("\\textbf{\\ours{} (ours)}", sum([[other[s][1][k] for k in ("psnr", "ssim", "lpips")] for s in ("kitti75", "kitti50", "kitti25")], []), True))
body = tex_table(krows, 9, [True, True, False] * 3, [2, 3, 3] * 3)
open(P / "tables" / "kitti.tex", "w").write("\n".join(body) + "\n")

for key, fn in (("nuscenes", "nuscenes"), ("pandaset", "pandaset")):
    rws = [(NAMES[m], v, False) for m, v in pub[key].items()]
    rep_m, our_m = other[key]
    rws.append(("IDSplat (reproduced)", [rep_m[k] for k in ("psnr", "ssim", "lpips")], False))
    if our_m:
        rws.append(("\\textbf{\\ours{} (ours)}", [our_m[k] for k in ("psnr", "ssim", "lpips")], True))
    open(P / "tables" / f"{fn}.tex", "w").write("\n".join(tex_table(rws, 3, [True, True, False], [2, 3, 3])) + "\n")

# ---------------------------------------------------------------- Table: ablation (Waymo, all 8 scenes, cumulative)
abl = [("IDSplat~\\cite{lindstrom2026idsplat}", "repro", ""), ("+ pose, raw test poses", "camopt", ".nointerp"),
       ("+ traj.-consistent pose", "camopt", ""), ("+ temporal views", "astra", ""),
       ("+ lens rendering", "lens", ""), ("+ off-trajectory views", "sv", "")]
arows = []
for name, exp, suf in abl:
    m = waymo_mean(exp, suffix=suf)
    x = xcam_mean(exp) if not suf else None
    arows.append((name, [None if m is None else m["psnr"], None if m is None else m["ssim"], None if m is None else m["lpips"],
                         None if m is None else m["dpsnr"], None if x is None else x["psnr_cov_aff"]], exp == "sv"))
    if m is None:
        LOG.append(f"ablation row {exp}{suf}: incomplete")
lines = []
# bold the best value of each column (ties at the printed precision are all bold)
abl_best = []
for j, (d, hi) in enumerate(zip((2, 3, 3, 2, 2), (True, True, False, True, True))):
    col = [round(r[1][j], d) for r in arows if r[1][j] is not None]
    abl_best.append((max(col) if hi else min(col)) if col else None)
for i, (name, vals, is_ours) in enumerate(arows):
    cells = [fmt(v, d) for v, d in zip(vals, (2, 3, 3, 2, 2))]
    cells = [f"\\textbf{{{c}}}" if v is not None and round(v, d) == b else c
             for c, v, d, b in zip(cells, vals, (2, 3, 3, 2, 2), abl_best)]
    lines.append(f"{'\\rowcolor{oursbg}' if is_ours else ''}{name} & " + " & ".join(cells) + " \\\\")
open(P / "tables" / "ablation.tex", "w").write("\n".join(lines) + "\n")
for name, exp, suf in abl:
    m = waymo_mean(exp, suffix=suf)
    if m:
        NUM[f"Abl{exp.capitalize()}{'Raw' if suf else ''}PSNR"] = fmt(m["psnr"], 2)
        NUM[f"Abl{exp.capitalize()}{'Raw' if suf else ''}DPSNR"] = fmt(m["dpsnr"], 2)
    x = None if suf else xcam_mean(exp)
    if x:
        NUM[f"Abl{exp.capitalize()}Cam"] = fmt(x["psnr_cov_aff"], 2)

# remove one component from the final model, Waymo 802 / 806
loo = [("\\ours{} (full)", "sv"), ("w/o pose refinement", "svnopose"), ("w/o lens-consistent rendering", "svnolens"),
       ("w/o off-trajectory views", "lens"), ("w/o all virtual views", "svnovv")]
ll = []
loo_vals = []
for name, exp in loo:
    rs = [waymo_run(exp, s) for s in (802, 806)]
    xs = [xcam(exp, s) for s in (802, 806)]
    loo_vals.append((name, exp, rs, xs, [r["psnr"] if r else None for r in rs] + [nmean([r["psnr"] for r in rs]) if all(rs) else None,
                                                    nmean([x["psnr_cov_aff"] for x in xs]) if all(xs) else None]))
# bold the best value of each column
loo_best = [max((round(v[4][j], 2) for v in loo_vals if v[4][j] is not None), default=None) for j in range(4)]
for name, exp, rs, xs, vals in loo_vals:
    cells = [f"\\textbf{{{fmt(v, 2)}}}" if v is not None and round(v, 2) == b else fmt(v, 2) for v, b in zip(vals, loo_best)]
    ll.append(f"{'\\rowcolor{oursbg}' if exp == 'sv' else ''}{name} & " + " & ".join(cells) + " \\\\")
    if all(rs):
        NUM[f"Loo{exp.capitalize()}PSNR"] = fmt(vals[2], 2)
    if all(xs):
        NUM[f"Loo{exp.capitalize()}Cam"] = fmt(vals[3], 2)
open(P / "tables" / "loo.tex", "w").write("\n".join(ll) + "\n")

# ---------------------------------------------------------------- Table: held-out camera extrapolation
xm = [("IDSplat~\\cite{lindstrom2026idsplat}", "repro"), ("+ pose refinement", "camopt"), ("+ temporal views", "astra"),
      ("+ lens rendering", "lens"), ("\\textbf{\\ours{} (full)}", "sv")]
if not all(xcam("lens", s) for s in WAYMO):
    xm = [r for r in xm if r[1] != "lens"]
    LOG.append("xcam table: lens row incomplete, left out")
xs_avail = [s for s in WAYMO if all(xcam(e, s) for _, e in xm)]
xl = []
xvals = []
for name, exp in xm:
    m = xcam_mean(exp, xs_avail)
    xvals.append((name, [m["psnr"], m["psnr_cov"], m["psnr_cov_aff"], m["ssim_cov"], m["lpips_cov"]], exp == "sv"))
open(P / "tables" / "xcam.tex", "w").write("\n".join(tex_table(xvals, 5, [True, True, True, True, False], [2, 2, 2, 3, 3])) + "\n")
NUM["XcamScenes"] = str(len(xs_avail))
NUM["XcamOurs"] = fmt(xvals[-1][1][2], 2)
NUM["XcamRep"] = fmt(xvals[0][1][2], 2)
NUM["XcamGain"] = fmt(xvals[-1][1][2] - xvals[0][1][2], 2)
NUM["XcamWins"] = str(sum(xcam("sv", s)["psnr_cov_aff"] > xcam("repro", s)["psnr_cov_aff"] for s in xs_avail))
NUM["XcamCov"] = fmt(100 * xcam_mean("repro", xs_avail)["cov"], 0)
NUM["XcamCamoptDrop"] = fmt(abs(xvals[1][1][2] - xvals[0][1][2]), 2)
NUM["XcamTivcGain"] = fmt(xvals[2][1][2] - xvals[1][1][2], 2)

# ---------------------------------------------------------------- per-scene Waymo (appendix)
ps = []
for s in WAYMO:
    a, b = waymo_run("repro", s), waymo_run(our_exp, s)
    xa, xb = xcam("repro", s), xcam(our_exp, s)
    ps.append(f"{s} & {pair(a['psnr'], b['psnr'], 2)} & {pair(a['dpsnr'], b['dpsnr'], 2)} & "
              f"{pair(xa['psnr_cov_aff'] if xa else None, xb['psnr_cov_aff'] if xb else None, 2)} \\\\")
open(P / "tables" / "perscene.tex", "w").write("\n".join(ps) + "\n")
NUM["WaymoWins"] = str(sum(waymo_run(our_exp, s)["psnr"] > waymo_run("repro", s)["psnr"] for s in WAYMO))

# ---------------------------------------------------------------- per-sequence tables for KITTI / nuScenes / PandaSet (appendix)
def per_seq(pat, lp):
    a, b = misc_runs("repro", pat), misc_runs(our_exp, pat)
    return [(k, a[k], b[k]) for k in sorted(a) if k in b]


kl = []
for seq in ("0001", "0002", "0006"):
    cells = []
    for split in ("75", "50", "25"):
        _, a, b = per_seq(f"kitti{seq}-{split}", "lpips_alex")[0]
        cells.append(pair(a["psnr"], b["psnr"], 2))
    kl.append(f"{seq} & " + " & ".join(cells) + " \\\\")
open(P / "tables" / "kitti_perscene.tex", "w").write("\n".join(kl) + "\n")
wins, total = 0, 0
for pat, lp, fn, strip in (("nusscene-*", "lpips_vgg", "nus_perscene", "nusscene-"), ("panda*", "lpips_alex", "panda_perscene", "panda")):
    pl = []
    for k, a, b in per_seq(pat, lp):
        total += 1; wins += b["psnr"] > a["psnr"]
        pl.append(f"{k.replace(strip, '')} & {pair(a['psnr'], b['psnr'], 2)} & {pair(a['ssim'], b['ssim'], 3)} & {pair(a[lp], b[lp], 3, False)} \\\\")
    open(P / "tables" / f"{fn}.tex", "w").write("\n".join(pl) + "\n")
for pat in ("kitti*-*",):
    for k, a, b in per_seq(pat, "lpips_alex"):
        total += 1; wins += b["psnr"] > a["psnr"]
NUM["OtherWins"] = str(wins)
NUM["OtherTotal"] = str(total)

# ---------------------------------------------------------------- extra numbers quoted in the text
for key, label in (("kitti75", "KITTISeventyFive"), ("kitti50", "KITTIFifty"), ("kitti25", "KITTITwentyFive"), ("nuscenes", "NUS")):
    if other[key][1]:
        NUM[f"{label}GainADGS"] = fmt(other[key][1]["psnr"] - pub[key]["AD-GS"][0], 2)
        NUM[f"{label}RepPSNR"] = fmt(other[key][0]["psnr"], 2)
NUM["PANDAGainPub"] = fmt(other["pandaset"][1]["psnr"] - pub["pandaset"]["IDSplat"][0], 2)
NUM["PANDARepPSNR"] = fmt(other["pandaset"][0]["psnr"], 2)
NUM["WaymoGainLPIPS"] = fmt(rep["lpips"] - our["lpips"], 3)
NUM["WaymoBestPubPSNR"] = fmt(max(v[0] for v in pub["waymo"].values()), 2)
NUM["WaymoGainBestPub"] = fmt(our["psnr"] - max(v[0] for v in pub["waymo"].values()), 2)
fr = [waymo_run("repro", s)["fps"] for s in WAYMO]
fo = [waymo_run(our_exp, s)["fps"] for s in WAYMO]
NUM["FPSRep"] = fmt(nmean(fr), 1)
NUM["FPSOurs"] = fmt(nmean(fo), 1)
xa = xcam_mean("astra", xs_avail)
NUM["XcamGainAstra"] = fmt(xvals[-1][1][2] - xa["psnr_cov_aff"], 2)
NUM["XcamFullRep"] = fmt(xvals[0][1][0], 2)
NUM["XcamFullOurs"] = fmt(xvals[-1][1][0], 2)
NUM["FrontRepPSNR"] = fmt(rep["psnr"], 1)
NUM["XcamDrop"] = fmt(rep["psnr"] - xvals[0][1][2], 0)

# ---------------------------------------------------------------- numbers.tex + PAPER_TABLES.md
# gap between train and held-out PSNR (fig12), from training curves exported by tools/export_curves.py
CUR = HERE / "data" / "assets" / "curves.json"
if CUR.exists():
    C = json.load(open(CUR))
    both = [s for s in WAYMO if f"repro-{s}" in C and f"{our_exp}-{s}" in C]
    if both:
        gap = lambda k: C[k]["train"] and (float(np.mean([v for _, v in C[k]["train"][-40:]])) - C[k]["test"][-1][1])
        NUM["GapRep"] = fmt(nmean([gap(f"repro-{s}") for s in both]), 1)
        NUM["GapOurs"] = fmt(nmean([gap(f"{our_exp}-{s}") for s in both]), 1)
        NUM["GapSeq"] = str(both[0])
        NUM["GapN"] = str(len(both))
for k in ("GapRep", "GapOurs", "GapSeq", "GapN", "AblLensPSNR", "AblLensDPSNR", "AblLensCam", "LooSvnoposePSNR", "LooSvnoposeCam",
          "LooSvnolensPSNR", "LooSvnolensCam", "LooSvnovvPSNR", "LooSvnovvCam"):
    if k not in NUM:
        NUM[k] = "\\textcolor{red}{??}"
        LOG.append(f"number {k} missing")
with open(P / "numbers.tex", "w") as fh:
    for k, v in NUM.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


def md(rows, cols):
    out = ["| method | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, vals, _ in rows:
        clean = name.replace("\\ours{}", "ASTRA").replace("\\textbf{", "").replace("}", "").split("~\\cite")[0].replace("$^\\dagger$", "*")
        out.append(f"| {clean} | " + " | ".join(fmt(v, 3 if c in ("SSIM", "LPIPS", "SSIM cov", "LPIPS cov") else 2) for v, c in zip(vals, cols)) + " |")
    return "\n".join(out)


(HERE / "results").mkdir(exist_ok=True)
with open(HERE / "results" / "PAPER_TABLES.md", "w") as fh:
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
             md([("IDSplat (reproduced)",) + tuple(arows[0][1:])] + arows[1:], ["PSNR", "SSIM", "LPIPS", "DPSNR", "XCam"]) + "\n\n")
    fh.write("## Removing one component (Waymo 802 / 806; last column = held-out camera PSNR, covered pixels, affine)\n\n" +
             md([(n, v, e == "sv") for n, e, _, _, v in loo_vals], ["PSNR 802", "PSNR 806", "PSNR mean", "XCam"]) + "\n\n")
    fh.write(f"## Held-out camera extrapolation (Waymo, {len(xs_avail)} scenes, FRONT_LEFT/FRONT_RIGHT)\n\n" +
             md([("IDSplat (reproduced)",) + tuple(xvals[0][1:])] + xvals[1:], ["PSNR full", "PSNR cov", "PSNR cov+aff", "SSIM cov", "LPIPS cov"]) + "\n")
print("\n".join(LOG))
print(json.dumps(NUM, indent=0))
