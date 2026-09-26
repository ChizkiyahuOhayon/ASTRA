"""Can the object point cloud be split into one instance per vehicle, unsupervised?

Rigid per-instance motion needs an instance assignment, and AD-GS has none: its
dynamic branch is per-Gaussian.

Clustering the accumulated cloud in space alone does not work, and the sweep
below reproduces why: point density falls off with range, so no single DBSCAN
radius fits both the dense near vehicle and the sparse far one -- tight radii
shatter 11 of scene006's 19 tracks, loose radii merge 10 of them into one
126k-point blob.  Two cars that pass through the same stretch of road at
different times are spatially connected however far apart they are in time.

Every point carries its own frame, so the fix is to use it.  Within one frame a
vehicle is a compact 4.5 x 2 x 1.5 m blob and vehicles sit metres apart, which
makes the per-frame split trivial; all that is left is linking a blob to the
same vehicle in the next frame, and 100 ms of motion is small next to the gap
between cars, so a centroid gate suffices.

Ground-truth boxes score the result and never produce it.
"""
import glob
import os
from argparse import ArgumentParser
from collections import Counter, defaultdict

import numpy as np
from sklearn.cluster import DBSCAN


def load_object_points(ply_path):
    from plyfile import PlyData
    ply = PlyData.read(ply_path)["vertex"]
    fields = ply.data.dtype.names
    xyz = np.stack([ply["x"], ply["y"], ply["z"]], axis=1)
    time = ply["t"] if "t" in fields else np.zeros(len(xyz))
    obj = ply["obj"] if "obj" in fields else np.zeros(len(xyz))
    keep = obj > 0.5
    return xyz[keep].astype(np.float64), np.asarray(time)[keep].astype(int)


def gt_by_frame(gt_dir):
    """Index the ground-truth files by the point cloud's own frame numbering.

    storePly writes `fid - first_frame` into the `t` channel while the extractor
    names its files with a plain counter, so the two agree only as long as both
    walked the same frame window.  Rebuilding the index from the stored `fid`
    keeps that coincidence from being load-bearing.
    """
    files = sorted(glob.glob(os.path.join(gt_dir, "*.npz")))
    fids = np.array([int(np.load(f)["fid"]) for f in files])
    return {int(f - fids.min()): path for f, path in zip(fids, files)}


def link_instances(xyz, time, eps, min_samples, gate, memory):
    """One DBSCAN per frame, then greedy nearest-centroid linking across frames.

    The gate is on the residual from a constant-velocity prediction rather than
    on raw displacement: a vehicle covers one to three metres per frame in world
    coordinates, and the centroid of its visible points shifts again as the view
    angle changes, so a positional gate loose enough to follow a fast car is
    also loose enough to jump to its neighbour.  Predicting first leaves only
    the acceleration and the shape-change to absorb.

    `memory` frames of tolerance keeps a vehicle that is briefly occluded from
    being handed a fresh instance id when it reappears.
    """
    labels = np.full(len(xyz), -1, dtype=int)
    live = {}          # instance id -> (centroid, velocity, last frame seen)
    next_id = 0
    for t in np.unique(time):
        sel = np.where(time == t)[0]
        cl = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xyz[sel])
        blobs = [(sel[cl == c], xyz[sel[cl == c]].mean(axis=0))
                 for c in range(cl.max() + 1)]
        predicted = {i: cen + vel * (t - last)
                     for i, (cen, vel, last) in live.items() if t - last <= memory}

        pairs = sorted(
            ((np.linalg.norm(cen - predicted[i]), b, i)
             for b, (_, cen) in enumerate(blobs) for i in predicted),
            key=lambda r: r[0])
        taken_blob, taken_id = set(), set()
        for d, b, i in pairs:
            if d > gate or b in taken_blob or i in taken_id:
                continue
            taken_blob.add(b); taken_id.add(i)
            idx, cen = blobs[b]
            old_cen, _, last = live[i]
            labels[idx] = i
            live[i] = (cen, (cen - old_cen) / (t - last), t)
        for b, (idx, cen) in enumerate(blobs):
            if b not in taken_blob:
                labels[idx] = next_id
                live[next_id] = (cen, np.zeros(3), t)
                next_id += 1
    return labels


def score(labels, truth, name):
    n_clusters = labels.max() + 1
    print("\n--- {}: {} instances, {:.1%} unassigned".format(
        name, n_clusters, (labels < 0).mean()))
    rows = []
    for c in range(n_clusters):
        sel = labels == c
        lab = [l for l in truth[sel] if l != ""]
        if not lab:
            rows.append((c, int(sel.sum()), "-", 0.0, 0, 0, 0))
            continue
        top, count = Counter(lab).most_common(1)[0]
        rows.append((c, int(sel.sum()), top, count / len(lab), len(set(lab)),
                     len(lab), len(lab) - count))
    rows.sort(key=lambda r: -r[1])
    print("  %-4s %8s %-24s %8s %6s" % ("id", "points", "dominant track", "purity", "tracks"))
    for c, n, top, purity, ntracks, _, _ in rows[:12]:
        print("  %-4d %8d %-24s %8.2f %6d" % (c, n, str(top)[:24], purity, ntracks))
    by_track = defaultdict(Counter)
    for c, l in zip(labels, truth):
        if l != "" and c >= 0:
            by_track[l][c] += 1
    split = sum(1 for _, cs in by_track.items() if len(cs) > 1)
    mixed = sum(1 for r in rows if r[4] > 1)

    # Splitting a vehicle is harmless -- each piece is still a rigid part of a
    # rigid body.  Merging two vehicles is not: one rigid transform cannot serve
    # both.  What that costs is the minority-track points inside mixed
    # instances, and counting instances instead of points overstates it, since
    # a 1 m ground-truth margin mislabels a handful of points at box borders.
    contaminated = sum(r[6] for r in rows)
    labelled = int((truth != "").sum())
    print("  tracks: {}, split across instances: {}; instances mixing tracks: {}".format(
        len(by_track), split, mixed))
    print("  points a shared rigid transform would be wrong for: {} ({:.2%})".format(
        contaminated, contaminated / max(labelled, 1)))


def box_frames(corners):
    centre = corners.mean(axis=1)
    axes_raw = [corners[:, 0] - corners[:, 3], corners[:, 0] - corners[:, 1], corners[:, 0] - corners[:, 4]]
    half = np.stack([np.linalg.norm(a, axis=-1) for a in axes_raw], axis=1) / 2
    axes = np.stack([a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-6) for a in axes_raw], axis=1)
    return centre, axes, half


def main():
    ap = ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--min_samples", type=int, default=10)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--gate", type=float, default=3.0,
                    help="metres a vehicle centroid may move between frames")
    ap.add_argument("--memory", type=int, default=3,
                    help="frames an occluded instance stays linkable")
    args = ap.parse_args()

    xyz, time = load_object_points(args.ply)
    if not len(xyz):
        raise SystemExit("no object points in {}: is the `obj` channel present?".format(args.ply))
    print("object points: {}  frames {}..{}".format(len(xyz), time.min(), time.max()))
    frames = gt_by_frame(args.gt_dir)
    print("ground-truth frames: {}".format(len(frames)))

    # ground-truth label per point: nearest actor box at that point's own timestamp
    truth = np.full(len(xyz), "", dtype=object)
    for t in np.unique(time):
        path = frames.get(int(t))
        if path is None:
            continue
        gt = np.load(path)
        corners = gt["box_corners_world"]
        if not len(corners):
            continue
        sel = np.where(time == t)[0]
        centre, axes, half = box_frames(corners)
        delta = xyz[sel][:, None, :] - centre[None, :, :]
        local = np.abs(np.einsum("nbi,bai->nba", delta, axes))
        dist = np.linalg.norm(np.clip(local - half[None, :, :], 0.0, None), axis=-1)
        best = dist.argmin(axis=1)
        hit = dist[np.arange(len(sel)), best] <= args.margin
        ids = [str(b) for b in gt["box_id"]]
        truth[sel[hit]] = [ids[b] for b in best[hit]]

    tagged = truth != ""
    print("points inside a ground-truth actor: {} ({:.1%})".format(tagged.sum(), tagged.mean()))

    for eps in args.eps:
        score(DBSCAN(eps=eps, min_samples=args.min_samples).fit_predict(xyz),
              truth, "space-only eps={}".format(eps))

    for eps in args.eps:
        score(link_instances(xyz, time, eps, args.min_samples, args.gate, args.memory),
              truth, "per-frame + link eps={} gate={}".format(eps, args.gate))


if __name__ == "__main__":
    main()
