"""Mutual direct blob edges from two passes of the v5 geometric linker.

Clustering is shared by both passes. Rejected edges split tracks; they never
remove blobs or turn their points into DBSCAN noise. No annotations are read.
"""
import numpy as np
from sklearn.cluster import DBSCAN


def cluster_blobs(xyz, frames, eps, min_samples):
    blobs = []  # (frame, original point indices, centroid), chronological IDs
    for t in np.unique(frames):
        sel = np.where(frames == t)[0]
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xyz[sel])
        for c in range(labels.max() + 1):
            idx = sel[labels == c]
            blobs.append((t, idx, xyz[idx].mean(axis=0)))
    return blobs


def link_blob_edges(blobs, gate, memory, reverse=False):
    """v5 greedy constant-velocity pass, recording actual last-blob -> new edges.

    Reverse time uses s=-t, so the same positive-gap prediction and memory test
    apply. Within-frame blob order and stable distance tie breaking match v5.
    Edges are stored as sorted blob-ID pairs for direction-independent matching.
    """
    by_frame = {}
    for b, (t, _, _) in enumerate(blobs):
        by_frame.setdefault(t, []).append(b)
    live = {}  # birth blob -> (centroid, velocity in s, last s, last blob)
    edges = set()
    for frame in sorted(by_frame, reverse=reverse):
        t = -frame if reverse else frame
        predicted = {i: cen + vel * (t - last)
                     for i, (cen, vel, last, _) in live.items() if t - last <= memory}
        pairs = sorted(
            ((np.linalg.norm(blobs[b][2] - predicted[i]), b, i)
             for b in by_frame[frame] for i in predicted), key=lambda r: r[0])
        taken_blob, taken_id = set(), set()
        for d, b, i in pairs:
            if d > gate or b in taken_blob or i in taken_id:
                continue
            taken_blob.add(b)
            taken_id.add(i)
            cen = blobs[b][2]
            old_cen, _, last, previous = live[i]
            edges.add(tuple(sorted((previous, b))))
            live[i] = (cen, (cen - old_cen) / (t - last), t, b)
        for b in by_frame[frame]:
            if b not in taken_blob:
                live[b] = (blobs[b][2], np.zeros(3), t, b)
    return edges


def mutual_components(blobs, gate, memory):
    forward = link_blob_edges(blobs, gate, memory)
    backward = link_blob_edges(blobs, gate, memory, reverse=True)
    parent = list(range(len(blobs)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in sorted(forward & backward):
        a, b = root(a), root(b)
        parent[max(a, b)] = min(a, b)
    # Earliest blob orders the components, matching v5 IDs when all edges agree.
    roots = [root(i) for i in range(len(blobs))]
    ids = {r: k for k, r in enumerate(sorted(set(roots)))}
    return np.array([ids[r] for r in roots], dtype=int), forward, backward


def link_instances_bidirectional(xyz, frames, eps, min_samples, gate, memory):
    blobs = cluster_blobs(xyz, frames, eps, min_samples)
    components, _, _ = mutual_components(blobs, gate, memory)
    labels = np.full(len(xyz), -1, dtype=int)
    for component, (_, idx, _) in zip(components, blobs):
        labels[idx] = component
    return labels
