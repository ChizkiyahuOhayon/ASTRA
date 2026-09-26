"""Geometric initialisation of a tracklet's rigid trajectory by ICP.

Adapted from IDSplat (CVPR Findings 2026, https://github.com/zenseact/idsplat),
which registers each instance's points between observations and keeps a fitness
score per pair instead of trusting every frame equally. IDSplat registers lidar
points with learned descriptors; this project has no lidar and no descriptors,
so nearest-neighbour ICP stands in -- adjacent frames move little, which is the
regime where it converges.

Why registration rather than centroid differences: a frame's blob is a *different*
set of Gaussians from the previous frame's, because every Gaussian has a single
birth time. Points entering and leaving move the centroid without the object
moving at all, and a trajectory fitted to those centroids inherits that jitter.
Shape alignment is invariant to which points were sampled, and returns the
rotation for free.
"""
import numpy as np


def kabsch(source, target, weights=None):
    """Least-squares rigid transform taking `source` onto `target`."""
    if weights is None:
        weights = np.ones(len(source))
    weights = weights / max(weights.sum(), 1e-12)
    source_mean = (weights[:, None] * source).sum(0)
    target_mean = (weights[:, None] * target).sum(0)
    covariance = ((weights[:, None] * (source - source_mean)).T
                  @ (target - target_mean))
    u, _, vt = np.linalg.svd(covariance)
    # Reflections are valid SVD factors but not rotations; flip the last axis.
    flip = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, flip]) @ u.T
    return rotation, target_mean - rotation @ source_mean


def icp(source, target, threshold, iterations=12):
    """Rigid alignment of two unordered point sets, plus an inlier fitness.

    Starts from the centroid offset, then alternates nearest-neighbour
    correspondence with a Kabsch solve, keeping only pairs inside `threshold`.
    """
    rotation = np.eye(3)
    translation = target.mean(0) - source.mean(0)
    fitness = 0.0
    for _ in range(iterations):
        moved = source @ rotation.T + translation
        distances = np.linalg.norm(moved[:, None, :] - target[None, :, :], axis=-1)
        nearest = distances.argmin(1)
        residual = distances[np.arange(len(source)), nearest]
        inlier = residual < threshold
        fitness = float(inlier.mean())
        if inlier.sum() < 3:
            break
        candidate = kabsch(source[inlier], target[nearest[inlier]])
        if np.allclose(candidate[0], rotation, atol=1e-9) and \
                np.allclose(candidate[1], translation, atol=1e-9):
            rotation, translation = candidate
            break
        rotation, translation = candidate
    return rotation, translation, fitness


def rotation_vector(rotation):
    """Axis-angle of a rotation matrix, stable at and near the identity."""
    trace = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(trace)
    if angle < 1e-8:
        return np.zeros(3)
    axis = np.array([rotation[2, 1] - rotation[1, 2],
                     rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]])
    return axis * (angle / (2.0 * np.sin(angle)))


def register_track(frames, threshold, min_fitness=0.3, min_points=6):
    """Chain consecutive registrations into poses relative to the first frame.

    `frames` is a list of point arrays in observation order. The shared model
    evaluates `p(t) = pivot + current(t) + R(t) . u`, so the translation this
    returns is the motion of the pivot under the accumulated transform,
    `R_k . pivot + T_k - pivot`, and the rotation is its axis-angle vector.

    A step that does not register well contributes only its centroid offset and
    no rotation, so a bad frame degrades to the behaviour this project already
    had rather than injecting a wrong pose. The fitness of each step is returned
    so the trajectory fit can weight the frames it trusts.
    """
    pivot = frames[0].mean(0)
    rotation = np.eye(3)
    translation = np.zeros(3)
    translations = [np.zeros(3)]
    vectors = [np.zeros(3)]
    fitnesses = [1.0]
    for previous, current in zip(frames[:-1], frames[1:]):
        if len(previous) < min_points or len(current) < min_points:
            step_rotation = np.eye(3)
            step_translation = current.mean(0) - previous.mean(0)
            fitness = 0.0
        else:
            step_rotation, step_translation, fitness = icp(previous, current, threshold)
            if fitness < min_fitness:
                step_rotation = np.eye(3)
                step_translation = current.mean(0) - previous.mean(0)
        rotation = step_rotation @ rotation
        translation = step_rotation @ translation + step_translation
        translations.append(rotation @ pivot + translation - pivot)
        vectors.append(rotation_vector(rotation))
        fitnesses.append(fitness)
    return np.stack(translations), np.stack(vectors), np.asarray(fitnesses), pivot
