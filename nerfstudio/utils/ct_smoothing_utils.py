# Copyright 2026 the authors of IDSplat and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial
from typing import List, Optional, Tuple

import gtsam
import matplotlib.pyplot as plt
import numpy as np
from gtsam import symbol
from scipy.spatial.transform import Rotation as scipy_rot

SMALL_KAPPA_THRESHOLD = 1e-6


def error_ct_motion(
    dt: float, this: gtsam.CustomFactor, values: gtsam.Values, jacobians: Optional[gtsam.JacobianFactor]
) -> np.ndarray:
    LW1 = values.atPose3(this.keys()[0])
    LW2 = values.atPose3(this.keys()[1])
    v = values.atDouble(this.keys()[2])
    kappa = values.atDouble(this.keys()[3])
    LB = values.atPose3(this.keys()[4])
    LB_rot_pose = gtsam.Pose3(LB.rotation(), np.zeros(3))

    T1 = LW1.compose(LB_rot_pose.inverse())
    T2 = LW2.compose(LB_rot_pose.inverse())

    theta = kappa * v * dt
    ddtheta_dv = kappa * dt
    ddtheta_dk = v * dt

    if abs(kappa) < SMALL_KAPPA_THRESHOLD:
        dx = v * dt
        dy = 0.0
        dR = gtsam.Rot3()

        # for jacobians
        ddx_dv = dt
        ddy_dv = 0.0

        ddx_dk = 0.0
        ddy_dk = 0.5 * (v**2) * (dt**2)
    else:
        dx = np.sin(theta) / (kappa)
        dy = (1.0 - np.cos(theta)) / (kappa)
        dR = gtsam.Rot3.Rz(theta)

        # for jacobians
        ddx_dv = dt * np.cos(theta)
        ddy_dv = dt * np.sin(theta)
        ddx_dk = (theta * np.cos(theta) - np.sin(theta)) / (kappa**2)
        ddy_dk = (theta * np.sin(theta) + np.cos(theta) - 1.0) / (kappa**2)

    pred_rel = gtsam.Pose3(dR, np.array([dx, dy, 0.0]))
    Tij = T1.inverse().compose(T2)
    r_pose = pred_rel.inverse().compose(Tij)
    r = gtsam.Pose3.Logmap(r_pose)

    if jacobians is not None:
        J_log = gtsam.Pose3.LogmapDerivative(r_pose)
        dr_dTij = J_log
        dTij_dT1 = -gtsam.Pose3.AdjointMap(Tij.inverse())
        dTij_dT2 = np.eye(6)
        dT1_dLW1 = gtsam.Pose3.AdjointMap(LB_rot_pose)
        dT2_dLW2 = gtsam.Pose3.AdjointMap(LB_rot_pose)
        dr_dLW1 = dr_dTij @ dTij_dT1 @ dT1_dLW1
        dr_dLW2 = dr_dTij @ dTij_dT2 @ dT2_dLW2
        jacobians[0] = dr_dLW1
        jacobians[1] = dr_dLW2

        dr_dpred = -J_log @ gtsam.Pose3.AdjointMap(r_pose.inverse())
        dpred_dv = np.array([0.0, 0.0, ddtheta_dv, ddx_dv, ddy_dv, 0.0])
        dpred_dk = np.array([0.0, 0.0, ddtheta_dk, ddx_dk, ddy_dk, 0.0])
        dr_dv = dr_dpred @ dpred_dv
        dr_dk = dr_dpred @ dpred_dk
        jacobians[2] = dr_dv.reshape(6, 1)
        jacobians[3] = dr_dk.reshape(6, 1)

        dTij_dLB = gtsam.Pose3.AdjointMap(LB_rot_pose) @ (
            gtsam.Pose3.AdjointMap((LW1.inverse().compose(LW2)).inverse()) - np.eye(6)
        )
        dr_dLB = dr_dTij @ dTij_dLB
        dr_dLB[:, 3:6] = 0.0
        jacobians[4] = dr_dLB

        for j in jacobians:
            if np.isnan(j).any() or np.isinf(j).any():
                print("unstable jacobian")

    if np.isnan(r).any() or np.isinf(r).any():
        print("unstable residual")

    return r


def add_rw1_on_controls(graph, v_keys, kappa_keys, dt, qv=0.5, qk=0.00001):  # 0.05):
    """
    Adds BetweenFactorDouble(V(k), V(k+1), 0) and likewise for K(k).
    qv, qk are continuous-time noise intensities; discrete sigma = sqrt(q * dt).
    """
    sigma_v = (qv * dt) ** 0.5
    sigma_k = (qk * dt) ** 0.5
    mV = gtsam.noiseModel.Isotropic.Sigma(1, sigma_v)
    mK = gtsam.noiseModel.Isotropic.Sigma(1, sigma_k)
    for k in range(len(v_keys) - 1):
        graph.add(gtsam.BetweenFactorDouble(v_keys[k], v_keys[k + 1], 0.0, mV))
        graph.add(gtsam.BetweenFactorDouble(kappa_keys[k], kappa_keys[k + 1], 0.0, mK))


#########################################
# Not fixed yet
#########################################


def error_local_body_priors(
    this: gtsam.CustomFactor, values: gtsam.Values, jacobians: Optional[gtsam.JacobianFactor]
) -> np.ndarray:
    LW = values.atPose3(this.keys()[0])  # local-to-world transform
    LB = values.atPose3(this.keys()[1])  # local-to-body transform
    LB_rot_pose = gtsam.Pose3(LB.rotation(), np.zeros(3))
    T = LW.compose(LB_rot_pose.inverse())  # body-to-world transform
    res = gtsam.Pose3.Logmap(T)

    if jacobians is not None:
        dr_dT = gtsam.Pose3.LogmapDerivative(T)
        dT_dLW = gtsam.Pose3.AdjointMap(LB_rot_pose)
        dT_dLB = -gtsam.Pose3.AdjointMap(LB_rot_pose)

        jacobians[0] = dr_dT @ dT_dLW
        jacobians[1] = dr_dT @ dT_dLB
        jacobians[1][:, 3:6] = 0.0  # set translation derivatives to 0

    return res


def add_prior_on_k(graph, kappa_keys, sigma_k=0.01):
    """
    Adds PriorFactorDouble(K(k), 0) for each k.
    sigma_k is standard deviation of the prior.
    """
    mK = gtsam.noiseModel.Isotropic.Sigma(1, sigma_k)
    for k in range(len(kappa_keys)):
        graph.add(gtsam.PriorFactorDouble(kappa_keys[k], 0.0, mK))


# def add_random_walk_z_roll_pitch(graph, pose_keys, dt, qz=1e-3, q_roll=1e-4, q_pitch=1e-4, big=1e6):
#     """
#     Adds BetweenFactorPose3(X(k), X(k+1), I) with large sigmas on x,y,yaw and
#     RW1 sigmas on z, roll, pitch: sigma = sqrt(q * dt).
#     """
#     sigma_z = (qz * dt) ** 0.5
#     sigma_roll = (q_roll * dt) ** 0.5
#     sigma_pitch = (q_pitch * dt) ** 0.5
#     sigmas = np.array(
#         [
#             sigma_roll,
#             sigma_pitch,
#             big,
#             big,
#             big,
#             sigma_z,
#         ],
#         dtype=float,
#     )  # roll, pitch, yaw
#     model = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
#     Ipose = gtsam.Pose3()
#     for k in range(len(pose_keys) - 1):
#         graph.add(gtsam.BetweenFactorPose3(pose_keys[k], pose_keys[k + 1], Ipose, model))


#########################################


def build_graph_gtsam(
    Z_abs: np.ndarray,
    dt: np.ndarray,
    w_abs: np.ndarray,
    Ts_init: Optional[np.ndarray] = None,
    vs_init: Optional[np.ndarray] = None,
    kappas_init: Optional[np.ndarray] = None,
) -> Tuple[gtsam.NonlinearFactorGraph, gtsam.Values, List[Tuple[int, gtsam.PriorFactorPose3]]]:
    """
    Builds and returns a GTSAM graph + initial Values.

    Args:
        Z_abs: (N, 4, 4) array of absolute poses.
        dt: (N-1,) array of time differences.
        w_abs: (N,) array of weights for absolute poses.
        w_motion: scalar weight for motion factors.
        w_v_smooth: scalar weight for velocity smoothness factors.
        w_k_smooth: scalar weight for curvature smoothness factors.
        w_rp: scalar weight for roll/pitch smoothness factors.
        w_z: scalar weight for z smoothness factors.
        Ts_init: (N, 4, 4) array of initial poses (optional).
        vs_init: (N,) array of initial velocities (optional).
        kappas_init: (N,) array of initial curvatures (optional).

    Returns:
        graph: GTSAM graph.
        initial: GTSAM initial values.
    """
    N = Z_abs.shape[0]
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    # --- Add symbols to initial values ---
    for i in range(N):
        T_init = Ts_init[i] if Ts_init is not None else Z_abs[i]
        initial.insert(symbol("x", i), gtsam.Pose3(T_init))

    for i in range(N - 1):
        v_init = (
            float(vs_init[i])
            if vs_init is not None
            else np.linalg.norm(
                initial.atPose3(symbol("x", i + 1)).translation() - initial.atPose3(symbol("x", i)).translation()
            )
            / dt[i]
        )
        k_init = float(kappas_init[i]) if kappas_init is not None else 0.0
        initial.insert(symbol("v", i), v_init)
        initial.insert(symbol("k", i), k_init)
    initial.insert(symbol("v", N - 1), initial.atDouble(symbol("v", N - 2)))
    initial.insert(symbol("k", N - 1), initial.atDouble(symbol("k", N - 2)))

    directions = []
    for i in range(N - 1):
        T1 = initial.atPose3(symbol("x", i))
        T2 = initial.atPose3(symbol("x", i + 1))
        T_rel = T1.between(T2)
        t = T_rel.translation()
        dist = np.linalg.norm(t)
        if dist < 0.5:
            continue
        directions.append(t / dist)
    if len(directions) > 0:
        mean_direction = np.mean(directions, axis=0)
        mean_direction /= np.linalg.norm(mean_direction)
        x_axis = np.array([1.0, 0.0, 0.0])
        dot = np.dot(x_axis, mean_direction)
        if np.isclose(abs(dot), 1.0, atol=1e-3):
            rotation = np.eye(3) if dot > 0 else np.diag([-1, -1, 1])
        else:
            rotation = scipy_rot.align_vectors([x_axis], [mean_direction])[0].as_matrix()
        initial.insert(symbol("z", 0), gtsam.Pose3(gtsam.Rot3(rotation), np.zeros(3)))
    else:
        initial.insert(symbol("z", 0), gtsam.Pose3())  # local-to-body transform (shared across all poses)

    # --- Absolute pose factors (unary measurement likelihoods) ---
    abs_pose_factors = []
    for i in range(N):
        if w_abs[i] < 1e-8:
            continue
        # scale = 1.0 / np.sqrt(w_abs[i])
        scale = 1.0
        rot_sigma = np.array([0.1, 0.1, 0.1]) * scale
        trans_sigma = np.array([0.2, 0.2, 0.2]) * scale
        noise = gtsam.noiseModel.Diagonal.Sigmas(np.hstack([rot_sigma, trans_sigma]))
        c = 1.0  # 1.345  # threshold (in std-dev units, after whitening by 'base')
        huber = gtsam.noiseModel.mEstimator.Huber.Create(c)
        robust = gtsam.noiseModel.Robust.Create(huber, noise)
        factor = gtsam.PriorFactorPose3(symbol("x", i), gtsam.Pose3(Z_abs[i]), robust)
        graph.add(factor)
        abs_pose_factors.append((i, factor))

    # --- Motion factors (Pose_i, Pose_{i+1}, v_i, kappa_i) ---
    for i in range(N - 1):
        # noise_model = gtsam.noiseModel.Isotropic.Sigma(6, 1.0)
        noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))
        gf = gtsam.CustomFactor(
            noise_model,
            [symbol("x", i), symbol("x", i + 1), symbol("v", i), symbol("k", i), symbol("z", 0)],
            partial(error_ct_motion, dt[i]),
        )
        graph.add(gf)

    # --- Smoothness on v and kappa ---
    for i in range(N - 1):
        add_rw1_on_controls(graph, [symbol("v", i), symbol("v", i + 1)], [symbol("k", i), symbol("k", i + 1)], dt[i])

    # --- Local body priors ---
    for i in range(N):
        big = 1e6
        sigma_roll = 0.4
        sigma_pitch = 0.4
        sigmas = np.array([sigma_roll, sigma_pitch, big, big, big, big], dtype=float)
        noise_model = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
        gf = gtsam.CustomFactor(noise_model, [symbol("x", i), symbol("z", 0)], partial(error_local_body_priors))
        graph.add(gf)

    # --- Prior on kappa ---
    for i in range(N):
        add_prior_on_k(graph, [symbol("k", i)], sigma_k=0.01)

    return graph, initial, abs_pose_factors


def optimize_graph(graph: gtsam.NonlinearFactorGraph, initial: gtsam.Values, max_iterations: int = 100):
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SILENT")
    params.setMaxIterations(max_iterations)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()
    return result


def find_outliers(abs_factors, result, threshold=1.345):
    outliers = []
    for i, factor in abs_factors:
        # Get whitened error (already normalized by covariance)
        err = factor.whitenedError(result)
        norm = np.linalg.norm(err)
        if norm > threshold:
            outliers.append(i)
    return outliers


def get_smoothed_poses(
    Z_abs: np.ndarray,
    dt: np.ndarray,
    w_abs: np.ndarray,
    Ts_init: Optional[np.ndarray] = None,
    vs_init: Optional[np.ndarray] = None,
    kappas_init: Optional[np.ndarray] = None,
    iterative: bool = True,
) -> np.ndarray:
    """
    Get smoothed poses from absolute poses.

    Args:
        Z_abs: (N, 4, 4) array of absolute poses.
        dt: (N-1,) array of time differences.
        w_abs: (N,) array of weights for absolute poses.
        w_motion: scalar weight for motion factors.
        w_v_smooth: scalar weight for velocity smoothness factors.
        w_k_smooth: scalar weight for curvature smoothness factors.
        w_rp: scalar weight for roll/pitch smoothness factors.
        w_z: scalar weight for z smoothness factors.
        Ts_init: (N, 4, 4) array of initial poses (optional).
        vs_init: (N,) array of initial velocities (optional).
        kappas_init: (N,) array of initial curvatures (optional).

    Returns:
        (N, 4, 4) array of smoothed poses.
    """
    assert Z_abs.ndim == 3
    assert Z_abs.shape[1:] == (4, 4)
    N = Z_abs.shape[0]
    assert dt.shape == (N - 1,)
    assert w_abs.shape == (N,)
    assert Ts_init is None or Ts_init.shape == (N, 4, 4)
    assert vs_init is None or vs_init.shape == (N,)
    assert kappas_init is None or kappas_init.shape == (N,)

    graph, initial, abs_pose_factors = build_graph_gtsam(
        Z_abs,
        dt,
        w_abs,
        Ts_init=Ts_init,
        vs_init=vs_init,
        kappas_init=kappas_init,
    )

    if iterative:
        # Run a few iterations first
        initial_result = optimize_graph(graph, initial, max_iterations=1)

        # Find outliers
        outliers = find_outliers(abs_pose_factors, initial_result, threshold=1.345)

        # Rebuild without outliers
        w_abs_new = w_abs.copy()
        for i in outliers:
            w_abs_new[i] = 0.0

        graph, initial, abs_pose_factors = build_graph_gtsam(
            Z_abs,
            dt,
            w_abs_new,
            Ts_init=Ts_init,
            vs_init=vs_init,
            kappas_init=kappas_init,
        )

    # Get final results
    result = optimize_graph(graph, initial, max_iterations=10)

    smoothed_poses = []
    LB_opt_rot = result.atPose3(symbol("z", 0))
    for i in range(N):
        T_opt = result.atPose3(symbol("x", i))
        T_opt = T_opt.compose(LB_opt_rot.inverse())
        smoothed_poses.append(T_opt.matrix())
    smoothed_poses = np.stack(smoothed_poses)
    return smoothed_poses


#########################################
# Test data generation functions
#########################################


def generate_circle_data(N=50, radius=5.0, total_time=5.0, trans_noise_std=0.05, rot_noise_std=0.05):
    thetas = np.linspace(0, 2 * np.pi, N)
    dt = np.ones(N - 1) * (total_time / (N - 1))

    Z_abs = []
    for theta in thetas:
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = 0.0  # 0.5 * theta  # add slight upward spiral for fun

        # Facing tangent direction
        yaw = theta + np.pi / 2 + np.random.randn() * rot_noise_std
        R = gtsam.Rot3.RzRyRx(0, 0, yaw)
        T = gtsam.Pose3(R, np.array([x, y, z]) + np.random.randn(3) * trans_noise_std)
        Z_abs.append(T.matrix())

    Z_abs = np.stack(Z_abs)
    w_abs = np.ones(N)
    vs_init = np.ones(N - 1) * (2 * np.pi * radius / total_time)
    kappas_init = np.ones(N - 1) / radius
    Ts_init = Z_abs.copy()
    return Z_abs, dt, w_abs, vs_init, kappas_init, Ts_init


def generate_quarter_circle_data(N=50, radius=5.0, total_time=5.0, trans_noise_std=0.05, rot_noise_std=0.05):
    assert N % 4 == 0, "N must be divisible by 4"
    thetas = np.linspace(0, np.pi / 2, N // 2)
    dt = np.ones(N - 1) * (total_time / (N - 1))

    Z_abs = []
    for theta in thetas:
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = 0.0

        # Facing tangent direction
        yaw = theta + np.pi / 2 + np.random.randn() * rot_noise_std
        R = gtsam.Rot3.RzRyRx(0, 0, yaw)
        T = gtsam.Pose3(R, np.array([x, y, z]) + np.random.randn(3) * trans_noise_std)
        Z_abs.append(T.matrix())

    Z_abs = [Z_abs[0]] * (N // 4) + Z_abs + [Z_abs[-1]] * (N // 4)
    Z_abs = np.stack(Z_abs)
    w_abs = np.ones(N)
    w_abs[: N // 4] = 0.0
    w_abs[-N // 4 :] = 0.0
    vs_init = np.ones(N - 1) * (2 * np.pi * radius / total_time)
    kappas_init = np.ones(N - 1) / radius
    Ts_init = Z_abs.copy()
    return Z_abs, dt, w_abs, vs_init, kappas_init, Ts_init


def generate_stop_go_data(N=15, segment_lengths=[5, 5, 5], total_time=20.0, trans_noise_std=0.05, rot_noise_std=0.05):
    assert sum(segment_lengths) == N, "Segment lengths must sum to N"
    Z_abs = []
    dt = np.ones(N) * (total_time / N)
    pos = np.zeros(3)
    yaw = 0.0

    for i in range(N):
        if i < segment_lengths[0]:  # move straight
            v = 50 / 3.6
            pos += np.array([v * dt[i], 0, 0])
        elif i < (segment_lengths[0] + segment_lengths[1]):  # rotate in place
            yaw += np.deg2rad(45 / 10)
        else:  # move in new direction
            v = 30 / 3.6
            pos += np.array([v * dt[i] * np.cos(yaw), v * dt[i] * np.sin(yaw), 0])
        R = gtsam.Rot3.RzRyRx(0, 0, yaw + np.random.randn() * rot_noise_std)
        Z_abs.append(gtsam.Pose3(R, pos.copy() + np.random.randn(3) * trans_noise_std).matrix())

    Z_abs = np.stack(Z_abs)
    w_abs = np.ones(N)
    vs_init = np.ones(N - 1)
    kappas_init = np.zeros(N - 1)
    Ts_init = Z_abs.copy()
    return Z_abs, dt[:-1], w_abs, vs_init, kappas_init, Ts_init


def generate_z_noisy_pitch_outlier_data(
    N=50, total_length=60.0, total_time=5.0, z_noise_std=0.05, n_outliers=3, outlier_pitch_deg=90
):
    dt = np.ones(N - 1) * (total_time / (N - 1))
    Z_abs = []

    # Straight line
    xs = np.linspace(0, total_length, N)
    ys = np.zeros(N)
    zs = np.zeros(N) + np.random.randn(N) * z_noise_std

    # Choose random indices for pitch outliers
    outlier_indices = np.random.choice(N, size=n_outliers, replace=False)

    Z_abs = []
    for i in range(N):
        x, y, z = xs[i], ys[i], zs[i]

        # Mostly flat pitch, with occasional extreme angles
        if i in outlier_indices:
            pitch = np.deg2rad(outlier_pitch_deg) * np.random.choice([-1, 1])
        else:
            pitch = 0.0

        yaw = 0.0  # forward-facing
        roll = 0.0
        R = gtsam.Rot3.RzRyRx(roll, pitch, yaw)
        T = gtsam.Pose3(R, np.array([x, y, z]))
        Z_abs.append(T.matrix())

    Z_abs = np.stack(Z_abs)
    w_abs = np.ones(N)
    vs_init = np.ones(N - 1) * (total_length / (N - 1))
    kappas_init = np.zeros(N - 1)
    Ts_init = Z_abs.copy()

    return Z_abs, dt, w_abs, vs_init, kappas_init, Ts_init


def generate_local_body_mismatch_data(
    N=50,
    total_length=60.0,
    total_time=5.0,
):
    dt = np.ones(N - 1) * (total_time / (N - 1))
    Z_abs = []

    # Straight line along y-axis
    xs = np.zeros(N)
    ys = np.linspace(0, total_length, N)
    zs = np.zeros(N)

    Z_abs = []
    for i in range(N):
        x, y, z = xs[i], ys[i], zs[i]

        pitch = 0.0
        yaw = np.deg2rad(0)
        roll = 0.0
        R = gtsam.Rot3.RzRyRx(roll, pitch, yaw)
        T = gtsam.Pose3(R, np.array([x, y, z]))
        Z_abs.append(T.matrix())

    Z_abs = np.stack(Z_abs)
    w_abs = np.ones(N)
    Ts_init = Z_abs.copy()
    vs_init = np.linalg.norm(np.diff(Ts_init[..., :3, 3], axis=0), axis=-1) / dt
    vs_init += (np.random.rand(N - 1) * 2 - 1) * 0.0

    return Z_abs, dt, w_abs, vs_init, None, Ts_init


def plot_poses_with_arrows(
    Z_abs, optimized_poses, is_valid=None, arrow_scale=1.5, arrow_color="gray", filename="poses_plot.jpg"
):
    plt.figure(figsize=(10, 10))
    if is_valid is None:
        plt.plot(Z_abs[:, 0, 3], Z_abs[:, 1, 3], "o-", label="initial poses", alpha=0.6)
        plt.plot(optimized_poses[:, 0, 3], optimized_poses[:, 1, 3], "o--", label="smoothed", alpha=0.8)
    else:
        plt.plot(Z_abs[:, 0, 3], Z_abs[:, 1, 3], "o-", color="blue", alpha=0.2)
        plt.plot(Z_abs[is_valid, 0, 3], Z_abs[is_valid, 1, 3], "o-", color="blue", label="initial poses", alpha=0.5)

        plt.plot(optimized_poses[:, 0, 3], optimized_poses[:, 1, 3], "o--", color="orange", alpha=0.2)
        plt.plot(
            optimized_poses[is_valid, 0, 3],
            optimized_poses[is_valid, 1, 3],
            "o--",
            color="orange",
            label="smoothed",
            alpha=0.5,
        )

    def draw_arrows(Ts, color):
        xs = Ts[:, 0, 3]
        ys = Ts[:, 1, 3]

        # extract heading direction from rotation (x-axis of rotation matrix)
        dx = Ts[:, 0, 0]
        dy = Ts[:, 1, 0]

        plt.quiver(
            xs, ys, dx, dy, angles="xy", scale_units="xy", scale=1 / arrow_scale, color=color, width=0.005, alpha=0.8
        )

    draw_arrows(Z_abs, "tab:blue")
    draw_arrows(optimized_poses, "tab:orange")

    xmin = np.min(Z_abs[:, 0, 3])
    xmax = np.max(Z_abs[:, 0, 3])
    ymin = np.min(Z_abs[:, 1, 3])
    ymax = np.max(Z_abs[:, 1, 3])
    plt.xlim(xmin - 10.0, xmax + 10.0)
    plt.ylim(ymin - 10.0, ymax + 10.0)
    plt.legend()
    plt.gca().set_aspect("equal")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Continuous-Time Smoothing: Trajectory & Orientation")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_poses_with_pitch_arrows(Z_abs, optimized_poses, arrow_scale=1.5, filename="poses_pitch_plot.jpg"):
    plt.figure(figsize=(10, 6))
    plt.plot(Z_abs[:, 0, 3], Z_abs[:, 2, 3], "o-", label="initial poses", alpha=0.6)
    plt.plot(optimized_poses[:, 0, 3], optimized_poses[:, 2, 3], ".--", label="smoothed", alpha=0.8)

    def draw_pitch_arrows(Ts, color):
        xs = Ts[:, 0, 3]
        zs = Ts[:, 2, 3]

        # Forward direction in X–Z plane (x-axis of rotation matrix)
        dx = Ts[:, 0, 0]
        dz = Ts[:, 2, 0]

        plt.quiver(
            xs, zs, dx, dz, angles="xy", scale_units="xy", scale=1 / arrow_scale, color=color, width=0.005, alpha=0.8
        )

    draw_pitch_arrows(Z_abs, "tab:blue")
    draw_pitch_arrows(optimized_poses, "tab:orange")

    plt.legend()
    plt.axis("equal")
    plt.xlabel("X [m]")
    plt.ylabel("Z [m]")
    plt.title("CT Smoothing: Z Noise & Pitch Orientation")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def main():
    w_abs_mult = 1.0
    numerical_derivative_test()

    # Circle test
    N = 20
    Z_abs, dt, w_abs, vs_init, kappas_init, Ts_init = generate_circle_data(
        N=N, total_time=10.0, radius=15.0, trans_noise_std=1.0, rot_noise_std=0.25
    )
    smoothed_poses = get_smoothed_poses(Z_abs, dt, w_abs * w_abs_mult)
    plot_poses_with_arrows(Z_abs, smoothed_poses, filename="tmp_circle_data.jpg")

    # Quarter circle test
    N = 16
    Z_abs, dt, w_abs, vs_init, kappas_init, Ts_init = generate_quarter_circle_data(
        N=N, total_time=10.0, radius=10.0, trans_noise_std=1e-6, rot_noise_std=1e-6
    )
    smoothed_poses = get_smoothed_poses(Z_abs, dt, w_abs * w_abs_mult)
    plot_poses_with_arrows(Z_abs, smoothed_poses, filename="tmp_quarter_circle_data.jpg")

    # Stop-go test
    N = 15
    Z_abs, dt, w_abs, vs_init, kappas_init, Ts_init = generate_stop_go_data(
        N=N, trans_noise_std=1.0, rot_noise_std=0.15
    )
    smoothed_poses = get_smoothed_poses(Z_abs, dt, w_abs * w_abs_mult)
    plot_poses_with_arrows(Z_abs, smoothed_poses, is_valid=None, arrow_scale=3.0, filename="tmp_stop_go_data.jpg")

    # Z noisy pitch outlier test
    N = 10
    Z_abs, dt, w_abs, vs_init, kappas_init, Ts_init = generate_z_noisy_pitch_outlier_data(
        N=N, total_length=60.0, total_time=5.0, z_noise_std=0.5, n_outliers=3, outlier_pitch_deg=90
    )
    smoothed_poses = get_smoothed_poses(Z_abs, dt, w_abs * w_abs_mult)
    plot_poses_with_pitch_arrows(Z_abs, smoothed_poses, arrow_scale=3.0, filename="tmp_z_noisy_pitch_outlier_data.jpg")

    # Local body mismatch test
    N = 20
    Z_abs, dt, w_abs, vs_init, kappas_init, Ts_init = generate_local_body_mismatch_data(
        N=N, total_length=60.0, total_time=5.0
    )
    smoothed_poses = get_smoothed_poses(Z_abs, dt, w_abs * w_abs_mult, vs_init=vs_init)
    plot_poses_with_arrows(Z_abs, smoothed_poses, arrow_scale=3.0, filename="tmp_local_body_mismatch_data.jpg")


def numerical_derivative_test():
    """Test analytical vs numerical Jacobians with detailed output."""
    import gtsam

    np.random.seed(42)

    dt = 0.1

    # Create test values
    LW1 = gtsam.Pose3(gtsam.Rot3.Ypr(0.1, 0.05, 0.0), np.array([1.0, 2.0, 0.0]))
    LW2 = gtsam.Pose3(gtsam.Rot3.Ypr(0.3, 0.08, 0.0), np.array([1.5, 2.2, 0.0]))
    v = 2.0
    kappa = 0.5
    LB = gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.0, 0.2), np.array([0.0, 0.0, 0.0]))

    test_cases = [
        (2.0, 0.5, "Normal curvature"),
        (2.0, 1e-8, "Small curvature"),
    ]

    for v_test, kappa_test, description in test_cases:
        print(f"\n{'='*70}")
        print(f"Test: {description} (v={v_test}, κ={kappa_test})")
        print("=" * 70)

        keys = [
            gtsam.symbol("x", 0),
            gtsam.symbol("x", 1),
            gtsam.symbol("v", 0),
            gtsam.symbol("k", 0),
            gtsam.symbol("b", 0),
        ]

        class MockFactor:
            def __init__(self, keys):
                self._keys = keys

            def keys(self):
                return self._keys

        factor = MockFactor(keys)

        values = gtsam.Values()
        values.insert(keys[0], LW1)
        values.insert(keys[1], LW2)
        values.insert(keys[2], v_test)
        values.insert(keys[3], kappa_test)
        values.insert(keys[4], LB)

        # Analytical Jacobians
        jacobians_analytical = [np.zeros((6, 6)) for _ in range(5)]
        jacobians_analytical[2] = np.zeros((6, 1))
        jacobians_analytical[3] = np.zeros((6, 1))

        r_analytical = error_ct_motion(dt, factor, values, jacobians_analytical)
        print(f"\nResidual: {r_analytical}")

        # Numerical Jacobians
        eps = 1e-7
        param_names = ["LW1", "LW2", "v", "κ", "LB"]

        for i, (key, name) in enumerate(zip(keys, param_names)):
            if i < 2 or i == 4:  # Poses
                J_num = np.zeros((6, 6))
                for j in range(6):
                    delta = np.zeros(6)
                    delta[j] = eps

                    values_plus = gtsam.Values(values)
                    values_minus = gtsam.Values(values)

                    pose = values.atPose3(key)
                    values_plus.update(key, pose.retract(delta))
                    values_minus.update(key, pose.retract(-delta))

                    r_plus = error_ct_motion(dt, factor, values_plus, None)
                    r_minus = error_ct_motion(dt, factor, values_minus, None)

                    J_num[:, j] = (r_plus - r_minus) / (2 * eps)
            else:  # Scalars
                values_plus = gtsam.Values(values)
                values_minus = gtsam.Values(values)

                val = values.atDouble(key)
                values_plus.update(key, val + eps)
                values_minus.update(key, val - eps)

                r_plus = error_ct_motion(dt, factor, values_plus, None)
                r_minus = error_ct_motion(dt, factor, values_minus, None)

                J_num = (r_plus - r_minus) / (2 * eps)
                J_num = J_num.reshape(6, 1)

            # Compare
            J_ana = jacobians_analytical[i]
            diff = np.abs(J_ana - J_num)
            max_diff = np.max(diff)
            rel_error = max_diff / (np.max(np.abs(J_num)) + 1e-10)

            print(f"\n--- {name} ---")
            print(f"Max abs error: {max_diff:.6e}")
            print(f"Max rel error: {rel_error:.6e}")

            if max_diff > 1e-4:
                print("❌ MISMATCH!")
                print("Analytical:")
                print(J_ana)
                print("Numerical:")
                print(J_num)
                print("Difference:")
                print(diff)
            else:
                print("✅ Match")

    print("\n" + "=" * 70)
    print("Test complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
