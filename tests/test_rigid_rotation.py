"""v8: shared rigid rotation on top of the shared translation."""
import io
import math
import os
import unittest

import numpy as np
import torch

from utils.shared_motion import (SharedMotion, axis_angle_quaternion,
                                 quaternion_rotate)
from tests.test_singleton_static import four_route_motion


def rotating_cloud(turn=0.9, frames=10, points=24, seed=5):
    """A rigid body that both translates and yaws; no correspondence is given."""
    rng = np.random.RandomState(seed)
    shape = rng.uniform(-.6, .6, (points, 3))
    xyz, times = [], []
    for frame in range(frames):
        t = frame / (frames - 1)
        angle = turn * t
        rotation = np.array([[math.cos(angle), -math.sin(angle), 0],
                             [math.sin(angle), math.cos(angle), 0],
                             [0, 0, 1.]])
        xyz.append(shape @ rotation.T + [4 * t, 0, 0])
        times.extend([frame / 19] * points)
    return (torch.tensor(np.concatenate(xyz), dtype=torch.float64),
            torch.tensor(times, dtype=torch.float64)[:, None], shape)


class QuaternionTests(unittest.TestCase):
    def test_zero_rotation_is_the_identity_with_a_finite_derivative(self):
        vector = torch.zeros(4, 3, dtype=torch.float64, requires_grad=True)
        quaternion = axis_angle_quaternion(vector)
        torch.testing.assert_close(quaternion, torch.tensor([[1., 0, 0, 0]] * 4,
                                                            dtype=torch.float64))
        target = torch.randn(4, 3, dtype=torch.float64)
        quaternion_rotate(quaternion, target).sum().backward()
        self.assertTrue(torch.isfinite(vector.grad).all())
        self.assertGreater(vector.grad.abs().sum().item(), 0)

    def test_quaternion_is_unit_and_matches_an_explicit_rotation(self):
        angle = 0.7
        vector = torch.tensor([[0., 0., angle]], dtype=torch.float64)
        quaternion = axis_angle_quaternion(vector)
        torch.testing.assert_close(quaternion.norm(dim=-1), torch.ones(1, dtype=torch.float64))
        point = torch.tensor([[1., 2., 3.]], dtype=torch.float64)
        expected = torch.tensor([[math.cos(angle) - 2 * math.sin(angle),
                                  math.sin(angle) + 2 * math.cos(angle), 3.]],
                                dtype=torch.float64)
        torch.testing.assert_close(quaternion_rotate(quaternion, point), expected)

    def test_small_and_large_angles_stay_finite(self):
        for scale in (0., 1e-9, 1e-5, 1e-3, 1., 10.):
            vector = torch.full((3, 3), scale, dtype=torch.float64, requires_grad=True)
            quaternion = axis_angle_quaternion(vector)
            torch.testing.assert_close(quaternion.norm(dim=-1),
                                       torch.ones(3, dtype=torch.float64))
            quaternion.sum().backward()
            self.assertTrue(torch.isfinite(vector.grad).all(), scale)


class RigidRotationTests(unittest.TestCase):
    def setUp(self):
        self.device = os.environ.get('ADGS_TEST_DEVICE', 'cpu')

    def test_identity_controls_reproduce_the_translation_only_model(self):
        motion, birth, xyz = four_route_motion(True, torch.float64, self.device)
        velocity = torch.randn_like(xyz)
        for time in (-.2, 0., .23, .5, 1.3):
            plain = motion(time, birth, velocity)
            motion.rigid_rotation = True
            torch.testing.assert_close(motion(time, birth, velocity, xyz), plain,
                                       rtol=0, atol=0)
            bias = motion.rotation_bias(time, birth)
            torch.testing.assert_close(bias, torch.tensor([[1., 0, 0, 0]], dtype=torch.float64,
                                                          device=bias.device).expand_as(bias))
            motion.rigid_rotation = False

    def test_a_checkpoint_without_rotation_controls_stays_translation_only(self):
        motion, birth, xyz = four_route_motion(True, torch.float64, self.device)
        state = motion.state_dict()
        for key in ('rot_control', 'center'):
            self.assertIn(key, state)
            del state[key]
        stream = io.BytesIO()
        torch.save(state, stream)
        stream.seek(0)
        restored = SharedMotion.from_state(torch.load(stream, weights_only=True))
        self.assertFalse(restored.rigid_rotation)
        torch.testing.assert_close(restored.rot_control,
                                   torch.zeros_like(restored.control), rtol=0, atol=0)
        restored.rigid_rotation = True
        velocity = torch.randn_like(xyz)
        torch.testing.assert_close(restored(.4, birth, velocity, xyz),
                                   motion(.4, birth, velocity), rtol=0, atol=0)

    def test_only_supported_tracklets_rotate(self):
        motion, birth, xyz = four_route_motion(True, torch.float64, self.device)
        motion.rigid_rotation = True
        with torch.no_grad():
            motion.rot_control.add_(torch.randn_like(motion.rot_control) * .3)
        ids = motion.point_ids
        bias = motion.rotation_bias(.4, birth)
        cubic = motion.enabled[ids]
        identity = torch.tensor([1., 0, 0, 0], dtype=torch.float64, device=bias.device)
        torch.testing.assert_close(bias[~cubic], identity.expand_as(bias[~cubic]))
        self.assertGreater((bias[cubic] - identity).abs().max().item(), 1e-3)
        velocity = torch.zeros_like(xyz)
        rigid = motion(.4, birth, velocity, xyz)
        motion.rigid_rotation = False
        torch.testing.assert_close(rigid[~cubic], motion(.4, birth, velocity)[~cubic],
                                   rtol=0, atol=0)

    def test_rotation_is_exactly_zero_at_each_point_birth_time(self):
        motion, birth, xyz = four_route_motion(True, torch.float64, self.device)
        motion.rigid_rotation = True
        with torch.no_grad():
            motion.rot_control.add_(torch.randn_like(motion.rot_control) * .4)
        ids = motion.point_ids
        for value in torch.unique(birth):
            born = (birth.reshape(-1) == value) & motion.enabled[ids]
            if not born.any():
                continue
            bias = motion.rotation_bias(float(value), birth)[born]
            identity = torch.tensor([1., 0, 0, 0], dtype=torch.float64, device=bias.device)
            torch.testing.assert_close(bias, identity.expand_as(bias), atol=1e-12, rtol=0)

    def test_gradient_descent_recovers_a_yawing_rigid_body(self):
        xyz, birth, shape = rotating_cloud()
        motion = SharedMotion.from_points(xyz, birth, .05, support_gate=True,
                                          piecewise_fallback=True, bidirectional=True,
                                          singleton_static=True, rigid_rotation=True)
        # However DBSCAN splits the body, every part is seen in all ten frames.
        self.assertGreaterEqual(int(motion.enabled.sum()), 1)
        # Ask the model to explain the true rigid pose of frame 0's points at
        # every later time. Translation alone cannot: the body also yaws.
        # DBSCAN noise has no shared motion at all, so it is not compared here.
        born = (birth.reshape(-1) == 0.) & motion.enabled[motion.point_ids]
        self.assertGreater(int(born.sum()), 10)
        velocity = torch.zeros_like(xyz)

        def fit(rigid, steps=400):
            motion.rigid_rotation = rigid
            with torch.no_grad():
                motion.control.copy_(torch.zeros_like(motion.control))
                motion.rot_control.copy_(torch.zeros_like(motion.rot_control))
            optimizer = torch.optim.Adam(motion.parameters(), lr=.05)
            for _ in range(steps):
                loss = 0.
                for frame in range(1, 10):
                    time = frame / 19
                    angle = .9 * frame / 9
                    rotation = torch.tensor(
                        [[math.cos(angle), -math.sin(angle), 0.],
                         [math.sin(angle), math.cos(angle), 0.],
                         [0., 0., 1.]], dtype=torch.float64)
                    target = torch.tensor(shape) @ rotation.T + torch.tensor(
                        [4 * frame / 9, 0, 0], dtype=torch.float64)
                    moved = (xyz + motion(time, birth, velocity, xyz))[born]
                    loss = loss + (moved - target[born[:len(shape)]]).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            return float(loss.detach())

        translation_only = fit(False)
        rigid = fit(True)
        self.assertTrue(np.isfinite(rigid))
        # The rigid model must explain what translation structurally cannot.
        self.assertLess(rigid, translation_only / 10)
        self.assertLess(rigid, 5e-3)

    def test_rotation_controls_receive_finite_gradient_from_the_start(self):
        motion, birth, xyz = four_route_motion(True, torch.float64, self.device)
        motion.rigid_rotation = True
        velocity = torch.zeros_like(xyz)
        motion(.4, birth, velocity, xyz).square().sum().backward()
        self.assertTrue(torch.isfinite(motion.rot_control.grad).all())
        cubic = torch.where(motion.enabled)[0]
        self.assertGreater(motion.rot_control.grad[cubic].abs().sum().item(), 0)
        other = torch.tensor([g for g in range(len(motion.start)) if g not in cubic.tolist()])
        torch.testing.assert_close(motion.rot_control.grad[other],
                                   torch.zeros_like(motion.rot_control.grad[other]),
                                   rtol=0, atol=0)

    def test_rotation_survives_clone_split_prune_and_reload(self):
        motion, birth, xyz = four_route_motion(True, torch.float64, self.device)
        motion.rigid_rotation = True
        with torch.no_grad():
            motion.rot_control.add_(torch.randn_like(motion.rot_control) * .2)
        velocity = torch.randn_like(xyz)
        ids = motion.point_ids
        parents = torch.stack([torch.where(motion.enabled[ids])[0][0],
                               torch.where(motion.static_enabled[ids])[0][0],
                               torch.where(ids == 0)[0][0]])
        original = motion(.23, birth, velocity, xyz).detach()
        lineage = torch.arange(len(ids), device=ids.device)
        motion.append(motion.point_ids[parents])
        birth = torch.cat([birth, birth[parents]])
        velocity = torch.cat([velocity, velocity[parents]])
        xyz = torch.cat([xyz, xyz[parents]])
        lineage = torch.cat([lineage, lineage[parents]])
        keep = torch.ones(len(lineage), dtype=torch.bool, device=ids.device)
        keep[parents] = False
        motion.prune(keep)
        stream = io.BytesIO()
        torch.save(motion.state_dict(), stream)
        stream.seek(0)
        restored = SharedMotion.from_state(torch.load(stream, weights_only=True))
        restored.rigid_rotation = True
        torch.testing.assert_close(restored(.23, birth[keep], velocity[keep], xyz[keep]),
                                   original[lineage[keep]], rtol=0, atol=0)

    def test_the_basis_cache_stays_inert_with_rotation_on(self):
        plain, birth, xyz = four_route_motion(True, torch.float64, self.device)
        cached, _, _ = four_route_motion(True, torch.float64, self.device)
        cached.basis_cache = True
        for motion in (plain, cached):
            motion.rigid_rotation = True
            with torch.no_grad():
                motion.rot_control.copy_(torch.full_like(motion.rot_control, .17))
        velocity = torch.randn_like(xyz)
        with torch.no_grad():
            for _ in range(2):
                for time in (-.2, 0., .23, .5, 1.3):
                    torch.testing.assert_close(cached(time, birth, velocity, xyz),
                                               plain(time, birth, velocity, xyz),
                                               rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
