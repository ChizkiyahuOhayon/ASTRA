import io
import unittest

import numpy as np
import torch

from utils.shared_motion import SharedMotion, cubic_basis


class SharedMotionTests(unittest.TestCase):
    def make_motion(self):
        # Control locations of a uniform cubic spline reproduce a linear path.
        sites = (torch.arange(8, dtype=torch.float64) - 1) / 5
        velocity = torch.tensor([2., -3., 0.5], dtype=torch.float64)
        control = torch.stack([torch.zeros(8, 3, dtype=torch.float64),
                               sites[:, None] * velocity])
        return SharedMotion(control, torch.zeros(2, dtype=torch.float64),
                            torch.ones(2, dtype=torch.float64), torch.tensor([1, 1, 0]))

    def test_basis_constant_linear_and_extrapolation(self):
        time = torch.tensor([-2., 0., 0.1, 0.5, 0.999, 1., 3.], dtype=torch.float64)
        basis = cubic_basis(time)
        torch.testing.assert_close(basis.sum(-1), torch.ones_like(time))
        sites = (torch.arange(8, dtype=torch.float64) - 1) / 5
        torch.testing.assert_close(basis @ sites, time)

    def test_shared_velocity_and_birth_anchor(self):
        motion = self.make_motion()
        birth = torch.tensor([[0.1], [0.8], [0.2]], dtype=torch.float64)
        velocity = torch.tensor([2., -3., 0.5], dtype=torch.float64)
        expected = (0.4 - birth) * velocity
        expected[2] = 0
        fallback = torch.zeros(3, 3, dtype=torch.float64)
        torch.testing.assert_close(motion(0.4, birth, fallback), expected)
        for i in (0, 1):
            torch.testing.assert_close(motion(float(birth[i]), birth, fallback)[i], torch.zeros(3, dtype=torch.float64))

    def test_velocity_fallback_is_routed_only_to_unassigned_points(self):
        motion = self.make_motion()
        motion.residual_all.fill_(False)  # historical v2 ablation
        birth = torch.tensor([[0.1], [0.8], [0.2]], dtype=torch.float64)
        fallback = torch.tensor([[10., 10., 10.], [20., 20., 20.],
                                 [-1., 2., 0.5]], dtype=torch.float64,
                                requires_grad=True)
        expected = motion(0.4, birth, torch.zeros_like(fallback)).detach()
        expected[2] = fallback.detach()[2] * (0.4 - birth[2])
        output = motion(0.4, birth, fallback)
        torch.testing.assert_close(output, expected)
        output.sum().backward()
        torch.testing.assert_close(fallback.grad[:2], torch.zeros_like(fallback.grad[:2]))
        self.assertGreater(fallback.grad[2].abs().sum().item(), 0)

    def test_v1_residual_is_preserved_for_every_point(self):
        motion = self.make_motion()
        birth = torch.tensor([[0.1], [0.8], [0.2]], dtype=torch.float64)
        velocity = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
        shared = motion(0.4, birth, torch.zeros_like(velocity)).detach()
        output = motion(0.4, birth, velocity)
        torch.testing.assert_close(output, shared + velocity * (0.4 - birth))
        output.sum().backward()
        torch.testing.assert_close(velocity.grad, (0.4 - birth).expand_as(velocity))

    def test_disabled_group_has_residual_but_no_control_gradient(self):
        motion = self.make_motion()
        motion.enabled[1] = False
        birth = torch.tensor([[0.1], [0.8], [0.2]], dtype=torch.float64)
        velocity = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
        output = motion(0.4, birth, velocity)
        torch.testing.assert_close(output, velocity * (0.4 - birth))
        output.sum().backward()
        torch.testing.assert_close(motion.control.grad, torch.zeros_like(motion.control))
        self.assertTrue((velocity.grad.abs().sum(1) > 0).all())

    def test_legacy_checkpoint_requires_explicit_residual_version(self):
        state = self.make_motion().state_dict()
        for key in ('enabled', 'linear_enabled', 'piecewise_enabled',
                    'knot_time', 'knot_count', 'residual_all'):
            del state[key]
        with self.assertRaisesRegex(ValueError, 'Legacy shared checkpoint'):
            SharedMotion.from_state(state)
        self.assertTrue(SharedMotion.from_state(state, 'v1').residual_all)
        self.assertFalse(SharedMotion.from_state(state, 'v2').residual_all)

    def test_v3_checkpoint_defaults_to_no_linear_fallback(self):
        state = self.make_motion().state_dict()
        for key in ('linear_enabled', 'piecewise_enabled', 'knot_time', 'knot_count'):
            del state[key]
        restored = SharedMotion.from_state(state)
        torch.testing.assert_close(restored.linear_enabled,
                                   torch.zeros_like(restored.enabled))
        torch.testing.assert_close(restored.piecewise_enabled,
                                   torch.zeros_like(restored.enabled))

    def test_gate_survives_split_clone_prune_and_reload(self):
        motion = self.make_motion()
        # Third group is disabled, with a distinct nonzero shared trajectory.
        motion = SharedMotion(torch.cat([motion.control.detach(), motion.control[1:2].detach() * 2]),
                              torch.zeros(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64),
                              torch.tensor([1, 2, 0]), torch.tensor([False, True, False]))
        birth = torch.tensor([[0.1], [0.8], [0.2]], dtype=torch.float64)
        velocity = torch.ones(3, 3, dtype=torch.float64)
        original = motion(0.4, birth, velocity).detach()
        parents = torch.tensor([0, 1, 0, 1])  # split repeat order used by GaussianModel
        motion.append(motion.point_ids[parents])
        birth = torch.cat([birth, birth[parents]])
        velocity = torch.cat([velocity, velocity[parents]])
        motion.append(motion.point_ids[2:3])  # clone unassigned
        birth = torch.cat([birth, birth[2:3]])
        velocity = torch.cat([velocity, velocity[2:3]])
        keep = torch.tensor([False, False, True, True, False, True, True, True])
        motion.prune(keep)
        stream = io.BytesIO()
        torch.save(motion.state_dict(), stream)
        stream.seek(0)
        restored = SharedMotion.from_state(torch.load(stream, weights_only=True))
        torch.testing.assert_close(restored(0.4, birth[keep], velocity[keep]), original[[2, 0, 0, 1, 2]])

    def test_gradients_and_group_isolation(self):
        motion = self.make_motion()
        birth = torch.tensor([[0.1], [0.8], [0.2]], dtype=torch.float64)
        motion(0.4, birth, torch.zeros(3, 3, dtype=torch.float64)).square().sum().backward()
        self.assertTrue(torch.isfinite(motion.control.grad).all())
        self.assertGreater(motion.control.grad[1].abs().sum().item(), 0)
        self.assertEqual(motion.control.grad[0].abs().sum().item(), 0)

    def test_clone_prune_and_checkpoint_keep_correspondence(self):
        motion = self.make_motion()
        birth = torch.tensor([[0.1], [0.8], [0.2]], dtype=torch.float64)
        fallback = torch.zeros(3, 3, dtype=torch.float64)
        before = motion(0.4, birth, fallback).detach()
        motion.append(motion.point_ids[torch.tensor([1, 0])])
        birth = torch.cat([birth, birth[torch.tensor([1, 0])]])
        fallback = torch.cat([fallback, fallback[torch.tensor([1, 0])]])
        keep = torch.tensor([False, True, True, True, True])
        motion.prune(keep)
        birth = birth[keep]
        fallback = fallback[keep]
        expected = before[torch.tensor([1, 2, 1, 0])]
        torch.testing.assert_close(motion(0.4, birth, fallback), expected)
        stream = io.BytesIO()
        torch.save(motion.state_dict(), stream)
        stream.seek(0)
        restored = SharedMotion.from_state(torch.load(stream, weights_only=True))
        torch.testing.assert_close(restored(0.4, birth, fallback), expected)

    def test_training_point_initialization_recovers_two_motions(self):
        # Dense, separated synthetic objects with different velocities; no GT
        # files or predicted RGB are needed by the initializer.
        rng = np.random.RandomState(4)
        shape = rng.uniform(-0.08, 0.08, size=(20, 3))
        velocities = np.array([[1., 0., 0.], [0., -0.5, 0.]])
        points, times, identities = [], [], []
        for frame in range(10):
            t = frame / 9
            for group in range(2):
                points.append(shape + [group*10, 0, 0] + t*velocities[group])
                times.extend([t]*len(shape))
                identities.extend([group]*len(shape))
        xyz = torch.tensor(np.concatenate(points), dtype=torch.float64)
        birth = torch.tensor(times, dtype=torch.float64)[:, None]
        motion = SharedMotion.from_points(xyz, birth, frame_gap=0.1)
        expected = (0.45-birth) * torch.tensor(velocities[np.array(identities)])
        fallback = torch.zeros_like(xyz)
        torch.testing.assert_close(motion(0.45, birth, fallback), expected, atol=1e-6, rtol=1e-6)
        self.assertEqual(motion.control.shape[0], 3)

    def test_support_gate_boundary_and_v1_reduction(self):
        rng = np.random.RandomState(6)
        shape = rng.uniform(-.08, .08, (20, 3))
        points, times = [], []
        for frame in range(8):
            for group in range(2):
                if group == 0 and frame == 7:
                    continue
                t = frame / 9
                points.append(shape + [10 * group + t, 0, 0])
                times.extend([t] * len(shape))
        xyz = torch.tensor(np.concatenate(points), dtype=torch.float64)
        birth = torch.tensor(times, dtype=torch.float64)[:, None]
        v1 = SharedMotion.from_points(xyz, birth, .1)
        v3 = SharedMotion.from_points(xyz, birth, .1, support_gate=True)
        torch.testing.assert_close(v1.control, v3.control)
        torch.testing.assert_close(v1.point_ids, v3.point_ids)
        self.assertEqual(v3.enabled.tolist(), [False, False, True])
        velocity = torch.ones_like(xyz) * .3
        output = v3(.4, birth, velocity)
        enabled = v3.enabled[v3.point_ids]
        torch.testing.assert_close(output[enabled], v1(.4, birth, velocity)[enabled])
        torch.testing.assert_close(output[~enabled], (velocity * (.4 - birth))[~enabled])

    def test_support_adaptive_linear_fallback(self):
        rng = np.random.RandomState(6)
        shape = rng.uniform(-.08, .08, (20, 3))
        points, times = [], []
        for frame in range(8):
            for group in range(2):
                if group == 0 and frame == 7:
                    continue
                t = frame / 9
                points.append(shape + [10 * group + t, 0, 0])
                times.extend([t] * len(shape))
        xyz = torch.tensor(np.concatenate(points), dtype=torch.float64)
        birth = torch.tensor(times, dtype=torch.float64)[:, None]
        v3 = SharedMotion.from_points(xyz, birth, .1, support_gate=True)
        v4 = SharedMotion.from_points(xyz, birth, .1, support_gate=True, linear_fallback=True)
        torch.testing.assert_close(v3.enabled, v4.enabled)
        torch.testing.assert_close(v3.point_ids, v4.point_ids)
        self.assertEqual(v4.linear_enabled.tolist(), [False, True, False])
        velocity = torch.ones_like(xyz) * .3
        short = v4.linear_enabled[v4.point_ids]
        expected = torch.zeros_like(xyz)
        expected[:, 0] = .4 - birth[:, 0]
        torch.testing.assert_close(v4(.4, birth, velocity)[short],
                                   (expected + velocity * (.4 - birth))[short],
                                   atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(v4(.4, birth, velocity)[~short],
                                   v3(.4, birth, velocity)[~short])

    def test_support_adaptive_piecewise_fallback(self):
        rng = np.random.RandomState(6)
        shape = rng.uniform(-.08, .08, (20, 3))
        points, times = [], []
        for frame in range(8):
            t = frame / 9
            if frame < 7:
                points.append(shape + [t, t * t, 0])
                times.extend([t] * len(shape))
            points.append(shape + [10 + t, 0, 0])
            times.extend([t] * len(shape))
        xyz = torch.tensor(np.concatenate(points), dtype=torch.float64)
        birth = torch.tensor(times, dtype=torch.float64)[:, None]
        v5 = SharedMotion.from_points(xyz, birth, .1, support_gate=True,
                                      piecewise_fallback=True)
        self.assertEqual(v5.piecewise_enabled.tolist(), [False, True, False])
        self.assertEqual(v5.knot_count.tolist(), [0, 7, 0])
        short = v5.piecewise_enabled[v5.point_ids]
        query = 3 / 9
        trajectory = torch.cat((birth, birth.square(), torch.zeros_like(birth)), dim=1)
        target = torch.tensor([query, query * query, 0.], dtype=torch.float64)
        velocity = torch.ones_like(xyz) * .3
        torch.testing.assert_close(
            v5(query, birth, velocity)[short],
            (target - trajectory + velocity * (query - birth))[short],
            atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
