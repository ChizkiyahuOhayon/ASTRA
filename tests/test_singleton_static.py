"""v7: grouped single-frame tracklets are static; the basis cache is inert."""
import io
import os
import unittest

import numpy as np
import torch

from utils.shared_motion import SharedMotion


def four_route_cloud():
    """Cubic, piecewise, grouped singleton and DBSCAN-noise points in one cloud."""
    rng = np.random.RandomState(11)
    shape = rng.uniform(-.08, .08, (20, 3))
    points, times = [], []
    for frame in range(10):
        t = frame / 19
        points.append(shape + [t, 0, 0])           # ten frames -> cubic
        times.extend([t] * len(shape))
        if frame < 3:
            points.append(shape + [20 + t, 0, 0])  # three frames -> piecewise
            times.extend([t] * len(shape))
    points.append(shape + [40., 0, 0])             # one frame -> grouped singleton
    times.extend([0.] * len(shape))
    points.append(np.array([[100., 100., 100.]]))  # below min_samples -> noise
    times.append(0.)
    xyz = torch.tensor(np.concatenate(points), dtype=torch.float64)
    birth = torch.tensor(times, dtype=torch.float64)[:, None]
    return xyz, birth


def four_route_motion(singleton_static=True, dtype=torch.float64, device='cpu'):
    xyz, birth = four_route_cloud()
    motion = SharedMotion.from_points(xyz.to(dtype), birth.to(dtype), .05, support_gate=True,
                                      piecewise_fallback=True, bidirectional=True,
                                      singleton_static=singleton_static)
    return motion.to(device), birth.to(device=device, dtype=dtype), xyz.to(device=device, dtype=dtype)


class SingletonStaticTests(unittest.TestCase):
    def setUp(self):
        self.device = os.environ.get('ADGS_TEST_DEVICE', 'cpu')

    def test_only_grouped_single_frame_tracklets_become_static(self):
        motion, _, _ = four_route_motion(device=self.device)
        self.assertEqual(int(motion.enabled.sum()), 1)
        self.assertEqual(int(motion.piecewise_enabled.sum()), 1)
        self.assertEqual(int(motion.static_enabled.sum()), 1)
        self.assertFalse(bool(motion.linear_enabled.any()))
        self.assertFalse(bool(motion.static_enabled[0]))  # noise group is never static
        self.assertFalse(bool((motion.static_enabled & (motion.enabled | motion.piecewise_enabled)).any()))
        ids = motion.point_ids
        self.assertEqual(int(motion.static_enabled[ids].sum()), 20)
        self.assertEqual(int((ids == 0).sum()), 1)
        self.assertEqual(int(motion.enabled[ids].sum()), 200)
        self.assertEqual(int(motion.piecewise_enabled[ids].sum()), 60)

    def test_v6_routes_are_untouched_and_singletons_are_the_only_change(self):
        v6, birth, xyz = four_route_motion(singleton_static=False, device=self.device)
        v7, _, _ = four_route_motion(singleton_static=True, device=self.device)
        self.assertFalse(bool(v6.static_enabled.any()))
        for key in ('control', 'start', 'duration', 'point_ids', 'knot_time', 'knot_count'):
            torch.testing.assert_close(v6.state_dict()[key], v7.state_dict()[key], rtol=0, atol=0)
        singleton = v7.static_enabled[v7.point_ids]
        torch.testing.assert_close(v6.enabled & ~v6.static_enabled, v7.enabled)
        velocity = torch.randn_like(xyz)
        for time in (-.1, 0., .23, .5, 1.1):
            a, b = v6(time, birth, velocity), v7(time, birth, velocity)
            torch.testing.assert_close(a[~singleton], b[~singleton], rtol=0, atol=0)
            torch.testing.assert_close(b[singleton], torch.zeros_like(b[singleton]), rtol=0, atol=0)
            torch.testing.assert_close(a[singleton], (velocity * (time - birth))[singleton])

    def test_static_points_never_move_and_take_no_gradient(self):
        motion, birth, xyz = four_route_motion(device=self.device)
        velocity = torch.randn_like(xyz, requires_grad=True)
        singleton = motion.static_enabled[motion.point_ids]
        for time in (-.4, 0., .37, 1., 2.):
            output = motion(time, birth, velocity)
            torch.testing.assert_close(output[singleton], torch.zeros_like(output[singleton]),
                                       rtol=0, atol=0)
            output.square().sum().backward()
            torch.testing.assert_close(velocity.grad[singleton],
                                       torch.zeros_like(velocity.grad[singleton]), rtol=0, atol=0)
            self.assertGreater(velocity.grad[~singleton].abs().sum().item(), 0)
            static_group = torch.where(motion.static_enabled)[0]
            torch.testing.assert_close(motion.control.grad[static_group],
                                       torch.zeros_like(motion.control.grad[static_group]),
                                       rtol=0, atol=0)
            velocity.grad = None
            motion.control.grad = None

    def test_unassigned_and_noise_keep_their_residual(self):
        motion, birth, xyz = four_route_motion(device=self.device)
        velocity = torch.randn_like(xyz, requires_grad=True)
        noise = motion.point_ids == 0
        output = motion(.31, birth, velocity)
        torch.testing.assert_close(output[noise], (velocity * (.31 - birth))[noise])
        output.sum().backward()
        self.assertGreater(velocity.grad[noise].abs().sum().item(), 0)

    def test_static_overrides_the_support_gate_being_off(self):
        xyz, birth = four_route_cloud()
        motion = SharedMotion.from_points(xyz, birth, .05, bidirectional=True,
                                          singleton_static=True).to(self.device)
        birth, xyz = birth.to(self.device), xyz.to(self.device)
        static_group = torch.where(motion.static_enabled)[0]
        self.assertEqual(len(static_group), 1)
        self.assertFalse(bool(motion.enabled[static_group].any()))
        singleton = motion.static_enabled[motion.point_ids]
        velocity = torch.randn_like(xyz)
        output = motion(.5, birth, velocity)
        torch.testing.assert_close(output[singleton], torch.zeros_like(output[singleton]),
                                   rtol=0, atol=0)

    def test_children_of_a_static_parent_stay_static_through_clone_split_prune(self):
        motion, birth, xyz = four_route_motion(device=self.device)
        ids = motion.point_ids
        velocity = torch.randn_like(xyz)
        parents = torch.stack([torch.where(motion.enabled[ids])[0][0],
                               torch.where(motion.piecewise_enabled[ids])[0][0],
                               torch.where(motion.static_enabled[ids])[0][0],
                               torch.where(ids == 0)[0][0]])
        original = motion(.23, birth, velocity).detach()
        lineage = torch.arange(len(ids), device=ids.device)
        for source in (parents, parents.repeat(2)):
            motion.append(motion.point_ids[source])
            birth = torch.cat([birth, birth[source]])
            velocity = torch.cat([velocity, velocity[source]])
            lineage = torch.cat([lineage, lineage[source]])
        keep = torch.ones(len(lineage), dtype=torch.bool, device=ids.device)
        keep[parents] = False
        motion.prune(keep)
        output = motion(.23, birth[keep], velocity[keep])
        torch.testing.assert_close(output, original[lineage[keep]], rtol=0, atol=0)
        static_child = motion.static_enabled[motion.point_ids]
        self.assertEqual(int(static_child.sum()), 20 + 2)  # parent pruned, 3 copies made
        torch.testing.assert_close(output[static_child], torch.zeros_like(output[static_child]),
                                   rtol=0, atol=0)

    def test_v6_checkpoint_without_the_field_keeps_the_old_residual_semantics(self):
        v6, birth, xyz = four_route_motion(singleton_static=False, device=self.device)
        state = v6.state_dict()
        del state['static_enabled']  # v1-v6 files have no such key
        stream = io.BytesIO()
        torch.save(state, stream)
        stream.seek(0)
        restored = SharedMotion.from_state(torch.load(stream, weights_only=True))
        self.assertFalse(bool(restored.static_enabled.any()))
        velocity = torch.randn_like(xyz)
        torch.testing.assert_close(restored(.42, birth, velocity), v6(.42, birth, velocity),
                                   rtol=0, atol=0)

    def test_legacy_v1_v2_checkpoints_still_require_an_explicit_choice(self):
        state = four_route_motion(singleton_static=False, device=self.device)[0].state_dict()
        for key in ('enabled', 'linear_enabled', 'piecewise_enabled', 'static_enabled',
                    'knot_time', 'knot_count', 'residual_all'):
            del state[key]
        with self.assertRaisesRegex(ValueError, 'Legacy shared checkpoint'):
            SharedMotion.from_state(state)
        self.assertTrue(SharedMotion.from_state(state, 'v1').residual_all)
        self.assertFalse(SharedMotion.from_state(state, 'v2').static_enabled.any())

    def test_v7_checkpoint_round_trips_the_static_route(self):
        motion, birth, xyz = four_route_motion(device=self.device)
        velocity = torch.randn_like(xyz)
        expected = motion(.61, birth, velocity).detach()
        stream = io.BytesIO()
        torch.save(motion.state_dict(), stream)
        stream.seek(0)
        state = torch.load(stream, weights_only=True)
        self.assertIn('static_enabled', state)
        self.assertNotIn('basis_cache', state)
        restored = SharedMotion.from_state(state)
        torch.testing.assert_close(restored.static_enabled, motion.static_enabled, rtol=0, atol=0)
        torch.testing.assert_close(restored(.61, birth, velocity), expected, rtol=0, atol=0)


class BirthBasisCacheTests(unittest.TestCase):
    def setUp(self):
        self.device = os.environ.get('ADGS_TEST_DEVICE', 'cpu')

    def cached_pair(self, dtype, singleton_static=True):
        plain, birth, xyz = four_route_motion(singleton_static, dtype, self.device)
        cached, _, _ = four_route_motion(singleton_static, dtype, self.device)
        cached.basis_cache = True
        return plain, cached, birth, xyz

    def test_cache_matches_forward_and_both_gradients_bitwise(self):
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            if self.device == 'cuda':
                torch.use_deterministic_algorithms(True)
            for singleton_static in (False, True):
                for dtype in (torch.float32, torch.float64):
                    plain, cached, birth, xyz = self.cached_pair(dtype, singleton_static)
                    for time in (-.2, 0., .23, .5, 1.3):
                        weight = torch.randn_like(xyz)
                        va = xyz.detach().clone().requires_grad_(True)
                        vb = xyz.detach().clone().requires_grad_(True)
                        a, b = plain(time, birth, va), cached(time, birth, vb)
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                        (a * weight).sum().backward()
                        (b * weight).sum().backward()
                        torch.testing.assert_close(plain.control.grad, cached.control.grad,
                                                   rtol=0, atol=0)
                        torch.testing.assert_close(va.grad, vb.grad, rtol=0, atol=0)
                        plain.control.grad = cached.control.grad = None
        finally:
            torch.use_deterministic_algorithms(previous)

    def test_cache_is_rebuilt_after_clone_split_and_prune(self):
        plain, cached, birth, xyz = self.cached_pair(torch.float64)
        velocity = torch.ones_like(xyz)
        ids = plain.point_ids
        parents = torch.stack([torch.where(plain.enabled[ids])[0][0],
                               torch.where(plain.piecewise_enabled[ids])[0][0],
                               torch.where(plain.static_enabled[ids])[0][0],
                               torch.where(ids == 0)[0][0]])
        cached(.23, birth, velocity)  # populate the cache before the topology changes
        self.assertIsNotNone(cached._plan)
        for motion in (plain, cached):
            motion.append(motion.point_ids[parents])
        self.assertIsNone(cached._plan)
        birth = torch.cat([birth, birth[parents]])
        velocity = torch.cat([velocity, velocity[parents]])
        cached(.23, birth, velocity)
        keep = torch.ones(len(birth), dtype=torch.bool, device=birth.device)
        keep[parents] = False
        for motion in (plain, cached):
            motion.prune(keep)
        torch.testing.assert_close(cached(.23, birth[keep], velocity[keep]),
                                   plain(.23, birth[keep], velocity[keep]), rtol=0, atol=0)

    def test_cache_follows_a_new_birth_tensor_without_an_explicit_reset(self):
        plain, cached, birth, xyz = self.cached_pair(torch.float64)
        velocity = torch.ones_like(xyz)
        cached(.23, birth, velocity)
        shifted = (birth + 0.).detach()  # same values, a different tensor object
        torch.testing.assert_close(cached(.23, shifted, velocity),
                                   plain(.23, shifted, velocity), rtol=0, atol=0)
        moved = birth * 0. + .1
        torch.testing.assert_close(cached(.23, moved, velocity),
                                   plain(.23, moved, velocity), rtol=0, atol=0)

    def test_training_steps_do_not_stale_the_cache(self):
        # Toggle one model rather than training two: CUDA reductions are not
        # bit-reproducible across separate graphs and Adam magnifies that.
        _, cached, birth, xyz = self.cached_pair(torch.float64)
        velocity = torch.ones_like(xyz)
        optimizer = torch.optim.Adam(cached.parameters(), lr=1e-2)
        for step in range(5):
            with torch.no_grad():
                value = cached(.7, birth, velocity).clone()
                cached.basis_cache = False
                torch.testing.assert_close(cached(.7, birth, velocity), value,
                                           rtol=0, atol=0)
                cached.basis_cache = True
            loss = cached(.1 * step, birth, velocity).square().sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    def test_cache_survives_a_state_dict_round_trip_and_device_moves(self):
        plain, cached, birth, xyz = self.cached_pair(torch.float64)
        velocity = torch.ones_like(xyz)
        cached(.23, birth, velocity)
        self.assertNotIn('basis_cache', cached.state_dict())
        cached.to(torch.float64)
        self.assertIsNone(cached._plan)
        torch.testing.assert_close(cached(.23, birth, velocity), plain(.23, birth, velocity),
                                   rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
