"""v7.1: the inference-time birth-anchor memo must be completely inert."""
import io
import os
import unittest

import torch

from tests.test_singleton_static import four_route_motion


class FrozenReferenceTests(unittest.TestCase):
    def setUp(self):
        self.device = os.environ.get('ADGS_TEST_DEVICE', 'cpu')

    def pair(self, dtype=torch.float64):
        plain, birth, xyz = four_route_motion(True, dtype, self.device)
        cached, _, _ = four_route_motion(True, dtype, self.device)
        cached.basis_cache = True
        return plain, cached, birth, xyz

    def test_inference_forward_is_bitwise_equal_with_and_without_the_memo(self):
        plain, cached, birth, xyz = self.pair()
        velocity = torch.randn_like(xyz)
        with torch.no_grad():
            for _ in range(3):                       # the memo is hit from call two on
                for time in (-.2, 0., .23, .5, 1.3):
                    torch.testing.assert_close(cached(time, birth, velocity),
                                               plain(time, birth, velocity), rtol=0, atol=0)
        self.assertIsNotNone(cached._reference_memo)

    def test_the_memo_is_never_taken_while_autograd_is_on(self):
        _, cached, birth, xyz = self.pair()
        velocity = torch.randn_like(xyz)
        cached(.23, birth, velocity)
        self.assertIsNone(cached._reference_memo)
        # Two graphs in one step must not share a freed intermediate.
        first = cached(.23, birth, velocity).square().sum()
        second = cached(.51, birth, velocity).square().sum()
        first.backward()
        second.backward()
        self.assertTrue(torch.isfinite(cached.control.grad).all())

    def test_an_optimiser_step_invalidates_the_memo(self):
        # One model, toggled: comparing two separately trained copies would
        # measure CUDA reduction nondeterminism, which Adam amplifies to a full
        # learning rate wherever the true gradient of a control slot is zero.
        _, motion, birth, xyz = self.pair()
        velocity = torch.ones_like(xyz)
        optimizer = torch.optim.Adam(motion.parameters(), lr=5e-2)
        for step in range(4):
            with torch.no_grad():
                memoised = motion(.4, birth, velocity).clone()
                motion.basis_cache = False
                torch.testing.assert_close(motion(.4, birth, velocity), memoised,
                                           rtol=0, atol=0)
                motion.basis_cache = True
                torch.testing.assert_close(motion(.4, birth, velocity), memoised,
                                           rtol=0, atol=0)
            loss = motion(.1 * step, birth, velocity).square().sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    def test_writing_the_controls_in_place_invalidates_the_memo(self):
        plain, cached, birth, xyz = self.pair()
        velocity = torch.ones_like(xyz)
        with torch.no_grad():
            cached(.4, birth, velocity)
            replacement = plain.control.detach() * 1.7 + .3
            for motion in (plain, cached):
                motion.control.copy_(replacement)
            torch.testing.assert_close(cached(.4, birth, velocity),
                                       plain(.4, birth, velocity), rtol=0, atol=0)

    def test_clone_split_prune_invalidate_the_memo(self):
        plain, cached, birth, xyz = self.pair()
        velocity = torch.ones_like(xyz)
        ids = plain.point_ids
        parents = torch.stack([torch.where(plain.enabled[ids])[0][0],
                               torch.where(plain.piecewise_enabled[ids])[0][0],
                               torch.where(plain.static_enabled[ids])[0][0],
                               torch.where(ids == 0)[0][0]])
        with torch.no_grad():
            cached(.23, birth, velocity)
            self.assertIsNotNone(cached._reference_memo)
            for motion in (plain, cached):
                motion.append(motion.point_ids[parents])
            self.assertIsNone(cached._reference_memo)
            birth = torch.cat([birth, birth[parents]])
            velocity = torch.cat([velocity, velocity[parents]])
            torch.testing.assert_close(cached(.23, birth, velocity),
                                       plain(.23, birth, velocity), rtol=0, atol=0)
            keep = torch.ones(len(birth), dtype=torch.bool, device=birth.device)
            keep[parents] = False
            for motion in (plain, cached):
                motion.prune(keep)
            self.assertIsNone(cached._reference_memo)
            torch.testing.assert_close(cached(.23, birth[keep], velocity[keep]),
                                       plain(.23, birth[keep], velocity[keep]), rtol=0, atol=0)

    def test_a_new_birth_tensor_invalidates_the_memo(self):
        plain, cached, birth, xyz = self.pair()
        velocity = torch.ones_like(xyz)
        with torch.no_grad():
            cached(.23, birth, velocity)
            moved = birth * 0. + .1
            torch.testing.assert_close(cached(.23, moved, velocity),
                                       plain(.23, moved, velocity), rtol=0, atol=0)

    def test_the_memo_never_reaches_the_checkpoint(self):
        plain, cached, birth, xyz = self.pair()
        velocity = torch.ones_like(xyz)
        with torch.no_grad():
            cached(.23, birth, velocity)
        state = cached.state_dict()
        self.assertNotIn('_reference_memo', state)
        self.assertNotIn('basis_cache', state)
        self.assertEqual(sorted(state), sorted(plain.state_dict()))
        stream = io.BytesIO()
        torch.save(state, stream)
        stream.seek(0)
        restored = type(cached).from_state(torch.load(stream, weights_only=True))
        self.assertIsNone(restored._reference_memo)
        self.assertFalse(restored.basis_cache)
        with torch.no_grad():
            torch.testing.assert_close(restored(.23, birth, velocity),
                                       plain(.23, birth, velocity), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
