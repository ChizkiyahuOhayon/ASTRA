"""Shared-motion composition; the real AD-GS basis is checked by CUDA lifecycle."""
import io
import unittest

import torch

from utils.shared_motion import SharedMotion


class ADGSResidualTests(unittest.TestCase):
    def make_motion(self, enabled=True):
        motion = SharedMotion(torch.randn(3, 8, 3, dtype=torch.float64),
                              torch.zeros(3, dtype=torch.float64),
                              torch.ones(3, dtype=torch.float64),
                              torch.tensor([0, 1, 2]),
                              torch.tensor([False, enabled, False]),
                              adgs_residual=True)
        birth = torch.tensor([[.1], [.3], [.6]], dtype=torch.float64)
        return motion, birth

    def test_disabled_shared_preserves_residual_exactly(self):
        motion, birth = self.make_motion(False)
        residual = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
        out = motion(.4, birth, torch.zeros_like(residual), residual_displacement=residual)
        torch.testing.assert_close(out, residual, rtol=0, atol=0)
        out.sum().backward()
        torch.testing.assert_close(residual.grad, torch.ones_like(residual))
        self.assertEqual(motion.control.grad.abs().sum(), 0)

    def test_additive_shared_and_gradients(self):
        motion, birth = self.make_motion()
        state = dict(motion.state_dict())
        del state['_adgs_residual']
        old = SharedMotion.from_state(state)
        residual = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
        zero = torch.zeros_like(residual)
        shared = old(.4, birth, zero)
        out = motion(.4, birth, zero, residual_displacement=residual)
        torch.testing.assert_close(out, shared + residual, rtol=0, atol=0)
        out.sum().backward()
        shared.sum().backward()
        torch.testing.assert_close(motion.control.grad, old.control.grad, rtol=0, atol=0)
        torch.testing.assert_close(residual.grad, torch.ones_like(residual))

    def test_cache_and_topology_preserve_composition(self):
        motion, birth = self.make_motion()
        for change in ('initial', 'append', 'prune'):
            if change == 'append':
                motion.append(torch.tensor([1, 0]))
                birth = torch.cat([birth, birth[:2]])
            if change == 'prune':
                keep = torch.tensor([True, False, True, True, True])
                motion.prune(keep)
                birth = birth[keep]
            residual = torch.randn(len(birth), 3, dtype=torch.float64, requires_grad=True)
            outputs = []
            for cache in (False, True, True):
                motion.basis_cache = cache
                out = motion(.4, birth, torch.zeros_like(residual), residual_displacement=residual)
                grads = torch.autograd.grad(out.square().sum(), (motion.control, residual))
                outputs.append((out, *grads))
            for actual in outputs[1:]:
                for a, b in zip(outputs[0], actual):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_checkpoint_and_legacy_semantics(self):
        motion, birth = self.make_motion()
        buf = io.BytesIO()
        torch.save(motion.state_dict(), buf)
        buf.seek(0)
        restored = SharedMotion.from_state(torch.load(buf, weights_only=True))
        self.assertTrue(restored.adgs_residual)
        residual = torch.randn(3, 3, dtype=torch.float64)
        for t in (-.2, .4, 1.2):
            torch.testing.assert_close(
                restored(t, birth, residual, residual_displacement=residual),
                motion(t, birth, residual, residual_displacement=residual), rtol=0, atol=0)
        state = dict(motion.state_dict())
        del state['_adgs_residual']
        self.assertFalse(SharedMotion.from_state(state).adgs_residual)

    def test_new_mode_cannot_silently_render_as_velocity(self):
        motion, birth = self.make_motion()
        with self.assertRaisesRegex(ValueError, 'evaluated displacement'):
            motion(.4, birth, torch.zeros(3, 3))


class ADGSUngatedTests(unittest.TestCase):
    """Supported points keep the translation-only model; the rest keep AD-GS."""

    def make_pair(self):
        args = (torch.randn(3, 8, 3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64),
                torch.ones(3, dtype=torch.float64), torch.tensor([0, 1, 2, 1]),
                torch.tensor([False, True, False]))
        stable = SharedMotion(*args)
        hybrid = SharedMotion(*args, adgs_residual=True, adgs_ungated=True)
        birth = torch.tensor([[.1], [.3], [.6], [.2]], dtype=torch.float64)
        return stable, hybrid, birth

    def test_routes_match_their_reference_models_exactly(self):
        stable, hybrid, birth = self.make_pair()
        velocity = torch.randn(4, 3, dtype=torch.float64)
        residual = torch.randn(4, 3, dtype=torch.float64)
        active = hybrid.active_points()
        torch.testing.assert_close(active, torch.tensor([False, True, False, True]))
        for t in (-.2, .4, 1.2):
            out = hybrid(t, birth, velocity, residual_displacement=residual)
            torch.testing.assert_close(out[active], stable(t, birth, velocity)[active], rtol=0, atol=0)
            torch.testing.assert_close(out[~active], residual[~active], rtol=0, atol=0)

    def test_gradients_follow_the_route(self):
        _, hybrid, birth = self.make_pair()
        velocity = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        residual = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        hybrid(.4, birth, velocity, residual_displacement=residual).sum().backward()
        active = hybrid.active_points()
        self.assertEqual(velocity.grad[~active].abs().sum(), 0)
        self.assertEqual(residual.grad[active].abs().sum(), 0)
        self.assertGreater(velocity.grad[active].abs().sum(), 0)

    def test_checkpoint_keeps_the_route(self):
        _, hybrid, birth = self.make_pair()
        buf = io.BytesIO()
        torch.save(hybrid.state_dict(), buf)
        buf.seek(0)
        restored = SharedMotion.from_state(torch.load(buf, weights_only=True))
        self.assertTrue(restored.adgs_ungated and restored.adgs_residual)
        velocity, residual = torch.randn(4, 3, dtype=torch.float64), torch.randn(4, 3, dtype=torch.float64)
        torch.testing.assert_close(restored(.4, birth, velocity, residual_displacement=residual),
                                   hybrid(.4, birth, velocity, residual_displacement=residual),
                                   rtol=0, atol=0)

    def test_ungated_requires_residual_basis(self):
        with self.assertRaisesRegex(ValueError, 'residual basis'):
            SharedMotion(torch.zeros(2, 8, 3), torch.zeros(2), torch.ones(2),
                         torch.tensor([0, 1]), adgs_ungated=True)


if __name__ == '__main__':
    unittest.main()
