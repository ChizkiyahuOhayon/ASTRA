import io
import os
import unittest

import numpy as np
import torch

from scripts.probe.probe_instance_clustering import link_instances
from utils.bidirectional_linking import (cluster_blobs, link_blob_edges,
                                         mutual_components, link_instances_bidirectional)
from utils.shared_motion import SharedMotion


class BidirectionalLinkingTests(unittest.TestCase):
    def cloud(self, positions, frames=None):
        if frames is None:
            frames = range(len(positions))
        return (np.repeat(np.array(positions, dtype=float)[:, None] * [1., 0., 0.], 4, axis=0),
                np.repeat(np.array(list(frames)), 4))

    def labels(self, xyz, frames, reverse=False):
        fn = link_instances_bidirectional if reverse else link_instances
        return fn(xyz, frames, .5, 4, 2.5, 10)

    def test_mutual_direct_edges_reject_one_sided_link_without_losing_points(self):
        xyz, frames = self.cloud([0., 1., 4.])
        blobs = cluster_blobs(xyz, frames, .5, 4)
        components, forward, backward = mutual_components(blobs, 2.5, 10)
        self.assertEqual(forward, {(0, 1), (1, 2)})
        self.assertEqual(backward, {(0, 1)})
        self.assertEqual(components.tolist(), [0, 0, 1])
        self.assertEqual(self.labels(xyz, frames).tolist(), [0] * 12)
        labels = self.labels(xyz, frames, True)
        self.assertEqual(labels.tolist(), [0] * 8 + [1] * 4)
        self.assertTrue((labels >= 0).all())

    def test_reversing_time_preserves_partition(self):
        xyz, frames = self.cloud([0., 1., 4., 4.5, 5.])
        a = self.labels(xyz, frames, True)
        b = self.labels(xyz, -frames, True)
        np.testing.assert_array_equal(a[:, None] == a, b[:, None] == b)
        reverse_rows = np.arange(len(xyz))[::-1]
        b = self.labels(xyz[reverse_rows], frames[reverse_rows], True)[reverse_rows]
        np.testing.assert_array_equal(a[:, None] == a, b[:, None] == b)

    def test_occlusion_edge_uses_exact_memory_boundary(self):
        for gap, expected in [(10, [0] * 8), (11, [0] * 4 + [1] * 4)]:
            xyz, frames = self.cloud([0., 0.], [0, gap])
            self.assertEqual(self.labels(xyz, frames, True).tolist(), expected)

    def test_empty_noise_and_isolated_blobs_are_retained(self):
        self.assertEqual(len(self.labels(np.empty((0, 3)), np.array([]), True)), 0)
        xyz, frames = self.cloud([0., 100.], [0, 20])
        xyz = np.concatenate([xyz, [[50., 50., 50.]]])
        frames = np.append(frames, 0)
        self.assertEqual(self.labels(xyz, frames, True).tolist(), [0] * 4 + [1] * 4 + [-1])

    def test_each_pass_reproduces_legacy_direct_edges_including_ties(self):
        xyz, frames = self.cloud([0., 2., 1., 2., 3., 3.], [0, 0, 1, 2, 3, 13])
        blobs = cluster_blobs(xyz, frames, .5, 4)
        for reverse in (False, True):
            old = self.labels(xyz, -frames if reverse else frames)
            expected = set()
            for group in np.unique(old[old >= 0]):
                nodes = [i for i, (_, idx, _) in enumerate(blobs) if old[idx[0]] == group]
                nodes.sort(key=lambda i: blobs[i][0], reverse=reverse)
                expected.update(tuple(sorted(pair)) for pair in zip(nodes[:-1], nodes[1:]))
            self.assertEqual(link_blob_edges(blobs, 2.5, 10, reverse), expected)

    def agreement_motion(self, dtype):
        rng = np.random.RandomState(9)
        shape = rng.uniform(-.08, .08, (20, 3))
        xyz, times = [], []
        for frame in range(12):
            t = frame / 19
            for group, length in enumerate([7, 12, 1]):
                if frame < length:
                    xyz.append(shape + [10 * group + t, .2*t, 0])
                    times.extend([t] * len(shape))
        xyz.append(np.array([[100., 100., 100.]]))
        times.append(0.)
        xyz = torch.tensor(np.concatenate(xyz), dtype=dtype)
        birth = torch.tensor(times, dtype=dtype)[:, None]
        v5 = SharedMotion.from_points(xyz, birth, .05, support_gate=True, piecewise_fallback=True)
        v6 = SharedMotion.from_points(xyz, birth, .05, support_gate=True,
                                      piecewise_fallback=True, bidirectional=True)
        for k, value in v5.state_dict().items():
            torch.testing.assert_close(value, v6.state_dict()[k], rtol=0, atol=0)
        device = os.environ.get('ADGS_TEST_DEVICE', 'cpu')
        return v5.to(device), v6.to(device), birth.to(device), xyz.to(device)

    def test_agreement_is_exact_v5_forward_and_control_gradient(self):
        # Required only for the test's bitwise comparison of duplicate CUDA
        # graphs, not a change to production training's determinism settings.
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            if os.environ.get('ADGS_TEST_DEVICE') == 'cuda':
                torch.use_deterministic_algorithms(True)
            for dtype in (torch.float32, torch.float64):
                v5, v6, birth, xyz = self.agreement_motion(dtype)
                velocity = torch.ones_like(xyz) * .2
                for time in [-.1, .0, .23, .5, 1.1]:
                    a, b = v5(time, birth, velocity), v6(time, birth, velocity)
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                    ga, = torch.autograd.grad(a.square().sum(), v5.control)
                    gb, = torch.autograd.grad(b.square().sum(), v6.control)
                    torch.testing.assert_close(ga, gb, rtol=0, atol=0)
        finally:
            torch.use_deterministic_algorithms(previous)

    def test_split_tracks_keep_v5_motion_routing(self):
        xyz, frames = self.cloud([0., 1., 4.])
        birth = torch.tensor(frames / 19, dtype=torch.float64)[:, None]
        motion = SharedMotion.from_points(torch.tensor(xyz), birth, .05, support_gate=True,
                                          piecewise_fallback=True, bidirectional=True)
        self.assertEqual(motion.enabled.tolist(), [False, False, False])
        self.assertEqual(motion.piecewise_enabled.tolist(), [False, True, False])
        self.assertEqual(motion.knot_count.tolist(), [0, 2, 0])
        self.assertTrue(motion.residual_all)
        velocity = torch.ones_like(torch.tensor(xyz))
        out = motion(.2, birth, velocity)
        singleton = motion.point_ids == 2
        torch.testing.assert_close(out[singleton], (velocity * (.2 - birth))[singleton])

    def test_v6_all_routes_survive_append_prune_checkpoint(self):
        _, motion, birth, xyz = self.agreement_motion(torch.float64)
        velocity = torch.ones_like(xyz)
        ids = motion.point_ids
        active = motion.enabled | motion.piecewise_enabled
        parents = torch.stack([torch.where(motion.enabled[ids])[0][0],
                               torch.where(motion.piecewise_enabled[ids])[0][0],
                               torch.where((ids > 0) & ~active[ids])[0][0],
                               torch.where(ids == 0)[0][0]])
        original = motion(.23, birth, velocity).detach()
        # Clone, then split-repeat, followed by pruning the original parents.
        lineage = torch.arange(len(ids), device=ids.device)
        for source in (parents, parents.repeat(2)):
            motion.append(motion.point_ids[source])
            birth = torch.cat([birth, birth[source]])
            velocity = torch.cat([velocity, velocity[source]])
            lineage = torch.cat([lineage, lineage[source]])
        keep = torch.ones(len(lineage), dtype=torch.bool, device=ids.device)
        keep[parents] = False
        motion.prune(keep)
        stream = io.BytesIO()
        torch.save(motion.state_dict(), stream)
        stream.seek(0)
        restored = SharedMotion.from_state(torch.load(stream, weights_only=True))
        output = restored(.23, birth[keep], velocity[keep])
        torch.testing.assert_close(output, original[lineage[keep]], rtol=0, atol=0)
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(restored.control.grad).all())


if __name__ == '__main__':
    unittest.main()
