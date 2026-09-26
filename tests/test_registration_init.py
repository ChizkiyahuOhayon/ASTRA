"""v10: ICP-registered initialisation of the shared rigid trajectory."""
import math
import os
import unittest

import numpy as np
import torch

from utils.shared_motion import SharedMotion
from utils.rigid_registration import icp, kabsch, register_track, rotation_vector


def yawing_track(frames=10, points=160, sample=110, turn=1.2, speed=2.0, seed=3):
    """A rigid body that drives and yaws, sampled differently in every frame.

    Each Gaussian in this project has a single birth time, so the blob seen at
    one frame is not the blob seen at the next. Re-sampling here reproduces
    that, which is precisely the case blob-centroid differencing gets wrong.
    """
    rng = np.random.RandomState(seed)
    shape = rng.uniform(-.7, .7, (points, 3))  # dense enough for DBSCAN eps=0.5
    xyz, times = [], []
    for frame in range(frames):
        ratio = frame / (frames - 1)
        angle = turn * ratio
        rotation = np.array([[math.cos(angle), -math.sin(angle), 0.],
                             [math.sin(angle), math.cos(angle), 0.],
                             [0., 0., 1.]])
        subset = shape[rng.choice(points, sample, replace=False)]
        xyz.append(subset @ rotation.T + [speed * ratio, .4 * ratio, 0])
        times.extend([frame / 19] * sample)
    return (torch.tensor(np.concatenate(xyz), dtype=torch.float64),
            torch.tensor(times, dtype=torch.float64)[:, None], shape, turn, speed)


class RigidRegistrationTests(unittest.TestCase):
    def test_kabsch_recovers_an_exact_rigid_transform(self):
        rng = np.random.RandomState(1)
        source = rng.uniform(-1, 1, (25, 3))
        angle = .7
        rotation = np.array([[math.cos(angle), -math.sin(angle), 0.],
                             [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
        offset = np.array([1.5, -.3, .2])
        estimated, translation = kabsch(source, source @ rotation.T + offset)
        np.testing.assert_allclose(estimated, rotation, atol=1e-10)
        np.testing.assert_allclose(translation, offset, atol=1e-10)

    def test_kabsch_never_returns_a_reflection(self):
        rng = np.random.RandomState(2)
        source = rng.uniform(-1, 1, (30, 3))
        target = source * [1, 1, -1]          # a reflection, not a rotation
        rotation, _ = kabsch(source, target)
        self.assertGreater(np.linalg.det(rotation), 0)

    def test_rotation_vector_round_trips_through_kabsch(self):
        for angle in (0.0, 1e-9, .3, 2.5):
            rotation = np.array([[math.cos(angle), -math.sin(angle), 0.],
                                 [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
            vector = rotation_vector(rotation)
            self.assertTrue(np.isfinite(vector).all())
            np.testing.assert_allclose(vector[2], angle, atol=1e-6)

    def test_icp_aligns_resampled_shapes(self):
        rng = np.random.RandomState(4)
        shape = rng.uniform(-1.2, 1.2, (60, 3))
        angle = .3
        rotation = np.array([[math.cos(angle), -math.sin(angle), 0.],
                             [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
        source = shape[rng.choice(60, 45, replace=False)]
        target = shape[rng.choice(60, 45, replace=False)] @ rotation.T + [1.0, 0, 0]
        estimated, _, fitness = icp(source, target, threshold=.35)
        self.assertGreater(fitness, .5)
        self.assertLess(abs(rotation_vector(estimated)[2] - angle), .05)

    def test_a_bad_pair_degrades_to_the_centroid_offset(self):
        rng = np.random.RandomState(5)
        frames = [rng.uniform(-1, 1, (20, 3)), rng.uniform(-1, 1, (20, 3)) + [40, 0, 0]]
        offsets, rotations, fitness, _ = register_track(frames, threshold=.1)
        self.assertEqual(fitness[1], 0.0)
        np.testing.assert_allclose(rotations[1], np.zeros(3), atol=0)
        np.testing.assert_allclose(offsets[1], frames[1].mean(0) - frames[0].mean(0), atol=1e-9)

    def test_chained_registration_tracks_a_long_yaw(self):
        xyz, birth, _, turn, _ = yawing_track()
        times = birth.numpy().reshape(-1)
        frames = [xyz.numpy()[times == t] for t in np.unique(times)]
        _, rotations, fitness, _ = register_track(frames, threshold=.25)
        self.assertGreater(fitness[1:].mean(), .5)
        self.assertLess(abs(rotations[-1][2] - turn), .05)


class RegistrationInitTests(unittest.TestCase):
    def setUp(self):
        self.device = os.environ.get('ADGS_TEST_DEVICE', 'cpu')

    def build(self, registration_init):
        xyz, birth, shape, turn, speed = yawing_track()
        motion = SharedMotion.from_points(
            xyz, birth, .05, support_gate=True, bidirectional=True,
            singleton_static=True, rigid_rotation=True,
            registration_init=registration_init, registration_threshold=.25)
        motion.rigid_rotation = True
        return motion, xyz, birth, shape, turn, speed

    def test_the_flag_off_leaves_the_rotation_controls_at_identity(self):
        motion, xyz, birth, _, _, _ = self.build(False)
        torch.testing.assert_close(motion.rot_control,
                                   torch.zeros_like(motion.rot_control), rtol=0, atol=0)
        velocity = torch.zeros_like(xyz)
        plain = motion(.3, birth, velocity)
        torch.testing.assert_close(motion(.3, birth, velocity, xyz), plain, rtol=0, atol=0)

    def test_registration_initialises_a_real_rotation(self):
        motion, _, _, _, turn, _ = self.build(True)
        self.assertGreaterEqual(int(motion.enabled.sum()), 1)
        self.assertGreater(float(motion.rot_control.abs().max()), .3 * turn)
        self.assertTrue(torch.isfinite(motion.rot_control).all())

    def test_the_initialisation_alone_explains_the_yaw(self):
        # No training at all: compare how well each initialisation already
        # places frame 0's points at their true pose in every later frame.
        def residual(registration_init):
            motion, xyz, birth, shape, turn, speed = self.build(registration_init)
            born = (birth.reshape(-1) == 0.) & motion.enabled[motion.point_ids]
            self.assertGreater(int(born.sum()), 10)
            base = xyz[born]
            total = 0.
            for frame in range(1, 10):
                ratio = frame / 9
                angle = turn * ratio
                rotation = torch.tensor(
                    [[math.cos(angle), -math.sin(angle), 0.],
                     [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]],
                    dtype=torch.float64)
                # Where frame 0's own points truly are at this later time.
                target = base @ rotation.T + torch.tensor(
                    [speed * ratio, .4 * ratio, 0.], dtype=torch.float64)
                moved = (xyz + motion(frame / 19, birth, torch.zeros_like(xyz), xyz))[born]
                total += float((moved - target).square().mean())
            return total
        centroid = residual(False)
        registered = residual(True)
        self.assertLess(registered, centroid / 5)


if __name__ == '__main__':
    unittest.main()
