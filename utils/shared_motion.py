"""Support-matched tracklet translation with all-point local velocity residual.

Grouping reuses the project's geometric tracker; no ground-truth boxes are read.
Grouped single-frame tracklets carry no velocity evidence, so v7 pins them to
their birth anchor: neither a shared trajectory nor a per-point residual.
This is an experimental representation, not a validated quality improvement.
"""
import numpy as np
import torch
from torch import nn


def cubic_basis(time, controls=8):
    """Uniform cubic B-spline basis, with linear continuation outside [0, 1]."""
    intervals = controls - 3
    bounded = time.clamp(0, 1)
    cell = (bounded * intervals).long().clamp(max=intervals - 1)
    u = bounded * intervals - cell
    weights = torch.stack(((1-u)**3, 3*u**3-6*u**2+4,
                           -3*u**3+3*u**2+3*u+1, u**3), dim=-1) / 6
    derivative = torch.stack((-3*(1-u)**2, 9*u**2-12*u,
                              -9*u**2+6*u+3, 3*u**2), dim=-1) / 6
    weights = weights + (time-bounded)[..., None] * intervals * derivative
    index = cell[..., None] + torch.arange(4, device=time.device)
    return time.new_zeros((*time.shape, controls)).scatter(-1, index, weights)


def axis_angle_quaternion(vector):
    """Unit quaternion of a rotation vector, smooth and finite at zero.

    The squared norm is softened before the square root so that a zero rotation
    -- the initial state of every tracklet -- still has a defined derivative;
    the vector part keeps its exact half-angle gradient there.
    """
    angle = torch.sqrt((vector * vector).sum(-1, keepdim=True) + 1e-16)
    half = 0.5 * angle
    # sin(a/2)/a, taken from its series where the quotient would cancel badly.
    scale = torch.where(angle > 1e-4, torch.sin(half) / angle, 0.5 - angle * angle / 48)
    return torch.cat([torch.cos(half), vector * scale], dim=-1)


def quaternion_rotate(quaternion, vector):
    """Rotate a vector by a unit quaternion without forming a matrix."""
    real, imaginary = quaternion[..., :1], quaternion[..., 1:]
    cross = 2 * torch.cross(imaginary, vector, dim=-1)
    return vector + real * cross + torch.cross(imaginary, cross, dim=-1)


class SharedMotion(nn.Module):
    def __init__(self, control, start, duration, point_ids, enabled=None,
                 residual_all=True, linear_enabled=None, piecewise_enabled=None,
                 knot_time=None, knot_count=None, static_enabled=None,
                 rot_control=None, center=None, adgs_residual=False, adgs_ungated=False):
        super().__init__()
        if adgs_ungated and not adgs_residual:
            raise ValueError('ungated AD-GS motion needs the AD-GS residual basis')
        # Persist the residual semantics; the Python flags avoid a CUDA sync per frame.
        # adgs_ungated: supported tracklets keep the low-DoF shared trajectory plus
        # a per-point velocity; only points outside the support gate use AD-GS.
        self.adgs_residual = bool(adgs_residual)
        self.adgs_ungated = bool(adgs_ungated)
        self.register_buffer('_adgs_residual', torch.tensor(
            self.adgs_residual, device=control.device, dtype=torch.bool))
        self.register_buffer('_adgs_ungated', torch.tensor(
            self.adgs_ungated, device=control.device, dtype=torch.bool))
        self.control = nn.Parameter(control)
        # Rigid rotation shares the translation's spline basis and birth
        # anchoring. Zero controls mean identity, so a fresh model and a v1-v7
        # checkpoint both start numerically identical to translation-only.
        if rot_control is None:
            rot_control = torch.zeros_like(control)
        self.rot_control = nn.Parameter(rot_control)
        if center is None:
            center = torch.zeros_like(control[:, 0])
        self.register_buffer('center', center)
        self.register_buffer('start', start)
        self.register_buffer('duration', duration)
        self.register_buffer('point_ids', point_ids)
        if enabled is None:
            enabled = torch.ones_like(start, dtype=torch.bool)
            enabled[0] = False
        self.register_buffer('enabled', enabled)
        if linear_enabled is None:
            linear_enabled = torch.zeros_like(enabled)
        self.register_buffer('linear_enabled', linear_enabled)
        if piecewise_enabled is None:
            piecewise_enabled = torch.zeros_like(enabled)
        self.register_buffer('piecewise_enabled', piecewise_enabled)
        # Absent in v1-v6 checkpoints: those keep the historical all-residual
        # singleton semantics instead of being silently frozen.
        if static_enabled is None:
            static_enabled = torch.zeros_like(enabled)
        self.register_buffer('static_enabled', static_enabled)
        if knot_time is None:
            knot_time = torch.zeros_like(control[..., 0])
        self.register_buffer('knot_time', knot_time)
        if knot_count is None:
            knot_count = torch.zeros_like(start, dtype=torch.long)
        self.register_buffer('knot_count', knot_count)
        self.register_buffer('residual_all', torch.as_tensor(residual_all, device=control.device, dtype=torch.bool))
        # Engineering only, never stored in the checkpoint: the cached plan is
        # exactly the tensors the uncached path recomputes every call.
        self.basis_cache = False
        # Runtime switch, never stored: a checkpoint carries zero rotation
        # controls when it was trained without them, which is already identity.
        self.rigid_rotation = False
        self._plan = None
        self._plan_birth = None
        self._reference_memo = None

    @classmethod
    def from_points(cls, xyz, time, frame_gap, support_gate=False,
                    linear_fallback=False, piecewise_fallback=False,
                    bidirectional=False, singleton_static=False,
                    rigid_rotation=False, registration_init=False,
                    registration_threshold=0.5, adgs_residual=False, adgs_ungated=False):
        # The reader retains 30% of object points: use four neighbours instead
        # of the full-cloud probe's ten. Other association settings are reused.
        if bidirectional:
            from utils.bidirectional_linking import link_instances_bidirectional as link_instances
        else:
            from scripts.probe.probe_instance_clustering import link_instances

        points = xyz.detach().cpu().numpy().astype(np.float64)
        times = time.detach().cpu().numpy().reshape(-1)
        frames = np.rint(times * (round(1 / frame_gap) - 1)).astype(int)
        labels = link_instances(points, frames, eps=0.5, min_samples=4,
                                gate=2.5, memory=10)
        count = int(labels.max() + 2) if len(labels) else 1
        control = np.zeros((count, 8, 3), dtype=np.float32)
        rot_control = np.zeros((count, 8, 3), dtype=np.float32)
        start = np.zeros(count, dtype=np.float32)
        duration = np.full(count, frame_gap, dtype=np.float32)
        enabled = np.ones(count, dtype=bool)
        enabled[0] = False
        linear_enabled = np.zeros(count, dtype=bool)
        piecewise_enabled = np.zeros(count, dtype=bool)
        # Group 0 collects DBSCAN noise and unassigned points; it is never
        # static, so those points keep their own local residual velocity.
        static_enabled = np.zeros(count, dtype=bool)
        center = np.zeros((count, 3), dtype=np.float32)
        knot_time = np.full((count, control.shape[1]), 2., dtype=np.float32)
        knot_count = np.zeros(count, dtype=np.int64)
        if linear_fallback and piecewise_fallback:
            raise ValueError('choose one short-track fallback')
        curvature = np.diff(np.eye(8), n=2, axis=0)
        if registration_init:
            from utils.rigid_registration import register_track
        for group in range(1, count):
            selected = labels == group - 1
            observed = np.unique(times[selected])
            # One observed frame gives no velocity evidence; no new threshold.
            static_enabled[group] = singleton_static and len(observed) < 2
            # Eight free spline controls need at least eight observations.
            # The initialization's curvature prior is not training evidence.
            enabled[group] = (not support_gate or len(observed) >= control.shape[1]) \
                and not static_enabled[group]
            linear_enabled[group] = support_gate and linear_fallback and 2 <= len(observed) < control.shape[1]
            piecewise_enabled[group] = support_gate and piecewise_fallback and 2 <= len(observed) < control.shape[1]
            start[group] = observed[0]
            duration[group] = max(observed[-1] - observed[0], frame_gap)
            if len(observed) < 2:
                continue
            blobs = [points[selected & (times == t)] for t in observed]
            centers = np.stack([blob.mean(0) for blob in blobs])
            # Pivot of the rigid rotation: the tracklet centroid at its start.
            center[group] = centers[0]
            rotations = None
            confidence = np.ones(len(observed))
            if registration_init:
                # A frame's blob is a different sample of Gaussians from the
                # previous frame's, so its centroid moves without the object
                # moving. Aligning the shapes is invariant to that and returns
                # the rotation too. Frames that register badly fall back to the
                # centroid offset, which is exactly the old behaviour.
                offsets, rotations, confidence, _ = register_track(
                    blobs, registration_threshold)
            else:
                offsets = centers - centers[0]
            local_time = torch.from_numpy(
                (observed.astype(np.float64) - start[group]) / duration[group])
            basis = cubic_basis(local_time).numpy()
            # Smooth centroid initialization; zero curvature leaves constant
            # velocity unpenalized. This is initialization, not an added loss.
            weight = np.concatenate([np.clip(confidence, 0.05, 1.0), np.ones(6)])[:, None]
            design = np.concatenate([basis, 0.1 * curvature]) * weight
            target = np.concatenate([offsets, np.zeros((6, 3))]) * weight
            control[group] = np.linalg.lstsq(design, target, rcond=None)[0]
            if rotations is not None and rigid_rotation:
                rot_target = np.concatenate([rotations, np.zeros((6, 3))]) * weight
                rot_control[group] = np.linalg.lstsq(design, rot_target, rcond=None)[0]
            if linear_enabled[group]:
                target = offsets
                slope, offset = np.linalg.lstsq(
                    np.stack([local_time.numpy(), np.ones(len(observed))], axis=1),
                    target, rcond=None)[0]
                control[group, 0] = offset
                control[group, -1] = offset + slope
            if piecewise_enabled[group]:
                knot_count[group] = len(observed)
                knot_time[group, :len(observed)] = local_time.numpy()
                control[group, :len(observed)] = offsets
        result = cls(torch.as_tensor(control, device=xyz.device, dtype=xyz.dtype),
                     torch.as_tensor(start, device=xyz.device, dtype=xyz.dtype),
                     torch.as_tensor(duration, device=xyz.device, dtype=xyz.dtype),
                     torch.as_tensor(labels + 1, device=xyz.device, dtype=torch.long),
                     torch.as_tensor(enabled, device=xyz.device),
                     linear_enabled=torch.as_tensor(linear_enabled, device=xyz.device),
                     piecewise_enabled=torch.as_tensor(piecewise_enabled, device=xyz.device),
                     knot_time=torch.as_tensor(knot_time, device=xyz.device, dtype=xyz.dtype),
                     knot_count=torch.as_tensor(knot_count, device=xyz.device),
                     static_enabled=torch.as_tensor(static_enabled, device=xyz.device),
                     center=torch.as_tensor(center, device=xyz.device, dtype=xyz.dtype),
                     rot_control=torch.as_tensor(rot_control, device=xyz.device, dtype=xyz.dtype),
                     adgs_residual=adgs_residual, adgs_ungated=adgs_ungated)
        result.rigid_rotation = rigid_rotation
        print('Shared motion: %d tracklets, %d unassigned / %d points' %
              (count - 1, int((labels < 0).sum()), len(labels)))
        if support_gate:
            active = enabled | linear_enabled | piecewise_enabled
            print('Shared support gate: %d enabled tracklets, %d residual-only points' %
                  (int(enabled.sum()), int((~active[labels + 1]).sum())))
        if linear_fallback:
            print('Shared linear fallback: %d tracklets, %d points' %
                  (int(linear_enabled.sum()), int(linear_enabled[labels + 1].sum())))
        if piecewise_fallback:
            print('Shared piecewise fallback: %d tracklets, %d points' %
                  (int(piecewise_enabled.sum()), int(piecewise_enabled[labels + 1].sum())))
        if singleton_static:
            print('Shared singleton static: %d tracklets, %d birth-anchored points' %
                  (int(static_enabled.sum()), int(static_enabled[labels + 1].sum())))
        if rigid_rotation:
            print('Shared rigid rotation: %d tracklets, %d points, |rot| max %.3f' %
                  (int(enabled.sum()), int(enabled[labels + 1].sum()),
                   float(np.abs(rot_control).max())))
        if registration_init:
            print('Shared registration init: threshold %.2f m' % registration_threshold)
        return result

    def _piecewise_plan(self, time, ids):
        """Knot interval and interpolation weight; independent of the controls."""
        knots = self.knot_time[ids]
        count = self.knot_count[ids]
        right = (time[:, None] >= knots).sum(-1).clamp(min=1)
        right = torch.minimum(right, count - 1)
        left = right - 1
        t0 = knots.gather(1, left[:, None]).squeeze(1)
        t1 = knots.gather(1, right[:, None]).squeeze(1)
        return left, right, ((time - t0) / (t1 - t0))[:, None]

    def _piecewise_eval(self, ids, left, right, weight):
        controls = self.control[ids]
        c0 = controls.gather(1, left[:, None, None].expand(-1, 1, 3)).squeeze(1)
        c1 = controls.gather(1, right[:, None, None].expand(-1, 1, 3)).squeeze(1)
        return c0 + weight * (c1 - c0)

    def _piecewise(self, time, ids):
        return self._piecewise_eval(ids, *self._piecewise_plan(time, ids))

    def _build_birth_plan(self, birth_time):
        """Route index sets, birth basis and piecewise weights.

        Everything here is fixed by group ID, birth time and knot time; nothing
        depends on the learnable controls, so a training step never stales it.
        The 8-wide birth basis is stored dense on purpose: a 4-wide support
        gather would reassociate the reduction and stop being bitwise equal.
        """
        ids = self.point_ids
        reference_time = (birth_time.reshape(-1) - self.start[ids]) / self.duration[ids]
        cubic_points = torch.where(self.enabled[ids])[0]
        linear_points = torch.where(self.linear_enabled[ids])[0]
        piecewise_points = torch.where(self.piecewise_enabled[ids])[0]
        # Per-point route masks and the group-level piecewise index set are also
        # fixed by the route buffers, so each cached call drops one more
        # `nonzero` synchronisation and a handful of elementwise kernels.
        active = (self.enabled | self.linear_enabled | self.piecewise_enabled) & ~self.static_enabled
        return (cubic_points, ids[cubic_points], cubic_basis(reference_time[cubic_points]),
                linear_points, ids[linear_points], reference_time[linear_points, None],
                piecewise_points, ids[piecewise_points],
                self._piecewise_plan(reference_time[piecewise_points], ids[piecewise_points]),
                torch.where(self.piecewise_enabled)[0], active[ids, None],
                (~self.static_enabled[ids, None]), (~(ids > 0)[:, None]))

    def birth_plan(self, birth_time):
        if not self.basis_cache:
            return self._build_birth_plan(birth_time)
        # Birth times are a fixed buffer that clone/split/prune replace with a
        # new tensor, so object identity is a sound and O(1) validity test.
        if self._plan is None or birth_time is not self._plan_birth:
            self._plan = self._build_birth_plan(birth_time)
            self._plan_birth = birth_time
        return self._plan

    def _reference(self, plan, fallback_velocity):
        (cubic_points, cubic_ids, reference_basis, linear_points, linear_ids,
         linear_time, piecewise_points, piecewise_ids, piecewise_plan) = plan[:9]
        reference = torch.zeros_like(fallback_velocity)
        reference = reference.index_copy(
            0, cubic_points,
            (reference_basis[..., None] * self.control[cubic_ids]).sum(-2))
        linear = self.control[linear_ids, 0] + linear_time * (
            self.control[linear_ids, -1] - self.control[linear_ids, 0])
        reference = reference.index_copy(0, linear_points, linear)
        return reference.index_copy(
            0, piecewise_points,
            self._piecewise_eval(piecewise_ids, *piecewise_plan))

    def frozen_reference(self, plan, fallback_velocity):
        """Birth-anchor offsets, reused while the controls cannot change.

        At inference the controls are frozen, so every Gaussian's birth-anchor
        offset is the same tensor on every frame and one evaluation serves the
        whole sequence. Only used when autograd is off, so a training step can
        never read a stale value or a freed graph; validity is an O(1) test of
        the plan object, the control tensor's identity and its in-place version
        counter, which every optimiser step and every `copy_` bumps.
        """
        if torch.is_grad_enabled():
            # Every parameter update is preceded by a forward with autograd on,
            # so dropping the memo here is what actually guarantees freshness.
            # `control._version` is only a cheap extra guard: torch 2.0.1's
            # foreach Adam updates parameters in place without bumping it.
            self._reference_memo = None
            return self._reference(plan, fallback_velocity)
        if not self.basis_cache:
            return self._reference(plan, fallback_velocity)
        memo = self._reference_memo
        if (memo is not None and memo[0] is plan and memo[1] is self.control
                and memo[2] == self.control._version
                and memo[3].shape == fallback_velocity.shape
                and memo[3].dtype == fallback_velocity.dtype):
            return memo[3]
        value = self._reference(plan, fallback_velocity)
        # A list, not an attribute per field: nn.Module would register a bare
        # Parameter attribute as a parameter and leak it into the state dict.
        self._reference_memo = [plan, self.control, self.control._version, value]
        return value

    def relative_rotation(self, time, birth_time, plan=None):
        """Unit quaternion taking each point from its birth pose to time t.

        Uses the translation's own basis and birth anchoring, so a tracklet
        whose rotation controls are zero returns the identity for every point
        and every query time. Only tracklets that cleared the support gate
        rotate; short and single-frame tracklets have no rotation evidence, so
        the work and the cached birth basis are shared with the cubic route.
        """
        plan = self.birth_plan(birth_time) if plan is None else plan
        cubic_points, cubic_ids, reference_basis = plan[0], plan[1], plan[2]
        local_time = (time - self.start) / self.duration
        current = (cubic_basis(local_time)[..., None] * self.rot_control).sum(-2)
        birth = (reference_basis[..., None] * self.rot_control[cubic_ids]).sum(-2)
        vector = self.rot_control.new_zeros((len(self.point_ids), 3))
        vector = vector.index_copy(0, cubic_points, current[cubic_ids] - birth)
        return axis_angle_quaternion(vector)

    def active_points(self):
        """Per-point mask of Gaussians carried by a shared trajectory."""
        active = (self.enabled | self.linear_enabled | self.piecewise_enabled) & ~self.static_enabled
        return active[self.point_ids]

    def rotation_bias(self, time, birth_time):
        """Per-point rotation to compose onto the Gaussian orientation."""
        if not self.rigid_rotation:
            return None
        return self.relative_rotation(time, birth_time)

    def forward(self, time, birth_time, fallback_velocity, xyz=None, residual_displacement=None):
        if self.adgs_residual and residual_displacement is None:
            raise ValueError('AD-GS residual mode requires the evaluated displacement')
        # Evaluate current time once per tracklet; birth-time anchoring keeps
        # each observed point at its own reference position at initialization.
        local_time = (time - self.start) / self.duration
        current_basis = cubic_basis(local_time)
        current = (current_basis[..., None] * self.control).sum(-2)
        linear = self.control[:, 0] + local_time[:, None] * (self.control[:, -1] - self.control[:, 0])
        current = torch.where(self.linear_enabled[:, None], linear, current)
        plan = self.birth_plan(birth_time)
        piecewise_groups = plan[9]
        current = current.index_copy(
            0, piecewise_groups, self._piecewise(local_time[piecewise_groups], piecewise_groups))
        reference = self.frozen_reference(plan, fallback_velocity)
        # A grouped single-frame tracklet gets neither the shared trajectory
        # nor its own residual; unassigned and noise points keep the residual.
        active, not_static, unassigned = plan[10], plan[11], plan[12]
        shared = (current[self.point_ids] - reference) * active
        velocity = fallback_velocity * (time - birth_time)
        fallback = residual_displacement if self.adgs_residual else velocity
        if self.adgs_ungated:
            fallback = torch.where(active, velocity, fallback)
        translation = shared + fallback * ((self.residual_all | unassigned) & not_static)
        if not self.rigid_rotation or xyz is None:
            return translation
        # Rigid term: the point's offset from the tracklet centre at its own
        # birth, carried by the relative rotation. Identity controls cancel it
        # exactly, so this is a strict superset of the translation-only model.
        offset = xyz - self.center[self.point_ids] - reference
        rotated = quaternion_rotate(self.relative_rotation(time, birth_time, plan), offset)
        return translation + (rotated - offset) * active

    def append(self, parent_ids):
        self.point_ids = torch.cat([self.point_ids, parent_ids])
        self._plan = self._plan_birth = self._reference_memo = None

    def prune(self, keep):
        self.point_ids = self.point_ids[keep]
        self._plan = self._plan_birth = self._reference_memo = None

    def _apply(self, *args, **kwargs):
        # Buffers may change device or dtype; no cached tensor may survive it.
        self._plan = self._plan_birth = self._reference_memo = None
        return super()._apply(*args, **kwargs)

    @classmethod
    def from_state(cls, state, legacy_residual=None):
        # Historical v1/v2 have identical keys but different residual semantics.
        # Require an explicit choice rather than silently corrupting old renders.
        residual_all = state.get('residual_all')
        if residual_all is None:
            if legacy_residual not in ('v1', 'v2'):
                raise ValueError('Legacy shared checkpoint: specify --shared_motion_legacy v1 or v2')
            residual_all = legacy_residual == 'v1'
        return cls(state['control'], state['start'], state['duration'], state['point_ids'],
                   state.get('enabled'), residual_all, state.get('linear_enabled'),
                   state.get('piecewise_enabled'), state.get('knot_time'),
                   state.get('knot_count'), state.get('static_enabled'),
                   state.get('rot_control'), state.get('center'),
                   adgs_residual=state.get('_adgs_residual', False),
                   adgs_ungated=state.get('_adgs_ungated', False))
