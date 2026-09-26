import os
import torch
import torch.nn as nn
from utils.func_utils import get_func_result, get_param_num


class PhotometricSpline:
    """Time-continuous camera response.

    A driving camera runs auto-exposure and auto-white-balance, so the same radiance
    is recorded with a different gain and black level at every timestamp.  We model
    that with the *same* B-spline basis the motion model already uses:

        I_obs(t) = exp(a(t)) . I_render + b(t),      [a(t), b(t)] = B-spline(t)

    Because the response is a smooth function of time rather than a per-image latent
    code, it is defined at held-out timestamps too -- nothing is optimised at test
    time and no test pixel is ever observed.  Zero-initialised, so the module is the
    identity at iteration 0 and the whole thing collapses back to AD-GS when
    ``ctrl_pts == 0``.
    """

    def __init__(self, ctrl_pts: int = 0, order: int = 3, num_cam: int = 1, frame_num: int = 0,
                 downsample_ratio: int = 3, center: bool = True):
        if ctrl_pts < 0:   # auto: one control point every `downsample_ratio` frames
            ctrl_pts = max(order + 1, frame_num // (num_cam * downsample_ratio))
        self.ctrl_pts = ctrl_pts
        self.enabled = ctrl_pts > 0
        self.num_cam = num_cam
        self.optimizer = None
        self.param = None
        self.mode = 'spline'
        self.const = None
        # Centre the control points so the curve averages to zero over the sequence.
        # Without this the response and the radiance field share an exact affine gauge:
        # any global (gain, bias) can be absorbed either way at zero training cost, and
        # the optimiser happily drifts into it (measured: bias +0.13, i.e. 13% of range),
        # which then clips against the [0,1] range at evaluation time.
        # Centred, the module can only express *relative* photometric drift, which is the
        # only thing it is supposed to model.
        self.center = center
        if not self.enabled:
            self.order_args = None
            return
        order = min(order, ctrl_pts - 1)
        # [bspline_ctrl_pts, bspline_order, poly, fft, quat_ctrl_pts, quat_order]
        self.order_args = [ctrl_pts, order, 0, 0, 0, 0]
        num_param = get_param_num(self.order_args)
        param = torch.zeros((num_cam, 6, num_param), dtype=torch.float32, device='cuda')
        self.param = nn.Parameter(param.requires_grad_(True))

    def get_response(self, time, cam_id: int = 0):
        """Returns (gain, bias), each broadcastable to a 3xHxW image."""
        param = self.param - self.param.mean(dim=-1, keepdim=True) if self.center else self.param
        coef = get_func_result(float(time), param, self.order_args)  # num_cam, 6
        coef = coef[cam_id]
        return torch.exp(coef[:3]).reshape(3, 1, 1), coef[3:].reshape(3, 1, 1)

    def freeze(self, mode, times=None):
        """Evaluation-time behaviour of the response curve.

        'spline'   -- evaluate the curve at the view's own timestamp (default)
        'mean'     -- use the curve's average over `times` (the training frames), i.e. a
                      single global calibration with no interpolation involved
        'identity' -- gain 1, bias 0
        The three share one set of trained weights, so they can be compared by
        re-rendering a finished checkpoint -- no retraining needed.
"""
        self.mode = mode
        self.const = None
        if not self.enabled or mode == 'spline':
            return
        if mode == 'identity':
            self.const = (torch.ones(3, 1, 1, device='cuda'), torch.zeros(3, 1, 1, device='cuda'))
        elif mode == 'mean':
            assert times is not None and len(times) > 0
            with torch.no_grad():
                gs, bs = zip(*[self.get_response(t) for t in times])
                self.const = (torch.stack(gs).mean(0), torch.stack(bs).mean(0))
        else:
            raise ValueError('unknown photo_test_mode: ' + mode)

    def apply(self, image, viewpoint_camera):
        if not self.enabled:
            return image
        if self.const is not None:
            return image * self.const[0] + self.const[1]
        gain, bias = self.get_response(viewpoint_camera.time, getattr(viewpoint_camera, 'cam_id', 0))
        return image * gain + bias

    def training_setup(self, training_args):
        if not self.enabled:
            return
        self.optimizer = torch.optim.Adam(
            [{'params': [self.param], 'lr': training_args.photo_lr, "name": "photo"}], lr=0.0, eps=1e-15)

    def step(self):
        if self.enabled:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

    def save_weights(self, path):
        if self.enabled:
            torch.save(self.param, path)

    def load_weights(self, path):
        if self.enabled and os.path.exists(path):
            self.param = nn.Parameter(torch.load(path, map_location='cuda').requires_grad_(True))
