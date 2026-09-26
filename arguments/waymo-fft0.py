# waymo.py, but the per-Gaussian Fourier colour deformation is switched off.
# Used to test how much of AD-GS's time-varying colour is really a *global*
# camera-response effect that the photometric B-spline can explain on its own.
num_cam = 1
order_args = dict(
    # bspline(ctrl_pts, order - 1), poly, fft, slerp(ctrl_pts, order - 1)
    xyz = [None, 5, 0, 6, 0, 0],
    rotation=[0, 0, 0, 0, None, 5],
    shs=[0, 0, 0, 0, 0, 0],
    background=[0, 0, 0, 0, 0, 0],
)
