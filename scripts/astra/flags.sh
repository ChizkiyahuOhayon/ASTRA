#!/bin/bash
# ASTRA component flags (paper Sec. 3). Source this file; every component is a plain IDSplat config flag.

# Trajectory-consistent pose refinement (Sec. 3.2): per-frame SO(3)xR3 camera corrections, carried to unseen
# timestamps by interpolating the corrections of the bracketing training frames. Lidar poses stay fixed.
POSE=(--pipeline.model.camera_optimizer.mode SO3xR3
      --pipeline.model.camopt_cameras_only True
      --pipeline.model.camopt_interp_eval True)

# Lens-consistent rendering (Sec. 3.3): render through the calibrated lens (no-op on rectified datasets).
LENS=(--pipeline.model.lens_distortion True)

# Virtual-view consistency (Sec. 3.4): virtual cameras between neighbouring training frames, half of them moved
# off the trajectory (yaw +-20 deg, sideways +-1 m), supervised by agreeing warps of the two real neighbours.
VVC=(--pipeline.tivc_lambda 0.2
     --pipeline.tivc_consistency 0.05
     --pipeline.tivc_min_alpha 0.95
     --pipeline.tivc_yaw 20
     --pipeline.tivc_lateral 1.0
     --pipeline.tivc_spatial_prob 0.5)

# Progressive Gaussian dropout (Sec. 3.5): drop rate grows linearly to 0.2 over training.
DROP=(--pipeline.model.gauss_dropout 0.2)

ASTRA=("${POSE[@]}" "${LENS[@]}" "${VVC[@]}" "${DROP[@]}")

# method_flags <astra|idsplat|astra-nopose|astra-novvc|astra-nolens|astra-nodrop|astra-temporal>
method_flags() {
  case "$1" in
    astra)          METHOD_FLAGS=("${ASTRA[@]}") ;;
    idsplat)        METHOD_FLAGS=() ;;
    astra-nopose)   METHOD_FLAGS=("${LENS[@]}" "${VVC[@]}" "${DROP[@]}") ;;
    astra-novvc)    METHOD_FLAGS=("${POSE[@]}" "${LENS[@]}" "${DROP[@]}") ;;
    astra-nolens)   METHOD_FLAGS=("${POSE[@]}" "${VVC[@]}" "${DROP[@]}") ;;
    astra-nodrop)   METHOD_FLAGS=("${POSE[@]}" "${LENS[@]}" "${VVC[@]}") ;;
    astra-temporal) METHOD_FLAGS=("${POSE[@]}" "${LENS[@]}" "${VVC[@]:0:6}" "${DROP[@]}") ;;  # no off-trajectory views
    *) echo "unknown method $1" >&2; return 1 ;;
  esac
}
