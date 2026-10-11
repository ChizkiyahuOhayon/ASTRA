#!/bin/bash
# ASTRA component flags (paper Sec. 3). Source this file; every component is a plain IDSplat config flag.
# One configuration is used on every dataset (Waymo, KITTI, nuScenes, PandaSet).

# (1) Trajectory-consistent pose refinement (Sec. 3.2): per-frame SO(3)xR3 camera corrections, carried to unseen
#     timestamps by interpolating the corrections of the bracketing training frames. Lidar poses stay fixed.
POSE=(--pipeline.model.camera_optimizer.mode SO3xR3
      --pipeline.model.camopt_cameras_only True
      --pipeline.model.camopt_interp_eval True)

# (2) Lens-consistent rendering (Sec. 3.3): render through the calibrated lens (no-op on rectified datasets).
LENS=(--pipeline.model.lens_distortion True)

# (3) Virtual-view consistency (Sec. 3.4): virtual cameras between neighbouring training frames (temporal views),
#     half of them moved off the trajectory (yaw +-20 deg, sideways +-1 m), supervised by agreeing warps of the two
#     real neighbours.
VVC_TEMPORAL=(--pipeline.tivc_lambda 0.2
              --pipeline.tivc_consistency 0.05
              --pipeline.tivc_min_alpha 0.95)
VVC_OFFTRAJ=(--pipeline.tivc_yaw 20
             --pipeline.tivc_lateral 1.0
             --pipeline.tivc_spatial_prob 0.5)
VVC=("${VVC_TEMPORAL[@]}" "${VVC_OFFTRAJ[@]}")

ASTRA=("${POSE[@]}" "${LENS[@]}" "${VVC[@]}")

# method_flags <astra|idsplat|astra-pose|astra-temporal|astra-nooff|astra-nopose|astra-nolens|astra-novvc>
#   astra-pose     = IDSplat + (1)                       (ablation row 3)
#   astra-temporal = IDSplat + (1) + temporal views      (ablation row 4)
#   astra-nooff    = ASTRA without off-trajectory views  (ablation row 5)
#   astra-no*      = ASTRA with one component removed    (leave-one-out table)
method_flags() {
  case "$1" in
    astra)          METHOD_FLAGS=("${ASTRA[@]}") ;;
    idsplat)        METHOD_FLAGS=() ;;
    astra-pose)     METHOD_FLAGS=("${POSE[@]}") ;;
    astra-temporal) METHOD_FLAGS=("${POSE[@]}" "${VVC_TEMPORAL[@]}") ;;
    astra-nooff)    METHOD_FLAGS=("${POSE[@]}" "${LENS[@]}" "${VVC_TEMPORAL[@]}") ;;
    astra-nopose)   METHOD_FLAGS=("${LENS[@]}" "${VVC[@]}") ;;
    astra-nolens)   METHOD_FLAGS=("${POSE[@]}" "${VVC[@]}") ;;
    astra-novvc)    METHOD_FLAGS=("${POSE[@]}" "${LENS[@]}") ;;
    *) echo "unknown method $1" >&2; return 1 ;;
  esac
}
