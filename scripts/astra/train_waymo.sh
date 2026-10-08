#!/bin/bash
# Train ASTRA (or a baseline / ablation) on one Waymo sequence of the AD-GS / StreetGS split (front camera,
# every 4th frame held out, 30k steps).
#
#   bash scripts/astra/train_waymo.sh --scene 802 --data /path/to/wod --masks_dir /path/to/masks/waymo \
#        [--method astra|idsplat|astra-nopose|...] [--gt_masks_dir /path/to/wod_gt_masks --tfrecord_dir /path/to/wod/v1]
#        [--out outputs]
#
# --scene      scene id 800..807 from scripts/arrays/wod_id_to_seq_adgs_split.txt
# --data       WOD v2 root (the folder that contains validation/)
# --masks_dir  Grounded-SAM-2 tracked masks (tools/make_masks_wod.py, see docs/DATA.md)
# --gt_masks_dir  optional: dynamic GT masks for DPSNR (tools/make_gt_masks.py); evaluation only
# --tfrecord_dir  WOD v1 tfrecords of the same segments (required by the parser together with --gt_masks_dir)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/flags.sh"
SCENE=""; DATA=""; MASKS=""; METHOD=astra; GT=""; TFR=""; OUT=outputs; EXTRA=()
while [[ $# -gt 0 ]]; do case "$1" in
  --scene) SCENE="$2"; shift 2;; --data) DATA="$2"; shift 2;; --masks_dir) MASKS="$2"; shift 2;;
  --method) METHOD="$2"; shift 2;; --gt_masks_dir) GT="$2"; shift 2;; --tfrecord_dir) TFR="$2"; shift 2;;
  --out) OUT="$2"; shift 2;;
  *) EXTRA+=("$1"); shift;; esac; done
: "${SCENE:?--scene is required}"; : "${DATA:?--data is required}"; : "${MASKS:?--masks_dir is required}"
method_flags "$METHOD"
IFS=, read -r _ SEG START END _ <<< "$(grep "^$SCENE," "$HERE/../arrays/wod_id_to_seq_adgs_split.txt")"
SEQ=$(ls "$DATA/validation/camera_image" | grep "^${SEG#seg}" | head -1); SEQ=${SEQ%.parquet}
[[ -n "$SEQ" ]] || { echo "segment ${SEG#seg} not found under $DATA/validation/camera_image" >&2; exit 1; }
GTARGS=(); [[ -n "$GT" ]] && GTARGS=(--dynamic_gt_masks_dir "$GT" --tfrecord_dir "${TFR:?--tfrecord_dir is required with --gt_masks_dir}")
uv run python nerfstudio/scripts/train.py idsplat-wod \
  --output-dir "$OUT" --vis tensorboard --experiment-name "$METHOD-$SCENE" \
  --pipeline.datamanager.oversample_cameras_factor=12 --pipeline.datamanager.camera_res_scale_factor=1.0 \
  "${METHOD_FLAGS[@]}" "${EXTRA[@]}" \
  wod-data --data "$DATA" --sequence "$SEQ" --start_frame "$START" --end_frame "$END" \
  --cameras FRONT --train_split_fraction 0.75 --parquet_dir validation \
  --object_masks_dir "$MASKS" --remove_points_outside_camera_frustum True --load_cuboids False \
  --output_folder "$DATA/images" "${GTARGS[@]}"
