#!/bin/bash
# Train on nuScenes with the AD-GS protocol: three front cameras, frames 10..69, every 4th frame held out.
#
#   bash scripts/astra/train_nuscenes.sh --scene scene-0230 --data /path/to/nuscenes --masks_dir /path/to/masks/nuscenes \
#        [--method astra|idsplat|...] [--out outputs]
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/flags.sh"
SCENE=""; DATA=""; MASKS=""; METHOD=astra; OUT=outputs; EXTRA=()
while [[ $# -gt 0 ]]; do case "$1" in
  --scene) SCENE="$2"; shift 2;; --data) DATA="$2"; shift 2;; --masks_dir) MASKS="$2"; shift 2;;
  --method) METHOD="$2"; shift 2;; --out) OUT="$2"; shift 2;; *) EXTRA+=("$1"); shift;; esac; done
: "${SCENE:?--scene is required}"; : "${DATA:?--data is required}"; : "${MASKS:?--masks_dir is required}"
method_flags "$METHOD"
uv run python nerfstudio/scripts/train.py idsplat --output-dir "$OUT" --vis tensorboard \
  --experiment-name "$METHOD-nus$SCENE" "${METHOD_FLAGS[@]}" "${EXTRA[@]}" \
  nuscenes-data --data "$DATA" --sequence "$SCENE" --cameras FRONT FRONT_LEFT FRONT_RIGHT \
  --start_frame 10 --end_frame 69 --object_masks_dir "$MASKS" --train_eval_split_type EVERY_N --split_every_n 4 \
  --load_cuboids False --remove_points_outside_camera_frustum True
