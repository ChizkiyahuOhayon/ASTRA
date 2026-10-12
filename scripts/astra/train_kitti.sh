#!/bin/bash
# Train on KITTI-MOT with the AD-GS protocol (both stereo cameras).
#
#   bash scripts/astra/train_kitti.sh --sequence 0001 --split 75 --data /path/to/kitti --masks_dir /path/to/masks/kitti \
#        [--method astra|idsplat|...] [--out outputs] [--left_only]
#
# --split 75 | 50 | 25   percentage of frames used for training (AD-GS nvs-75 / nvs-50 / nvs-25)
# --left_only            train on the left camera (image_02) only; the right camera is then a real held-out view
#                        0.54 m to the side (scripts/astra/eval_offpath.sh --mode stereo)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/flags.sh"
SEQ=""; SPLIT=75; DATA=""; MASKS=""; METHOD=astra; OUT=outputs; EXTRA=(); CAMS=(); TAG=kitti
while [[ $# -gt 0 ]]; do case "$1" in
  --sequence) SEQ="$2"; shift 2;; --split) SPLIT="$2"; shift 2;; --data) DATA="$2"; shift 2;;
  --masks_dir) MASKS="$2"; shift 2;; --method) METHOD="$2"; shift 2;; --out) OUT="$2"; shift 2;;
  --left_only) CAMS=(--cameras image_02); TAG=kittiL; shift;;
  *) EXTRA+=("$1"); shift;; esac; done
: "${SEQ:?--sequence is required}"; : "${DATA:?--data is required}"; : "${MASKS:?--masks_dir is required}"
method_flags "$METHOD"
case $SEQ in 0001) S=380; E=431;; 0002) S=140; E=224;; 0006) S=65; E=120;; *) echo "AD-GS uses 0001/0002/0006" >&2; exit 1;; esac
case $SPLIT in 75) SP=(--split_every_n 4);; 50) SP=(--split_every_n 2);; 25) SP=(--split_every_n 4 --split_every_n_holds_out False);; esac
uv run python nerfstudio/scripts/train.py idsplat --output-dir "$OUT" --vis tensorboard \
  --experiment-name "$METHOD-$TAG$SEQ-$SPLIT" "${METHOD_FLAGS[@]}" "${EXTRA[@]}" \
  kittimot-data --data "$DATA" --sequence "$SEQ" --start_frame $S --end_frame $E "${CAMS[@]}" \
  --object_masks_dir "$MASKS" --train_eval_split_type EVERY_N "${SP[@]}" --load_cuboids False \
  --remove_points_outside_camera_frustum True
