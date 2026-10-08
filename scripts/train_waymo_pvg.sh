#!/bin/bash
# Train IDSplat on Waymo Open Dataset — PVG split.
#
# Required arguments:
#   --sequence   Segment name, e.g. 102353
#   --data       Path to the WOD parquet root (directory containing the 'training' folder)
#   --masks_dir  Path to the Grounded-SAM-2 tracked masks directory
#
# Optional arguments:
#   --start_frame  Override start frame (default: looked up from scripts/arrays/wod_id_to_seq_pvg.txt)
#   --end_frame    Override end frame
#
# Example:
#   bash scripts/train_waymo_pvg.sh \
#       --sequence 102353 \
#       --data /path/to/wod \
#       --masks_dir /path/to/masks

set -e

ARRAY_FILE="$(dirname "$0")/arrays/wod_id_to_seq_pvg.txt"
SEQUENCE=""
DATA=""
MASKS_DIR=""
START_FRAME=""
END_FRAME=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sequence)   SEQUENCE="$2";   shift 2 ;;
        --data)       DATA="$2";       shift 2 ;;
        --masks_dir)  MASKS_DIR="$2";  shift 2 ;;
        --start_frame) START_FRAME="$2"; shift 2 ;;
        --end_frame)  END_FRAME="$2";  shift 2 ;;
        *)            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

: "${SEQUENCE:?'--sequence is required'}"
: "${DATA:?'--data is required'}"
: "${MASKS_DIR:?'--masks_dir is required'}"

if [[ -z "$START_FRAME" || -z "$END_FRAME" ]]; then
    LINE=$(grep -m1 ",seg${SEQUENCE}," "$ARRAY_FILE")
    if [[ -z "$LINE" ]]; then
        echo "Error: sequence ${SEQUENCE} not found in ${ARRAY_FILE}" >&2
        exit 1
    fi
    IFS=',' read -r _scene_id _seg_name _start _end _type <<< "$LINE"
    START_FRAME="${START_FRAME:-$_start}"
    END_FRAME="${END_FRAME:-$_end}"
fi

uv run python nerfstudio/scripts/train.py idsplat-wod \
    --output-dir outputs \
    --vis wandb \
    --experiment-name pvg-${SEQUENCE} \
    "${EXTRA_ARGS[@]}" \
    wod-data \
    --data "$DATA" \
    --sequence "$SEQUENCE" \
    --start_frame "$START_FRAME" \
    --end_frame "$END_FRAME" \
    --cameras FRONT FRONT_LEFT FRONT_RIGHT \
    --train_split_fraction 0.75 \
    --parquet_dir training \
    --object_masks_dir "$MASKS_DIR" \
    --remove_points_outside_camera_frustum True
