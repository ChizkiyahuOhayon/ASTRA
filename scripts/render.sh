#!/bin/bash
# Render output videos from a trained IDSplat model.
#
# Required arguments:
#   --config       Path to the config.yml from the training output directory
#   --output_path  Path to save the output video
#
# Optional arguments:
#   --split        Data split to render: train, eval, or train+eval (default: train+eval)
#   --sensor_index Camera index to render (default: 0)
#
# Example:
#   bash scripts/render.sh \
#       --config outputs/notr-102319/idsplat/<timestamp>/config.yml \
#       --output_path renders/notr-102319.mp4

set -e

CONFIG=""
OUTPUT_PATH=""
SPLIT="train+eval"
SENSOR_INDEX=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)       CONFIG="$2";       shift 2 ;;
        --output_path)  OUTPUT_PATH="$2";  shift 2 ;;
        --split)        SPLIT="$2";        shift 2 ;;
        --sensor_index) SENSOR_INDEX="$2"; shift 2 ;;
        *)              EXTRA_ARGS+=("$1"); shift ;;
    esac
done

: "${CONFIG:?'--config is required'}"
: "${OUTPUT_PATH:?'--output_path is required'}"

uv run python nerfstudio/scripts/render.py \
    interpolate \
    --load-config "$CONFIG" \
    --output-path "$OUTPUT_PATH" \
    --rendered-output-names rgb \
    --pose-source train \
    --sensor-index "$SENSOR_INDEX" \
    --split "$SPLIT" \
    --frame-rate 10 \
    --interpolation-steps 1 \
    "${EXTRA_ARGS[@]}"
