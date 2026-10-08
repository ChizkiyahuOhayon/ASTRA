#!/bin/bash
# Evaluate a trained IDSplat model.
#
# Required arguments:
#   --config   Path to the config.yml from the training output directory
#
# Example:
#   bash scripts/eval.sh --config outputs/notr-102319/idsplat/<timestamp>/config.yml

set -e

CONFIG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

: "${CONFIG:?'--config is required'}"

uv run python nerfstudio/scripts/eval.py \
    --load-config "$CONFIG"
