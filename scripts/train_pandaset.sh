#!/bin/bash
# Train IDSplat on PandaSet.
#
# Required arguments:
#   --sequence   Sequence number, e.g. 001
#   --data       Path to the PandaSet root directory
#   --masks_dir  Path to the Grounded-SAM-2 tracked masks directory
#
# Example:
#   bash scripts/train_pandaset.sh \
#       --sequence 001 \
#       --data /path/to/pandaset \
#       --masks_dir /path/to/masks

set -e

SEQUENCE=""
DATA=""
MASKS_DIR=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sequence)  SEQUENCE="$2";  shift 2 ;;
        --data)      DATA="$2";      shift 2 ;;
        --masks_dir) MASKS_DIR="$2"; shift 2 ;;
        *)           EXTRA_ARGS+=("$1"); shift ;;
    esac
done

: "${SEQUENCE:?'--sequence is required'}"
: "${DATA:?'--data is required'}"
: "${MASKS_DIR:?'--masks_dir is required'}"

uv run python nerfstudio/scripts/train.py idsplat \
    --output-dir outputs \
    --vis wandb \
    --experiment-name pandaset-${SEQUENCE} \
    --pipeline.model.mcmc_cap_max=10_000_000 \
    --pipeline.model.max_num_seed_points=5_000_000 \
    --pipeline.model.n_extra_random_points=1_000_000 \
    --pipeline.model.n_far_points=1_000_000 \
    "${EXTRA_ARGS[@]}" \
    pandaset-data \
    --data "$DATA" \
    --sequence "$SEQUENCE" \
    --train_split_fraction 0.5 \
    --object_masks_dir "$MASKS_DIR"
