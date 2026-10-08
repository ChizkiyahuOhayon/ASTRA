#!/bin/bash
# Train on PandaSet with the IDSplat protocol: all six cameras, half of the frames held out.
#
#   bash scripts/astra/train_pandaset.sh --sequence 001 --data /path/to/data/pandaset --masks_dir /path/to/masks/pandaset \
#        [--method astra|idsplat|...] [--out outputs]
# Note: the PandaSet parser derives mask paths by splitting on "data/pandaset/", so --data must end in data/pandaset.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/flags.sh"
SEQ=""; DATA=""; MASKS=""; METHOD=astra; OUT=outputs; EXTRA=()
while [[ $# -gt 0 ]]; do case "$1" in
  --sequence) SEQ="$2"; shift 2;; --data) DATA="$2"; shift 2;; --masks_dir) MASKS="$2"; shift 2;;
  --method) METHOD="$2"; shift 2;; --out) OUT="$2"; shift 2;; *) EXTRA+=("$1"); shift;; esac; done
: "${SEQ:?--sequence is required}"; : "${DATA:?--data is required}"; : "${MASKS:?--masks_dir is required}"
method_flags "$METHOD"
uv run python nerfstudio/scripts/train.py idsplat --output-dir "$OUT" --vis tensorboard \
  --experiment-name "$METHOD-panda$SEQ" \
  --pipeline.model.mcmc_cap_max=10_000_000 --pipeline.model.max_num_seed_points=5_000_000 \
  --pipeline.model.n_extra_random_points=1_000_000 --pipeline.model.n_far_points=1_000_000 \
  "${METHOD_FLAGS[@]}" "${EXTRA[@]}" \
  pandaset-data --data "$DATA" --sequence "$SEQ" --train_split_fraction 0.5 --object_masks_dir "$MASKS"
