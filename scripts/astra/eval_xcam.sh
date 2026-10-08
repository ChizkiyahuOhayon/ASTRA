#!/bin/bash
# Held-out-camera benchmark (paper Sec. 4.3): render a front-camera model at the real FRONT_LEFT / FRONT_RIGHT
# cameras and score co-visible pixels. Run the IDSplat baseline of a scene FIRST: its depth defines the shared
# co-visibility masks (<mask_dir>/<scene>.npz) that every other method is scored on.
#   bash scripts/astra/eval_xcam.sh --config outputs/idsplat-802/idsplat-wod/<time>/config.yml \
#        --mask_dir results/xcam/masks --out results/xcam/idsplat-802.json
set -e
CFG=""; MASKS=""; OUT=""; EVERY=2
while [[ $# -gt 0 ]]; do case "$1" in --config) CFG="$2"; shift 2;; --mask_dir) MASKS="$2"; shift 2;;
  --out) OUT="$2"; shift 2;; --every) EVERY="$2"; shift 2;; *) echo "unknown $1"; exit 1;; esac; done
: "${CFG:?}"; : "${MASKS:?}"; : "${OUT:?}"
mkdir -p "$(dirname "$OUT")"
uv run python tools/xcam_eval.py "$CFG" "$OUT" "$MASKS" "$EVERY"
