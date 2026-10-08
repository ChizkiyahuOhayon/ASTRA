#!/bin/bash
# Waymo held-out frames: PSNR / SSIM / LPIPS-VGG / DPSNR (IDSplat eval code).
#   bash scripts/astra/eval_waymo.sh --config outputs/astra-802/idsplat-wod/<time>/config.yml --data /path/to/wod \
#        --gt_masks_dir /path/to/wod_gt_masks --out results/astra-802.json
set -e
CFG=""; DATA=""; GT=""; OUT=""
while [[ $# -gt 0 ]]; do case "$1" in --config) CFG="$2"; shift 2;; --data) DATA="$2"; shift 2;;
  --gt_masks_dir) GT="$2"; shift 2;; --out) OUT="$2"; shift 2;; *) echo "unknown $1"; exit 1;; esac; done
: "${CFG:?}"; : "${DATA:?}"; : "${OUT:?}"
GTARGS=(); [[ -n "$GT" ]] && GTARGS=(--dynamic-gt-masks-dir "$GT")
uv run python nerfstudio/scripts/eval.py --load-config "$CFG" --output-path "$OUT" --data-root-path "$DATA" "${GTARGS[@]}"
