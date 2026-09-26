#!/usr/bin/env bash
# Waymo: 8 scenes, front camera, nvs-75 (every 4th frame held out), 30k iterations.
# Reports PSNR / SSIM / LPIPS and DPSNR on ground-truth moving-object masks.
#
#   bash scripts/benchmark/waymo.sh                 # ASTRA  -> output/waymo/astra
#   METHOD=adgs bash scripts/benchmark/waymo.sh     # AD-GS  -> output/waymo/adgs
set -euo pipefail
source "$(dirname "$0")/common.sh"
ASTRA_FLAGS=("${OBJECTS[@]}" "${BACKGROUND[@]}")
DATA=${DATA:-data/waymo} MASKS=${MASKS:-data/waymo_dynamic_masks} OUT=${OUT:-output/waymo/$METHOD}

for scene in scene006 scene026 scene090 scene105 scene108 scene134 scene150 scene181; do
    train_and_render arguments/waymo.py "$DATA/$scene" "$OUT/$scene" 30000
    python scripts/eval/dynamic_psnr.py -m "$OUT/$scene" -s "$DATA/$scene" --masks "$MASKS" --iteration 30000
done
python scripts/benchmark/summarize.py "$OUT"
