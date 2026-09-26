#!/usr/bin/env bash
# KITTI-MOT: sequences 0001 / 0002 / 0006, stereo, nvs-75 / 50 / 25 splits, 60k iterations.
# The table reports each split averaged over the three sequences, as in AD-GS.
#
#   bash scripts/benchmark/kitti.sh                 # ASTRA  -> output/kitti/astra
#   METHOD=adgs bash scripts/benchmark/kitti.sh     # AD-GS  -> output/kitti/adgs
set -euo pipefail
source "$(dirname "$0")/common.sh"
ASTRA_FLAGS=("${OBJECTS[@]}" "${ROUTING[@]}" "${BACKGROUND[@]}")
DATA=${DATA:-data/kitti} OUT=${OUT:-output/kitti/$METHOD}

for split in 75 50 25; do
    for seq in 0001 0002 0006; do
        train_and_render "arguments/kitti-$split.py" "$DATA/$seq" "$OUT/$seq-nvs$split" 60000 --split_mode "nvs-$split"
    done
done
python scripts/benchmark/summarize.py "$OUT"
