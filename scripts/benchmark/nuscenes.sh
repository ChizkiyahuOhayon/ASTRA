#!/usr/bin/env bash
# nuScenes: 6 scenes, three front cameras, frames 10-69, 60k iterations.
#
#   bash scripts/benchmark/nuscenes.sh              # ASTRA  -> output/nuscenes/astra
#   METHOD=adgs bash scripts/benchmark/nuscenes.sh  # AD-GS  -> output/nuscenes/adgs
set -euo pipefail
source "$(dirname "$0")/common.sh"
ASTRA_FLAGS=("${OBJECTS[@]}" "${ROUTING[@]}" "${BACKGROUND[@]}")
DATA=${DATA:-data/nuscenes} OUT=${OUT:-output/nuscenes/$METHOD}

for scene in scene-0230 scene-0242 scene-0255 scene-0295 scene-0518 scene-0749; do
    train_and_render arguments/nuscenes.py "$DATA/$scene" "$OUT/$scene" 60000
done
python scripts/benchmark/summarize.py "$OUT"
