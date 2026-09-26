#!/usr/bin/env bash
# Download the preprocessed KITTI-MOT and nuScenes scenes into data/.
#
#   pip install gdown
#   bash scripts/download_data.sh [kitti|nuscenes|all]
#
# Each scene is a .tar.gz split into <= 1 GB parts with a .sha256 file; parts are verified,
# joined and unpacked, then deleted. Waymo is not hosted (see README, Option B).
set -euo pipefail

KITTI_URL=https://drive.google.com/drive/folders/1eSEmEeyZMrbEaDofmBXG3tTI8HE69nMd
NUSCENES_URL=https://drive.google.com/drive/folders/1R632rPzT9GoHislQAb7TkNzBpZCxzQOf
which=${1:-all}
cd "$(dirname "$0")/.."

fetch() {  # fetch <dataset> <drive folder url>
    local dataset=$1 url=$2 tmp=data/_download/$1
    mkdir -p "$tmp" "data/$dataset"
    gdown --folder "$url" -O "$tmp" --remaining-ok
    tmp=$(dirname "$(find "$tmp" -name '*.sha256' | head -1)")
    for sums in "$tmp"/*.sha256; do
        scene=$(basename "$sums" .sha256)
        if [[ -d data/$dataset/$scene ]]; then echo "data/$dataset/$scene exists, skipping"; continue; fi
        (cd "$tmp" && sha256sum -c "$scene.sha256")
        cat "$tmp/$scene".tar.gz.part-* | tar xz -C "data/$dataset"
        rm -f "$tmp/$scene".tar.gz.part-*
        echo "data/$dataset/$scene ready"
    done
}

[[ $which == all || $which == kitti ]] && fetch kitti "$KITTI_URL"
[[ $which == all || $which == nuscenes ]] && fetch nuscenes "$NUSCENES_URL"
echo done
