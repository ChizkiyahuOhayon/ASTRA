#!/bin/bash
# Off-path tests scored against real measurements (paper Sec. 4.3, Tab. 5). No model-made mask is involved.
#
#   side    Waymo model trained on FRONT, rendered at the real FRONT_LEFT / FRONT_RIGHT cameras; rendered depth is
#           scored against the LiDAR returns of the same moment on points that some FRONT training image saw.
#   stereo  KITTI model trained on the left camera only (train_kitti.sh --left_only), rendered at the real right
#           camera 0.54 m to the side: PSNR / SSIM / LPIPS against the real right image, and LiDAR depth error.
#
#   bash scripts/astra/eval_offpath.sh --config outputs/astra-802/idsplat-wod/<time>/config.yml --mode side \
#        --out results/offpath/astra-802.side.json
#   bash scripts/astra/eval_offpath.sh --config outputs/astra-kittiL0001-75/idsplat/<time>/config.yml --mode stereo \
#        --out results/offpath/astra-kittiL0001-75.stereo.json
set -e
CFG=""; MODE=""; OUT=""
while [[ $# -gt 0 ]]; do case "$1" in --config) CFG="$2"; shift 2;; --mode) MODE="$2"; shift 2;;
  --out) OUT="$2"; shift 2;; *) echo "unknown $1"; exit 1;; esac; done
: "${CFG:?--config is required}"; : "${MODE:?--mode side|stereo is required}"; : "${OUT:?--out is required}"
mkdir -p "$(dirname "$OUT")"
uv run python tools/offpath_eval.py "$CFG" "$OUT" "$MODE"
