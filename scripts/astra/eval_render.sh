#!/bin/bash
# KITTI / nuScenes / PandaSet held-out frames: dump the test renders, then PSNR / SSIM / LPIPS-VGG / LPIPS-Alex.
#   bash scripts/astra/eval_render.sh --config outputs/astra-kitti0001-75/idsplat/<time>/config.yml --out results/astra-kitti0001-75
set -e
CFG=""; OUT=""
while [[ $# -gt 0 ]]; do case "$1" in --config) CFG="$2"; shift 2;; --out) OUT="$2"; shift 2;; *) echo "unknown $1"; exit 1;; esac; done
: "${CFG:?}"; : "${OUT:?}"
uv run python tools/render_dump.py "$CFG" "$OUT/renders" - - 100000
uv run python tools/metrics_dump.py "$OUT/renders" | tail -1 | tee "$OUT/metrics.json"
