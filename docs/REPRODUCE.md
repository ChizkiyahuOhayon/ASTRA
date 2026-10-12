# Reproducing the paper

All runs use one GPU (we used an RTX 4090 48 GB), 30k steps, and the hyperparameters in `scripts/astra/flags.sh`.
Two trainings can share a 48 GB GPU on Waymo, KITTI and nuScenes; PandaSet needs the GPU alone.
Set the paths once:

```bash
D=data; M=data/masks; O=outputs; R=results; mkdir -p $R/xcam
cfg() { ls -d $O/$1/*/*/config.yml | tail -1; }   # newest config of a run
```

## Table 1: Waymo (8 sequences)

```bash
for m in idsplat astra; do for s in 800 801 802 803 804 805 806 807; do
  bash scripts/astra/train_waymo.sh --scene $s --method $m --data $D/wod --masks_dir $M/waymo --gt_masks_dir $D/wod_gt_masks --tfrecord_dir $D/wod/v1
  bash scripts/astra/eval_waymo.sh --config $(cfg $m-$s) --data $D/wod --gt_masks_dir $D/wod_gt_masks --out $R/$m-$s.json
done; done
```

## Table 2: KITTI (3 sequences x 3 splits)

```bash
for m in idsplat astra; do for split in 75 50 25; do for q in 0001 0002 0006; do
  bash scripts/astra/train_kitti.sh --sequence $q --split $split --method $m --data $D/kitti --masks_dir $M/kitti
  bash scripts/astra/eval_render.sh --config $(cfg $m-kitti$q-$split) --out $R/$m-kitti$q-$split
done; done; done
```

## Table 3: nuScenes (6 scenes) and PandaSet (10 sequences)

```bash
for m in idsplat astra; do
  for sc in scene-0230 scene-0242 scene-0255 scene-0295 scene-0518 scene-0749; do
    bash scripts/astra/train_nuscenes.sh --scene $sc --method $m --data $D/nuscenes --masks_dir $M/nuscenes
    bash scripts/astra/eval_render.sh --config $(cfg $m-nus$sc) --out $R/$m-nus$sc
  done
  for q in 001 011 016 028 053 063 084 106 123 158; do
    bash scripts/astra/train_pandaset.sh --sequence $q --method $m --data $D/pandaset --masks_dir $M/pandaset
    bash scripts/astra/eval_render.sh --config $(cfg $m-panda$q) --out $R/$m-panda$q
  done
done
```

## Table 4: held-out cameras

Needs the Waymo runs of Table 1. Run IDSplat first in every scene: its depth defines the co-visibility mask that
all methods are scored on.

```bash
for m in idsplat astra; do for s in 800 801 802 803 804 805 806 807; do
  bash scripts/astra/eval_xcam.sh --config $(cfg $m-$s) --mask_dir $R/xcam/masks --out $R/xcam/$m-$s.json
done; done
python tools/xcam_table.py $R/xcam idsplat astra
```

## Table 5: off-path tests with real measurements

Waymo side-camera depth needs the Waymo runs of Table 1. The KITTI right-camera test needs left-camera-only runs.

```bash
for m in idsplat astra; do for s in 800 801 802 803 804 805 806 807; do
  bash scripts/astra/eval_offpath.sh --config $(cfg $m-$s) --mode side --out $R/offpath/$m-$s.side.json
done; done
for m in idsplat astra; do for q in 0001 0002 0006; do
  bash scripts/astra/train_kitti.sh --sequence $q --split 75 --data data/kitti --masks_dir data/masks/kitti \
       --method $m --left_only
  bash scripts/astra/eval_offpath.sh --config $(cfg $m-kittiL$q-75) --mode stereo --out $R/offpath/$m-kittiL$q-75.stereo.json
done; done
```

Our result files are in `results_offpath.tar.gz` on the Drive folder; the per-sequence numbers are in
[`RESULTS.md`](RESULTS.md#off-path-tests-with-real-measurements).

## Tables 6 and 7: ablations

The rows of Table 6 add one component at a time. Each row is a `--method` of `scripts/astra/flags.sh`, trained and
evaluated like Table 1 (add `eval_xcam.sh` for the last column):

| Row | how to run |
|---|---|
| IDSplat | `--method idsplat` |
| + pose refinement, raw test poses | the next row's model, evaluated with `--pipeline.model.camopt_interp_eval False` (edit `config.yml`) |
| + trajectory-consistent pose refinement | `--method astra-pose` |
| + temporal virtual views | `--method astra-temporal` |
| + lens-consistent rendering | `--method astra-nooff` |
| + off-trajectory virtual views (ASTRA) | `--method astra` |

Table 7 (sequences 802 and 806) removes one component from ASTRA: `astra-nopose`, `astra-nolens`, `astra-nooff`
and `astra-novvc`.

## Evaluating the released checkpoints

Download an archive from the [GitHub release](https://github.com/ChizkiyahuOhayon/ASTRA/releases/tag/checkpoints-v1),
check it against `SHA256SUMS`, unpack it into `outputs/` and point its config at your data:

```bash
sha256sum -c SHA256SUMS --ignore-missing
tar xf astra-kitti0001-75.tar -C outputs/
uv run python tools/relocate_run.py outputs/astra-kitti0001-75 --data data/kitti --masks_dir data/masks/kitti
bash scripts/astra/eval_render.sh --config $(cfg astra-kitti0001-75) --out results/astra-kitti0001-75
```

The last command reproduces the paper number of that run exactly (25.72 dB PSNR for KITTI 0001, 75% split).

## From result files to tables

`tools/make_tables.py <results_dir> <paper_dir>` reads the JSON files written above (plus `published.json`, the
numbers copied from the AD-GS and IDSplat papers) and writes the LaTeX tables and `PAPER_TABLES.md`. Our own
result files are in the Drive folder (`results.tar.gz`) if you want to check a number without retraining.

Our result files use the internal run names of the project:

| file prefix | method |
|---|---|
| `repro` | IDSplat (reproduced) |
| `camopt` | + trajectory-consistent pose refinement (`astra-pose`); `*.nointerp.json` = raw test poses |
| `astra` | + temporal virtual views (`astra-temporal`) |
| `lens` | + lens-consistent rendering (`astra-nooff`) |
| `sv` | ASTRA (`astra`) |
| `svnopose`, `svnolens`, `svnovv` | ASTRA without pose refinement / lens / any virtual views |
