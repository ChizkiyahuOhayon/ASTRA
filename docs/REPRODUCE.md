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

## Tables 5 and 6: ablations

The cumulative rows of Table 5 are the following methods of `scripts/astra/flags.sh`, trained and evaluated like
Table 1 (add `eval_xcam.sh` for the last column):

| Row | how to run |
|---|---|
| IDSplat | `--method idsplat` |
| + pose refinement, raw test poses | the next row's model, evaluated with `--pipeline.model.camopt_interp_eval False` (edit `config.yml`) |
| + trajectory-consistent pose refinement | `--method idsplat` plus the `POSE` flags |
| + temporal virtual views | `POSE` + the first three `VVC` flags |
| + lens + dropout | `--method astra-temporal` |
| + off-trajectory views (full) | `--method astra` |

Table 6 (sequences 802 and 806) removes one component from `astra-temporal`: `astra-nopose`, `astra-nolens`,
`astra-nodrop` with the off-trajectory flags removed (`--pipeline.tivc_spatial_prob 0`).

## From result files to tables

`tools/make_tables.py <results_dir> <paper_dir>` reads the JSON files written above (plus `published.json`, the
numbers copied from the AD-GS and IDSplat papers) and writes the LaTeX tables and `PAPER_TABLES.md`. Our own
result files are in the Drive folder (`results.tar.gz`) if you want to check a number without retraining.

Our result files use the internal run names of the project: `repro` = IDSplat (reproduced), `camopt` = + pose
refinement, `astra` = + temporal virtual views, `gdl` = + lens + dropout, `stivc` = ASTRA (full); the
`*.nointerp.json` files are the raw-test-pose evaluations.
