"""Tracked instance masks for ASTRA / IDSplat from WOD v2 parquet (Grounding DINO + SAM 2, continuous ids).

    make_masks_wod.py <wod_validation_dir> <segment> <seg_id> <start> <end> <out_dir> [--step 5]

Keyframes every `step` frames: Grounding DINO boxes -> SAM 2 masks; each keyframe mask is matched to
the masks propagated from earlier keyframes (IoU) to keep its instance id, unmatched ones get a new id.
SAM 2's video predictor then propagates all tracked objects to the next keyframe. Output follows the
IDSplat README: <out>/<seg_id>/<cam_id>/{json_data/mask_XXX_<cam>.json, mask_data/mask_XXX.npy}.
No annotation is read: only camera images.
"""
import argparse
import io
import json
import os

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

CLASSES = ["car", "truck", "van", "bus", "train", "human", "cyclist", "bicycle", "pedestrian"]
FRONT, CAM_ID = 1, "0"  # WOD camera_name FRONT; IDSplat CAM_TYPE_TO_ID["FRONT"]


def export_frames(wod, segment, start, end, frame_dir):
    table = pq.read_table(os.path.join(wod, "camera_image", segment + ".parquet"),
                          columns=["key.frame_timestamp_micros", "key.camera_name", "[CameraImageComponent].image"])
    df = table.to_pandas()
    df = df[df["key.camera_name"] == FRONT].sort_values("key.frame_timestamp_micros").reset_index(drop=True)
    os.makedirs(frame_dir, exist_ok=True)
    frames = list(range(start, min(end, len(df) - 1) + 1))
    for i, f in enumerate(frames):  # SAM 2 wants a folder of consecutively numbered jpgs
        path = os.path.join(frame_dir, "%05d.jpg" % i)
        if not os.path.exists(path):
            Image.open(io.BytesIO(df.loc[f, "[CameraImageComponent].image"])).convert("RGB").save(path, quality=95)
    return frames


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    return inter / max(np.logical_or(a, b).sum(), 1)


def main():
    ap = argparse.ArgumentParser()
    for k in ["wod", "segment", "seg_id"]:
        ap.add_argument(k)
    ap.add_argument("start", type=int)
    ap.add_argument("end", type=int)
    ap.add_argument("out")
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--box_threshold", type=float, default=0.35)
    ap.add_argument("--text_threshold", type=float, default=0.25)
    ap.add_argument("--match_iou", type=float, default=0.5)
    ap.add_argument("--sam2_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--sam2_ckpt", required=True)
    args = ap.parse_args()

    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    frame_dir = os.path.join(args.out, "frames", args.segment)
    frames = export_frames(args.wod, args.segment, args.start, args.end, frame_dir)
    names = sorted(os.listdir(frame_dir))[: len(frames)]
    print("frames", len(frames), flush=True)

    dev = "cuda"
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    detector = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(dev)
    image_predictor = SAM2ImagePredictor(build_sam2(args.sam2_cfg, args.sam2_ckpt, device=dev))
    video_predictor = build_sam2_video_predictor(args.sam2_cfg, args.sam2_ckpt, device=dev)
    prompt = ". ".join(CLASSES) + "."

    root = os.path.join(args.out, args.seg_id, CAM_ID)
    os.makedirs(os.path.join(root, "json_data"), exist_ok=True)
    os.makedirs(os.path.join(root, "mask_data"), exist_ok=True)

    tracks = {}          # id -> class name
    propagated = {}      # frame index (local) -> {id: bool mask}
    next_id = 1
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = video_predictor.init_state(video_path=frame_dir, offload_video_to_cpu=True, async_loading_frames=True)
        for key in range(0, len(frames), args.step):
            image = Image.open(os.path.join(frame_dir, names[key])).convert("RGB")
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(dev)
            with torch.autocast("cuda", enabled=False):
                outputs = detector(**inputs)
            det = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=args.box_threshold, text_threshold=args.text_threshold,
                target_sizes=[image.size[::-1]])[0]
            boxes = det["boxes"].float().cpu().numpy()
            labels = [l if l in CLASSES else (l.split()[0] if l.split() and l.split()[0] in CLASSES else "car")
                      for l in det.get("text_labels", det["labels"])]
            current = dict(propagated.get(key, {}))
            if len(boxes):
                image_predictor.set_image(np.array(image))
                masks, _, _ = image_predictor.predict(box=boxes, multimask_output=False)
                masks = masks.reshape(len(boxes), *masks.shape[-2:]) > 0
                for m, lab in zip(masks, labels):
                    best, best_iou = None, args.match_iou
                    for tid, pm in current.items():
                        v = iou(m, pm)
                        if v > best_iou:
                            best, best_iou = tid, v
                    if best is None:
                        best = next_id
                        next_id += 1
                        tracks[best] = lab
                    current[best] = m
            # re-seed the video predictor from this keyframe and propagate to the next one
            video_predictor.reset_state(state)
            for tid, m in current.items():
                video_predictor.add_new_mask(state, frame_idx=key, obj_id=tid, mask=m)
            propagated[key] = current
            if current:
                stop = min(key + args.step, len(frames) - 1)
                for f, obj_ids, logits in video_predictor.propagate_in_video(
                        state, start_frame_idx=key, max_frame_num_to_track=stop - key):
                    if f == key:
                        continue
                    propagated[f] = {int(o): (logits[i, 0] > 0).cpu().numpy() for i, o in enumerate(obj_ids)}
            print("keyframe %d: %d detections, %d tracked, %d ids" % (key, len(boxes), len(current), next_id - 1),
                  flush=True)

    h, w = np.array(Image.open(os.path.join(frame_dir, names[0]))).shape[:2]
    for local, f in enumerate(frames):
        canvas = np.zeros((h, w), dtype=np.uint16)
        labels = {}
        for tid, m in sorted(propagated.get(local, {}).items()):
            if m.any():
                canvas[m & (canvas == 0)] = tid
                labels[str(tid)] = {"instance_id": tid, "class_name": tracks[tid]}
        name = "mask_%03d.npy" % f
        np.save(os.path.join(root, "mask_data", name), canvas)
        with open(os.path.join(root, "json_data", "mask_%03d_%s.json" % (f, CAM_ID)), "w") as fp:
            json.dump({"mask_name": name, "labels": labels}, fp)
    print("done", args.segment, "ids", next_id - 1, flush=True)


if __name__ == "__main__":
    main()
