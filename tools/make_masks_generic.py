"""Tracked instance masks (Grounding DINO + SAM 2, continuous ids) for any ordered image list.

    make_masks_generic.py <image_list.txt> <out_root> --sam2_ckpt <ckpt> [--step 5]

Same tracker as make_masks_wod.py (WOD). <image_list.txt> holds one camera's images in time order. Output:
<out_root>/json_data/mask_<stem>.json and <out_root>/mask_data/mask_<stem>.npy, the layout read by the
KITTI-MOT and nuScenes parsers. No annotation is read: only camera images.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_masks_wod import CLASSES, iou  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_list")
    ap.add_argument("out_root")
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--box_threshold", type=float, default=0.35)
    ap.add_argument("--text_threshold", type=float, default=0.25)
    ap.add_argument("--match_iou", type=float, default=0.5)
    ap.add_argument("--sam2_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--sam2_ckpt", required=True)
    args = ap.parse_args()

    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    paths = [p.strip() for p in open(args.image_list) if p.strip()]
    frame_dir = os.path.join(args.out_root, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    for i, p in enumerate(paths):  # SAM 2 wants a folder of consecutively numbered jpgs
        dst = os.path.join(frame_dir, "%05d.jpg" % i)
        if not os.path.exists(dst):
            Image.open(p).convert("RGB").save(dst, quality=95)
    names = ["%05d.jpg" % i for i in range(len(paths))]
    print("frames", len(paths), flush=True)

    dev = "cuda"
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    detector = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(dev)
    image_predictor = SAM2ImagePredictor(build_sam2(args.sam2_cfg, args.sam2_ckpt, device=dev))
    video_predictor = build_sam2_video_predictor(args.sam2_cfg, args.sam2_ckpt, device=dev)
    prompt = ". ".join(CLASSES) + "."
    os.makedirs(os.path.join(args.out_root, "json_data"), exist_ok=True)
    os.makedirs(os.path.join(args.out_root, "mask_data"), exist_ok=True)

    tracks, propagated, next_id = {}, {}, 1
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = video_predictor.init_state(video_path=frame_dir, offload_video_to_cpu=True, async_loading_frames=True)
        for key in range(0, len(paths), args.step):
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
                        best, next_id = next_id, next_id + 1
                        tracks[best] = lab
                    current[best] = m
            video_predictor.reset_state(state)
            for tid, m in current.items():
                video_predictor.add_new_mask(state, frame_idx=key, obj_id=tid, mask=m)
            propagated[key] = current
            if current:
                stop = min(key + args.step, len(paths) - 1)
                for f, obj_ids, logits in video_predictor.propagate_in_video(
                        state, start_frame_idx=key, max_frame_num_to_track=stop - key):
                    if f != key:
                        propagated[f] = {int(o): (logits[i, 0] > 0).cpu().numpy() for i, o in enumerate(obj_ids)}
            print("keyframe %d: %d detections, %d tracked, %d ids" % (key, len(boxes), len(current), next_id - 1),
                  flush=True)

    h, w = np.array(Image.open(os.path.join(frame_dir, names[0]))).shape[:2]
    for local, p in enumerate(paths):
        stem = os.path.splitext(os.path.basename(p))[0]
        canvas = np.zeros((h, w), dtype=np.uint16)
        labels = {}
        for tid, m in sorted(propagated.get(local, {}).items()):
            if m.any():
                canvas[m & (canvas == 0)] = tid
                labels[str(tid)] = {"instance_id": tid, "class_name": tracks[tid]}
        name = "mask_%s.npy" % stem
        np.save(os.path.join(args.out_root, "mask_data", name), canvas)
        with open(os.path.join(args.out_root, "json_data", "mask_%s.json" % stem), "w") as fp:
            json.dump({"mask_name": name, "labels": labels}, fp)
    print("done", args.out_root, "ids", next_id - 1, flush=True)


if __name__ == "__main__":
    main()
