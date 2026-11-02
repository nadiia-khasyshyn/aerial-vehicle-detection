"""Intersect two tiled detection runs over a clip into ground truth.

`agree_coco.py`'s rule applied to whole frames: keep a box only where an
open-vocabulary and a stock COCO detector both put one. What both miss, this
misses too, so the result is a starting point for review.

    python labeling/tiled_yoloworld.py --videos clip.mp4 --frame-stride 10 \
        --weights yolov8x-worldv2.pt --out ow
    python labeling/tiled_yoloworld.py --videos clip.mp4 --frame-stride 10 \
        --weights yolov8x.pt --out coco
    python labeling/agree_frames.py --a ow/detections/clip.csv \
        --b coco/detections/clip.csv --out gt_clip.csv
"""

import argparse
import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from avd import iou as box_iou   # noqa: E402

COCO_VEHICLES = {"bicycle", "car", "motorcycle", "bus", "truck"}


def read(path, keep_classes, min_conf):
    per_frame = collections.defaultdict(list)
    for row in csv.DictReader(open(path)):
        if keep_classes and row.get("class") not in keep_classes:
            continue
        if min_conf and float(row.get("conf", 1.0)) < min_conf:
            continue
        per_frame[int(row["frame"])].append(
            tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2")))
    return per_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="open-vocabulary detections csv")
    ap.add_argument("--b", required=True, help="stock COCO detections csv")
    ap.add_argument("--iou", type=float, default=0.5,
                    help="IoU at which the two runs count as agreeing")
    ap.add_argument("--conf-a", type=float, default=0.0)
    ap.add_argument("--conf-b", type=float, default=0.0)
    ap.add_argument("--coco-classes", action="store_true", default=True,
                    help="keep only vehicle classes from the COCO run")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    first = read(args.a, None, args.conf_a)
    second = read(args.b, COCO_VEHICLES if args.coco_classes else None,
                  args.conf_b)

    frames = sorted(set(first) | set(second))
    rows, agreed, only_a, only_b = [], 0, 0, 0
    for frame in frames:
        boxes_a, boxes_b = first.get(frame, []), second.get(frame, [])
        matched = set()
        for box in boxes_a:
            hit = next((i for i, other in enumerate(boxes_b)
                        if i not in matched and box_iou(box, other) >= args.iou),
                       None)
            if hit is None:
                only_a += 1
                continue
            matched.add(hit)
            agreed += 1
            rows.append([frame] + [int(v) for v in box])
        only_b += len(boxes_b) - len(matched)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "x1", "y1", "x2", "y2"])
        writer.writerows(rows)

    print(f"{out}  {len(frames)} frames, {agreed} agreed boxes")
    print(f"  proposed by one run only: {only_a} open-vocabulary, {only_b} COCO")


if __name__ == "__main__":
    main()
