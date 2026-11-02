"""Measure how far the proposal model is from the labels that ship.

Runs the open-vocabulary detector over the dataset tiles and diffs its boxes
against the labels in the split: how many it would add, how many it misses,
how many it places well enough to keep. The point is to put a number on what
the human review and the filters actually contribute, instead of asserting it.

A box counts as agreeing when IoU >= --iou; proposals that overlap a label
less than that are "misplaced" rather than "extra", because they found the
vehicle and failed on the box.

    python labeling/proposal_vs_labels.py --split dataset/train \
        --weights yolov8x-worldv2.pt --conf 0.1 --limit 600
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from avd import get, iou as box_iou   # noqa: E402

VEHICLE_PROMPTS = list(get("labels.vehicle_prompts"))
DISTRACTOR_PROMPTS = list(get("labels.distractor_prompts"))
TILE = int(get("tiling.tile"))


def labels_of(path, tile):
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                cx, cy, w, h = (float(v) * tile for v in parts[1:5])
                out.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--weights", default="yolov8x-worldv2.pt")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--tile", type=int, default=TILE)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=0, help="sample this many tiles")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    model.set_classes(VEHICLE_PROMPTS + DISTRACTOR_PROMPTS)
    n_vehicle = len(VEHICLE_PROMPTS)

    split = Path(args.split)
    tiles = sorted((split / "images").glob("*.jpg"))
    if args.limit and len(tiles) > args.limit:
        random.seed(args.seed)
        tiles = random.sample(tiles, args.limit)

    agree = missed = misplaced = extra = labelled = proposed = 0
    for start in range(0, len(tiles), args.batch):
        chunk = tiles[start:start + args.batch]
        results = model.predict([str(p) for p in chunk], device=args.device,
                                imgsz=args.tile, conf=args.conf, verbose=False)
        for path, res in zip(chunk, results):
            gt = labels_of(split / "labels" / f"{path.stem}.txt", args.tile)
            labelled += len(gt)
            boxes = []
            for box, cls in zip(res.boxes.xyxy.cpu().tolist(),
                                res.boxes.cls.cpu().tolist()):
                if int(cls) < n_vehicle:
                    boxes.append(box)
            proposed += len(boxes)
            used = set()
            for b in boxes:
                hit = next((i for i, g in enumerate(gt)
                            if i not in used and box_iou(b, g) >= args.iou), None)
                if hit is not None:
                    used.add(hit)
                    agree += 1
                elif any(box_iou(b, g) > 0.1 for g in gt):
                    misplaced += 1
                else:
                    extra += 1
            missed += len(gt) - len(used)
        done = min(start + args.batch, len(tiles))
        if done % (args.batch * 10) == 0:
            print(f"  {done}/{len(tiles)} tiles", flush=True)

    print(f"\ntiles {len(tiles)} | labels {labelled} | proposals {proposed}")
    print(f"  agree with a label (IoU>={args.iou}): {agree} "
          f"({100 * agree / max(1, labelled):.0f}% of labels)")
    print(f"  proposed but no label at all:         {extra}")
    print(f"  overlaps a label, box too far off:    {misplaced}")
    print(f"  labels the proposer never found:      {missed} "
          f"({100 * missed / max(1, labelled):.0f}% of labels)")


if __name__ == "__main__":
    main()
