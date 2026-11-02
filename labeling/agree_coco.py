"""Accept the boxes an open-vocabulary and a stock COCO detector agree on.

Both models are general-purpose: YOLO-World is open-vocabulary, the second
is a stock COCO checkpoint. Their failure modes differ — YOLO-World follows
the prompt and scatters low-confidence noise over kerbs and rooftops, the COCO
detector knows what a car looks like but was trained on ground-level views and
misses small top-down vehicles — so agreement is a stronger signal than either
model's own confidence.

    python labeling/agree_coco.py --images dataset/train/images \
        --labels dataset/train/labels --limit 400
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from avd import get, iou as box_iou   # noqa: E402

VEHICLE_PROMPTS = list(get("labels.vehicle_prompts"))
DISTRACTOR_PROMPTS = list(get("labels.distractor_prompts"))
COCO_VEHICLE_IDS = {1, 2, 3, 5, 7}       # bicycle, car, motorcycle, bus, truck
TILE = int(get("tiling.tile"))


def read_labels(path, tile):
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                cx, cy, w, h = (float(v) * tile for v in parts[1:5])
                out.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def predict(model, paths, imgsz, conf, device, batch, keep=None):
    """Boxes per image, optionally filtered to a set of class indices."""
    per_image = []
    for start in range(0, len(paths), batch):
        chunk = [str(p) for p in paths[start:start + batch]]
        for res in model.predict(chunk, device=device, imgsz=imgsz, conf=conf,
                                 verbose=False):
            boxes = []
            for box, cls in zip(res.boxes.xyxy.cpu().tolist(),
                                res.boxes.cls.cpu().tolist()):
                if keep is None or int(cls) in keep:
                    boxes.append(box)
            per_image.append(boxes)
    return per_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--world", default="yolov8x-worldv2.pt")
    ap.add_argument("--coco", default="yolov8x.pt")
    ap.add_argument("--conf-world", type=float, default=0.10)
    ap.add_argument("--conf-coco", type=float, default=0.10)
    ap.add_argument("--iou", type=float, default=0.5,
                    help="IoU at which the two detectors count as agreeing")
    ap.add_argument("--tile", type=int, default=TILE)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="write the agreed boxes as YOLO labels here")
    args = ap.parse_args()

    from ultralytics import YOLO, YOLOWorld

    paths = sorted(Path(args.images).glob("*.jpg"))
    if args.limit and len(paths) > args.limit:
        random.seed(args.seed)
        paths = sorted(random.sample(paths, args.limit))

    world = YOLOWorld(args.world)
    world.set_classes(VEHICLE_PROMPTS + DISTRACTOR_PROMPTS)
    world_boxes = predict(world, paths, args.tile, args.conf_world,
                          args.device, args.batch, keep=set(range(len(VEHICLE_PROMPTS))))
    coco_boxes = predict(YOLO(args.coco), paths, args.tile, args.conf_coco,
                         args.device, args.batch, keep=COCO_VEHICLE_IDS)

    out_dir = Path(args.out) if args.out else None
    if out_dir:
        for sub in ("images", "labels"):
            (out_dir / sub).mkdir(parents=True, exist_ok=True)

    agreed = world_only = coco_only = 0
    agreed_in_labels = agreed_extra = labels_total = labels_found = 0
    for path, wb, cb in zip(paths, world_boxes, coco_boxes):
        gt = read_labels(Path(args.labels) / f"{path.stem}.txt", args.tile)
        labels_total += len(gt)
        matched_c = set()
        keep = []
        for b in wb:
            hit = next((i for i, c in enumerate(cb)
                        if i not in matched_c and box_iou(b, c) >= args.iou), None)
            if hit is None:
                world_only += 1
                continue
            matched_c.add(hit)
            keep.append(b)
        agreed += len(keep)
        coco_only += len(cb) - len(matched_c)

        used = set()
        for b in keep:
            hit = next((i for i, g in enumerate(gt)
                        if i not in used and box_iou(b, g) >= args.iou), None)
            if hit is None:
                agreed_extra += 1
            else:
                used.add(hit)
                agreed_in_labels += 1
        labels_found += len(used)

        if out_dir and keep:
            lines = []
            for x1, y1, x2, y2 in keep:
                w, h = (x2 - x1) / args.tile, (y2 - y1) / args.tile
                lines.append(f"0 {(x1 + x2) / 2 / args.tile:.6f} "
                             f"{(y1 + y2) / 2 / args.tile:.6f} {w:.6f} {h:.6f}")
            (out_dir / "labels" / f"{path.stem}.txt").write_text("\n".join(lines) + "\n")

    print(f"tiles {len(paths)} | labels in the split {labels_total}")
    print(f"  agreed boxes (both detectors, IoU>={args.iou}): {agreed}")
    print(f"    of those matching a label: {agreed_in_labels} "
          f"({100 * agreed_in_labels / max(1, agreed):.0f}% precision against the labels)")
    print(f"    of those with no label:    {agreed_extra}")
    print(f"  labels recovered by agreement alone: {labels_found} "
          f"({100 * labels_found / max(1, labels_total):.0f}% of the split)")
    print(f"  proposed by one detector only: {world_only} open-vocabulary, {coco_only} COCO")


if __name__ == "__main__":
    main()
