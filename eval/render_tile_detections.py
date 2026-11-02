"""Contact sheet of validation tiles with the detector's output drawn on them.

Yellow: ground truth. Green: a detection that matches a label (IoU >= --iou).
Red: a detection that matches nothing. A label nobody found keeps its yellow
box alone — so a miss and a false alarm are told apart by eye rather than by
the caption.

Tiles are taken at even spacing through the sorted split instead of chosen,
which keeps the sheet honest about the average case.

    python eval/render_tile_detections.py --split dataset/val \
        --weights weights/clip/best.pt --conf 0.5 --out examples/val_detections.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from avd import get, iou as box_iou   # noqa: E402

TILE = int(get("tiling.tile"))
GT_COLOR = (0, 215, 255)       # BGR: yellow
HIT_COLOR = (80, 230, 80)      # green, matched a label
EXTRA_COLOR = (60, 60, 255)    # red, matched nothing


def labels_of(path, tile):
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                cx, cy, w, h = (float(v) * tile for v in parts[1:5])
                out.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def draw_tile(path, gt, dets, iou_thr, size):
    img = cv2.imread(str(path))
    if img is None:
        return None, (0, 0, 0)
    for x1, y1, x2, y2 in gt:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), GT_COLOR, 1)

    used = set()
    hits = extra = 0
    for x1, y1, x2, y2 in dets:
        match = next((i for i, g in enumerate(gt)
                      if i not in used and box_iou((x1, y1, x2, y2), g) >= iou_thr),
                     None)
        box = (int(x1), int(y1)), (int(x2), int(y2))
        if match is None:
            cv2.rectangle(img, *box, EXTRA_COLOR, 2)
            extra += 1
        else:
            used.add(match)
            cv2.rectangle(img, *box, HIT_COLOR, 2)
            hits += 1
    missed = len(gt) - len(used)

    img = cv2.resize(img, (size, size))
    caption = f"{hits} found"
    if missed:
        caption += f", {missed} missed"
    if extra:
        caption += f", {extra} extra"
    if not gt and not dets:
        caption = "empty tile, no detections"
    cv2.rectangle(img, (0, 0), (size, 24), (0, 0, 0), -1)
    cv2.putText(img, caption, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img, (hits, missed, extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="a dataset split directory")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--tile", type=int, default=TILE)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cell", type=int, default=380)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--min-boxes", type=int, default=1,
                    help="skip tiles with fewer labels than this")
    ap.add_argument("--offset", type=int, default=0,
                    help="start the even spacing here, for a different sheet")
    ap.add_argument("--out", default="examples/val_detections.jpg")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)

    split = Path(args.split)
    tiles = sorted((split / "images").glob("*.jpg"))
    labels = {p: labels_of(split / "labels" / f"{p.stem}.txt", args.tile)
              for p in tiles}
    pool = [p for p in tiles if len(labels[p]) >= args.min_boxes]
    if not pool:
        sys.exit(f"no tiles in {split} with >= {args.min_boxes} labels")

    want = args.cols * args.rows
    step = max(1, len(pool) // want)
    chosen = pool[args.offset::step][:want]

    results = model.predict([str(p) for p in chosen], device=args.device,
                            imgsz=args.tile, conf=args.conf, verbose=False)

    cells, total = [], [0, 0, 0]
    for path, res in zip(chosen, results):
        dets = res.boxes.xyxy.cpu().tolist()
        cell, (h, m, e) = draw_tile(path, labels[path], dets, args.iou, args.cell)
        if cell is None:
            continue
        cells.append(cell)
        total[0] += h
        total[1] += m
        total[2] += e

    rows = [cv2.hconcat(cells[i:i + args.cols])
            for i in range(0, len(cells), args.cols)
            if len(cells[i:i + args.cols]) == args.cols]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.vconcat(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])

    print(f"{out}  {len(cells)} tiles from {len(pool)} candidates "
          f"(every {step}th from {args.offset})")
    print(f"  on this sheet: {total[0]} found, {total[1]} missed, {total[2]} extra")


if __name__ == "__main__":
    main()
