"""Promote picked YOLO-World additions into the accepted set.

The first pass saved only crop images, so the box coordinates have to be
recovered. Re-running the detector on just the affected tiles gives them back;
crops are matched by exact width, height and confidence rather than by the
index in the filename, because the index depended on an accepted set that has
since changed.
"""

import argparse
import collections
import csv
import glob
import re
from pathlib import Path

import cv2
import torch
from PIL import Image

import classify_crops as C
from separate_similar import feats
from ultralytics import YOLOWorld
from yw_additions import PROMPTS

CROP_RE = re.compile(r"^(?P<tile>.+)_w\d+_\w+_c(?P<conf>[\d.]+)\.jpg$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", nargs="+", required=True)
    ap.add_argument("--pool", default="crops_yw_new")
    ap.add_argument("--images", default="dataset_v234/images")
    ap.add_argument("--accepted", default="crops_classified/vehicle_ge080")
    ap.add_argument("--scores", default="crops_classified/scores.csv")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--sim", type=float, required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--max-area-frac", type=float, default=0.15)
    ap.add_argument("--max-aspect", type=float, default=5.0)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    root = Path.cwd()
    pool = sorted(glob.glob(str(root / args.pool / "*.jpg")))
    clf = C.load_classifier(C.CKPT, args.device)
    E, _ = feats(clf, args.query, args.device)
    F, pool = feats(clf, pool, args.device)
    sim = (F @ E.T).max(dim=1).values.numpy()
    picked = [p for p, s in zip(pool, sim) if s >= args.sim]
    print(f"{len(picked)} crops at sim>={args.sim}")

    want = collections.defaultdict(list)
    for p in picked:
        m = CROP_RE.match(Path(p).name)
        if not m:
            continue
        w, h = Image.open(p).size
        want[m["tile"]].append((p, w, h, float(m["conf"])))
    print(f"spanning {len(want)} tiles", flush=True)

    model = YOLOWorld(args.weights)
    model.set_classes(PROMPTS)
    lim = args.max_area_frac * args.imgsz * args.imgsz

    rows, unmatched = [], 0
    tiles = sorted(want)
    for i in range(0, len(tiles), 16):
        chunk = tiles[i:i + 16]
        imgs = [cv2.imread(str(root / args.images / f"{t}.jpg")) for t in chunk]
        res = model.predict(imgs, device=args.device, imgsz=args.imgsz,
                            conf=args.conf, iou=0.6, verbose=False)
        for tile, r in zip(chunk, res):
            b = r.boxes
            if len(b) == 0:
                continue
            xyxy = b.xyxy.cpu()
            cf = b.conf.cpu()
            bw = xyxy[:, 2] - xyxy[:, 0]
            bh = xyxy[:, 3] - xyxy[:, 1]
            aspect = torch.maximum(bw / bh.clamp(min=1), bh / bw.clamp(min=1))
            ok = (bw * bh < lim) & (aspect < args.max_aspect)
            xyxy, cf = xyxy[ok], cf[ok]
            # Same clipping the crops were cut with, so sizes line up exactly.
            boxes = []
            for (x1, y1, x2, y2), c in zip(xyxy.tolist(), cf.tolist()):
                X1, Y1 = max(int(x1), 0), max(int(y1), 0)
                X2, Y2 = min(int(x2), args.imgsz), min(int(y2), args.imgsz)
                boxes.append((X1, Y1, X2, Y2, c))
            for path, w, h, c in want[tile]:
                hit = [bx for bx in boxes
                       if bx[2] - bx[0] == w and bx[3] - bx[1] == h
                       and abs(bx[4] - c) < 5e-4]
                if not hit:
                    unmatched += 1
                    continue
                X1, Y1, X2, Y2, cc = hit[0]
                rows.append((Path(path).name, tile, X1, Y1, X2, Y2, f"{cc:.4f}", path))

    print(f"recovered {len(rows)} boxes, {unmatched} unmatched")
    if not rows:
        return

    with open(root / args.scores, "a", newline="") as fh:
        w = csv.writer(fh)
        for name, tile, x1, y1, x2, y2, cc, _ in rows:
            w.writerow([name, tile, x1, y1, x2, y2, cc])

    dst = root / args.accepted
    for name, *_, path in rows:
        Path(path).rename(dst / name)
    print(f"moved {len(rows)} crops -> {dst}")


if __name__ == "__main__":
    main()
