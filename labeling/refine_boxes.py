"""Refit every accepted box to the object it contains.

A detector box is where the model guessed the edges; a mask is where the
pixels of the object actually stop. Prompting SAM with the existing box and
taking the mask's extent turns one into the other. The guards matter more than
the refinement: a mask that leaks onto a shadow or a neighbouring vehicle would
silently corrupt an annotation, so a new box that disagrees too much with the
old one is discarded rather than trusted.
"""

import argparse
import collections
import csv
import re
from pathlib import Path

import cv2
import numpy as np

from ultralytics import SAM

NAME_RE = re.compile(r"^(?P<tile>\d+_f\d+_x\d+_y\d+)_")


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accepted", nargs="+", required=True)
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--images", default="dataset_v1/images")
    ap.add_argument("--out", default="crops_v1_low/scores_refined.csv")
    ap.add_argument("--weights", default="mobile_sam.pt")
    ap.add_argument("--min-iou", type=float, default=0.5)
    ap.add_argument("--area-lo", type=float, default=0.4)
    ap.add_argument("--area-hi", type=float, default=2.5)
    ap.add_argument("--chunk", type=int, default=32,
                    help="boxes prompted per call; SAM encodes the tile once")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    root = Path.cwd()
    kept = {p.name for d in args.accepted for p in (root / d).glob("*.jpg")}
    per_tile = collections.defaultdict(list)
    rows = {}
    for r in (r for f in args.scores for r in csv.DictReader(open(root / f))):
        if r["crop"] not in kept:
            continue
        m = NAME_RE.match(r["crop"])
        if not m:
            continue
        box = [int(r[k]) for k in ("x1", "y1", "x2", "y2")]
        per_tile[m["tile"]].append((r["crop"], box))
        rows[r["crop"]] = r
    n_box = sum(len(v) for v in per_tile)
    print(f"{n_box} boxes over {len(per_tile)} tiles", flush=True)

    model = SAM(args.weights)
    out = []
    n_changed = n_guard = n_nomask = 0
    shifts, areas = [], []

    for i, tile in enumerate(sorted(per_tile)):
        img = cv2.imread(str(root / args.images / f"{tile}.jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        items = per_tile[tile]
        for s in range(0, len(items), args.chunk):
            part = items[s:s + args.chunk]
            res = model.predict(img, bboxes=[b for _, b in part],
                                device=args.device, verbose=False)
            masks = res[0].masks
            data = masks.data.cpu().numpy() if masks is not None else []
            for (name, b), m in zip(part, data):
                ys, xs = np.nonzero(m > 0.5)
                r = rows[name]
                nb = None
                if len(xs) >= 4:
                    nb = [int(xs.min()), int(ys.min()),
                          int(xs.max()) + 1, int(ys.max()) + 1]
                    ao = (b[2] - b[0]) * (b[3] - b[1])
                    an = (nb[2] - nb[0]) * (nb[3] - nb[1])
                    ratio = an / ao if ao else 0
                    ok = (nb[2] - nb[0] >= 3 and nb[3] - nb[1] >= 3
                          and 0 <= nb[0] and 0 <= nb[1] and nb[2] <= W and nb[3] <= H
                          and iou(b, nb) >= args.min_iou
                          and args.area_lo <= ratio <= args.area_hi)
                    if ok:
                        shifts.append(iou(b, nb))
                        areas.append(ratio)
                        n_changed += nb != b
                    else:
                        n_guard += 1
                        nb = None
                else:
                    n_nomask += 1
                fb = nb or b
                out.append([name, r["tile"], r.get("source", ""), *fb,
                            r.get("det_conf", ""), r.get("p_vehicle", ""),
                            int(nb is not None), *b])
            if len(data) < len(part):
                n_nomask += len(part) - len(data)
        if i % 100 == 0:
            print(f"  {i}/{len(per_tile)} tiles", flush=True)

    with open(root / args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["crop", "tile", "source", "x1", "y1", "x2", "y2", "det_conf",
                    "p_vehicle", "refined", "ox1", "oy1", "ox2", "oy2"])
        w.writerows(out)

    print(f"\nrefined {len(shifts)} boxes ({n_changed} actually moved)")
    print(f"kept original: {n_guard} failed the guards, {n_nomask} had no mask")
    if shifts:
        print(f"IoU old/new   median {np.median(shifts):.2f}  "
              f"p10 {np.percentile(shifts, 10):.2f}")
        print(f"area new/old  median {np.median(areas):.2f}  "
              f"p10 {np.percentile(areas, 10):.2f}  p90 {np.percentile(areas, 90):.2f}")
    print(f"-> {root / args.out}")


if __name__ == "__main__":
    main()
