"""Replace oversized annotations with what is actually inside them.

A box covering a stretch of road with one car in it passes a "is there a
vehicle here" check and fails as an annotation. Re-running the detectors
inside the box at high resolution answers the better question - where exactly
is the vehicle - and usually returns one or more tight boxes. A parent whose
children cover most of it was tight already and is kept; one that yields
nothing is set aside for review rather than deleted.
"""

import argparse
import csv
import re
from pathlib import Path

import cv2
import torch
from torchvision.ops import box_iou, nms

from pipeline_v1 import PROMPTS
from ultralytics import YOLO, YOLOE

NAME_RE = re.compile(r"^(?P<tile>\d+_f\d+_x\d+_y\d+)_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accepted", nargs="+", required=True)
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--images", default="dataset_v1/images")
    ap.add_argument("--out", default="crops_v1_low/scores_split.csv")
    ap.add_argument("--custom", default="weights/labeler_custom.pt")
    ap.add_argument("--yoloe", default="yoloe-11l-seg.pt")
    ap.add_argument("--area-frac", type=float, default=0.08,
                    help="boxes above this share of the tile get re-examined")
    ap.add_argument("--cover", type=float, default=0.6,
                    help="children covering more than this of the parent mean it was tight")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    root = Path.cwd()
    kept = {p.name for d in args.accepted for p in (root / d).glob("*.jpg")}
    rows = []
    for f in args.scores:
        for r in csv.DictReader(open(root / f)):
            if r["crop"] in kept:
                rows.append(r)
    lim = args.area_frac * args.tile * args.tile
    big = [r for r in rows
           if (int(r["x2"]) - int(r["x1"])) * (int(r["y2"]) - int(r["y1"])) > lim]
    print(f"{len(rows)} accepted, {len(big)} larger than {args.area_frac:.0%} of a tile",
          flush=True)

    custom = YOLO(args.custom)
    yoloe = YOLOE(args.yoloe)
    yoloe.set_classes(PROMPTS, yoloe.get_text_pe(PROMPTS))

    out, n_split, n_kept, n_none = [], 0, 0, 0
    cache = {}
    for i, r in enumerate(big):
        tile = NAME_RE.match(r["crop"])["tile"]
        if tile not in cache:
            cache.clear()
            cache[tile] = cv2.imread(str(root / args.images / f"{tile}.jpg"))
        img = cache[tile]
        if img is None:
            continue
        x1, y1, x2, y2 = (int(r[k]) for k in ("x1", "y1", "x2", "y2"))
        pad = 8
        cx1, cy1 = max(x1 - pad, 0), max(y1 - pad, 0)
        cx2, cy2 = min(x2 + pad, img.shape[1]), min(y2 + pad, img.shape[0])
        sub = img[cy1:cy2, cx1:cx2]
        if sub.size == 0:
            continue
        kids = []
        for model in (custom, yoloe):
            res = model.predict(sub, device=args.device, imgsz=args.imgsz,
                                conf=args.conf, iou=0.6, verbose=False)[0]
            b = res.boxes
            if len(b):
                kids.append(torch.cat([b.xyxy.cpu(), b.conf.cpu()[:, None]], 1))
        if not kids:
            n_none += 1
            out.append((r, None))
            continue
        k = torch.cat(kids)
        k = k[nms(k[:, :4], k[:, 4], 0.5)]
        area_p = (x2 - x1) * (y2 - y1)
        area_k = ((k[:, 2] - k[:, 0]) * (k[:, 3] - k[:, 1])).sum().item()
        if area_k / area_p > args.cover or len(k) == 0:
            n_kept += 1
            out.append((r, None))
            continue
        n_split += 1
        for j, (bx, cf) in enumerate(zip(k[:, :4].tolist(), k[:, 4].tolist())):
            out.append((r, (cx1 + int(bx[0]), cy1 + int(bx[1]),
                            cx1 + int(bx[2]), cy1 + int(bx[3]), cf, j)))
        if i % 100 == 0:
            print(f"  {i}/{len(big)}", flush=True)

    with open(root / args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["crop", "tile", "source", "x1", "y1", "x2", "y2",
                    "det_conf", "p_vehicle", "parent"])
        for r, kid in out:
            if kid is None:
                continue
            x1, y1, x2, y2, cf, j = kid
            name = f"{r['tile']}_s{j:03d}_{abs(hash(r['crop'])) % 9973:04d}" \
                   f"_c{cf:.3f}_p1.000.jpg"
            w.writerow([name, r["tile"], "split", x1, y1, x2, y2,
                        f"{cf:.4f}", "1.0000", r["crop"]])
    print(f"\nsplit {n_split} parents into tighter boxes")
    print(f"kept as they were: {n_kept} already tight, {n_none} yielded nothing")
    print(f"-> {root / args.out}")


if __name__ == "__main__":
    main()
