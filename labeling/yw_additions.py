"""Second pass with YOLO-World over the cut tiles, keeping only what is new.

The custom detector is blind to two-wheelers and misses dark vehicles;
YOLO-World sees those but is noisy, so it runs at a high threshold here and
every box that already matches an accepted detection is dropped. What is left
is the candidate additions, saved as crops to be looked at.
"""

import argparse
import collections
import csv
import re
from pathlib import Path

import cv2
import torch
from torchvision.ops import box_iou

from ultralytics import YOLOWorld

PROMPTS = [
    "car seen from above",
    "van seen from above",
    "bus seen from above",
    "truck seen from above",
    "motorcycle seen from above",
]
NAME_RE = re.compile(r"^(?P<tile>.+)_b\d+_p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="dataset_v234/images")
    ap.add_argument("--accepted", default="crops_classified/vehicle_ge080")
    ap.add_argument("--scores", default="crops_classified/scores.csv")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="crops_yw_new")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--iou-dup", type=float, default=0.4,
                    help="above this overlap with an accepted box it is not new")
    ap.add_argument("--max-area-frac", type=float, default=0.15)
    ap.add_argument("--max-aspect", type=float, default=5.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    root = Path.cwd()
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    kept = {p.name for p in (root / args.accepted).glob("*.jpg")}
    accepted = collections.defaultdict(list)
    for r in csv.DictReader(open(root / args.scores)):
        if r["crop"] not in kept:
            continue
        m = NAME_RE.match(r["crop"])
        if m:
            accepted[m["tile"]].append([int(r[k]) for k in ("x1", "y1", "x2", "y2")])
    print(f"{len(kept)} accepted boxes over {len(accepted)} tiles", flush=True)

    model = YOLOWorld(args.weights)
    model.set_classes(PROMPTS)   # before the first predict
    names = model.model.names

    paths = sorted((root / args.images).glob("*.jpg"))
    lim = args.max_area_frac * args.imgsz * args.imgsz
    n_raw = n_new = 0
    per_video = collections.Counter()

    for i in range(0, len(paths), args.batch):
        chunk = paths[i:i + args.batch]
        imgs = [cv2.imread(str(p)) for p in chunk]
        res = model.predict(imgs, device=args.device, imgsz=args.imgsz,
                            conf=args.conf, iou=0.6, verbose=False)
        for p, img, r in zip(chunk, imgs, res):
            b = r.boxes
            if len(b) == 0:
                continue
            xyxy = b.xyxy.cpu()
            conf = b.conf.cpu()
            cls = b.cls.cpu().long()
            bw = xyxy[:, 2] - xyxy[:, 0]
            bh = xyxy[:, 3] - xyxy[:, 1]
            aspect = torch.maximum(bw / bh.clamp(min=1), bh / bw.clamp(min=1))
            ok = (bw * bh < lim) & (aspect < args.max_aspect)
            if not ok.any():
                continue
            xyxy, conf, cls = xyxy[ok], conf[ok], cls[ok]
            n_raw += len(xyxy)

            old = accepted.get(p.stem, [])
            if old:
                iou = box_iou(xyxy, torch.tensor(old, dtype=torch.float32))
                fresh = (iou.max(dim=1).values < args.iou_dup)
            else:
                fresh = torch.ones(len(xyxy), dtype=torch.bool)
            if not fresh.any():
                continue

            for j, (box, c, k) in enumerate(zip(xyxy[fresh].tolist(),
                                                conf[fresh].tolist(),
                                                cls[fresh].tolist())):
                x1, y1, x2, y2 = (max(int(box[0]), 0), max(int(box[1]), 0),
                                  min(int(box[2]), img.shape[1]),
                                  min(int(box[3]), img.shape[0]))
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                tag = names[int(k)].split()[0]
                cv2.imwrite(str(out / f"{p.stem}_w{j:03d}_{tag}_c{c:.3f}.jpg"),
                            img[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 95])
                n_new += 1
                per_video[p.stem.split("_")[0]] += 1
        if i % (args.batch * 40) == 0:
            print(f"  {i}/{len(paths)} tiles, {n_new} new", flush=True)

    print(f"\nYOLO-World @conf {args.conf}: {n_raw} boxes after geometry filter")
    print(f"new (IoU<{args.iou_dup} with every accepted box): {n_new}")
    for k, v in sorted(per_video.items()):
        print(f"  {k}: {v}")
    print(f"crops: {out}")


if __name__ == "__main__":
    main()
