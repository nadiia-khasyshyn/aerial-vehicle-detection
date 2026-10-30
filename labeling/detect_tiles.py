"""Run one model over a set of cut tiles and write YOLO labels + a preview.

Separate from build_dataset.py on purpose: the tiles are already on disk, so
this re-labels them without touching the videos.
"""

import argparse
import shutil
import time
from pathlib import Path

import cv2
import torch

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="dataset_512/images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="weights/labeler_custom.pt")
    ap.add_argument("--prefix", nargs="*", default=None,
                    help="keep only tiles whose name starts with one of these")
    ap.add_argument("--conf", type=float, default=0.08)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--thickness", type=int, default=3)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--copy-images", action="store_true")
    args = ap.parse_args()

    root = Path.cwd()
    paths = sorted((root / args.images).glob("*.jpg"))
    if args.prefix:
        paths = [p for p in paths if any(p.name.startswith(x) for x in args.prefix)]
    out = root / args.out
    vis = root / f"{args.out}_vis"
    (out / "labels").mkdir(parents=True, exist_ok=True)
    vis.mkdir(parents=True, exist_ok=True)
    if args.copy_images:
        (out / "images").mkdir(parents=True, exist_ok=True)
    print(f"{len(paths)} tiles, conf {args.conf}, imgsz {args.imgsz}", flush=True)

    model = YOLO(args.weights)
    names = model.names
    t0 = time.time()
    n_boxes = 0
    per_video = {}
    empty = 0

    for i in range(0, len(paths), args.batch):
        chunk = paths[i:i + args.batch]
        imgs = [cv2.imread(str(p)) for p in chunk]
        results = model.predict(imgs, device=args.device, imgsz=args.imgsz,
                                conf=args.conf, iou=0.6, verbose=False)
        for p, img, res in zip(chunk, imgs, results):
            b = res.boxes
            xyxy = b.xyxy.cpu() if len(b) else torch.zeros((0, 4))
            cls = b.cls.cpu().long() if len(b) else torch.zeros(0, dtype=torch.long)
            conf = b.conf.cpu() if len(b) else torch.zeros(0)
            h, w = img.shape[:2]
            with open(out / "labels" / f"{p.stem}.txt", "w") as fh:
                for x1, y1, x2, y2 in xyxy.tolist():
                    fh.write(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                             f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
            shown = img.copy()
            for (x1, y1, x2, y2), c, s in zip(xyxy.tolist(), cls.tolist(), conf.tolist()):
                col = (0, 255, 0) if int(c) == 1 else (0, 200, 255)
                cv2.rectangle(shown, (int(x1), int(y1)), (int(x2), int(y2)),
                              col, args.thickness)
            cv2.imwrite(str(vis / f"{p.stem}.jpg"), shown, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if args.copy_images:
                shutil.copy2(p, out / "images" / p.name)
            n_boxes += len(xyxy)
            if len(xyxy) == 0:
                empty += 1
            key = p.name.split("_")[0]
            per_video[key] = per_video.get(key, 0) + len(xyxy)

    with open(out / "data.yaml", "w") as fh:
        base = out / "images" if args.copy_images else root / args.images
        fh.write(f"path: {out}\ntrain: {base}\nval: {base}\nnc: 1\nnames:\n  0: vehicle\n")

    print(f"\n{len(paths)} tiles, {n_boxes} boxes "
          f"({n_boxes / max(len(paths), 1):.1f} per tile) in {time.time() - t0:.0f}s")
    print(f"empty tiles: {empty}")
    for k, v in sorted(per_video.items()):
        print(f"  {k}: {v} boxes")
    print(f"labels: {out}/labels\nvisualisation: {vis}  "
          f"(green {names.get(1, 'cls1')}, orange {names.get(0, 'cls0')})")


if __name__ == "__main__":
    main()
