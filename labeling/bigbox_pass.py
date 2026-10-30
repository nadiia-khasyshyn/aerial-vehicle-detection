"""Recover the detections the area filter threw away.

The pipeline capped boxes at 15% of a tile to keep roofs and awnings out. On a
512 tile that cap sits below the size of a car filmed from this altitude, so it
was deleting the easiest objects in the clip. This pass re-runs the same two
detectors with the cap raised and keeps only boxes that overlap nothing already
known, which leaves exactly what the cap had removed.
"""

import argparse
import collections
import csv
import re
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms
from torchvision.ops import box_iou

import classify_crops as C
from pipeline_v1 import PROMPTS
from ultralytics import YOLO, YOLOE

NAME_RE = re.compile(r"^(?P<tile>\d+_f\d+_x\d+_y\d+)_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="dataset_v1/images")
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--out", default="crops_v1_low/big_new")
    ap.add_argument("--out-scores", default="crops_v1_low/scores_big.csv")
    ap.add_argument("--custom", default="weights/labeler_custom.pt")
    ap.add_argument("--yoloe", default="yoloe-11l-seg.pt")
    ap.add_argument("--conf-custom", type=float, default=0.05)
    ap.add_argument("--conf-yoloe", type=float, default=0.03)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--iou-dup", type=float, default=0.4)
    ap.add_argument("--max-area-frac", type=float, default=0.45)
    ap.add_argument("--min-area-frac", type=float, default=0.15,
                    help="0 to keep everything; 0.15 isolates what the old cap cut")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--max-aspect", type=float, default=5.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    root = Path.cwd()
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    known = collections.defaultdict(list)
    for r in (r for f in args.scores for r in csv.DictReader(open(root / f))):
        m = NAME_RE.match(r["crop"])
        if m:
            known[m["tile"]].append([int(r[k]) for k in ("x1", "y1", "x2", "y2")])
    print(f"{sum(len(v) for v in known.values())} known boxes", flush=True)

    custom = YOLO(args.custom)
    yoloe = YOLOE(args.yoloe)
    yoloe.set_classes(PROMPTS, yoloe.get_text_pe(PROMPTS))
    clf = C.load_classifier(C.CKPT, args.device)
    tf = transforms.Compose([transforms.Resize((96, 96)), transforms.ToTensor(),
                             transforms.Normalize(C.MEAN, C.STD)])

    csv_f = open(root / args.out_scores, "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["crop", "tile", "source", "x1", "y1", "x2", "y2",
                    "det_conf", "p_vehicle", "area_frac"])

    paths = sorted((root / args.images).glob("*.jpg"))
    lim = args.max_area_frac * args.tile * args.tile
    n_new = 0
    hist = collections.Counter()

    for s in range(0, len(paths), args.batch):
        chunk = paths[s:s + args.batch]
        imgs = [cv2.imread(str(p)) for p in chunk]
        rc = custom.predict(imgs, device=args.device, imgsz=args.imgsz,
                            conf=args.conf_custom, iou=0.6, verbose=False)
        re_ = yoloe.predict(imgs, device=args.device, imgsz=args.imgsz,
                            conf=args.conf_yoloe, iou=0.6, verbose=False)
        for p, img, a, b in zip(chunk, imgs, rc, re_):
            boxes, confs, srcs = [], [], []
            for res, tag in ((a, "custom"), (b, "yoloe")):
                bb = res.boxes
                if not len(bb):
                    continue
                xy, cf = bb.xyxy.cpu(), bb.conf.cpu()
                bw, bh = xy[:, 2] - xy[:, 0], xy[:, 3] - xy[:, 1]
                asp = torch.maximum(bw / bh.clamp(min=1), bh / bw.clamp(min=1))
                ok = (bw * bh < lim) & (asp < args.max_aspect)
                boxes.append(xy[ok]); confs.append(cf[ok]); srcs += [tag] * int(ok.sum())
            if not boxes:
                continue
            xy = torch.cat(boxes); cf = torch.cat(confs)
            old = known.get(p.stem, [])
            if old:
                fresh = box_iou(xy, torch.tensor(old, dtype=torch.float32)
                                ).max(dim=1).values < args.iou_dup
                xy, cf = xy[fresh], cf[fresh]
                srcs = [s_ for s_, f in zip(srcs, fresh.tolist()) if f]
            if not len(xy):
                continue
            if args.min_area_frac > 0:
                keep = ((xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1])
                        >= args.min_area_frac * args.tile * args.tile)
                xy, cf = xy[keep], cf[keep]
                srcs = [s_ for s_, k in zip(srcs, keep.tolist()) if k]
                if not len(xy):
                    continue

            crops, meta = [], []
            H, W = img.shape[:2]
            for i, (bx, c, src) in enumerate(zip(xy.tolist(), cf.tolist(), srcs)):
                x1, y1 = max(int(bx[0]), 0), max(int(bx[1]), 0)
                x2, y2 = min(int(bx[2]), W), min(int(bx[3]), H)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                crop = Image.fromarray(cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
                crops.append(tf(crop))
                meta.append((i, src, c, x1, y1, x2, y2, crop))
            if not crops:
                continue
            with torch.no_grad():
                pv = torch.softmax(clf(torch.stack(crops).to(args.device)),
                                   dim=1)[:, 1].cpu().tolist()
            for (i, src, c, x1, y1, x2, y2, crop), score in zip(meta, pv):
                af = (x2 - x1) * (y2 - y1) / (args.tile ** 2)
                name = f"{p.stem}_g{i:03d}_{src[0]}_c{c:.3f}_p{score:.3f}.jpg"
                crop.save(out / name, quality=95)
                csv_w.writerow([name, p.stem, src, x1, y1, x2, y2,
                                f"{c:.4f}", f"{score:.4f}", f"{af:.3f}"])
                n_new += 1
                hist[src] += 1
        if s % (args.batch * 20) == 0:
            print(f"  {s}/{len(paths)} tiles, {n_new} recovered", flush=True)

    csv_f.close()
    print(f"\n{n_new} boxes the old cap had deleted: {dict(hist)}")
    print(f"crops: {out}\nscores: {root / args.out_scores}")


if __name__ == "__main__":
    main()
