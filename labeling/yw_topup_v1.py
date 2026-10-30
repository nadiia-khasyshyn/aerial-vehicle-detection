"""Third pass over video 1's tiles: YOLO-World looking for what nothing found.

Custom + YOLOE already ran on these tiles, so a box here is only interesting
if it overlaps nothing either of them proposed - accepted or rejected alike.
Suppressing against the rejected boxes too matters: those were already judged
once, and re-surfacing them would just replay the same contact sheets. Every
survivor is cropped, scored by the classifier and written with its box, so
promoting a pick later needs no coordinate recovery.
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
from ultralytics import YOLOWorld

# Measured per prompt on a 313-tile sample: motorcycle 72% of proposals were
# vehicles, car 59%, truck 39%, bus 12%, van 0%. Bus and van are dropped - bus
# was matching tent canopies, kerbs and market stalls, van matched nothing the
# truck prompt did not already find.
PROMPTS = [
    "car seen from above",
    "truck seen from above",
    "motorcycle seen from above",
    "scooter seen from above",
]
NAME_RE = re.compile(r"^(?P<tile>\d+_f\d+_x\d+_y\d+)_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="dataset_v1/images")
    ap.add_argument("--scores", nargs="+",
                    default=["crops_v1_low/scores.csv"],
                    help="every csv whose boxes count as already known")
    ap.add_argument("--accepted", nargs="+",
                    default=["crops_v1_low/vehicle_ge080", "crops_v1_low/recovered"])
    ap.add_argument("--out", default="crops_v1_low/yw_new")
    ap.add_argument("--out-scores", default="crops_v1_low/scores_yw.csv")
    ap.add_argument("--weights", default="yolov8x-worldv2.pt")
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--iou-dup", type=float, default=0.4)
    ap.add_argument("--max-area-frac", type=float, default=0.15)
    ap.add_argument("--max-aspect", type=float, default=5.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    root = Path.cwd()
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    kept = {p.name for d in args.accepted for p in (root / d).glob("*.jpg")}
    known = collections.defaultdict(list)
    n_acc = 0
    for r in (r for f in args.scores
              for r in csv.DictReader(open(root / f))):
        m = NAME_RE.match(r["crop"])
        if not m:
            continue
        known[m["tile"]].append([int(r[k]) for k in ("x1", "y1", "x2", "y2")])
        n_acc += r["crop"] in kept
    n_known = sum(len(v) for v in known.values())
    print(f"{n_known} known boxes ({n_acc} of them accepted) over {len(known)} tiles",
          flush=True)

    model = YOLOWorld(args.weights)
    model.set_classes(PROMPTS)   # before the first predict
    names = model.model.names

    clf = C.load_classifier(C.CKPT, args.device)
    tf = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(C.MEAN, C.STD),
    ])

    csv_f = open(root / args.out_scores, "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["crop", "tile", "source", "x1", "y1", "x2", "y2",
                    "det_conf", "p_vehicle"])

    paths = sorted((root / args.images).glob("*.jpg"))
    lim = args.max_area_frac * args.imgsz * args.imgsz
    n_raw = n_new = 0
    hist = collections.Counter()
    per_cls = collections.Counter()

    for i in range(0, len(paths), args.batch):
        chunk = paths[i:i + args.batch]
        imgs = [cv2.imread(str(p)) for p in chunk]
        res = model.predict(imgs, device=args.device, imgsz=args.imgsz,
                            conf=args.conf, iou=0.6, verbose=False)
        for p, img, r in zip(chunk, imgs, res):
            b = r.boxes
            if len(b) == 0:
                continue
            xyxy, conf, cls = b.xyxy.cpu(), b.conf.cpu(), b.cls.cpu().long()
            bw = xyxy[:, 2] - xyxy[:, 0]
            bh = xyxy[:, 3] - xyxy[:, 1]
            aspect = torch.maximum(bw / bh.clamp(min=1), bh / bw.clamp(min=1))
            ok = (bw * bh < lim) & (aspect < args.max_aspect)
            if not ok.any():
                continue
            xyxy, conf, cls = xyxy[ok], conf[ok], cls[ok]
            n_raw += len(xyxy)

            old = known.get(p.stem, [])
            if old:
                iou = box_iou(xyxy, torch.tensor(old, dtype=torch.float32))
                fresh = iou.max(dim=1).values < args.iou_dup
            else:
                fresh = torch.ones(len(xyxy), dtype=torch.bool)
            if not fresh.any():
                continue
            xyxy, conf, cls = xyxy[fresh], conf[fresh], cls[fresh]

            crops, meta = [], []
            H, W = img.shape[:2]
            for j, (box, c, k) in enumerate(zip(xyxy.tolist(), conf.tolist(),
                                                cls.tolist())):
                x1, y1 = max(int(box[0]), 0), max(int(box[1]), 0)
                x2, y2 = min(int(box[2]), W), min(int(box[3]), H)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                crop = Image.fromarray(cv2.cvtColor(img[y1:y2, x1:x2],
                                                    cv2.COLOR_BGR2RGB))
                crops.append(tf(crop))
                meta.append((j, c, int(k), x1, y1, x2, y2, crop))
            if not crops:
                continue
            with torch.no_grad():
                pv = torch.softmax(clf(torch.stack(crops).to(args.device)),
                                   dim=1)[:, 1].cpu().tolist()

            for (j, c, k, x1, y1, x2, y2, crop), score in zip(meta, pv):
                tag = names[k].split()[0]
                name = f"{p.stem}_w{j:03d}_{tag}_c{c:.3f}_p{score:.3f}.jpg"
                crop.save(out / name, quality=95)
                csv_w.writerow([name, p.stem, "yoloworld", x1, y1, x2, y2,
                                f"{c:.4f}", f"{score:.4f}"])
                n_new += 1
                hist[min(int(score * 10), 9)] += 1
                per_cls[tag] += 1
        if i % (args.batch * 20) == 0:
            print(f"  {i}/{len(paths)} tiles, {n_new} new", flush=True)

    csv_f.close()
    print(f"\nYOLO-World @conf {args.conf}: {n_raw} boxes after geometry filter")
    print(f"new (IoU<{args.iou_dup} with every known box): {n_new}")
    print("by prompt: " + ", ".join(f"{k} {v}" for k, v in per_cls.most_common()))
    print("p_vehicle:")
    for d in range(10):
        if hist[d]:
            print(f"  {d/10:.1f}-{d/10+0.1:.1f}  {hist[d]}")
    print(f"crops: {out}\nscores: {root / args.out_scores}")


if __name__ == "__main__":
    main()
