"""Full pass over one clip's tiles: custom detector, YOLOE top-up, classifier.

The order matters. The custom model is the better detector but is blind to
two-wheelers, which is most of what moves through this intersection; YOLOE is
the only model measured here that sees them, so it runs second and only its
genuinely new boxes are kept. The classifier then scores every surviving crop,
so one score decides acceptance regardless of which detector proposed the box.
"""

import argparse
import csv
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms
from torchvision.ops import box_iou

import classify_crops as C
from ultralytics import YOLO, YOLOE

PROMPTS = [
    "car seen from above",
    "van seen from above",
    "bus seen from above",
    "truck seen from above",
    "motorcycle seen from above",
    "scooter seen from above",
]


def geometry_ok(xyxy, imgsz, max_area_frac, max_aspect):
    bw = xyxy[:, 2] - xyxy[:, 0]
    bh = xyxy[:, 3] - xyxy[:, 1]
    aspect = torch.maximum(bw / bh.clamp(min=1), bh / bw.clamp(min=1))
    return (bw * bh < max_area_frac * imgsz * imgsz) & (aspect < max_aspect)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="dataset_v1/images")
    ap.add_argument("--out", default="crops_v1")
    ap.add_argument("--labels", default="dataset_v1/labels_merged")
    ap.add_argument("--custom", default="weights/labeler_custom.pt")
    ap.add_argument("--yoloe", required=True)
    ap.add_argument("--conf-custom", type=float, default=0.10)
    ap.add_argument("--conf-yoloe", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--iou-dup", type=float, default=0.4)
    ap.add_argument("--max-area-frac", type=float, default=0.15)
    ap.add_argument("--max-aspect", type=float, default=5.0)
    ap.add_argument("--split", type=float, default=0.8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    root = Path.cwd()
    out = root / args.out
    hi = out / f"vehicle_ge{int(args.split * 100):03d}"
    lo = out / f"vehicle_lt{int(args.split * 100):03d}"
    lab = root / args.labels
    for d in (hi, lo, lab):
        d.mkdir(parents=True, exist_ok=True)
    paths = sorted((root / args.images).glob("*.jpg"))
    print(f"{len(paths)} tiles", flush=True)

    custom = YOLO(args.custom)
    yoloe = YOLOE(args.yoloe)
    yoloe.set_classes(PROMPTS, yoloe.get_text_pe(PROMPTS))

    clf = C.load_classifier(C.CKPT, args.device)
    tf = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(C.MEAN, C.STD),
    ])

    csv_f = open(out / "scores.csv", "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["crop", "tile", "source", "x1", "y1", "x2", "y2",
                    "det_conf", "p_vehicle"])

    n_c = n_e = n_new = n_hi = n_lo = 0

    for s in range(0, len(paths), args.batch):
        chunk = paths[s:s + args.batch]
        imgs = [cv2.imread(str(p)) for p in chunk]
        res_c = custom.predict(imgs, device=args.device, imgsz=args.imgsz,
                               conf=args.conf_custom, iou=0.6, verbose=False)
        res_e = yoloe.predict(imgs, device=args.device, imgsz=args.imgsz,
                              conf=args.conf_yoloe, iou=0.6, verbose=False)

        for p, img, rc, re_ in zip(chunk, imgs, res_c, res_e):
            bc = rc.boxes
            cb = bc.xyxy.cpu() if len(bc) else torch.zeros((0, 4))
            cc = bc.conf.cpu() if len(bc) else torch.zeros(0)
            if len(cb):
                keep = geometry_ok(cb, args.imgsz, args.max_area_frac, args.max_aspect)
                cb, cc = cb[keep], cc[keep]
            n_c += len(cb)

            be = re_.boxes
            eb = be.xyxy.cpu() if len(be) else torch.zeros((0, 4))
            ec = be.conf.cpu() if len(be) else torch.zeros(0)
            if len(eb):
                keep = geometry_ok(eb, args.imgsz, args.max_area_frac, args.max_aspect)
                eb, ec = eb[keep], ec[keep]
            n_e += len(eb)

            if len(eb) and len(cb):
                fresh = box_iou(eb, cb).max(dim=1).values < args.iou_dup
                eb, ec = eb[fresh], ec[fresh]
            n_new += len(eb)

            boxes = [(b, c, "custom") for b, c in zip(cb.tolist(), cc.tolist())]
            boxes += [(b, c, "yoloe") for b, c in zip(eb.tolist(), ec.tolist())]
            if not boxes:
                continue

            crops, meta = [], []
            H, W = img.shape[:2]
            for i, (b, c, src) in enumerate(boxes):
                x1, y1 = max(int(b[0]), 0), max(int(b[1]), 0)
                x2, y2 = min(int(b[2]), W), min(int(b[3]), H)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                crop = Image.fromarray(cv2.cvtColor(img[y1:y2, x1:x2],
                                                    cv2.COLOR_BGR2RGB))
                crops.append(tf(crop))
                meta.append((i, src, c, x1, y1, x2, y2, crop))
            if not crops:
                continue
            with torch.no_grad():
                pv = torch.softmax(clf(torch.stack(crops).to(args.device)),
                                   dim=1)[:, 1].cpu().tolist()

            lines = []
            for (i, src, c, x1, y1, x2, y2, crop), score in zip(meta, pv):
                name = f"{p.stem}_{src[0]}{i:03d}_c{c:.3f}_p{score:.3f}.jpg"
                crop.save((hi if score >= args.split else lo) / name, quality=95)
                csv_w.writerow([name, p.stem, src, x1, y1, x2, y2,
                                f"{c:.4f}", f"{score:.4f}"])
                if score >= args.split:
                    n_hi += 1
                    lines.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                                 f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
                else:
                    n_lo += 1
            (lab / f"{p.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

        if s % (args.batch * 20) == 0:
            print(f"  {s}/{len(paths)} tiles, {n_hi + n_lo} crops", flush=True)

    csv_f.close()
    total = n_hi + n_lo
    print(f"\ncustom @{args.conf_custom}: {n_c} boxes")
    print(f"yoloe  @{args.conf_yoloe}: {n_e} boxes, {n_new} of them new "
          f"(IoU<{args.iou_dup})")
    print(f"classified {total} crops")
    print(f"  p_vehicle >= {args.split}: {n_hi} ({n_hi / max(total,1)*100:.1f}%) -> {hi}")
    print(f"  p_vehicle <  {args.split}: {n_lo} ({n_lo / max(total,1)*100:.1f}%) -> {lo}")
    print(f"labels (accepted only): {lab}\nscores: {out / 'scores.csv'}")


if __name__ == "__main__":
    main()
