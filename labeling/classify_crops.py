"""Score detection crops with the 2-class vehicle classifier and split them.

Crops are taken exactly on the box, no padding, matching how the classifier's
own training crops were cut. Preprocessing mirrors the eval transform in
ml-pipeline/src/classifier/finetune_2cls_v3.py: PIL RGB, resize to 96, ImageNet
normalisation.
"""

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

CLASSES = ["non_vehicle", "vehicle"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
# MobileNetV3-L fine-tuned on vehicle / non-vehicle crops (96 px). Place the
# checkpoint in the workspace you run the pipeline from, or pass --ckpt.
CKPT = "weights/vehicle_classifier.pt"


def load_classifier(ckpt, device):
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASSES))
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.eval().to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="dataset_v234/images")
    ap.add_argument("--labels", default="dataset_v234/labels")
    ap.add_argument("--out", default="crops_classified")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--split", type=float, default=0.8)
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    root = Path.cwd()
    out = root / args.out
    hi = out / f"vehicle_ge{int(args.split * 100):03d}"
    lo = out / f"vehicle_lt{int(args.split * 100):03d}"
    for d in (hi, lo):
        d.mkdir(parents=True, exist_ok=True)

    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    model = load_classifier(args.ckpt, args.device)

    rows = []
    batch_t, batch_meta = [], []
    n_hi = n_lo = n_skip = 0
    csv_f = open(out / "scores.csv", "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["crop", "tile", "x1", "y1", "x2", "y2", "p_vehicle"])

    def flush():
        nonlocal n_hi, n_lo
        if not batch_t:
            return
        x = torch.stack(batch_t).to(args.device)
        with torch.no_grad():
            p = torch.softmax(model(x), dim=1)[:, 1].cpu()
        for (crop, name, box), score in zip(batch_meta, p.tolist()):
            dst = hi if score >= args.split else lo
            crop.save(dst / f"{name}_p{score:.3f}.jpg", quality=95)
            csv_w.writerow([f"{name}_p{score:.3f}.jpg", name.rsplit("_b", 1)[0],
                            *box, f"{score:.4f}"])
            if score >= args.split:
                n_hi += 1
            else:
                n_lo += 1
        batch_t.clear()
        batch_meta.clear()

    label_files = sorted((root / args.labels).glob("*.txt"))
    for li, lf in enumerate(label_files):
        img_p = root / args.images / f"{lf.stem}.jpg"
        if not img_p.exists():
            continue
        lines = [l for l in open(lf) if l.strip()]
        if not lines:
            continue
        img = Image.open(img_p).convert("RGB")
        W, H = img.size
        for bi, line in enumerate(lines):
            _, cx, cy, bw, bh = line.split()
            cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
            x1 = max(int(round((cx - bw / 2) * W)), 0)
            y1 = max(int(round((cy - bh / 2) * H)), 0)
            x2 = min(int(round((cx + bw / 2) * W)), W)
            y2 = min(int(round((cy + bh / 2) * H)), H)
            if x2 - x1 < 2 or y2 - y1 < 2:
                n_skip += 1
                continue
            crop = img.crop((x1, y1, x2, y2))   # no padding, box exactly
            batch_t.append(tf(crop))
            batch_meta.append((crop, f"{lf.stem}_b{bi:03d}", (x1, y1, x2, y2)))
            if len(batch_t) >= args.batch:
                flush()
        if li % 500 == 0:
            print(f"  {li}/{len(label_files)} tiles, {n_hi + n_lo} crops", flush=True)
    flush()
    csv_f.close()

    total = n_hi + n_lo
    print(f"\n{total} crops scored ({n_skip} skipped as degenerate)")
    print(f"  p_vehicle >= {args.split}: {n_hi} ({n_hi / max(total,1)*100:.1f}%)  -> {hi}")
    print(f"  p_vehicle <  {args.split}: {n_lo} ({n_lo / max(total,1)*100:.1f}%)  -> {lo}")
    print(f"scores: {out / 'scores.csv'}")


if __name__ == "__main__":
    main()
