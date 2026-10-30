"""Find crops that look like a given example and move them into their own bin.

Two steps on purpose. Without --apply it only ranks and writes a contact sheet,
because the cut-off is not guessable: embeddings pick up background as well as
object, so one example of a cone on grass and the same cone on asphalt sit far
apart, and the point where real vehicles start bleeding in moves per query.
Look at the sheet, then pass the threshold.

Similarity is cosine over the vehicle classifier's own pooled features, which
keeps everything on one backbone and needs no extra model.
"""

import argparse
import csv
import glob
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import classify_crops as C

TF = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(C.MEAN, C.STD),
])


def feats(model, paths, device, batch=256):
    """Embeddings, skipping files that vanished mid-run."""
    out, kept = [], []
    for i in range(0, len(paths), batch):
        imgs, ok = [], []
        for p in paths[i:i + batch]:
            try:
                imgs.append(TF(Image.open(p).convert("RGB")))
                ok.append(p)
            except Exception:
                pass
        if not imgs:
            continue
        with torch.no_grad():
            f = model.avgpool(model.features(torch.stack(imgs).to(device))).flatten(1)
        out.append(torch.nn.functional.normalize(f, dim=1).cpu())
        kept += ok
    return (torch.cat(out) if out else torch.zeros((0, 960))), kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", nargs="+", required=True,
                    help="example crop(s); a crop matches on its best one")
    ap.add_argument("--pool", default="crops_classified/vehicle_ge080")
    ap.add_argument("--dst-root", default="crops_classified",
                    help="where the bin folder and its csv are written")
    ap.add_argument("--name", required=True, help="bin folder name")
    ap.add_argument("--apply", type=float, default=None,
                    help="similarity cut-off; omit to only rank and preview")
    ap.add_argument("--sheet", type=int, default=96)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    root = Path.cwd()
    model = C.load_classifier(C.CKPT, args.device)
    queries = []
    for q in args.query:
        hits = glob.glob(str(root / "crops_classified" / "*" / q)) if "/" not in q else [q]
        queries += hits or [q]
    pool = sorted(glob.glob(str(root / args.pool / "*.jpg")))

    E, used = feats(model, queries, args.device)
    if len(used) != len(queries):
        print(f"WARNING: {len(queries)-len(used)} query file(s) unreadable")
    if len(used) == 0:
        raise SystemExit("no usable query images: " + "; ".join(queries[:3]))
    F, pool = feats(model, pool, args.device)
    sim = (F @ E.T).max(dim=1).values.numpy()
    order = np.argsort(-sim)

    print(f"{len(queries)} queries, {len(pool)} candidates")
    print("  " + "  ".join(f">={t}:{(sim >= t).sum()}"
                           for t in (0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4)))

    sheet = root / "examples" / f"rank_{args.name}.jpg"
    cells = []
    for i in order[:args.sheet]:
        c = cv2.resize(cv2.imread(pool[i]), (96, 96))
        cv2.putText(c, f"{sim[i]:.2f}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (0, 255, 255), 1)
        cells.append(c)
    cols = 8
    rows = [np.hstack(cells[r * cols:(r + 1) * cols])
            for r in range(len(cells) // cols)]
    cv2.imwrite(str(sheet), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"contact sheet: {sheet}")

    if args.apply is None:
        print("no --apply, nothing moved")
        return

    dst = root / args.dst_root / args.name
    dst.mkdir(parents=True, exist_ok=True)
    moved, gone = [], 0
    for p, s in zip(pool, sim):
        if s < args.apply:
            continue
        try:
            shutil.move(p, dst / Path(p).name)
            moved.append((Path(p).name, f"{s:.4f}"))
        except FileNotFoundError:
            gone += 1
    with open(root / args.dst_root / f"{args.name}.csv", "a", newline="") as fh:
        w = csv.writer(fh)
        if fh.tell() == 0:
            w.writerow(["crop", "sim"])
        w.writerows(sorted(moved, key=lambda r: -float(r[1])))
    print(f"moved {len(moved)} (sim>={args.apply}) -> {dst}"
          + (f", {gone} already gone" if gone else ""))


if __name__ == "__main__":
    main()
