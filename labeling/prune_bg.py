"""Thin the background set down to N tiles, keeping the variety.

Picking at random would keep the proportions of what is already there, and
what is already there is lopsided: a hovering drone contributes hundreds of
near-copies of one patch of grass, so random sampling returns that same
patch hundreds of times. Clustering the tiles in the classifier's feature
space and keeping one tile per cluster spends the budget on distinct scenes
instead of on repetitions, and what is dropped is moved aside, not deleted.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from torchvision import transforms

import classify_crops as C

TF = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(C.MEAN, C.STD),
])


def feats(model, paths, device, batch=256):
    out = []
    for i in range(0, len(paths), batch):
        imgs = [TF(Image.open(p).convert("RGB")) for p in paths[i:i + batch]]
        with torch.no_grad():
            f = model.avgpool(model.features(torch.stack(imgs).to(device))).flatten(1)
        out.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
        print(f"  {min(i + batch, len(paths))}/{len(paths)}", flush=True)
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="dataset_bg")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--dropped", default="dataset_bg_dropped")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    root = Path(__file__).parent
    img = root / args.dir / "images"
    lab = root / args.dir / "labels"
    paths = sorted(img.glob("*.jpg"))
    print(f"{len(paths)} фонових тайлів -> {args.n}")

    model = C.load_classifier(C.CKPT, args.device)
    F = feats(model, paths, args.device)

    km = KMeans(n_clusters=args.n, n_init=4, random_state=0).fit(F)
    keep = []
    for c in range(args.n):
        idx = np.flatnonzero(km.labels_ == c)
        if not len(idx):
            continue
        d = ((F[idx] - km.cluster_centers_[c]) ** 2).sum(1)
        keep.append(idx[d.argmin()])          # medoid: a real tile, not a mean
    keep = set(keep)

    dst = root / args.dropped / "images"
    dstl = root / args.dropped / "labels"
    for d in (dst, dstl):
        d.mkdir(parents=True, exist_ok=True)
    moved = 0
    for i, p in enumerate(paths):
        if i in keep:
            continue
        shutil.move(str(p), dst / p.name)
        t = lab / f"{p.stem}.txt"
        if t.exists():
            shutil.move(str(t), dstl / t.name)
        moved += 1

    left = sorted(img.glob("*.jpg"))
    per = {}
    for p in left:
        v = p.name.split("_")[0]
        per[v] = per.get(v, 0) + 1
    print(f"\nлишилось {len(left)}, відкладено {moved} -> {dst.parent}")
    for v in sorted(per, key=lambda k: -per[k]):
        print(f"  {v}: {per[v]}")


if __name__ == "__main__":
    main()
