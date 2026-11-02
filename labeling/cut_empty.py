"""Cut background tiles: the ones no detector found anything in.

A tiled detector spends most of its inference on empty ground, so the training
set needs empty ground too, or the model never learns to stay quiet. What
counts as empty is taken from the lowest-confidence run available: a tile where
even a 0.1 threshold found nothing is background, while one holding a faint car
is not, and mislabelling the second kind teaches the model to ignore real
vehicles.

Tiles are spread over frames rather than taken in runs, because consecutive
frames of a drone shot are nearly the same picture.
"""

import argparse
import collections
import csv
import random
import re
from pathlib import Path

import cv2

NAME_RE = re.compile(r"^(?P<tile>\d+_f(?P<frame>\d+)_x(?P<x>\d+)_y(?P<y>\d+))")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--exclude", nargs="+", required=True,
                    help="score csvs; any tile appearing there is not empty")
    ap.add_argument("--out", default="dataset_v2_bg")
    ap.add_argument("--total", type=int, default=2300)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--labels", action="store_true",
                    help="also write an empty .txt per tile")
    ap.add_argument("--min-edges", type=float, default=0.02,
                    help="skip near-uniform tiles: sky teaches nothing")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(__file__).parent
    vid = re.split(r"[-_]", Path(args.video).stem)[0]
    img_dir = root / args.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir = root / args.out / "labels"
    if args.labels:
        lab_dir.mkdir(parents=True, exist_ok=True)

    taken = set()
    for f in args.exclude:
        for r in csv.DictReader(open(root / f)):
            taken.add(r["tile"])
    print(f"{len(taken)} тайлів із детекціями виключено", flush=True)

    cap = cv2.VideoCapture(str(root / args.video))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    xs = list(range(0, max(W - args.tile, 0) + 1, args.stride))
    ys = list(range(0, max(H - args.tile, 0) + 1, args.stride))
    if xs[-1] + args.tile < W:
        xs.append(W - args.tile)
    if ys[-1] + args.tile < H:
        ys.append(H - args.tile)

    frames = [i for i in range(n_frames) if i % args.every == 0]
    cand = collections.defaultdict(list)
    for i in frames:
        for oy in ys:
            for ox in xs:
                name = f"{vid}_f{i:05d}_x{ox:04d}_y{oy:04d}"
                if name not in taken:
                    cand[i].append((name, ox, oy))
    n_cand = sum(len(v) for v in cand.values())
    print(f"{n_cand} порожніх кандидатів у {len(cand)} кадрах", flush=True)

    # spread the quota evenly over frames, then over positions within a frame
    rnd = random.Random(args.seed)
    per = max(1, round(args.total / max(len(cand), 1)))
    plan = collections.defaultdict(list)
    picked = 0
    for i in sorted(cand):
        items = cand[i][:]
        rnd.shuffle(items)
        for it in items[:per]:
            plan[i].append(it)
            picked += 1
    # top up if rounding left us short
    if picked < args.total:
        rest = [(i, it) for i in sorted(cand) for it in cand[i]
                if it not in plan[i]]
        rnd.shuffle(rest)
        for i, it in rest[:args.total - picked]:
            plan[i].append(it)
            picked += 1
    print(f"відібрано {picked}, ~{per} на кадр", flush=True)

    written = n_flat = 0
    idx = -1
    while True:
        if not cap.grab():
            break
        idx += 1
        if idx not in plan:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        for name, ox, oy in plan[idx]:
            sub = frame[oy:oy + args.tile, ox:ox + args.tile]
            edges = (cv2.Canny(cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY),
                               60, 160) > 0).mean()
            if edges < args.min_edges:
                n_flat += 1
                continue
            cv2.imwrite(str(img_dir / f"{name}.jpg"), sub,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            if args.labels:
                (lab_dir / f"{name}.txt").write_text("")
            written += 1
        if written % 500 < per:
            print(f"  {written}/{picked}", flush=True)
    cap.release()
    print(f"\n{written} порожніх тайлів -> {img_dir}")
    print(f"{n_flat} пропущено як однорідні (небо)")


if __name__ == "__main__":
    main()
