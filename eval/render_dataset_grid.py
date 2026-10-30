"""Contact sheets of labelled tiles, mixing clips so one sheet shows the
whole dataset rather than one scene.

    python eval/render_dataset_grid.py --dataset dataset --split train \
        --out examples --sheets 2 --cols 4 --rows 3
"""

import argparse
import collections
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from avd import parse_tile_name   # noqa: E402


def draw_tile(img_path, lbl_path, size):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    n = 0
    if lbl_path.exists():
        for line in lbl_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            n += 1
    img = cv2.resize(img, (size, size))
    name = parse_tile_name(img_path.stem)
    caption = f"{name.clip if name else '?'}  {n} box" + ("" if n == 1 else "es")
    cv2.rectangle(img, (0, 0), (size, 26), (0, 0, 0), -1)
    cv2.putText(img, caption, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="examples")
    ap.add_argument("--sheets", type=int, default=2)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-empty", action="store_true",
                    help="also show background tiles, which carry no boxes")
    args = ap.parse_args()

    root = Path(args.dataset) / args.split
    by_clip = collections.defaultdict(list)
    for img in sorted((root / "images").iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lbl = root / "labels" / f"{img.stem}.txt"
        if not args.with_empty and (not lbl.exists() or not lbl.read_text().strip()):
            continue
        name = parse_tile_name(img.stem)
        by_clip[name.clip if name else "?"].append((img, lbl))

    random.seed(args.seed)
    for clip in by_clip:
        random.shuffle(by_clip[clip])
    clips = sorted(by_clip)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cursor = {c: 0 for c in clips}
    for sheet in range(args.sheets):
        tiles = []
        # round-robin over clips so every sheet mixes the whole dataset
        while len(tiles) < args.cols * args.rows:
            progressed = False
            for clip in clips:
                if len(tiles) >= args.cols * args.rows:
                    break
                i = cursor[clip]
                if i < len(by_clip[clip]):
                    img, lbl = by_clip[clip][i]
                    cursor[clip] = i + 1
                    drawn = draw_tile(img, lbl, args.size)
                    if drawn is not None:
                        tiles.append(drawn)
                    progressed = True
            if not progressed:
                break
        if len(tiles) < args.cols:
            break
        rows = [cv2.hconcat(tiles[r * args.cols:(r + 1) * args.cols])
                for r in range(len(tiles) // args.cols)]
        path = out_dir / f"dataset_{args.split}_grid{sheet + 1}.jpg"
        cv2.imwrite(str(path), cv2.vconcat(rows))
        print(f"-> {path}")


if __name__ == "__main__":
    main()
