"""Add zoomed-in copies of labelled tiles so the model sees larger vehicles.

The training clips are shot from altitude: the median vehicle is 25 px across,
and nothing in the set looks like the 300-400 px vehicles of low-altitude
footage. Cropping a small window out of a tile and scaling it back up to the
tile size produces exactly those: the same vehicles, 2x or 4x bigger.

Only windows that fully contain at least one box are kept, so no vehicle is
sliced by the zoom crop, and boxes are re-projected into the new tile.

    python labeling/zoom_tiles.py --split dataset/train --out dataset_zoom \
        --zooms 2 4 --per-tile 1
"""

import argparse
import random
from pathlib import Path

import cv2


def boxes_of(label_path, tile):
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            cx, cy, w, h = (float(v) * tile for v in parts[1:5])
            out.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="e.g. dataset/train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--zooms", type=float, nargs="+", default=[2, 4],
                    help="zoom factors; with --sample one is drawn at random per crop")
    ap.add_argument("--per-tile", type=int, default=1,
                    help="zoomed crops to emit per source tile and zoom level")
    ap.add_argument("--min-box", type=float, default=0.0,
                    help="only zoom tiles whose largest box is at least this "
                         "many pixels: upscaling a 25 px car 6x invents detail "
                         "that is not in the source, a 150 px one does not")
    ap.add_argument("--sample", action="store_true",
                    help="draw one zoom per crop instead of emitting every "
                         "level, which spreads box sizes smoothly instead of "
                         "piling them at two values")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    src = Path(args.split)
    out = Path(args.out)
    for sub in ("images", "labels"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    written = 0
    for img_path in sorted((src / "images").iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        boxes = boxes_of(src / "labels" / f"{img_path.stem}.txt", args.tile)
        if not boxes:
            continue  # background tiles gain nothing from zooming
        biggest = max(max(b[2] - b[0], b[3] - b[1]) for b in boxes)
        if biggest < args.min_box:
            continue
        img = None
        levels = ([random.choice(args.zooms) for _ in range(args.per_tile)]
                  if args.sample else args.zooms)
        for zoom in levels:
            win = int(args.tile / zoom)
            for k in range(1 if args.sample else args.per_tile):
                fitting = [b for b in boxes
                           if b[2] - b[0] <= win and b[3] - b[1] <= win]
                if not fitting:
                    continue
                anchor = max(fitting, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                cx = (anchor[0] + anchor[2]) / 2
                cy = (anchor[1] + anchor[3]) / 2
                ox = int(min(max(0, cx - win / 2), args.tile - win))
                oy = int(min(max(0, cy - win / 2), args.tile - win))
                kept = []
                for b in boxes:
                    if b[0] >= ox and b[1] >= oy and b[2] <= ox + win and b[3] <= oy + win:
                        kept.append(((b[0] - ox) * zoom, (b[1] - oy) * zoom,
                                     (b[2] - ox) * zoom, (b[3] - oy) * zoom))
                if not kept:
                    continue
                if img is None:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        break
                crop = cv2.resize(img[oy:oy + win, ox:ox + win],
                                  (args.tile, args.tile), interpolation=cv2.INTER_CUBIC)
                stem = f"{img_path.stem}_z{zoom:g}_{k}_{ox}_{oy}"
                cv2.imwrite(str(out / "images" / f"{stem}.jpg"), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                lines = []
                for x1, y1, x2, y2 in kept:
                    w, h = (x2 - x1) / args.tile, (y2 - y1) / args.tile
                    lines.append(f"0 {(x1 + x2) / 2 / args.tile:.6f} "
                                 f"{(y1 + y2) / 2 / args.tile:.6f} {w:.6f} {h:.6f}")
                (out / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
                written += 1
    print(f"wrote {written} zoomed tiles to {out}")


if __name__ == "__main__":
    main()
