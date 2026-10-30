"""Turn a tiled YOLO split back into full-frame ground truth.

Tile names carry the frame and the tile origin
(`<clip>_f00582_x2304_y1648.jpg`), so the normalised box coordinates inside a
tile map straight back to pixels in the source frame. Boxes that the tiling
cut are stitched: the same vehicle seen in two neighbouring tiles becomes one
box again, so distances computed from box size stay honest.

    python eval/tiles_to_gt.py --split dataset/val --out gt_val.csv
"""

import argparse
import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from avd import parse_tile_name   # noqa: E402


def stitch(boxes, iom_threshold=0.5):
    """Merge boxes that overlap across a tile seam into their union."""
    boxes = sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    kept = []
    for b in boxes:
        merged = False
        for i, k in enumerate(kept):
            ix1, iy1 = max(b[0], k[0]), max(b[1], k[1])
            ix2, iy2 = min(b[2], k[2]), min(b[3], k[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            small = min((b[2] - b[0]) * (b[3] - b[1]), (k[2] - k[0]) * (k[3] - k[1]))
            if small and inter / small >= iom_threshold:
                kept[i] = (min(b[0], k[0]), min(b[1], k[1]),
                           max(b[2], k[2]), max(b[3], k[3]))
                merged = True
                break
        if not merged:
            kept.append(tuple(b))
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="e.g. dataset/val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=512)
    args = ap.parse_args()

    split = Path(args.split)
    per_frame = collections.defaultdict(list)
    tiles = 0
    for lbl in sorted((split / "labels").glob("*.txt")):
        name = parse_tile_name(lbl.stem)
        if name is None:
            continue
        tiles += 1
        ox, oy, frame = name.x, name.y, name.frame
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cx, cy, w, h = (float(v) * args.tile for v in parts[1:5])
            per_frame[frame].append((ox + cx - w / 2, oy + cy - h / 2,
                                     ox + cx + w / 2, oy + cy + h / 2))

    total = 0
    with open(args.out, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["frame", "x1", "y1", "x2", "y2"])
        for frame in sorted(per_frame):
            for b in stitch(per_frame[frame]):
                out.writerow([frame, *(int(round(v)) for v in b)])
                total += 1
    print(f"{tiles} tiles over {len(per_frame)} frames -> {total} stitched boxes")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
