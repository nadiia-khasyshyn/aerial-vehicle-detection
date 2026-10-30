"""Cut training tiles around whole vehicles, with 90-degree rotations.

Written for low-altitude footage where a vehicle is a large share of the tile:
instead of a fixed grid, each tile is centred on a box, so the vehicle lands
whole rather than sliced by a seam. Boxes that the tile still clips are kept
only when most of them is inside, because a sliver teaches the wrong size.

Rotation by 90/180/270 is free diversity here: seen from above a scene has no
gravity direction, and rotating by right angles keeps boxes axis-aligned, so
no box is inflated the way an arbitrary angle would inflate it.

    python labeling/cut_tiles_from_gt.py --video clip.mp4 --gt gt.csv \
        --out dataset_lowalt --rotations 0 90 180 270
"""

import argparse
import collections
import csv
import random
from pathlib import Path

import cv2


def clip_boxes(boxes, ox, oy, tile, min_visible=0.6):
    """Boxes re-projected into the tile, keeping those mostly inside it."""
    kept = []
    for x1, y1, x2, y2 in boxes:
        ix1, iy1 = max(x1, ox), max(y1, oy)
        ix2, iy2 = min(x2, ox + tile), min(y2, oy + tile)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        visible = ((ix2 - ix1) * (iy2 - iy1)) / max(1.0, (x2 - x1) * (y2 - y1))
        if visible < min_visible:
            continue
        kept.append((ix1 - ox, iy1 - oy, ix2 - ox, iy2 - oy))
    return kept


def rotate(img, boxes, degrees, tile):
    if degrees == 0:
        return img, boxes
    turns = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
             270: cv2.ROTATE_90_COUNTERCLOCKWISE}[degrees]
    out = []
    for x1, y1, x2, y2 in boxes:
        if degrees == 90:
            out.append((tile - y2, x1, tile - y1, x2))
        elif degrees == 180:
            out.append((tile - x2, tile - y2, tile - x1, tile - y1))
        else:
            out.append((y1, tile - x2, y2, tile - x1))
    return cv2.rotate(img, turns), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--gt", required=True, help="csv with frame,x1,y1,x2,y2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--rotations", type=int, nargs="+", default=[0, 90, 180, 270])
    ap.add_argument("--min-visible", type=float, default=0.6,
                    help="a clipped box is kept when this share of it is inside")
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="crop a tile/zoom window and scale it up, so vehicles "
                         "land larger than in the source frame")
    ap.add_argument("--jitter", type=int, default=40,
                    help="random offset of the tile around the anchor box, px")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="report how many tiles would be written and stop")
    args = ap.parse_args()

    random.seed(args.seed)
    per_frame = collections.defaultdict(list)
    for r in csv.DictReader(open(args.gt)):
        per_frame[int(r["frame"])].append(
            (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])))

    out = Path(args.out)
    if not args.dry_run:
        for sub in ("images", "labels"):
            (out / sub).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stem = Path(args.video).stem.split("-")[0].split("_")[0]
    tiles = boxes_out = skipped = 0

    for frame_idx in sorted(per_frame):
        boxes = per_frame[frame_idx]
        img = None
        taken = set()
        for anchor in boxes:
            aw, ah = anchor[2] - anchor[0], anchor[3] - anchor[1]
            if aw > args.tile or ah > args.tile:
                skipped += 1          # vehicle larger than a tile: no whole view exists
                continue
            win = int(args.tile / args.zoom)
            if aw > win or ah > win:
                skipped += 1      # the anchor would not fit the zoom window
                continue
            cx, cy = (anchor[0] + anchor[2]) / 2, (anchor[1] + anchor[3]) / 2
            jx = random.randint(-args.jitter, args.jitter) if args.jitter else 0
            jy = random.randint(-args.jitter, args.jitter) if args.jitter else 0
            ox = int(min(max(0, cx - win / 2 + jx), max(0, W - win)))
            oy = int(min(max(0, cy - win / 2 + jy), max(0, H - win)))
            # one tile per grid cell keeps near-identical crops out of the set
            key = (ox // 64, oy // 64)
            if key in taken:
                continue
            taken.add(key)
            kept = [tuple(v * args.zoom for v in b)
                    for b in clip_boxes(boxes, ox, oy, win, args.min_visible)]
            if not kept:
                continue
            if args.dry_run:
                tiles += len(args.rotations)
                boxes_out += len(kept) * len(args.rotations)
                continue
            if img is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, img = cap.read()
                if not ok:
                    break
            crop = img[oy:oy + win, ox:ox + win]
            if args.zoom != 1.0 and crop.size:
                crop = cv2.resize(crop, (args.tile, args.tile),
                                  interpolation=cv2.INTER_CUBIC)
            if crop.shape[0] != args.tile or crop.shape[1] != args.tile:
                crop = cv2.copyMakeBorder(crop, 0, args.tile - crop.shape[0],
                                          0, args.tile - crop.shape[1],
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))
            for deg in args.rotations:
                rimg, rboxes = rotate(crop, kept, deg, args.tile)
                name = f"{stem}_f{frame_idx:05d}_x{ox:04d}_y{oy:04d}r{deg}"
                cv2.imwrite(str(out / "images" / f"{name}.jpg"), rimg,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                lines = []
                for x1, y1, x2, y2 in rboxes:
                    w, h = (x2 - x1) / args.tile, (y2 - y1) / args.tile
                    lines.append(f"0 {(x1 + x2) / 2 / args.tile:.6f} "
                                 f"{(y1 + y2) / 2 / args.tile:.6f} {w:.6f} {h:.6f}")
                (out / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n")
                tiles += 1
                boxes_out += len(rboxes)

    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb} {tiles} tiles with {boxes_out} boxes "
          f"({len(args.rotations)} rotations each)")
    if skipped:
        print(f"{skipped} anchors skipped: the vehicle is larger than one tile")


if __name__ == "__main__":
    main()
