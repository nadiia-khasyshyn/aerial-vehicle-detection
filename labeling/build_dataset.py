"""Cut a 256 px tile dataset (YOLO format) from the detection CSVs.

Tiles overlap by default: on a flush 256 grid roughly a third of the boxes
straddle a grid line and 13.7% of them keep under 40% of the vehicle, because
the median box here is 36-69 px against a 256 px period. A 192 px step drops
that to 0.5%.

Writes two parallel trees:
  <out>/images, <out>/labels   the dataset itself
  <out>_vis/                   the same tiles with boxes drawn, for eyeballing
"""

import argparse
import csv
import collections
import random
from pathlib import Path

import cv2

VIDEOS = {
    "3405804": "3405804-uhd_3840_2160_30fps.mp4",
    "5382494": "5382494-uhd_3840_2160_24fps.mp4",
    "8457857": "8457857-uhd_3840_2160_24fps.mp4",
    "8968356": "8968356-hd_1920_1080_30fps.mp4",
}


def tile_origins(size, tile, stride):
    if size <= tile:
        return [0]
    xs = list(range(0, size - tile + 1, stride))
    if xs[-1] != size - tile:
        xs.append(size - tile)
    return xs


def load_boxes(csv_path, min_conf, min_box):
    """frame -> [(x1, y1, x2, y2)], confident and large enough to be real."""
    per_frame = collections.defaultdict(list)
    for r in csv.DictReader(open(csv_path)):
        if float(r["conf"]) < min_conf:
            continue
        x1, y1, x2, y2 = (int(r[k]) for k in ("x1", "y1", "x2", "y2"))
        if min(x2 - x1, y2 - y1) < min_box:
            continue
        per_frame[int(r["frame"])].append((x1, y1, x2, y2))
    return per_frame


def tile_boxes(boxes, ox, oy, tile, keep_frac):
    """Boxes clipped to one tile, dropping the fragments a cut leaves behind."""
    out = []
    for x1, y1, x2, y2 in boxes:
        ix1, iy1 = max(x1, ox), max(y1, oy)
        ix2, iy2 = min(x2, ox + tile), min(y2, oy + tile)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        # A sliver of a bonnet is not a vehicle; teaching the model otherwise
        # is worse than leaving the tile without that label.
        if (ix2 - ix1) * (iy2 - iy1) < keep_frac * (x2 - x1) * (y2 - y1):
            continue
        out.append((ix1 - ox, iy1 - oy, ix2 - ox, iy2 - oy))
    return out


def plan(per_frame, W, H, tile, stride, quota, rng):
    """Pick (frame, ox, oy) tiles, spread thin over as many frames as possible.

    Diversity is the scarce resource here, not tile count: the clips are
    17-31 s long, so taking every tile from a few frames would return the same
    vehicle dozens of times. Taking a few tiles from many frames does not.
    """
    xs, ys = tile_origins(W, tile, stride), tile_origins(H, tile, stride)
    occupied = {}
    for f, boxes in per_frame.items():
        tiles = [(ox, oy) for oy in ys for ox in xs
                 if tile_boxes(boxes, ox, oy, tile, 0.6)]
        if tiles:
            occupied[f] = tiles
    if not occupied:
        return []

    frames = sorted(occupied)
    per = max(1, -(-quota // len(frames)))
    chosen = []
    for f in frames:
        tiles = occupied[f]
        rng.shuffle(tiles)
        for ox, oy in tiles[:per]:
            chosen.append((f, ox, oy))
        if len(chosen) >= quota:
            break
    return chosen[:quota]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--det", default="output_c003/detections")
    p.add_argument("--out", default="dataset")
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--stride", type=int, default=192)
    p.add_argument("--total", type=int, default=5000)
    p.add_argument("--min-conf", type=float, default=0.10)
    p.add_argument("--min-box", type=int, default=8)
    p.add_argument("--keep-frac", type=float, default=0.6)
    p.add_argument("--videos", nargs="*", default=list(VIDEOS))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    root = Path.cwd()
    out = root / args.out
    vis = root / f"{args.out}_vis"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    vis.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    quota = args.total // len(args.videos)
    T = args.tile
    n_tiles = n_boxes = 0

    for name in args.videos:
        per_frame = load_boxes(root / args.det / f"{name}.csv",
                               args.min_conf, args.min_box)
        cap = cv2.VideoCapture(str(root / VIDEOS[name]))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        chosen = plan(per_frame, W, H, T, args.stride, quota, rng)
        by_frame = collections.defaultdict(list)
        for f, ox, oy in chosen:
            by_frame[f].append((ox, oy))

        wanted = sorted(by_frame)
        idx = -1
        made = 0
        for f in wanted:
            while idx < f:
                ok = cap.grab()
                idx += 1
                if not ok:
                    break
            ok, frame = cap.retrieve()
            if not ok:
                continue
            for ox, oy in by_frame[f]:
                boxes = tile_boxes(per_frame[f], ox, oy, T, args.keep_frac)
                if not boxes:
                    continue
                stem = f"{name}_f{f:05d}_x{ox:04d}_y{oy:04d}"
                crop = frame[oy:oy + T, ox:ox + T]
                cv2.imwrite(str(out / "images" / f"{stem}.jpg"), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                with open(out / "labels" / f"{stem}.txt", "w") as fh:
                    for x1, y1, x2, y2 in boxes:
                        fh.write(f"0 {(x1 + x2) / 2 / T:.6f} {(y1 + y2) / 2 / T:.6f} "
                                 f"{(x2 - x1) / T:.6f} {(y2 - y1) / T:.6f}\n")
                shown = crop.copy()
                for x1, y1, x2, y2 in boxes:
                    cv2.rectangle(shown, (x1, y1), (x2, y2), (255, 128, 0), 1)
                cv2.imwrite(str(vis / f"{stem}.jpg"), shown,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                made += 1
                n_boxes += len(boxes)
        cap.release()
        n_tiles += made
        print(f"  {name}: {made} tiles from {len(wanted)} frames", flush=True)

    with open(out / "data.yaml", "w") as fh:
        fh.write(f"path: {out}\ntrain: images\nval: images\nnc: 1\n"
                 f"names:\n  0: vehicle\n")

    print(f"\n{n_tiles} tiles, {n_boxes} boxes "
          f"({n_boxes / max(n_tiles, 1):.1f} per tile)")
    print(f"dataset: {out}\nvisualisation: {vis}")


if __name__ == "__main__":
    main()
