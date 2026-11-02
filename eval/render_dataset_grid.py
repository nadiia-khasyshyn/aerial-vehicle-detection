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
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
            n += 1
    img = cv2.resize(img, (size, size))
    name = parse_tile_name(img_path.stem)
    caption = f"{name.clip if name else '?'}  {n} box" + ("" if n == 1 else "es")
    cv2.rectangle(img, (0, 0), (size, 26), (0, 0, 0), -1)
    cv2.putText(img, caption, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img



def count_boxes(label_path):
    if not label_path.exists():
        return 0
    return sum(1 for line in label_path.read_text().splitlines() if line.strip())


def per_clip_sheet(by_clip, args, out_dir):
    """One row per clip, tiles taken from frames spread across the timeline.

    Picking at random puts near-duplicates in the same row: a hovering drone
    gives many tiles at the same grid position seconds apart. Splitting each
    clip's frames into as many buckets as there are columns, and taking the
    busiest tile in each bucket, keeps a row showing four different moments.
    """
    if args.stems:
        lookup = {img.stem: (img, lbl) for items in by_clip.values()
                  for img, lbl in items}
        stems = Path(args.stems).read_text().split()
        cells = [draw_tile(*lookup[stem], args.size) for stem in stems]
        width = len(stems) // args.cols
        sheet = cv2.vconcat([cv2.hconcat(cells[i:i + width])
                             for i in range(0, len(cells), width)])
        out = out_dir / "dataset_by_clip.jpg"
        cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"-> {out}  {len(cells)} tiles from {args.stems}")
        return

    rows, chosen = [], []
    random.seed(args.seed)
    for clip in sorted(by_clip):
        items = [(img, lbl) for img, lbl in by_clip[clip]
                 if count_boxes(lbl) >= args.min_boxes]
        if len(items) < args.cols:
            continue
        items.sort(key=lambda t: parse_tile_name(t[0].stem).frame)
        frames = [parse_tile_name(img.stem).frame for img, _ in items]
        lo, hi = frames[0], frames[-1] or 1
        buckets = collections.defaultdict(list)
        for (img, lbl), frame in zip(items, frames):
            slot = min(args.cols - 1,
                       int((frame - lo) / max(hi - lo, 1) * args.cols))
            buckets[slot].append((img, lbl))
        picked, used_positions = [], set()
        for slot in range(args.cols):
            pool = buckets.get(slot) or items
            # Busiest tile first, but not one taken from a tile position already
            # used in this row: the same patch of road seconds apart looks like
            # a duplicate even when the frames are far apart.
            ranked = sorted(pool, key=lambda t: -count_boxes(t[1]))
            if args.pick == "random":
                ranked = random.sample(ranked, len(ranked))
            choice = next((t for t in ranked
                           if (parse_tile_name(t[0].stem).x,
                               parse_tile_name(t[0].stem).y) not in used_positions),
                          ranked[0])
            name = parse_tile_name(choice[0].stem)
            used_positions.add((name.x, name.y))
            picked.append(choice)
        cells = [draw_tile(img, lbl, args.size) for img, lbl in picked]
        cells = [c for c in cells if c is not None]
        if len(cells) == args.cols:
            rows.append(cells if args.transpose else cv2.hconcat(cells))
            chosen.append([img.stem for img, _ in picked])
            print(f"  {clip}: frames "
                  f"{[parse_tile_name(i.stem).frame for i, _ in picked]}")
    if not rows:
        raise SystemExit("no clip had enough tiles")
    order = (zip(*chosen) if args.transpose else chosen)
    (out_dir / "dataset_by_clip.txt").write_text(
        "\n".join(stem for row in order for stem in row) + "\n")
    if args.transpose:
        # rows[clip][slot] -> one output row per slot, clips side by side
        sheet = cv2.vconcat([cv2.hconcat([r[slot] for r in rows])
                             for slot in range(args.cols)])
    else:
        sheet = cv2.vconcat(rows)
    out = out_dir / "dataset_by_clip.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"-> {out}  {len(rows)} clips x {args.cols} tiles")


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
    ap.add_argument("--per-clip", action="store_true",
                    help="one row per clip instead of sheets that mix clips")
    ap.add_argument("--min-boxes", type=int, default=1,
                    help="--per-clip: skip tiles with fewer boxes than this")
    ap.add_argument("--pick", choices=["busiest", "random"], default="busiest",
                    help="--per-clip: take the busiest tile in each time bucket, "
                         "or a random one (honours --seed) to reroll the sheet")
    ap.add_argument("--stems", default=None,
                    help="--per-clip: render exactly the tile stems in this file, "
                         "reading order, instead of picking them")
    ap.add_argument("--transpose", action="store_true",
                    help="--per-clip: lay clips across the sheet instead of down, "
                         "which makes a wide, short strip rather than a tall grid")
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

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.per_clip:
        per_clip_sheet(by_clip, args, out_dir)
        return

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
