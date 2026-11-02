"""Keep only the predictions inside the labelled tile area.

Tile-built ground truth covers ~16% of each frame, so a full-frame run is
mostly looking at unlabelled ground: unrestricted it reports 1233 false
positives and 0.25 mAP where the restricted file gives 69 and 0.66.

    python eval/restrict_to_tiles.py --pred eval_run/predictions.csv \
        --tiles dataset/val/images --clip 14710575 \
        --out eval_run/predictions_in_tiles.csv
"""

import argparse
import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from avd import get, parse_tile_name   # noqa: E402

TILE = int(get("tiling.tile"))


def tile_boxes(tiles_dir, clip, tile):
    """Labelled rectangles per frame: a tile only covers its own frame.

    The labelled area moves from frame to frame, so the union of all tiles
    would keep boxes on ground a given frame never had labelled.
    """
    per_frame = collections.defaultdict(list)
    for path in sorted(Path(tiles_dir).glob("*.jpg")):
        name = parse_tile_name(path.stem)
        if name is None or (clip and name.clip != clip):
            continue
        per_frame[name.frame].append(
            (name.x, name.y, name.x + tile, name.y + tile))
    return per_frame


def inside(box, regions, mode):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    for x1, y1, x2, y2 in regions:
        if mode == "centre":
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return True
        elif box[0] >= x1 and box[1] >= y1 and box[2] <= x2 and box[3] <= y2:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--tiles", required=True,
                    help="the split's images/ directory, for the tile grid")
    ap.add_argument("--clip", default=None,
                    help="keep tiles of this clip only (a split may mix clips)")
    ap.add_argument("--tile", type=int, default=TILE)
    ap.add_argument("--mode", choices=["centre", "contained"], default="centre",
                    help="'centre' keeps a box whose centre is in a labelled "
                         "tile; 'contained' demands the whole box")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    per_frame = tile_boxes(args.tiles, args.clip, args.tile)
    if not per_frame:
        sys.exit(f"no tiles for clip {args.clip} in {args.tiles}")

    rows = list(csv.DictReader(open(args.pred)))
    fields = rows[0].keys() if rows else ["frame", "time_s", "conf",
                                          "x1", "y1", "x2", "y2"]
    kept = [r for r in rows
            if inside([float(r[k]) for k in ("x1", "y1", "x2", "y2")],
                      per_frame.get(int(r["frame"]), []), args.mode)]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(kept)

    tiles = sum(len(v) for v in per_frame.values())
    print(f"{out}  {len(kept)} of {len(rows)} predictions kept "
          f"({tiles} labelled tiles over {len(per_frame)} frames, "
          f"{args.mode} rule)")


if __name__ == "__main__":
    main()
