"""Write YOLO labels for the tiles from whatever currently counts as accepted.

The labels pipeline_v1 wrote went stale the moment crops were moved by hand,
so they are rebuilt here from the same folders and coordinate tables the video
is rendered from. A tile with no accepted box still gets an empty file: that is
a tile stating it holds nothing, which a training set needs as much as a box.
"""

import argparse
import collections
import csv
import re
from pathlib import Path

NAME_RE = re.compile(r"^(?P<tile>\d+_f\d+_x\d+_y\d+)_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accepted", nargs="+", required=True)
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--images", default="dataset_v1/images")
    ap.add_argument("--out", default="dataset_v1/labels_low")
    ap.add_argument("--tile", type=int, default=512)
    args = ap.parse_args()

    root = Path.cwd()
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    kept = {p.name for d in args.accepted for p in (root / d).glob("*.jpg")}
    per_tile = collections.defaultdict(list)
    used = 0
    for r in (r for f in args.scores for r in csv.DictReader(open(root / f))):
        if r["crop"] not in kept:
            continue
        m = NAME_RE.match(r["crop"])
        if not m:
            continue
        x1, y1, x2, y2 = (int(r[k]) for k in ("x1", "y1", "x2", "y2"))
        per_tile[m["tile"]].append(
            f"0 {(x1 + x2) / 2 / args.tile:.6f} {(y1 + y2) / 2 / args.tile:.6f} "
            f"{(x2 - x1) / args.tile:.6f} {(y2 - y1) / args.tile:.6f}")
        used += 1

    for old in out.glob("*.txt"):
        old.unlink()
    tiles = sorted((root / args.images).glob("*.jpg"))
    empty = 0
    for t in tiles:
        lines = per_tile.get(t.stem, [])
        empty += not lines
        (out / f"{t.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    print(f"{len(kept)} accepted crops, {used} boxes over {len(per_tile)} tiles")
    print(f"{len(tiles)} label files written ({empty} empty) -> {out}")


if __name__ == "__main__":
    main()
