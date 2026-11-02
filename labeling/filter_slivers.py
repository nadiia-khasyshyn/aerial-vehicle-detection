"""Drop sliver boxes from YOLO labels: anything thinner than --min-side px.

Tile cutting leaves slivers where a vehicle barely pokes into a tile; a few
pixels of car teach noisy regression and edge-texture false positives.
Substantially visible clipped vehicles are kept on purpose — removing them
would teach the model that a half-visible car is background, which kills
recall at tile seams.

    python filter_slivers.py --labels dataset/train/labels --tile 512 --min-side 7
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from avd import get   # noqa: E402

TILE = int(get("tiling.tile"))
MIN_SIDE = float(get("labels.min_box_side_px"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--tile", type=int, default=TILE, help="tile size in px")
    ap.add_argument("--min-side", type=float, default=MIN_SIDE,
                    help="drop boxes whose shorter side is below this, in px")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total = dropped = files_changed = 0
    for path in sorted(Path(args.labels).glob("*.txt")):
        kept, changed = [], False
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            total += 1
            w, h = float(parts[3]) * args.tile, float(parts[4]) * args.tile
            if min(w, h) < args.min_side:
                dropped += 1
                changed = True
                continue
            kept.append(line)
        if changed:
            files_changed += 1
            if not args.dry_run:
                path.write_text("\n".join(kept) + ("\n" if kept else ""))

    verb = "would drop" if args.dry_run else "dropped"
    print(f"{verb} {dropped} of {total} boxes (shorter side < {args.min_side}px) "
          f"across {files_changed} files")


if __name__ == "__main__":
    main()
