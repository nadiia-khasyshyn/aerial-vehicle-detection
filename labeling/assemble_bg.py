"""Assemble the background half of the dataset from several clips.

Two constraints decide what goes in. The first is provenance: background
from a single clip teaches a single scene, so each clip gets a quota rather
than whatever it happens to yield. The second is repetition: a drone that
hovers produces hundreds of near-identical tiles at the same grid position,
which inflate the count without adding anything, so each position gives up
only a handful of frames, spread across the clip's timeline.
"""

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

NAME_RE = re.compile(r"^(?P<vid>\d+)_f(?P<frame>\d+)_x(?P<x>\d+)_y(?P<y>\d+)")


def pick(paths, quota, per_pos):
    by_pos = defaultdict(list)
    for p in paths:
        m = NAME_RE.match(p.name)
        by_pos[(m["x"], m["y"])].append((int(m["frame"]), p))
    chosen = []
    for pos in sorted(by_pos):
        items = sorted(by_pos[pos])
        n = min(per_pos, len(items))
        if n:
            step = len(items) / n
            chosen += [items[int(i * step)][1] for i in range(n)]
    chosen.sort(key=lambda p: p.name)
    if len(chosen) <= quota:
        return chosen, len(by_pos)
    step = len(chosen) / quota
    return [chosen[int(i * step)] for i in range(quota)], len(by_pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_bg")
    ap.add_argument("--spec", nargs="+", required=True,
                    help="vid:quota:per_pos:dir[,dir...]")
    ap.add_argument("--reject", nargs="*", default=[],
                    help="files listing tile names a sieve rejected")
    args = ap.parse_args()

    root = Path(__file__).parent
    img = root / args.out / "images"
    lab = root / args.out / "labels"
    for d in (img, lab):
        d.mkdir(parents=True, exist_ok=True)

    rejected = set()
    for f in args.reject:
        fp = Path(f)
        if fp.exists():
            rejected |= set(fp.read_text().split())
    print(f"{len(rejected)} тайлів відкинуто ситами, не беруться")

    total = 0
    for s in args.spec:
        vid, quota, per_pos, dirs = s.split(":", 3)
        paths = []
        for d in dirs.split(","):
            paths += [q for q in sorted((root / d).glob("*.jpg"))
                      if q.name not in rejected]
        sel, n_pos = pick(paths, int(quota), int(per_pos))
        for p in sel:
            shutil.copy2(p, img / p.name)
            (lab / f"{p.stem}.txt").write_text("")
        total += len(sel)
        print(f"{vid}: з {len(paths)} доступних взято {len(sel)} "
              f"({n_pos} позицій сітки, не більше {per_pos} кадрів з позиції)")
    print(f"\nвсього {total} фонових тайлів -> {img}")


if __name__ == "__main__":
    main()
