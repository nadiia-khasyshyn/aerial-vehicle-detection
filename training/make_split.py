"""Merge labelled tile sets into one dataset with a temporal train/val split.

The split is by frame index, not by random tile: consecutive frames show the
same vehicles in nearly the same places, so a random tile split would leak
most of the val set into training. Taking the last --val-frac of each video's
frame range as val keeps the two sets temporally disjoint while still covering
every scene.

Tile names are expected as <video>_f<frame>_x<ox>_y<oy>.jpg, which is what
labeling/build_dataset.py writes.
"""

import argparse
import collections
import re
import shutil
from pathlib import Path

STEM_RE = re.compile(r"^(?P<vid>\w+?)_f(?P<frame>\d+)_x\d+_y\d+$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True,
                    help="images_dir:labels_dir pairs, e.g. "
                         "dataset_v1/images:dataset_v1/labels_merged")
    ap.add_argument("--out", default="dataset_final")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--link", action="store_true",
                    help="symlink images instead of copying")
    args = ap.parse_args()

    root = Path.cwd()
    out = root / args.out
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Collect (image, label) pairs; a later source overrides an earlier one
    # for the same tile, so list sources oldest-first.
    pairs = {}
    for src in args.sources:
        img_dir, lab_dir = (root / p for p in src.split(":"))
        for img in sorted(img_dir.glob("*.jpg")):
            lab = lab_dir / f"{img.stem}.txt"
            if lab.exists():
                pairs[img.stem] = (img, lab)
    if not pairs:
        raise SystemExit("no image/label pairs found")

    # Frame threshold per video: frames at or above it go to val.
    frames = collections.defaultdict(set)
    for stem in pairs:
        m = STEM_RE.match(stem)
        if not m:
            raise SystemExit(f"tile name not understood: {stem}")
        frames[m["vid"]].add(int(m["frame"]))
    cut = {}
    for vid, fs in frames.items():
        fs = sorted(fs)
        cut[vid] = fs[int(len(fs) * (1 - args.val_frac))]

    counts = collections.Counter()
    boxes = collections.Counter()
    for stem, (img, lab) in sorted(pairs.items()):
        m = STEM_RE.match(stem)
        split = "val" if int(m["frame"]) >= cut[m["vid"]] else "train"
        dst_img = out / "images" / split / img.name
        if args.link:
            dst_img.unlink(missing_ok=True)
            dst_img.symlink_to(img.resolve())
        else:
            shutil.copy2(img, dst_img)
        shutil.copy2(lab, out / "labels" / split / lab.name)
        counts[split] += 1
        boxes[split] += sum(1 for l in open(lab) if l.strip())

    with open(out / "data.yaml", "w") as fh:
        fh.write(f"path: {out.resolve()}\n"
                 f"train: images/train\nval: images/val\n"
                 f"nc: 1\nnames:\n  0: vehicle\n")

    print(f"videos: {({v: f'val from frame {c}' for v, c in sorted(cut.items())})}")
    for split in ("train", "val"):
        print(f"{split}: {counts[split]} tiles, {boxes[split]} boxes")
    print(f"-> {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
