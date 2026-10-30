"""Assemble the final YOLO dataset from the labelled tile folders.

Each source is an `images:labels` pair produced by the labelling pipeline
(`dataset_main`, `dataset_bg`, ... in the workspace). Tiles are linked, not
copied, so the final dataset costs no extra disk; `--copy` writes real files
instead, which is what you want before rsyncing it to a rented GPU.

The split is by clip, not by frame: a whole video is held out for validation
so neighbouring near-duplicate frames cannot sit on both sides.

    python training/build_dataset.py \
        --train dataset_main/images:dataset_main/labels \
                dataset_bg/images:dataset_bg/labels \
        --val   dataset_val2/images:dataset_val2/labels \
                dataset_val2_bg/images:dataset_val2_bg/labels \
        --out dataset
"""

import argparse
import collections
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from avd import parse_tile_name   # noqa: E402


def link_pair(images, labels, dest, copy):
    img_dir, lbl_dir = Path(images), Path(labels)
    n = 0
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        for src, sub in ((img, "images"), (lbl, "labels")):
            out = dest / sub / src.name
            if out.exists() or out.is_symlink():
                out.unlink()
            if copy:
                shutil.copyfile(src.resolve(), out)
            else:
                out.symlink_to(src.resolve())
        n += 1
    return n


def clip_counts(dest):
    counts = collections.Counter()
    for img in (dest / "images").iterdir():
        name = parse_tile_name(img.stem)
        counts[name.clip if name else "?"] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True, metavar="IMAGES:LABELS")
    ap.add_argument("--val", nargs="+", required=True, metavar="IMAGES:LABELS")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--copy", action="store_true",
                    help="copy tiles instead of symlinking them")
    args = ap.parse_args()

    out = Path(args.out)
    for split, sources in (("train", args.train), ("val", args.val)):
        for sub in ("images", "labels"):
            (out / split / sub).mkdir(parents=True, exist_ok=True)
        total = 0
        for pair in sources:
            images, _, labels = pair.partition(":")
            total += link_pair(images, labels, out / split, args.copy)
        counts = clip_counts(out / split)
        boxes = sum(1 for p in (out / split / "labels").iterdir()
                    for line in p.read_text().splitlines() if line.strip())
        empty = sum(1 for p in (out / split / "labels").iterdir()
                    if not p.read_text().strip())
        print(f"{split}: {total} tiles, {boxes} boxes, {empty} background tiles")
        print(f"  clips: {dict(counts)}")

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        "# Tiled vehicle detection, 512x512 tiles cut at native resolution.\n"
        f"path: {out.resolve()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "names:\n"
        "  0: vehicle\n")
    print(f"-> {data_yaml}")


if __name__ == "__main__":
    main()
