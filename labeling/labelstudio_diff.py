"""Count label corrections for the README (rule 2): compare a Label Studio
JSON export against the original YOLO labels and report, per tile and in
total, how many boxes were added, removed, fixed and untouched.

Definitions, also worth quoting in the README:
- added:   in the export but matching no original box at IoU >= 0.5
- removed: in the originals but matching no exported box at IoU >= 0.5
- fixed:   matched a box at IoU >= 0.5 but below --same-iou (moved or resized)
- same:    matched at IoU >= --same-iou

    python labelstudio_diff.py --export export.json --labels dataset/train/labels
"""

import argparse
import collections
import json
import urllib.parse
from pathlib import Path


def yolo_boxes(label_path):
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                _, cx, cy, w, h = (float(v) for v in parts[:5])
                boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return boxes


def export_boxes(task):
    annotations = task.get("annotations") or []
    results = annotations[0]["result"] if annotations else \
        task.get("predictions", [{}])[0].get("result", [])
    boxes = []
    for r in results:
        if r.get("type") != "rectanglelabels":
            continue
        v = r["value"]
        x0, y0 = v["x"] / 100, v["y"] / 100
        boxes.append((x0, y0, x0 + v["width"] / 100, y0 + v["height"] / 100))
    return boxes


def stem_of(task):
    image = task["data"]["image"]
    path = urllib.parse.unquote(image.split("?d=")[-1])
    return Path(path).stem


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = ((a[2] - a[0]) * (a[3] - a[1]) +
            (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / area if area > 0 else 0.0


def greedy_match(old, new):
    pairs = sorted(((iou(o, n), i, j) for i, o in enumerate(old)
                    for j, n in enumerate(new)), reverse=True)
    used_old, used_new, matches = set(), set(), []
    for score, i, j in pairs:
        if score < 0.5 or i in used_old or j in used_new:
            continue
        used_old.add(i); used_new.add(j)
        matches.append((i, j, score))
    return matches, used_old, used_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="Label Studio JSON export")
    ap.add_argument("--labels", required=True, help="original YOLO labels dir")
    ap.add_argument("--same-iou", type=float, default=0.9,
                    help="IoU at or above which a box counts as untouched")
    args = ap.parse_args()

    tasks = json.loads(Path(args.export).read_text())
    labels_dir = Path(args.labels)
    totals = collections.Counter()
    for task in tasks:
        old = yolo_boxes(labels_dir / f"{stem_of(task)}.txt")
        new = export_boxes(task)
        matches, used_old, used_new = greedy_match(old, new)
        totals["added"] += len(new) - len(used_new)
        totals["removed"] += len(old) - len(used_old)
        totals["fixed"] += sum(1 for _, _, s in matches if s < args.same_iou)
        totals["same"] += sum(1 for _, _, s in matches if s >= args.same_iou)
        totals["tiles"] += 1

    print(f"tiles reviewed: {totals['tiles']}")
    print(f"added {totals['added']} / removed {totals['removed']} / "
          f"fixed {totals['fixed']} (IoU<{args.same_iou}) / "
          f"untouched {totals['same']}")


if __name__ == "__main__":
    main()
