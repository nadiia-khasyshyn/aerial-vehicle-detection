"""Write corrected Label Studio annotations back into a YOLO labels dir.

Only reviewed tiles (tasks with a submitted annotation) are written; the rest
keep their current labels. When a label file is a symlink into the original
dataset, the symlink is replaced by a real file, so the pristine originals
survive for labelstudio_diff.py's added/removed/fixed count.

    python labelstudio_to_yolo.py --export export.json --labels dataset/train/labels
"""

import argparse
import json
import urllib.parse
from pathlib import Path


def task_stem(task):
    image = task["data"]["image"]
    return Path(urllib.parse.unquote(image.split("?d=")[-1])).stem


def yolo_lines(task):
    annotations = task.get("annotations") or []
    if not annotations:
        return None  # not reviewed: keep the current label file
    lines = []
    for r in annotations[0].get("result", []):
        if r.get("type") != "rectanglelabels":
            continue
        v = r["value"]
        w, h = v["width"] / 100, v["height"] / 100
        cx, cy = v["x"] / 100 + w / 2, v["y"] / 100 + h / 2
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="Label Studio JSON export")
    ap.add_argument("--labels", required=True, help="YOLO labels dir to update")
    args = ap.parse_args()

    labels_dir = Path(args.labels)
    tasks = json.loads(Path(args.export).read_text())
    written = skipped = 0
    for task in tasks:
        lines = yolo_lines(task)
        if lines is None:
            skipped += 1
            continue
        out = labels_dir / f"{task_stem(task)}.txt"
        if out.is_symlink():
            out.unlink()  # break the link; originals stay pristine
        out.write_text("\n".join(lines) + ("\n" if lines else ""))
        written += 1
    print(f"updated {written} label files ({skipped} tasks without "
          f"a submitted annotation left untouched)")


if __name__ == "__main__":
    main()
